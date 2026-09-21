import json
import os
import time
import random
import logging
from typing import Optional

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 4
TOOL_LOG_LINES = 20
TOOL_INCIDENT_LIMIT = 8
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_recent_logs",
            "description": (
                "Fetch the most recent raw log lines for this incident's signature "
                "from the last N minutes. Use when you need more log context beyond "
                "the initial sample."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {
                        "type": "integer",
                        "description": "How many minutes back to fetch logs (max 60)",
                    }
                },
                "required": ["minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_related_incidents",
            "description": (
                "Get other open incidents in this project from the last N minutes. "
                "Use to detect cascades — if multiple services are failing together, "
                "raise severity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {
                        "type": "integer",
                        "description": "Look-back window in minutes (max 30)",
                    }
                },
                "required": ["minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_runbook",
            "description": (
                "Retrieve the full steps of a runbook by its id. "
                "Use when you have matched a runbook and need its detailed remediation steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "runbook_id": {
                        "type": "string",
                        "description": "The runbook id (e.g. 'db_connection_timeout')",
                    }
                },
                "required": ["runbook_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_incident_timeline",
            "description": (
                "Get a time-ordered list of all incidents in this project from the "
                "last N minutes. Use when you suspect a cascade and need to see "
                "which incident started first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {
                        "type": "integer",
                        "description": "Look-back window in minutes (max 30)",
                    }
                },
                "required": ["minutes"],
            },
        },
    },
]


class ToolExecutor:
    """Executes tool calls requested by the LLM."""

    def __init__(self, incident, project):
        self.incident = incident
        self.project = project

    def execute(self, tool_name: str, args: dict) -> str:
        """Dispatch tool call and return result as a JSON string."""
        try:
            if tool_name == "get_recent_logs":
                return self._get_recent_logs(int(args.get("minutes", 10)))
            elif tool_name == "get_related_incidents":
                return self._get_related_incidents(int(args.get("minutes", 15)))
            elif tool_name == "get_runbook":
                return self._get_runbook(args.get("runbook_id", ""))
            elif tool_name == "get_incident_timeline":
                return self._get_incident_timeline(int(args.get("minutes", 15)))
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as e:
            logger.warning("[INVESTIGATOR] Tool %s failed: %s", tool_name, e)
            return json.dumps({"error": str(e)})

    def _get_recent_logs(self, minutes: int) -> str:
        minutes = min(minutes, 60)
        lines = list(self.incident.sample_lines or [])[:TOOL_LOG_LINES]
        return json.dumps(
            {
                "incident_id": self.incident.id,
                "signature": self.incident.signature,
                "sample_count": len(lines),
                "lines": lines,
                "note": f"Showing up to {TOOL_LOG_LINES} stored sample lines (live Loki fetch available in Phase 5)",
            }
        )

    def _get_related_incidents(self, minutes: int) -> str:
        from datetime import datetime, timedelta
        from app.data.models import Incident
        from app.serving.models import Analysis
        from app.shared.database import SessionLocal

        minutes = min(minutes, 30)
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        db = SessionLocal()
        try:
            rows = (
                db.query(Incident)
                .filter(
                    Incident.project_id == self.project.id,
                    Incident.id != self.incident.id,
                    Incident.status == "open",
                    Incident.last_seen >= cutoff,
                )
                .order_by(Incident.last_seen.desc())
                .limit(TOOL_INCIDENT_LIMIT)
                .all()
            )
            result = []
            for row in rows:
                analysis = (
                    db.query(Analysis)
                    .filter(Analysis.incident_id == row.id)
                    .order_by(Analysis.created_at.desc())
                    .first()
                )
                result.append(
                    {
                        "id": row.id,
                        "source": row.source,
                        "signature": row.signature[:100],
                        "count": row.count,
                        "first_seen": row.first_seen.strftime("%H:%M:%S"),
                        "last_seen": row.last_seen.strftime("%H:%M:%S"),
                        "severity": analysis.severity if analysis else None,
                        "disposition": analysis.disposition if analysis else None,
                    }
                )
            return json.dumps({"window_minutes": minutes, "related": result})
        finally:
            db.close()

    def _get_runbook(self, runbook_id: str) -> str:
        from app.serving.runbook_loader import get_runbooks

        runbooks = get_runbooks()
        for rb in runbooks:
            if rb.id == runbook_id:
                return json.dumps(
                    {
                        "id": rb.id,
                        "name": rb.name,
                        "description": rb.description,
                        "default_severity": rb.default_severity,
                        "disposition": rb.disposition,
                        "steps": rb.steps,
                        "observe_threshold": rb.observe_threshold,
                    }
                )
        return json.dumps({"error": f"Runbook '{runbook_id}' not found"})

    def _get_incident_timeline(self, minutes: int) -> str:
        from datetime import datetime, timedelta
        from app.data.models import Incident
        from app.shared.database import SessionLocal

        minutes = min(minutes, 30)
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        db = SessionLocal()
        try:
            rows = (
                db.query(Incident)
                .filter(
                    Incident.project_id == self.project.id,
                    Incident.status == "open",
                    Incident.first_seen >= cutoff,
                )
                .order_by(Incident.first_seen.asc())
                .limit(20)
                .all()
            )
            timeline = [
                {
                    "id": row.id,
                    "source": row.source,
                    "signature": row.signature[:80],
                    "first_seen": row.first_seen.strftime("%H:%M:%S"),
                    "count": row.count,
                    "is_current": row.id == self.incident.id,
                }
                for row in rows
            ]
            return json.dumps({"window_minutes": minutes, "timeline": timeline})
        finally:
            db.close()


SYSTEM_PROMPT = """You are an expert SRE autonomous agent investigating a production incident.

You have access to tools to gather evidence before making your final diagnosis.
Use them strategically — you have at most {max_iter} rounds.

Investigation strategy:
1. If you see a DB/connection error, call get_related_incidents to detect cascades
2. If you need more log context, call get_recent_logs
3. If a runbook ID was suggested, call get_runbook for its steps
4. If timing of incidents matters, call get_incident_timeline
5. When you have enough evidence, produce your final JSON analysis

CRITICAL: After your investigation, you MUST output a JSON object (no markdown, no prose) with:
{{
  "severity": "low|medium|high|critical",
  "disposition": "NO_ACTION|OBSERVE|NEEDS_DEV|NEEDS_ONCALL|ESCALATE",
  "confidence": 0.0-1.0,
  "summary": "2-3 sentence summary",
  "suspected_root_cause": "short explanation of the most likely underlying cause, or null",
  "next_steps": ["step1", "step2", "step3"],
  "ticket_title": "concise title under 100 chars",
  "ticket_body": "detailed description for developers"
}}

Severity rules:
- CRITICAL: OOM, heap exhaustion, segfaults, service completely down
- HIGH: DB connection errors, NPE, major features broken, cascade detected
- MEDIUM: Partial degradation, intermittent errors
- LOW: Single occurrence, cosmetic, known noise

Disposition rules:
- ESCALATE: page on-call NOW (critical/high + cascades)
- NEEDS_ONCALL: notify on-call during business hours
- NEEDS_DEV: create a dev ticket
- OBSERVE: watch for recurrence
- NO_ACTION: known noise, ignore
"""

USER_PROMPT = """Investigate this incident:

Source: {source} | Environment: {environment}
Count: {count} | First seen: {first_seen} | Last seen: {last_seen}

Initial evidence:
{evidence_context}

Investigate using the available tools, then output your final JSON analysis."""


def _make_llm_client(project=None):
    """Return a raw Groq client (not LangChain) for tool-calling support."""
    try:
        from groq import Groq
    except ImportError:
        logger.error("[INVESTIGATOR] groq package not installed")
        return None, None

    key = (project.groq_api_key if project else None) or ""
    if not key:
        keys = [
            os.getenv(v, "").strip()
            for v in ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"]
            if os.getenv(v, "").strip()
        ]
        key = random.choice(keys) if keys else ""

    if not key:
        return None, None

    return Groq(api_key=key), _configured_groq_models()


def _configured_groq_models() -> list[str]:
    primary = os.getenv("GROQ_MODEL", "").strip()
    fallbacks_raw = os.getenv("GROQ_MODEL_FALLBACKS", "").strip()
    fallbacks = [m.strip() for m in fallbacks_raw.split(",") if m.strip()]

    models = []
    if primary:
        models.append(primary)
    models.append(DEFAULT_GROQ_MODEL)
    models.extend(fallbacks)

    deduped = []
    seen = set()
    for model in models:
        if model not in seen:
            deduped.append(model)
            seen.add(model)
    return deduped


def _is_rate_limit_error(error: Exception) -> bool:
    text = str(error).lower()
    return "429" in text or "rate limit" in text or "rate_limit_exceeded" in text


class InvestigationLoop:
    """
    Runs the multi-turn tool-calling investigation loop.
    Falls back to single-shot analysis if Groq tool-calling fails.
    """

    def investigate(self, incident, project, evidence=None) -> Optional[object]:
        """
        Run the full investigation loop.

        Returns an IncidentAnalysis-compatible object, or None on failure.
        Always falls back to the standard decision_engine if the loop fails.
        """
        from app.serving.decision_engine import (
            get_decision_engine,
            validate_analysis,
            IncidentAnalysis,
        )

        t0 = time.time()
        self._last_tool_calls = []
        self._last_iterations = 0
        self._last_fallback = False
        lf = self._make_langfuse(project)
        trace = self._start_trace(lf, incident)

        client, models = _make_llm_client(project)
        if not client:
            logger.info(
                "[INVESTIGATOR] No Groq client — falling back to decision_engine"
            )
            self._last_fallback = True
            return self._fallback(incident, project, evidence)

        executor = ToolExecutor(incident, project)

        evidence_context = (
            evidence.as_prompt_context()
            if evidence
            else ("\n".join(incident.sample_lines[:3] or ["(no logs)"]))
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(max_iter=MAX_ITERATIONS),
            },
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    source=incident.source,
                    environment=incident.environment,
                    count=incident.count,
                    first_seen=incident.first_seen.strftime("%Y-%m-%d %H:%M:%S"),
                    last_seen=incident.last_seen.strftime("%Y-%m-%d %H:%M:%S"),
                    evidence_context=evidence_context,
                ),
            },
        ]

        tool_calls_made = []
        final_text = None

        try:
            for iteration in range(MAX_ITERATIONS + 1):
                self._last_iterations = iteration
                span = self._start_span(trace, f"iteration-{iteration}", messages)

                response = None
                last_error = None
                for model_name in models:
                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            tools=TOOLS if iteration < MAX_ITERATIONS else None,
                            tool_choice="auto" if iteration < MAX_ITERATIONS else None,
                            temperature=0.2,
                            max_tokens=1500,
                        )
                        break
                    except Exception as e:
                        last_error = e
                        if _is_rate_limit_error(e) and model_name != models[-1]:
                            logger.warning(
                                "[INVESTIGATOR] Model %s rate-limited — trying fallback",
                                model_name,
                            )
                            continue
                        raise

                if response is None and last_error:
                    raise last_error

                msg = response.choices[0].message
                self._end_span(span, msg)

                if not msg.tool_calls:
                    final_text = msg.content or ""
                    logger.info(
                        "[INVESTIGATOR] Final answer after %d iterations, %d tool calls",
                        iteration,
                        len(tool_calls_made),
                    )
                    break

                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )

                for tc in msg.tool_calls:
                    args = {}
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        pass

                    tool_result = executor.execute(tc.function.name, args)
                    tool_calls_made.append(
                        {
                            "tool": tc.function.name,
                            "args": args,
                        }
                    )

                    logger.info(
                        "[INVESTIGATOR] Tool called: %s(%s)",
                        tc.function.name,
                        args,
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result,
                        }
                    )

                self._last_tool_calls = list(tool_calls_made)

            if not final_text:
                logger.warning("[INVESTIGATOR] No final text after loop — falling back")
                self._last_fallback = True
                return self._fallback(incident, project, evidence)

            analysis = self._parse_final(final_text, incident)
            if not analysis:
                self._last_fallback = True
                return self._fallback(incident, project, evidence)

            elapsed_ms = int((time.time() - t0) * 1000)
            self._update_trace(trace, analysis, tool_calls_made, elapsed_ms)

            logger.info(
                "[INVESTIGATOR] Done in %dms — %s/%s — %d tool calls",
                elapsed_ms,
                analysis.severity,
                analysis.disposition,
                len(tool_calls_made),
            )
            return analysis

        except Exception as e:
            logger.error("[INVESTIGATOR] Loop failed: %s — falling back", e)
            self._update_trace(trace, None, tool_calls_made, 0, error=str(e))
            self._last_tool_calls = list(tool_calls_made)
            self._last_fallback = True
            return self._fallback(incident, project, evidence)

    def _fallback(self, incident, project, evidence):
        """Fall back to the standard single-shot decision engine."""
        logger.info("[INVESTIGATOR] Using fallback decision_engine for %s", incident.id)
        from app.serving.decision_engine import get_decision_engine

        return get_decision_engine().analyze_incident(
            incident, project=project, evidence=evidence
        )

    def _parse_final(self, text: str, incident) -> Optional[object]:
        from app.serving.decision_engine import validate_analysis, IncidentAnalysis
        import re

        clean = re.sub(r"```(?:json)?", "", text).strip()

        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            logger.warning("[INVESTIGATOR] No JSON found in final response")
            return None

        try:
            data = json.loads(match.group())
            analysis = IncidentAnalysis(
                severity=data.get("severity", "medium"),
                disposition=data.get("disposition", "OBSERVE"),
                confidence=float(data.get("confidence", 0.6)),
                summary=data.get("summary", ""),
                suspected_root_cause=data.get("suspected_root_cause"),
                next_steps=data.get("next_steps", []),
                ticket_title=data.get("ticket_title", "")[:100],
                ticket_body=data.get("ticket_body", ""),
            )
            return validate_analysis(analysis, incident)
        except Exception as e:
            logger.warning(
                "[INVESTIGATOR] Failed to parse final JSON: %s | text: %s",
                e,
                text[:200],
            )
            return None

    def _make_langfuse(self, project):
        try:
            from langfuse import Langfuse

            pk = (project.langfuse_public_key if project else None) or os.getenv(
                "LANGFUSE_PUBLIC_KEY", ""
            )
            sk = (project.langfuse_secret_key if project else None) or os.getenv(
                "LANGFUSE_SECRET_KEY", ""
            )
            host = (project.langfuse_host if project else None) or os.getenv(
                "LANGFUSE_HOST", "https://cloud.langfuse.com"
            )
            if not pk or not sk:
                return None
            return Langfuse(public_key=pk, secret_key=sk, host=host)
        except Exception:
            return None

    def _start_trace(self, lf, incident):
        if not lf:
            return _NoOp()
        try:
            return lf.trace(
                name="investigation-loop",
                metadata={"incident_id": str(incident.id), "source": incident.source},
            )
        except Exception:
            return _NoOp()

    def _start_span(self, trace, name, messages):
        try:
            return trace.span(name=name, input={"message_count": len(messages)})
        except Exception:
            return _NoOp()

    def _end_span(self, span, msg):
        try:
            has_tools = bool(getattr(msg, "tool_calls", None))
            span.end(
                output={
                    "has_tool_calls": has_tools,
                    "content_len": len(msg.content or ""),
                }
            )
        except Exception:
            pass

    def _update_trace(self, trace, analysis, tool_calls, elapsed_ms, error=None):
        try:
            meta = {"tool_calls": tool_calls, "elapsed_ms": elapsed_ms}
            if analysis:
                meta.update(
                    {"severity": analysis.severity, "disposition": analysis.disposition}
                )
            if error:
                meta["error"] = error
            trace.update(metadata=meta)
        except Exception:
            pass


class _NoOp:
    def span(self, *a, **kw):
        return _NoOp()

    def generation(self, *a, **kw):
        return _NoOp()

    def update(self, *a, **kw):
        return self

    def end(self, *a, **kw):
        return self


_investigation_loop: Optional[InvestigationLoop] = None


def get_investigation_loop() -> InvestigationLoop:
    global _investigation_loop
    if _investigation_loop is None:
        _investigation_loop = InvestigationLoop()
    return _investigation_loop
