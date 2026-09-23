import json
import logging
import time
from typing import Optional

from app.serving.local_model import (
    AdapterLoadError,
    BaseModelLoadError,
    LocalInferenceError,
)
from app.serving.model_runtime import get_model_runtime
from app.shared.observability import trace_operation

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 4
TOOL_LOG_LINES = 20
TOOL_INCIDENT_LIMIT = 8
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
                "note": f"Showing up to {TOOL_LOG_LINES} stored sample lines",
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
3. If timing of incidents matters, call get_incident_timeline
4. When you have enough evidence, produce your final JSON analysis

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


class InvestigationLoop:
    """
    Runs the multi-turn tool-calling investigation loop.
    Falls back to single-shot local analysis if tool-calling output is unusable.
    """

    def investigate(self, incident, project, evidence=None) -> Optional[object]:
        """
        Run the full investigation loop.

        Returns an IncidentAnalysis-compatible object, or None on failure.
        Always falls back to the standard decision_engine if the loop fails.
        """

        t0 = time.time()
        self._last_tool_calls = []
        self._last_iterations = 0
        self._last_fallback = False
        runtime_session = get_model_runtime().resolve_project_model(
            getattr(project, "id", None), project=project
        )

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
                with trace_operation(
                    "investigation_iteration",
                    plane="serving",
                    metadata={
                        "project_id": getattr(project, "id", None),
                        "incident_id": str(incident.id),
                        "base_model": getattr(
                            runtime_session,
                            "base_model",
                            runtime_session.default_model,
                        ),
                        "decision_source": "local_llm",
                    },
                ) as span:
                    response = runtime_session.complete_with_tools(
                        model=runtime_session.default_model,
                        messages=messages,
                        tools=TOOLS if iteration < MAX_ITERATIONS else None,
                        tool_choice="auto" if iteration < MAX_ITERATIONS else None,
                        temperature=0.2,
                        max_tokens=1500,
                    )
                    msg = response
                    span.metrics(
                        {
                            "iteration": iteration,
                            "message_count": len(messages),
                            "has_tool_calls": int(bool(msg.tool_calls)),
                        }
                    )

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
                                    "name": tc.name,
                                    "arguments": tc.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )

                for tc in msg.tool_calls:
                    args = {}
                    try:
                        args = json.loads(tc.arguments or "{}")
                    except json.JSONDecodeError:
                        pass

                    tool_result = executor.execute(tc.name, args)
                    tool_calls_made.append(
                        {
                            "tool": tc.name,
                            "args": args,
                        }
                    )

                    logger.info(
                        "[INVESTIGATOR] Tool called: %s(%s)",
                        tc.name,
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
            self._record_result(
                incident, project, analysis, tool_calls_made, elapsed_ms
            )

            logger.info(
                "[INVESTIGATOR] Done in %dms — %s/%s — %d tool calls",
                elapsed_ms,
                analysis.severity,
                analysis.disposition,
                len(tool_calls_made),
            )
            return analysis

        except (AdapterLoadError, BaseModelLoadError, LocalInferenceError) as e:
            logger.error("[INVESTIGATOR] Local model failure: %s", e)
            self._record_result(incident, project, None, tool_calls_made, 0, error=e)
            self._last_tool_calls = list(tool_calls_made)
            return None
        except Exception as e:
            logger.error("[INVESTIGATOR] Loop failed: %s — falling back", e)
            self._record_result(incident, project, None, tool_calls_made, 0, error=e)
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
        import re

        from app.serving.decision_engine import IncidentAnalysis, validate_analysis

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

    def _record_result(
        self,
        incident,
        project,
        analysis,
        tool_calls,
        elapsed_ms,
        error=None,
    ):
        with trace_operation(
            "investigation_result",
            plane="serving",
            metadata={
                "project_id": getattr(project, "id", None),
                "incident_id": str(incident.id),
                "source": incident.source,
                "decision_source": "local_llm",
                "policy_result": (
                    analysis.disposition if analysis is not None else "failed"
                ),
                "result": analysis.severity if analysis is not None else "error",
                "error_type": type(error).__name__ if error is not None else None,
            },
        ) as span:
            span.metrics(
                {
                    "elapsed_ms": elapsed_ms,
                    "tool_call_count": len(tool_calls),
                }
            )


_investigation_loop: Optional[InvestigationLoop] = None


def get_investigation_loop() -> InvestigationLoop:
    global _investigation_loop
    if _investigation_loop is None:
        _investigation_loop = InvestigationLoop()
    return _investigation_loop
