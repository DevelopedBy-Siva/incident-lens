import os
import time
from typing import Optional

from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.serving.model_runtime import get_model_runtime

load_dotenv()


def _make_langfuse(project=None):
    try:
        from langfuse import Langfuse

        public_key = (project.langfuse_public_key if project else None) or os.getenv(
            "LANGFUSE_PUBLIC_KEY", ""
        )
        secret_key = (project.langfuse_secret_key if project else None) or os.getenv(
            "LANGFUSE_SECRET_KEY", ""
        )
        host = (project.langfuse_host if project else None) or os.getenv(
            "LANGFUSE_HOST", "https://cloud.langfuse.com"
        )
        if not public_key or not secret_key:
            return None
        return Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    except ImportError:
        return None


class _NoOpTrace:
    def span(self, *a, **kw):
        return _NoOpSpan()

    def generation(self, *a, **kw):
        return _NoOpSpan()

    def update(self, *a, **kw):
        return self

    def end(self, *a, **kw):
        return self


class _NoOpSpan:
    def end(self, *a, **kw):
        return self

    def update(self, *a, **kw):
        return self


def _langfuse_usage_payload(response) -> Optional[dict]:
    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

    if not input_tokens and not output_tokens and not total_tokens:
        return None

    return {
        "input": input_tokens,
        "output": output_tokens,
        "total": total_tokens,
        "unit": "TOKENS",
    }


class IncidentAnalysis(BaseModel):
    severity: str = Field(description="Severity level: low, medium, high, or critical")
    disposition: str = Field(
        description="NO_ACTION, OBSERVE, NEEDS_DEV, NEEDS_ONCALL, or ESCALATE"
    )
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    summary: str = Field(description="2-3 sentence summary of the issue")
    suspected_root_cause: Optional[str] = Field(
        default=None,
        description="Short suspected root cause statement, or null if unclear",
    )
    next_steps: list[str] = Field(description="3-5 concrete action items")
    ticket_title: str = Field(description="Concise ticket title (max 100 chars)")
    ticket_body: str = Field(description="Detailed ticket description for developers")


class RootCauseResult(BaseModel):
    has_cause: bool = Field(description="True if an earlier incident caused this one")
    cause_incident_id: Optional[str] = Field(default=None)
    cause_explanation: Optional[str] = Field(default=None)
    confidence: float = Field(description="Confidence between 0.0 and 1.0")


def validate_analysis(analysis: IncidentAnalysis, incident) -> IncidentAnalysis:
    critical_patterns = [
        "outofmemoryerror",
        "heap space",
        "segmentation fault",
        "segfault",
        "core dumped",
        "fatal error",
        "stack overflow",
    ]
    high_patterns = [
        "database connection",
        "connection refused",
        "timeout",
        "null pointer",
        "exception",
        "failed to connect",
        "connection pool",
    ]

    sample_text = " ".join(incident.sample_lines or []).lower()
    full_text = f"{incident.signature.lower()} {sample_text}"

    if any(p in full_text for p in critical_patterns):
        if analysis.severity not in ["critical", "high"]:
            analysis.severity = "critical"
        if analysis.disposition not in ["ESCALATE", "NEEDS_ONCALL"]:
            analysis.disposition = "ESCALATE"
    elif any(p in full_text for p in high_patterns):
        if analysis.severity == "low":
            analysis.severity = "high"

    analysis.severity = analysis.severity.lower().strip()
    analysis.disposition = analysis.disposition.upper().strip()

    if analysis.severity == "critical" and analysis.disposition not in [
        "ESCALATE",
        "NEEDS_ONCALL",
    ]:
        analysis.disposition = "ESCALATE"
    if analysis.severity == "high" and analysis.disposition not in [
        "ESCALATE",
        "NEEDS_ONCALL",
        "NEEDS_DEV",
    ]:
        analysis.disposition = "NEEDS_ONCALL"
    if analysis.disposition == "ESCALATE" and analysis.severity not in [
        "critical",
        "high",
    ]:
        analysis.severity = "high"

    analysis.confidence = max(0.0, min(1.0, analysis.confidence))
    if analysis.suspected_root_cause is not None:
        analysis.suspected_root_cause = analysis.suspected_root_cause.strip() or None

    if not analysis.ticket_title or not analysis.ticket_title.strip():
        analysis.ticket_title = f"{incident.source} - {incident.signature[:50]}"
    if len(analysis.ticket_title) > 100:
        analysis.ticket_title = analysis.ticket_title[:97] + "..."

    return analysis


class DecisionEngine:
    def __init__(self):
        self.parser = PydanticOutputParser(pydantic_object=IncidentAnalysis)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert SRE analyzing production incidents.

You will receive:
1. Core incident metadata (source, environment, count, timestamps)
2. An evidence bundle containing:
   - Sample log lines from this incident
   - Other open incidents currently firing in the same system
   - A matched runbook with recommended steps (if found)
   - A known root cause link (if already established)

Use ALL of this evidence when deciding severity and disposition.
Key reasoning rules:
- If multiple related incidents are firing together, treat this as a potential cascade — raise severity
- If a runbook matched with high score, bias toward its disposition and steps
- If a root cause is already known, reflect that in the summary
- If count is low (< 3) and no related incidents, prefer OBSERVE over ESCALATE

Severity guidelines:
- CRITICAL: Service down, data loss, OutOfMemoryError, heap space, segfaults, fatal errors
- HIGH: Database connection errors, null pointer exceptions, major features broken
- MEDIUM: Feature partially broken, intermittent errors
- LOW: Minor issues, cosmetic, affects few users

Disposition guidelines:
- ESCALATE: CRITICAL/HIGH — page on-call immediately
- NEEDS_ONCALL: HIGH — notify on-call during business hours
- NEEDS_DEV: MEDIUM/HIGH — standard dev ticket
- OBSERVE: LOW/MEDIUM — monitor for patterns
- NO_ACTION: LOW — known noise

CRITICAL RULES:
- CRITICAL or HIGH severity → ESCALATE or NEEDS_ONCALL
- OutOfMemoryError, heap space, segfaults → ALWAYS CRITICAL + ESCALATE
- DB connection errors, NPE → ALWAYS HIGH minimum
- ALWAYS provide ticket_title

{format_instructions}""",
                ),
                (
                    "human",
                    """Analyze this incident:

Source: {source} | Environment: {environment}
Count: {count} | First seen: {first_seen} | Last seen: {last_seen}

--- EVIDENCE ---
{evidence_context}
--- END EVIDENCE ---

{format_instructions}""",
                ),
            ]
        )

        self.root_cause_parser = PydanticOutputParser(pydantic_object=RootCauseResult)
        self.root_cause_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert SRE specializing in distributed systems failure analysis.
Determine whether a NEW incident was caused by one of several EARLIER incidents.

Key principles:
- Infrastructure failures cascade: DB pool exhaustion → timeouts → NullPointerExceptions
- Typical propagation: 30s to 5 minutes between cause and effect
- Only assign a cause if technically plausible and clear
- Not every incident has a cause

{format_instructions}""",
                ),
                (
                    "human",
                    """NEW INCIDENT:
ID: {new_id} | Source: {new_source} | First seen: {new_first_seen}
Signature: {new_signature}
Sample: {new_sample_logs}

EARLIER INCIDENTS (oldest first):
{earlier_incidents}

Was the new incident caused by one of the earlier incidents?
- YES: has_cause=true, provide cause_incident_id and cause_explanation
- NO: has_cause=false, null the other fields
- cause_incident_id MUST be one of the IDs listed above

{format_instructions}""",
                ),
            ]
        )

    def analyze_incident(
        self,
        incident,
        project=None,
        evidence=None,
    ) -> Optional[IncidentAnalysis]:
        """
        Analyze an incident using the LLM.

        Args:
            incident: Incident ORM object
            project:  Project ORM object (for credentials)
            evidence: EvidenceBundle from app.data.evidence.build_evidence()
                      If None, falls back to incident.sample_lines[:3] (old behaviour)
        """
        t0 = time.time()
        lf = _make_langfuse(project)
        trace = (
            lf.trace(
                name="incident-analysis",
                metadata={
                    "incident_id": str(incident.id),
                    "source": incident.source,
                    "count": incident.count,
                    "has_evidence_bundle": evidence is not None,
                },
            )
            if lf
            else _NoOpTrace()
        )

        if evidence is not None:
            evidence_context = evidence.as_prompt_context()
        else:
            sample_logs = (
                "\n".join(incident.sample_lines[:3]) if incident.sample_lines else "N/A"
            )
            evidence_context = f"=== Sample log lines ===\n{sample_logs}"

        formatted = self.prompt.format_messages(
            format_instructions=self.parser.get_format_instructions(),
            source=incident.source,
            environment=incident.environment,
            count=incident.count,
            first_seen=incident.first_seen.strftime("%Y-%m-%d %H:%M:%S"),
            last_seen=incident.last_seen.strftime("%Y-%m-%d %H:%M:%S"),
            evidence_context=evidence_context,
        )

        runtime_session = get_model_runtime().resolve_project_model(
            getattr(project, "id", None), project=project
        )
        model_name = runtime_session.default_model

        try:
            gen = trace.generation(
                name="llm-analysis",
                model=model_name,
                model_parameters={"temperature": 0.3},
                input=evidence_context,
            )

            response = runtime_session.complete(
                model=model_name,
                messages=formatted,
                temperature=0.3,
            )
            elapsed_ms = int((time.time() - t0) * 1000)

            usage_payload = _langfuse_usage_payload(response)
            if usage_payload is None:
                gen.end(output=response.content)
            else:
                gen.end(output=response.content, usage=usage_payload)

            analysis = self.parser.parse(response.content)
            analysis = validate_analysis(analysis, incident)

            trace.update(
                metadata={
                    "severity": analysis.severity,
                    "disposition": analysis.disposition,
                    "latency_ms": elapsed_ms,
                    "analysis_source": "llm",
                    "model": model_name,
                    "evidence_related_count": (
                        len(evidence.related_incidents) if evidence else 0
                    ),
                }
            )
            return analysis
        except Exception as error:
            trace.update(metadata={"error": str(error)})
            print(f"[LLM] local analyze_incident failed: {error}")
            return None

    def chain_root_cause(
        self, new_incident, earlier_incidents, project=None
    ) -> Optional[RootCauseResult]:
        if not earlier_incidents:
            return None

        lf = _make_langfuse(project)
        trace = (
            lf.trace(
                name="root-cause-chaining",
                metadata={
                    "new_incident_id": str(new_incident.id),
                    "candidates": len(earlier_incidents),
                },
            )
            if lf
            else _NoOpTrace()
        )

        earlier_blocks = []
        for inc in earlier_incidents:
            sample = "\n  ".join((inc.sample_lines or [])[:2]) or "N/A"
            earlier_blocks.append(
                f"ID: {inc.id}\n  Source: {inc.source}\n"
                f"  First seen: {inc.first_seen.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"  Signature: {inc.signature}\n  Sample: {sample}"
            )

        formatted = self.root_cause_prompt.format_messages(
            format_instructions=self.root_cause_parser.get_format_instructions(),
            new_id=new_incident.id,
            new_source=new_incident.source,
            new_environment=new_incident.environment,
            new_first_seen=new_incident.first_seen.strftime("%Y-%m-%d %H:%M:%S"),
            new_signature=new_incident.signature,
            new_sample_logs="\n".join((new_incident.sample_lines or [])[:3]) or "N/A",
            earlier_incidents="\n\n".join(earlier_blocks),
        )

        runtime_session = get_model_runtime().resolve_project_model(
            getattr(project, "id", None), project=project
        )
        model_name = runtime_session.default_model

        try:
            gen = trace.generation(name="root-cause-llm", model=model_name)
            response = runtime_session.complete(
                model=model_name,
                messages=formatted,
                temperature=0.3,
            )
            gen.end(output=response.content)

            result = self.root_cause_parser.parse(response.content)
            valid_ids = {inc.id for inc in earlier_incidents}
            if result.has_cause and result.cause_incident_id not in valid_ids:
                print("[CHAIN] LLM returned invalid cause_incident_id — discarding")
                return None

            trace.update(
                metadata={
                    "has_cause": result.has_cause,
                    "confidence": result.confidence,
                    "model": model_name,
                }
            )
            return result
        except Exception as error:
            trace.update(metadata={"error": str(error)})
            print(f"[CHAIN] local chain_root_cause failed: {error}")
            return None


_decision_engine: Optional[DecisionEngine] = None


def get_decision_engine() -> DecisionEngine:
    global _decision_engine
    if _decision_engine is None:
        _decision_engine = DecisionEngine()
    return _decision_engine
