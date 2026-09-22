import time
from typing import Optional

from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.serving.model_runtime import get_model_runtime
from app.shared.observability import trace_operation

load_dotenv()


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

        with trace_operation(
            "decision_generation",
            plane="serving",
            metadata={
                "project_id": getattr(project, "id", None),
                "incident_id": str(incident.id),
                "source": incident.source,
                "decision_source": "local_llm",
            },
        ) as trace:
            runtime_session = get_model_runtime().resolve_project_model(
                getattr(project, "id", None), project=project
            )
            model_name = runtime_session.default_model
            trace.tags(
                {
                    "base_model": getattr(runtime_session, "base_model", model_name),
                    "adapter_version": (
                        runtime_session.active_artifact.version
                        if getattr(runtime_session, "active_artifact", None)
                        else None
                    ),
                }
            )

            try:
                response = runtime_session.complete(
                    model=model_name,
                    messages=formatted,
                    temperature=0.3,
                )
                elapsed_ms = int((time.time() - t0) * 1000)

                with trace_operation(
                    "decision_validation",
                    plane="serving",
                    metadata={
                        "project_id": getattr(project, "id", None),
                        "incident_id": str(incident.id),
                        "decision_source": "local_llm",
                    },
                ):
                    analysis = self.parser.parse(response.content)
                    analysis = validate_analysis(analysis, incident)

                trace.tags(
                    {
                        "policy_result": analysis.disposition,
                        "result": analysis.severity,
                    }
                )
                trace.metrics({"inference_latency_ms": elapsed_ms})
                return analysis
            except Exception as error:
                trace.tag("error_type", type(error).__name__)
                print(f"[LLM] local analyze_incident failed: {error}")
                return None

    def chain_root_cause(
        self, new_incident, earlier_incidents, project=None
    ) -> Optional[RootCauseResult]:
        if not earlier_incidents:
            return None

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

        with trace_operation(
            "root_cause_chaining",
            plane="serving",
            metadata={
                "project_id": getattr(project, "id", None),
                "incident_id": str(new_incident.id),
                "candidate_count": len(earlier_incidents),
                "decision_source": "local_llm",
            },
        ) as trace:
            runtime_session = get_model_runtime().resolve_project_model(
                getattr(project, "id", None), project=project
            )
            model_name = runtime_session.default_model

            try:
                response = runtime_session.complete(
                    model=model_name,
                    messages=formatted,
                    temperature=0.3,
                )
                result = self.root_cause_parser.parse(response.content)
                valid_ids = {inc.id for inc in earlier_incidents}
                if result.has_cause and result.cause_incident_id not in valid_ids:
                    print("[CHAIN] LLM returned invalid cause_incident_id — discarding")
                    return None

                trace.tag("result", "cause_found" if result.has_cause else "no_cause")
                return result
            except Exception as error:
                trace.tag("error_type", type(error).__name__)
                print(f"[CHAIN] local chain_root_cause failed: {error}")
                return None


_decision_engine: Optional[DecisionEngine] = None


def get_decision_engine() -> DecisionEngine:
    global _decision_engine
    if _decision_engine is None:
        _decision_engine = DecisionEngine()
    return _decision_engine
