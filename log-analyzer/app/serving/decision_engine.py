"""Compatibility facade for the one incident-analysis contract."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.serving.triage import classify
from app.shared.incident_policy import DISPOSITIONS, SEVERITIES


class IncidentAnalysis(BaseModel):
    severity: str
    disposition: str
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    suspected_root_cause: Optional[str] = None
    next_steps: list[str]
    ticket_title: str
    ticket_body: str


def validate_analysis(analysis: IncidentAnalysis, incident) -> IncidentAnalysis:
    severity = analysis.severity.lower().strip()
    disposition = analysis.disposition.upper().strip()
    if severity not in SEVERITIES or disposition not in DISPOSITIONS:
        raise ValueError("Model returned an invalid severity or disposition")
    analysis.severity = severity
    analysis.disposition = disposition
    analysis.confidence = max(0.0, min(1.0, float(analysis.confidence)))
    analysis.next_steps = [step.strip() for step in analysis.next_steps if isinstance(step, str) and step.strip()]
    analysis.summary = analysis.summary.strip()
    analysis.ticket_title = analysis.ticket_title.strip()[:100]
    analysis.ticket_body = analysis.ticket_body.strip()
    return analysis


class DecisionEngine:
    """Keep callers working while delegating production analysis to InvestigationLoop."""

    def analyze_incident(self, incident, project=None, evidence=None) -> IncidentAnalysis:
        result = classify(incident, evidence)
        return IncidentAnalysis(
            severity=result.severity,
            disposition=result.disposition,
            confidence=result.confidence,
            summary=result.summary,
            suspected_root_cause=result.root_cause,
            next_steps=result.next_steps,
            ticket_title=result.ticket_title,
            ticket_body=result.summary,
        )

    def chain_root_cause(self, new_incident, earlier_incidents, project=None):
        return None


_engine: DecisionEngine | None = None


def get_decision_engine() -> DecisionEngine:
    global _engine
    if _engine is None:
        _engine = DecisionEngine()
    return _engine
