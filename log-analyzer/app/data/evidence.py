import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from app.shared.observability import trace_operation

logger = logging.getLogger(__name__)

MAX_INCIDENT_SAMPLES = 8

MAX_RELATED_INCIDENTS = 5
RELATED_INCIDENT_WINDOW_MINUTES = 15

LOG_LINE_MAX_CHARS = 300


@dataclass
class RelatedIncident:
    id: str
    signature: str
    source: str
    count: int
    first_seen: datetime
    severity: Optional[str] = None
    disposition: Optional[str] = None
    ticket_title: Optional[str] = None


@dataclass
class EvidenceBundle:
    sample_lines: list[str] = field(default_factory=list)

    related_incidents: list[RelatedIncident] = field(default_factory=list)

    runbook_id: Optional[str] = None
    runbook_name: Optional[str] = None
    runbook_steps: list[str] = field(default_factory=list)
    runbook_score: float = 0.0
    runbook_selection_source: Optional[str] = None
    candidate_runbooks: list[str] = field(default_factory=list)

    root_cause_id: Optional[str] = None
    root_cause_signature: Optional[str] = None
    root_cause_explanation: Optional[str] = None

    gathered_at: datetime = field(default_factory=datetime.utcnow)

    def as_prompt_context(self) -> str:
        """
        Renders the bundle as a structured string ready to be injected
        into the LLM prompt. Keeps formatting dense but readable.
        """
        parts: list[str] = []

        parts.append("=== Sample log lines ===")
        if self.sample_lines:
            for line in self.sample_lines:
                parts.append(f"  {line[:LOG_LINE_MAX_CHARS]}")
        else:
            parts.append("  (none)")

        parts.append("\n=== Related open incidents (same project, last 15min) ===")
        if self.related_incidents:
            for ri in self.related_incidents:
                sev_disp = ""
                if ri.severity or ri.disposition:
                    sev_disp = f" [{ri.severity or '?'}/{ri.disposition or '?'}]"
                parts.append(
                    f"  • [{ri.source}] {ri.signature[:80]}{sev_disp}"
                    f" — count={ri.count} first_seen={ri.first_seen.strftime('%H:%M:%S')}"
                )
        else:
            parts.append("  (none)")

        if self.root_cause_id:
            parts.append("\n=== Known root cause ===")
            parts.append(f"  Incident: {self.root_cause_id}")
            parts.append(f"  Signature: {self.root_cause_signature}")
            if self.root_cause_explanation:
                parts.append(f"  Explanation: {self.root_cause_explanation}")

        return "\n".join(parts)


def build_evidence(incident, project) -> EvidenceBundle:
    """
    Gather all available evidence for an incident.

    Args:
        incident:  Incident ORM object
        project:   Project ORM object (used for scoping queries)

    Returns:
        EvidenceBundle — always returns something, never raises.
    """
    with trace_operation(
        "evidence_generation",
        plane="data",
        metadata={
            "project_id": getattr(project, "id", None),
            "incident_id": str(incident.id),
            "source": incident.source,
        },
    ) as span:
        bundle = _build_evidence(incident, project)
        span.metrics(
            {
                "sample_count": len(bundle.sample_lines),
                "related_incident_count": len(bundle.related_incidents),
            }
        )
        span.tag("result", "completed")
        return bundle


def _build_evidence(incident, project) -> EvidenceBundle:
    bundle = EvidenceBundle()

    try:
        raw_lines = incident.sample_lines or []
        bundle.sample_lines = [
            line[:LOG_LINE_MAX_CHARS] for line in raw_lines[:MAX_INCIDENT_SAMPLES]
        ]
    except Exception as e:
        logger.warning("[EVIDENCE] Failed to gather sample lines: %s", e)

    try:
        bundle.related_incidents = _fetch_related_incidents(incident, project)
    except Exception as e:
        logger.warning("[EVIDENCE] Failed to fetch related incidents: %s", e)

    try:
        if incident.root_cause_incident_id:
            bundle.root_cause_id = incident.root_cause_incident_id
            bundle.root_cause_explanation = incident.cause_explanation or ""
            root_sig = _fetch_incident_signature(incident.root_cause_incident_id)
            bundle.root_cause_signature = root_sig
    except Exception as e:
        logger.warning("[EVIDENCE] Failed to fetch root cause: %s", e)

    logger.info(
        "[EVIDENCE] Built bundle for %s — %d samples, %d related",
        incident.id,
        len(bundle.sample_lines),
        len(bundle.related_incidents),
    )

    return bundle


def _fetch_related_incidents(incident, project) -> list[RelatedIncident]:
    from app.data.models import Incident
    from app.serving.models import Analysis
    from app.shared.database import SessionLocal
    from sqlalchemy import func

    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(minutes=RELATED_INCIDENT_WINDOW_MINUTES)
        rows = (
            db.query(Incident)
            .filter(
                Incident.project_id == project.id,
                Incident.id != incident.id,
                Incident.status == "open",
                Incident.last_seen >= cutoff,
            )
            .order_by(Incident.last_seen.desc())
            .limit(MAX_RELATED_INCIDENTS)
            .all()
        )

        related = []
        for row in rows:
            analysis = (
                db.query(Analysis)
                .filter(Analysis.incident_id == row.id)
                .order_by(Analysis.created_at.desc())
                .first()
            )
            related.append(
                RelatedIncident(
                    id=row.id,
                    signature=row.signature,
                    source=row.source,
                    count=row.count,
                    first_seen=row.first_seen,
                    severity=analysis.severity if analysis else None,
                    disposition=analysis.disposition if analysis else None,
                    ticket_title=analysis.ticket_title if analysis else None,
                )
            )
        return related
    finally:
        db.close()


def _fetch_incident_signature(incident_id: str) -> Optional[str]:
    from app.data.models import Incident
    from app.shared.database import SessionLocal

    db = SessionLocal()
    try:
        row = db.query(Incident).filter(Incident.id == incident_id).first()
        return row.signature if row else None
    finally:
        db.close()
