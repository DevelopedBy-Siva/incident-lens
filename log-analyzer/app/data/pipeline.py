"""Turn a chronological batch of log events into evidence-backed incidents."""

from __future__ import annotations

import logging
from datetime import datetime

from app.control.models import Project
from app.data.clustering import cluster_log_db
from app.data.parser import ParsedLog, resolve_source
from app.data.signatures import generate_signature
from app.shared.database import SessionLocal
from sqlalchemy.exc import IntegrityError


ACTIONABLE_LEVELS = {"WARN", "ERROR", "CRITICAL"}
REANALYSE_COUNTS = {2, 5, 10, 20, 50}


logger = logging.getLogger(__name__)


def process_log_batch(payload: dict) -> dict[str, int]:
    """Cluster all events first, then analyze each affected incident once.

    This preserves the evidence sequence collected during a poll/replay instead
    of issuing an analysis for the first line and losing its following symptoms.
    """
    from app.serving.orchestrator import analyze_incident

    project = payload.get("_project") or _project(payload["project_id"])
    if project is None:
        return {"incidents_created": 0, "incidents_updated": 0, "failed": 1}

    totals = {"incidents_created": 0, "incidents_updated": 0, "failed": 0}
    affected = {}
    processed_provider_ids: list[str] = []
    events = payload.get("events") or [
        {"message": raw} for raw in payload.get("logs", [])
    ]
    for event in events:
        try:
            raw = event.get("message") if isinstance(event, dict) else event
            provider_id = event.get("provider_id") if isinstance(event, dict) else None
            if provider_id and _provider_event_is_processed(project.id, str(provider_id)):
                logger.debug(
                    "[PIPELINE] project=%s provider_event=%s skipped=duplicate",
                    project.id,
                    provider_id,
                )
                continue
            parsed = ParsedLog(
                raw,
                observed_at=_event_timestamp(event),
                attributes=event.get("attributes") if isinstance(event, dict) else None,
            )
            if parsed.level not in ACTIONABLE_LEVELS:
                if provider_id:
                    processed_provider_ids.append(str(provider_id))
                logger.debug(
                    "[PIPELINE] project=%s provider_event=%s skipped=level:%s",
                    project.id,
                    provider_id or "local",
                    parsed.level,
                )
                continue
            source = resolve_source(parsed, payload.get("source", "unknown"))
            incident, created = cluster_log_db(
                project_id=project.id,
                source=source,
                environment=payload.get("environment", "prod"),
                parsed_log=parsed,
                signature=generate_signature(source, parsed),
            )
            totals["incidents_created" if created else "incidents_updated"] += 1
            if provider_id:
                processed_provider_ids.append(str(provider_id))
            affected[incident.id] = (incident, created or incident.count in REANALYSE_COUNTS)
            logger.info(
                "[PIPELINE] project=%s incident=%s event=%s persisted=%s count=%s",
                project.id,
                incident.id,
                provider_id or "local",
                "created" if created else "updated",
                incident.count,
            )
        except Exception:
            totals["failed"] += 1
            logger.exception(
                "[PIPELINE] project=%s event=%s stage=cluster failed",
                project.id,
                provider_id or "local",
            )

    try:
        missing_analyses = _incident_ids_without_analysis(affected)
    except Exception:
        # The analysis attempt itself reports a useful failure and can recover
        # through SQLAlchemy's connection handling. Do not make a best-effort
        # retry lookup prevent every affected incident from being analyzed.
        missing_analyses = set(affected)
    for incident_id, (incident, should_analyze) in affected.items():
        if not should_analyze and incident_id not in missing_analyses:
            continue
        try:
            logger.info(
                "[PIPELINE] project=%s incident=%s stage=analysis starting",
                project.id,
                incident_id,
            )
            analyze_incident(incident, project=project, force=True)
            logger.info(
                "[PIPELINE] project=%s incident=%s stage=analysis completed",
                project.id,
                incident_id,
            )
        except Exception:
            totals["failed"] += 1
            logger.exception(
                "[PIPELINE] project=%s incident=%s stage=analysis failed",
                project.id,
                incident_id,
            )

    # The prior implementation claimed provider IDs before analysis. A native
    # model failure then made the event permanently invisible after restart.
    # Commit the idempotency ledger only after the complete batch is durable.
    if totals["failed"] == 0:
        _mark_provider_events_processed(project.id, processed_provider_ids)
    else:
        logger.warning(
            "[PIPELINE] project=%s batch failed=%s; provider IDs left retryable",
            project.id,
            totals["failed"],
        )
    logger.info(
        "[PIPELINE] project=%s batch complete created=%s updated=%s failed=%s",
        project.id,
        totals["incidents_created"],
        totals["incidents_updated"],
        totals["failed"],
    )
    return totals


def process_pending_incidents(project, *, limit: int = 50) -> dict[str, int]:
    """Retry durable incidents that were interrupted before analysis persisted.

    This recovers rows created by older versions that wrote the provider-event
    ledger before invoking the model.  It also makes the recovery path visible
    in normal worker logs instead of requiring a manual database intervention.
    """
    from app.data.models import Incident
    from app.serving.models import Analysis
    from app.serving.orchestrator import analyze_incident

    db = SessionLocal()
    try:
        incidents = (
            db.query(Incident)
            .outerjoin(Analysis, Analysis.incident_id == Incident.id)
            .filter(
                Incident.project_id == project.id,
                Incident.status == "open",
                Analysis.id.is_(None),
            )
            .order_by(Incident.last_seen.asc())
            .limit(limit)
            .all()
        )
        for incident in incidents:
            db.expunge(incident)
    finally:
        db.close()

    totals = {"recovered": 0, "failed": 0}
    for incident in incidents:
        try:
            logger.info(
                "[RECOVERY] project=%s incident=%s stage=analysis starting",
                project.id,
                incident.id,
            )
            analyze_incident(incident, project=project, force=True)
            totals["recovered"] += 1
            logger.info(
                "[RECOVERY] project=%s incident=%s stage=analysis completed",
                project.id,
                incident.id,
            )
        except Exception:
            totals["failed"] += 1
            logger.exception(
                "[RECOVERY] project=%s incident=%s stage=analysis failed",
                project.id,
                incident.id,
            )
    return totals


def _incident_ids_without_analysis(affected: dict) -> set[str]:
    """Return affected incident IDs that need a retry after an interrupted analysis.

    This handles a retry after an interruption between incident persistence and
    analysis persistence, including records created before the recovery worker
    has completed its next polling cycle.
    """
    if not affected:
        return set()
    from app.serving.models import Analysis

    db = SessionLocal()
    try:
        incident_ids = list(affected)
        analysed = {
            incident_id
            for (incident_id,) in (
                db.query(Analysis.incident_id)
                .filter(Analysis.incident_id.in_(incident_ids))
                .all()
            )
        }
        return set(incident_ids) - analysed
    finally:
        db.close()


def _project(project_id: str):
    db = SessionLocal()
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            db.expunge(project)
        return project
    finally:
        db.close()


def _provider_event_is_processed(project_id: str, provider_id: str) -> bool:
    """Return whether a completed Datadog event is already in the ledger."""
    from app.data.models import IngestedLog

    db = SessionLocal()
    try:
        return (
            db.query(IngestedLog.id)
            .filter(
                IngestedLog.project_id == project_id,
                IngestedLog.provider_id == provider_id,
            )
            .first()
            is not None
        )
    finally:
        db.close()


def _mark_provider_events_processed(project_id: str, provider_ids: list[str]) -> None:
    """Record completed provider events without converting a race into failure."""
    if not provider_ids:
        return
    from app.data.models import IngestedLog

    db = SessionLocal()
    try:
        for provider_id in set(provider_ids):
            db.add(
                IngestedLog(
                    project_id=project_id,
                    provider_id=provider_id,
                    observed_at=datetime.utcnow(),
                )
            )
        db.commit()
        logger.debug(
            "[PIPELINE] project=%s provider_events=%s ledger=committed",
            project_id,
            len(set(provider_ids)),
        )
    except IntegrityError:
        # Another analyzer/process may have completed the same batch first.
        # The records are still safely deduplicated, so this is not a failure.
        db.rollback()
        logger.info(
            "[PIPELINE] project=%s provider event ledger already committed by another worker",
            project_id,
        )
    except Exception:
        db.rollback()
        logger.exception(
            "[PIPELINE] project=%s provider event ledger persistence failed",
            project_id,
        )
    finally:
        db.close()


def _event_timestamp(event) -> datetime | None:
    if not isinstance(event, dict) or not isinstance(event.get("timestamp"), str):
        return None
    try:
        return datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
    except ValueError:
        return None
