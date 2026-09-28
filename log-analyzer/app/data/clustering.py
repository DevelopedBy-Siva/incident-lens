"""Persist a bounded incident stream. One signature maps to one recent incident."""

from datetime import datetime, timedelta, timezone

from app.data.models import Incident
from app.shared.database import SessionLocal


CLUSTER_WINDOW_MINUTES = 10
MAX_SAMPLES = 20


def cluster_log_db(project_id, source, environment, parsed_log, signature):
    """Append an event to its service/entity incident window using event time."""
    observed_at = parsed_log.timestamp
    now = (
        observed_at.astimezone(timezone.utc).replace(tzinfo=None)
        if getattr(observed_at, "tzinfo", None)
        else observed_at
    )
    db = SessionLocal()
    try:
        incident = (
            db.query(Incident)
            .filter(
                Incident.project_id == project_id,
                Incident.source == source,
                Incident.environment == environment,
                Incident.signature == signature,
                Incident.status == "open",
                Incident.last_seen >= now - timedelta(minutes=CLUSTER_WINDOW_MINUTES),
            )
            .order_by(Incident.last_seen.desc())
            .first()
        )
        if incident is None:
            incident = Incident(
                project_id=project_id,
                source=source,
                environment=environment,
                signature=signature,
                first_seen=now,
                last_seen=now,
                count=1,
                sample_lines=[parsed_log.raw],
                auto_tags=_observed_tags(parsed_log),
                status="open",
            )
            db.add(incident)
            db.commit()
            db.refresh(incident)
            return incident, True

        samples = list(incident.sample_lines or [])
        if parsed_log.raw not in samples:
            incident.sample_lines = (samples + [parsed_log.raw])[-MAX_SAMPLES:]
        incident.auto_tags = {**(incident.auto_tags or {}), **_observed_tags(parsed_log)}
        incident.count += 1
        incident.last_seen = now
        db.commit()
        db.refresh(incident)
        return incident, False
    finally:
        db.close()


def _observed_tags(parsed_log) -> dict[str, str]:
    allowed = {
        "host", "hostname", "pod", "container", "namespace", "region", "zone",
        "domain", "endpoint", "topic", "partition", "deployment_version", "trace_id",
        "request_id", "peer", "dependency",
    }
    attributes = getattr(parsed_log, "attributes", {}) or {}
    tags = {
        key: str(value)
        for key, value in attributes.items()
        if key in allowed and value not in (None, "", [], {})
    }
    if getattr(parsed_log, "entity", None):
        tags["pod" if "-" in parsed_log.entity else "host"] = parsed_log.entity
    return tags
