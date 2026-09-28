"""Persist a bounded incident stream. One signature maps to one recent incident."""

from datetime import datetime, timedelta

from app.data.models import Incident
from app.shared.database import SessionLocal


CLUSTER_WINDOW_MINUTES = 10
MAX_SAMPLES = 20


def cluster_log_db(project_id, source, environment, parsed_log, signature):
    now = datetime.utcnow()
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
                status="open",
            )
            db.add(incident)
            db.commit()
            db.refresh(incident)
            return incident, True

        samples = list(incident.sample_lines or [])
        if parsed_log.raw not in samples:
            incident.sample_lines = (samples + [parsed_log.raw])[-MAX_SAMPLES:]
        incident.count += 1
        incident.last_seen = now
        db.commit()
        db.refresh(incident)
        return incident, False
    finally:
        db.close()
