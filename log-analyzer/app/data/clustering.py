"""Cluster normalized log events into incidents."""

from datetime import datetime, timedelta

from app.data.models import Incident
from app.shared.database import SessionLocal


CLUSTER_WINDOW_MINUTES = 2
MAX_SAMPLES = 10


def cluster_log_db(project_id, source, environment, parsed_log, signature):
    db = SessionLocal()
    try:
        window_start = datetime.utcnow() - timedelta(minutes=CLUSTER_WINDOW_MINUTES)
        incident = (
            db.query(Incident)
            .filter(
                Incident.project_id == project_id,
                Incident.signature == signature,
                Incident.status == "open",
                Incident.last_seen >= window_start,
            )
            .order_by(Incident.last_seen.desc())
            .first()
        )
        if incident:
            incident.count += 1
            incident.last_seen = datetime.utcnow()
            if len(incident.sample_lines or []) < MAX_SAMPLES:
                lines = list(incident.sample_lines or [])
                lines.append(parsed_log.raw)
                incident.sample_lines = lines
            db.commit()
            db.refresh(incident)
            return incident, False

        new_incident = Incident(
            project_id=project_id,
            source=source,
            environment=environment,
            signature=signature,
            first_seen=datetime.utcnow(),
            last_seen=datetime.utcnow(),
            count=1,
            sample_lines=[parsed_log.raw],
            status="open",
        )
        db.add(new_incident)
        db.commit()
        db.refresh(new_incident)
        return new_incident, True
    finally:
        db.close()
