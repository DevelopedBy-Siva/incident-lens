"""Incident-state operations shared by ingestion and serving workflows."""

from datetime import datetime, timedelta
import logging

from app.data.models import Incident
from app.shared.database import SessionLocal


logger = logging.getLogger(__name__)

ROOT_CAUSE_LOOKBACK_MINUTES = 10


def fetch_recent_root_cause_candidates(project_id, new_incident_id):
    cutoff = datetime.utcnow() - timedelta(minutes=ROOT_CAUSE_LOOKBACK_MINUTES)
    db = SessionLocal()
    try:
        candidates = (
            db.query(Incident)
            .filter(
                Incident.project_id == project_id,
                Incident.id != new_incident_id,
                Incident.status == "open",
                Incident.first_seen >= cutoff,
                Incident.root_cause_incident_id == None,
            )
            .order_by(Incident.first_seen.asc())
            .all()
        )
        for candidate in candidates:
            db.expunge(candidate)
        return candidates
    finally:
        db.close()


def apply_root_cause_chain(new_incident_id, cause_incident_id, explanation):
    db = SessionLocal()
    try:
        incident = db.query(Incident).filter(Incident.id == new_incident_id).first()
        if incident:
            incident.root_cause_incident_id = cause_incident_id
            incident.cause_explanation = explanation
            db.commit()
            print(f"[CHAIN] {new_incident_id} → {cause_incident_id}")
    except Exception as e:
        db.rollback()
        print(f"[CHAIN] Failed: {e}")
    finally:
        db.close()


def stamp_actioned(incident_id: str):
    db = SessionLocal()
    try:
        row = db.query(Incident).filter(Incident.id == incident_id).first()
        if row and hasattr(row, "last_actioned_at"):
            row.last_actioned_at = datetime.utcnow()
            db.commit()
    except Exception as e:
        db.rollback()
        logger.warning("[TASKS] stamp_actioned failed for %s: %s", incident_id, e)
    finally:
        db.close()
