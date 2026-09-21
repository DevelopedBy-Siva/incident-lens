"""Control-plane maintenance commands for incident-processing data."""
from app.data.models import Incident
from app.serving.models import Analysis, ActionLog, InvestigationRun
from app.shared.database import SessionLocal
import logging

logger = logging.getLogger(__name__)


def cleanup_all_data():
    """Empty incident-processing tables used for demos and local reset flows."""
    db = SessionLocal()

    try:
        db.query(ActionLog).delete()
        db.query(InvestigationRun).delete()
        db.query(Analysis).delete()
        db.query(Incident).delete()

        db.commit()
        logger.info("All incident-processing data deleted")

    except Exception as e:
        db.rollback()
        logger.error("Cleanup failed: %s", e)

    finally:
        db.close()
