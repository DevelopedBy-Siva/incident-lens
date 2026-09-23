"""Control-plane maintenance commands for incident-processing data."""

import logging
import os
import shutil
from pathlib import Path

from app.control.models import Project
from app.data.models import Incident
from app.serving.models import ActionLog, Analysis, InvestigationRun
from app.shared.database import SessionLocal
from app.training.artifact_metadata import configured_artifact_metadata_writer
from app.training.dataset_storage import configured_dataset_storage
from app.training.models import Dataset, ModelArtifact, TrainingJob

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


def cleanup_project_data(db, project_id: str) -> dict[str, int]:
    """Delete incident-processing records owned by one project."""
    incident_ids = db.query(Incident.id).filter(Incident.project_id == project_id)
    counts = {
        "action_logs": db.query(ActionLog)
        .filter(ActionLog.project_id == project_id)
        .delete(synchronize_session=False),
        "investigation_runs": db.query(InvestigationRun)
        .filter(InvestigationRun.project_id == project_id)
        .delete(synchronize_session=False),
        "analyses": db.query(Analysis)
        .filter(Analysis.incident_id.in_(incident_ids))
        .delete(synchronize_session=False),
        "incidents": db.query(Incident)
        .filter(Incident.project_id == project_id)
        .delete(synchronize_session=False),
    }
    db.commit()
    return counts


def delete_project(db, project_id: str) -> dict[str, int]:
    """Delete a project, all owned database records, and local model data."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise LookupError("Project not found")

    dataset_keys = [
        row[0]
        for row in db.query(Dataset.storage_key)
        .filter(Dataset.project_id == project_id)
        .all()
    ]
    artifact_versions = [
        row[0]
        for row in db.query(ModelArtifact.artifact_version)
        .filter(ModelArtifact.project_id == project_id)
        .all()
    ]

    incident_ids = db.query(Incident.id).filter(Incident.project_id == project_id)
    counts = {
        "action_logs": db.query(ActionLog)
        .filter(ActionLog.project_id == project_id)
        .delete(synchronize_session=False),
        "investigation_runs": db.query(InvestigationRun)
        .filter(InvestigationRun.project_id == project_id)
        .delete(synchronize_session=False),
        "analyses": db.query(Analysis)
        .filter(Analysis.incident_id.in_(incident_ids))
        .delete(synchronize_session=False),
        "incidents": db.query(Incident)
        .filter(Incident.project_id == project_id)
        .delete(synchronize_session=False),
        "training_jobs": db.query(TrainingJob)
        .filter(TrainingJob.project_id == project_id)
        .delete(synchronize_session=False),
        "model_artifacts": db.query(ModelArtifact)
        .filter(ModelArtifact.project_id == project_id)
        .delete(synchronize_session=False),
        "datasets": db.query(Dataset)
        .filter(Dataset.project_id == project_id)
        .delete(synchronize_session=False),
    }
    db.delete(project)
    db.commit()

    dataset_storage = configured_dataset_storage()
    for storage_key in dataset_keys:
        dataset_storage.delete(storage_key)
    artifact_storage = configured_artifact_metadata_writer()
    for artifact_version in artifact_versions:
        artifact_storage.remove(project_id, artifact_version)

    # Also clear legacy local directories when filesystem storage is selected.
    if not os.getenv("S3_BUCKET", "").strip():
        _remove_project_directory(
            Path(os.getenv("ARTIFACT_STORAGE_PATH", "artifacts")), project_id
        )
        _remove_project_directory(
            Path(os.getenv("DATASET_STORAGE_PATH", "datasets")) / "projects",
            project_id,
        )
    return counts


def _remove_project_directory(root: Path, project_id: str) -> None:
    root = root.expanduser().resolve()
    target = (root / project_id).resolve()
    if root not in target.parents:
        raise ValueError("Project storage path escapes configured root")
    if target.exists():
        shutil.rmtree(target)
