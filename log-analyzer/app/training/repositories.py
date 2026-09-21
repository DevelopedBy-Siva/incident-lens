import re
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.data.models import Incident
from app.serving.models import Analysis, ActionLog, InvestigationRun
from app.training.models import (
    Dataset,
    DatasetStatus,
    ModelArtifact,
    TrainingJob,
)

DATASET_VERSION_PATTERN = re.compile(r"^dataset-v(\d+)$")


@dataclass(frozen=True)
class DatasetSourceRecord:
    incident: Incident
    analysis: Analysis | None
    investigation: InvestigationRun | None
    action_log: ActionLog | None


class DatasetSourceRepository:
    """Read persisted incident facts used to construct training examples."""

    def __init__(self, db: Session):
        self.db = db

    def list_for_project(self, project_id: str) -> list[DatasetSourceRecord]:
        latest_analysis_id = (
            self.db.query(Analysis.id)
            .filter(Analysis.incident_id == Incident.id)
            .order_by(Analysis.created_at.desc(), Analysis.id.desc())
            .limit(1)
            .correlate(Incident)
            .scalar_subquery()
        )
        latest_investigation_id = (
            self.db.query(InvestigationRun.id)
            .filter(
                InvestigationRun.incident_id == Incident.id,
                InvestigationRun.project_id == project_id,
            )
            .order_by(InvestigationRun.started_at.desc(), InvestigationRun.id.desc())
            .limit(1)
            .correlate(Incident)
            .scalar_subquery()
        )
        latest_action_id = (
            self.db.query(ActionLog.id)
            .filter(
                ActionLog.incident_id == Incident.id,
                ActionLog.project_id == project_id,
            )
            .order_by(ActionLog.actioned_at.desc(), ActionLog.id.desc())
            .limit(1)
            .correlate(Incident)
            .scalar_subquery()
        )

        rows = (
            self.db.query(Incident, Analysis, InvestigationRun, ActionLog)
            .outerjoin(Analysis, Analysis.id == latest_analysis_id)
            .outerjoin(InvestigationRun, InvestigationRun.id == latest_investigation_id)
            .outerjoin(ActionLog, ActionLog.id == latest_action_id)
            .filter(Incident.project_id == project_id)
            .order_by(Incident.first_seen.asc(), Incident.id.asc())
            .all()
        )
        return [DatasetSourceRecord(*row) for row in rows]


class DatasetRepository:
    """Persistence operations for immutable dataset metadata."""

    def __init__(self, db: Session):
        self.db = db

    def add(self, dataset: Dataset) -> Dataset:
        self.db.add(dataset)
        return dataset

    def get_for_project(self, dataset_id: str, project_id: str) -> Dataset | None:
        return (
            self.db.query(Dataset)
            .filter(Dataset.id == dataset_id, Dataset.project_id == project_id)
            .first()
        )

    def list_for_project(self, project_id: str) -> list[Dataset]:
        return (
            self.db.query(Dataset)
            .filter(Dataset.project_id == project_id)
            .order_by(Dataset.created_at.desc())
            .all()
        )

    def next_version(self, project_id: str) -> str:
        versions = (
            self.db.query(Dataset.dataset_version)
            .filter(Dataset.project_id == project_id)
            .all()
        )
        version_numbers = [
            int(match.group(1))
            for (version,) in versions
            if (match := DATASET_VERSION_PATTERN.fullmatch(version))
        ]
        return f"dataset-v{max(version_numbers, default=0) + 1}"

    def set_status(
        self,
        dataset: Dataset,
        status: DatasetStatus,
        record_count: int | None = None,
    ) -> Dataset:
        dataset.status = status
        if record_count is not None:
            dataset.record_count = record_count
        return dataset


class TrainingJobRepository:
    """Persistence operations for training job metadata."""

    def __init__(self, db: Session):
        self.db = db

    def add(self, job: TrainingJob) -> TrainingJob:
        self.db.add(job)
        return job

    def get_for_project(self, job_id: str, project_id: str) -> TrainingJob | None:
        return (
            self.db.query(TrainingJob)
            .filter(TrainingJob.id == job_id, TrainingJob.project_id == project_id)
            .first()
        )

    def list_for_project(self, project_id: str) -> list[TrainingJob]:
        return (
            self.db.query(TrainingJob)
            .filter(TrainingJob.project_id == project_id)
            .order_by(TrainingJob.created_at.desc())
            .all()
        )


class ModelArtifactRepository:
    """Persistence operations for model artifact metadata."""

    def __init__(self, db: Session):
        self.db = db

    def add(self, artifact: ModelArtifact) -> ModelArtifact:
        self.db.add(artifact)
        return artifact

    def get_for_project(
        self, artifact_id: str, project_id: str
    ) -> ModelArtifact | None:
        return (
            self.db.query(ModelArtifact)
            .filter(
                ModelArtifact.id == artifact_id,
                ModelArtifact.project_id == project_id,
            )
            .first()
        )

    def list_for_project(self, project_id: str) -> list[ModelArtifact]:
        return (
            self.db.query(ModelArtifact)
            .filter(ModelArtifact.project_id == project_id)
            .order_by(ModelArtifact.created_at.desc())
            .all()
        )
