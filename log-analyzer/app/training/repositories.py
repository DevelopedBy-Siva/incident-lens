from sqlalchemy.orm import Session

from app.training.models import Dataset, ModelArtifact, TrainingJob


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
