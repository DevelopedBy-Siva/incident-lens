from sqlalchemy.orm import Session

from app.control.models import Project
from app.control.repositories import ProjectRepository
from app.training.models import (
    Dataset,
    DatasetStatus,
    ModelArtifact,
    ModelArtifactStatus,
    TrainingJob,
)
from app.training.repositories import (
    DatasetRepository,
    ModelArtifactRepository,
    TrainingJobRepository,
)


class ModelManagementService:
    """Coordinates model-lifecycle metadata without running training or inference."""

    def __init__(self, db: Session):
        self.db = db
        self.projects = ProjectRepository(db)
        self.datasets = DatasetRepository(db)
        self.training_jobs = TrainingJobRepository(db)
        self.artifacts = ModelArtifactRepository(db)

    def create_dataset_metadata(
        self,
        project_id: str,
        dataset_version: str,
        storage_key: str,
        status: DatasetStatus = DatasetStatus.VALIDATING,
        record_count: int = 0,
    ) -> Dataset:
        self._require_project(project_id)
        dataset = Dataset(
            project_id=project_id,
            dataset_version=dataset_version,
            storage_key=storage_key,
            status=status,
            record_count=record_count,
        )
        return self._commit(self.datasets.add(dataset))

    def mark_dataset_ready(
        self, project_id: str, dataset_id: str, record_count: int
    ) -> Dataset:
        if record_count < 0:
            raise ValueError("Dataset record count cannot be negative")
        dataset = self._require_dataset(project_id, dataset_id)
        self._require_validating_dataset(dataset)
        return self._commit(
            self.datasets.set_status(
                dataset, DatasetStatus.READY, record_count=record_count
            )
        )

    def mark_dataset_failed(self, project_id: str, dataset_id: str) -> Dataset:
        dataset = self._require_dataset(project_id, dataset_id)
        self._require_validating_dataset(dataset)
        return self._commit(
            self.datasets.set_status(dataset, DatasetStatus.FAILED, record_count=0)
        )

    def create_training_job(self, project_id: str, dataset_id: str) -> TrainingJob:
        self._require_project(project_id)
        if not self.datasets.get_for_project(dataset_id, project_id):
            raise ValueError("Dataset does not belong to project")
        job = TrainingJob(project_id=project_id, dataset_id=dataset_id)
        return self._commit(self.training_jobs.add(job))

    def register_model_artifact(
        self,
        project_id: str,
        artifact_version: str,
        dataset_id: str,
        adapter_path: str | None = None,
        evaluation_score: float | None = None,
        status: ModelArtifactStatus = ModelArtifactStatus.READY,
    ) -> ModelArtifact:
        project = self._require_project(project_id)
        if not self.datasets.get_for_project(dataset_id, project_id):
            raise ValueError("Dataset does not belong to project")
        artifact = ModelArtifact(
            project_id=project_id,
            artifact_version=artifact_version,
            base_model=project.base_model,
            adapter_path=adapter_path,
            dataset_id=dataset_id,
            evaluation_score=evaluation_score,
            status=status,
        )
        return self._commit(self.artifacts.add(artifact))

    def update_active_artifact(
        self, project_id: str, artifact_id: str | None
    ) -> Project:
        project = self._require_project(project_id)
        if artifact_id is not None:
            artifact = self.artifacts.get_for_project(artifact_id, project_id)
            if not artifact:
                raise ValueError("Artifact does not belong to project")
            if artifact.status != ModelArtifactStatus.READY:
                raise ValueError("Only READY artifacts can be activated")
        return self._commit(self.projects.set_active_artifact(project, artifact_id))

    def _require_project(self, project_id: str) -> Project:
        project = self.projects.get(project_id)
        if not project:
            raise ValueError("Project not found")
        return project

    def _require_dataset(self, project_id: str, dataset_id: str) -> Dataset:
        dataset = self.datasets.get_for_project(dataset_id, project_id)
        if not dataset:
            raise ValueError("Dataset not found")
        return dataset

    @staticmethod
    def _require_validating_dataset(dataset: Dataset) -> None:
        if dataset.status != DatasetStatus.VALIDATING:
            raise ValueError("Only VALIDATING datasets can change lifecycle status")

    def _commit(self, entity):
        try:
            self.db.commit()
            self.db.refresh(entity)
            return entity
        except Exception:
            self.db.rollback()
            raise
