import logging
from datetime import datetime
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.control.models import Project
from app.control.repositories import ProjectRepository
from app.training.artifact_metadata import (
    ArtifactMetadataWriter,
    configured_artifact_metadata_writer,
)
from app.training.dataset_storage import DatasetStorage, configured_dataset_storage
from app.training.evaluation import (
    BasicEvaluationService,
    EvaluationResult,
    EvaluationService,
)
from app.training.models import (
    Dataset,
    DatasetStatus,
    ModelArtifact,
    ModelArtifactStatus,
    TrainingJob,
    TrainingJobStatus,
)
from app.training.repositories import (
    DatasetRepository,
    ModelArtifactRepository,
    TrainingJobRepository,
)
from app.training.training_engine import (
    PlaceholderTrainingEngine,
    TrainingEngine,
    TrainingRequest,
    TrainingResult,
)

logger = logging.getLogger(__name__)


class TrainingJobNotFoundError(ValueError):
    pass


class TrainingJobStateError(ValueError):
    pass


class TrainingPipelineError(RuntimeError):
    pass


class TrainingWorker:
    """Synchronously execute queued training jobs and their lifecycle."""

    ARTIFACT_VERSION_RESERVATION_ATTEMPTS = 3

    def __init__(
        self,
        db: Session,
        dataset_storage: DatasetStorage | None = None,
        engine: TrainingEngine | None = None,
        evaluator: EvaluationService | None = None,
        artifact_writer: ArtifactMetadataWriter | None = None,
    ):
        self.db = db
        self.dataset_storage = dataset_storage or configured_dataset_storage()
        self.engine = engine or PlaceholderTrainingEngine()
        self.evaluator = evaluator or BasicEvaluationService()
        self.artifact_writer = artifact_writer or configured_artifact_metadata_writer()
        self.projects = ProjectRepository(db)
        self.datasets = DatasetRepository(db)
        self.jobs = TrainingJobRepository(db)
        self.artifacts = ModelArtifactRepository(db)

    def run_next(self, project_id: str | None = None) -> TrainingJob | None:
        job = self.jobs.next_queued(project_id)
        if not job:
            return None
        return self.run(job.id, job.project_id)

    def run(self, job_id: str, project_id: str) -> TrainingJob:
        job = self.jobs.get_for_project(job_id, project_id)
        if not job:
            raise TrainingJobNotFoundError("Training job not found")

        job = self.jobs.reserve(job_id, project_id, datetime.utcnow())
        if not job:
            raise TrainingJobStateError("Only QUEUED training jobs can be run")
        self._commit(job)

        artifact = None
        try:
            project = self._require_project(project_id)
            dataset = self._require_ready_dataset(project_id, job.dataset_id)
            dataset_content = self.dataset_storage.load(dataset.storage_key)
            training_result = self.engine.train(
                TrainingRequest(
                    project_id=project_id,
                    dataset_id=dataset.id,
                    dataset_version=dataset.dataset_version,
                    base_model=project.base_model,
                    dataset_content=dataset_content,
                    expected_record_count=dataset.record_count,
                )
            )
            if not training_result.succeeded:
                raise TrainingPipelineError(
                    training_result.error or "Training engine reported failure"
                )

            self.jobs.transition(job, TrainingJobStatus.EVALUATING)
            self._commit(job)
            evaluation = self.evaluator.evaluate(
                dataset, dataset_content, training_result
            )
            if not evaluation.passed:
                raise TrainingPipelineError("Training evaluation did not pass")

            artifact = self._register_artifact(project, dataset, evaluation)
            self.artifact_writer.write(
                project.id,
                artifact.artifact_version,
                self._artifact_metadata(
                    project, dataset, artifact, training_result, evaluation
                ),
            )
            return self._activate_and_complete(project, job, artifact)
        except Exception as exc:  # noqa: BLE001 - lifecycle failures must mark the job
            logger.warning("Training job %s failed: %s", job_id, exc)
            return self._fail(job_id, project_id, artifact)

    def _register_artifact(
        self,
        project: Project,
        dataset: Dataset,
        evaluation: EvaluationResult,
    ) -> ModelArtifact:
        for attempt in range(self.ARTIFACT_VERSION_RESERVATION_ATTEMPTS):
            artifact_version = self.artifacts.next_version(project.id)
            artifact = ModelArtifact(
                project_id=project.id,
                artifact_version=artifact_version,
                base_model=project.base_model,
                adapter_path=self.artifact_writer.artifact_path(
                    project.id, artifact_version
                ),
                dataset_id=dataset.id,
                evaluation_score=evaluation.score,
                status=ModelArtifactStatus.READY,
            )
            try:
                self.artifacts.add(artifact)
                return self._commit(artifact)
            except IntegrityError:
                self.db.rollback()
                if attempt == self.ARTIFACT_VERSION_RESERVATION_ATTEMPTS - 1:
                    raise

        raise RuntimeError("Unable to reserve an artifact version")

    def _activate_and_complete(
        self,
        project: Project,
        job: TrainingJob,
        artifact: ModelArtifact,
    ) -> TrainingJob:
        self.projects.set_active_artifact(project, artifact.id)
        self.jobs.transition(
            job,
            TrainingJobStatus.PASSED,
            artifact_id=artifact.id,
            finished_at=datetime.utcnow(),
        )
        try:
            self.db.commit()
            return job
        except Exception:
            self.db.rollback()
            raise

    def _fail(
        self,
        job_id: str,
        project_id: str,
        artifact: ModelArtifact | None,
    ) -> TrainingJob:
        self.db.rollback()
        job = self.jobs.get_for_project(job_id, project_id)
        if not job:
            raise TrainingJobNotFoundError("Training job not found while failing job")
        if artifact is not None:
            persisted_artifact = self.artifacts.get_for_project(artifact.id, project_id)
            if persisted_artifact:
                self.artifacts.set_status(
                    persisted_artifact, ModelArtifactStatus.FAILED
                )
        if job.status in {
            TrainingJobStatus.RUNNING,
            TrainingJobStatus.EVALUATING,
        }:
            self.jobs.transition(
                job, TrainingJobStatus.FAILED, finished_at=datetime.utcnow()
            )
        return self._commit(job)

    def _require_project(self, project_id: str) -> Project:
        project = self.projects.get(project_id)
        if not project:
            raise TrainingPipelineError("Project not found")
        return project

    def _require_ready_dataset(self, project_id: str, dataset_id: str) -> Dataset:
        dataset = self.datasets.get_for_project(dataset_id, project_id)
        if not dataset or dataset.status != DatasetStatus.READY:
            raise TrainingPipelineError("Training job dataset is not READY")
        return dataset

    @staticmethod
    def _artifact_metadata(
        project: Project,
        dataset: Dataset,
        artifact: ModelArtifact,
        training_result: TrainingResult,
        evaluation: EvaluationResult,
    ) -> dict[str, Any]:
        return {
            "schema_version": "incidentlens-model-artifact-v1",
            "project_id": project.id,
            "dataset_id": dataset.id,
            "dataset_version": dataset.dataset_version,
            "base_model": artifact.base_model,
            "artifact_id": artifact.id,
            "artifact_version": artifact.artifact_version,
            "adapter_path": artifact.adapter_path,
            "evaluation_score": artifact.evaluation_score,
            "evaluation_metrics": evaluation.metrics,
            "created_at": artifact.created_at.isoformat(),
            "training_engine": training_result.engine,
            "training_simulated": training_result.simulated,
            "training_metrics": training_result.metrics,
            "contains_model_weights": False,
        }

    def _commit(self, entity):
        try:
            self.db.commit()
            return entity
        except Exception:
            self.db.rollback()
            raise
