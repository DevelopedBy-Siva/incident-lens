import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.control.models import Project
from app.control.repositories import ProjectRepository
from app.shared.model_config import configured_base_model
from app.training.artifact_metadata import (
    ArtifactMetadataAlreadyExistsError,
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
    ARTIFACT_VERSION_PATTERN,
    DatasetRepository,
    ModelArtifactRepository,
    TrainingJobRepository,
)
from app.training.training_engine import (
    TrainingEngine,
    TrainingRequest,
    TrainingResult,
    configured_training_engine,
)
from app.training.training_profile import (
    LoraTrainingProfile,
    configured_training_profile,
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
        training_profile: LoraTrainingProfile | None = None,
    ):
        self.db = db
        self.dataset_storage = dataset_storage or configured_dataset_storage()
        self.engine = engine or configured_training_engine()
        self.evaluator = evaluator or BasicEvaluationService()
        self.artifact_writer = artifact_writer or configured_artifact_metadata_writer()
        self.training_profile = training_profile or configured_training_profile()
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
        artifact_version = None
        try:
            project = self._require_project(project_id)
            dataset = self._require_ready_dataset(project_id, job.dataset_id)
            dataset_content = self.dataset_storage.load(dataset.storage_key)
            base_model = configured_base_model()
            if project.base_model != base_model:
                project.base_model = base_model
                self._commit(project)
            artifact_version, adapter_path = self._prepare_artifact_output(project.id)
            training_result = self.engine.train(
                TrainingRequest(
                    project_id=project_id,
                    dataset_id=dataset.id,
                    dataset_version=dataset.dataset_version,
                    base_model=base_model,
                    dataset_content=dataset_content,
                    expected_record_count=dataset.record_count,
                    adapter_output_path=adapter_path,
                    profile=self.training_profile,
                )
            )
            if not training_result.succeeded:
                raise TrainingPipelineError(
                    training_result.error or "Training engine reported failure"
                )
            if (
                Path(training_result.adapter_path).resolve()
                != Path(adapter_path).resolve()
            ):
                raise TrainingPipelineError(
                    "Training engine wrote to an unexpected artifact path"
                )

            self.jobs.transition(job, TrainingJobStatus.EVALUATING)
            self._commit(job)
            evaluation = self.evaluator.evaluate(
                dataset, dataset_content, training_result
            )
            if not evaluation.passed:
                raise TrainingPipelineError("Training evaluation did not pass")

            artifact = self._register_artifact(
                project,
                dataset,
                artifact_version,
                adapter_path,
                evaluation,
            )
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
            if artifact is None and artifact_version is not None:
                self.artifact_writer.remove(project_id, artifact_version)
            return self._fail(job_id, project_id, artifact)

    def _register_artifact(
        self,
        project: Project,
        dataset: Dataset,
        artifact_version: str,
        adapter_path: str,
        evaluation: EvaluationResult,
    ) -> ModelArtifact:
        artifact = ModelArtifact(
            project_id=project.id,
            artifact_version=artifact_version,
            base_model=configured_base_model(),
            adapter_path=adapter_path,
            dataset_id=dataset.id,
            evaluation_score=evaluation.score,
            status=ModelArtifactStatus.READY,
        )
        try:
            self.artifacts.add(artifact)
            return self._commit(artifact)
        except IntegrityError:
            self.db.rollback()
            raise

    def _prepare_artifact_output(self, project_id: str) -> tuple[str, str]:
        candidate = self.artifacts.next_version(project_id)
        match = ARTIFACT_VERSION_PATTERN.fullmatch(candidate)
        version_number = int(match.group(1)) if match else 1
        for _ in range(self.ARTIFACT_VERSION_RESERVATION_ATTEMPTS):
            version = f"adapter-v{version_number}"
            try:
                return version, self.artifact_writer.prepare(project_id, version)
            except ArtifactMetadataAlreadyExistsError:
                version_number += 1
        raise TrainingPipelineError("Unable to reserve an artifact directory")

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

    def _artifact_metadata(
        self,
        project: Project,
        dataset: Dataset,
        artifact: ModelArtifact,
        training_result: TrainingResult,
        evaluation: EvaluationResult,
    ) -> dict[str, Any]:
        return {
            "schema_version": "incidentlens-lora-artifact-v1",
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
            "training_duration_seconds": training_result.metrics.get(
                "duration_seconds"
            ),
            "lora_configuration": self.training_profile.as_metadata(),
            "training_metrics": training_result.metrics,
            "framework_versions": training_result.framework_versions,
            "artifact_files": list(training_result.artifact_files),
            "contains_adapter_weights": True,
            "contains_base_model_weights": False,
        }

    def _commit(self, entity):
        try:
            self.db.commit()
            return entity
        except Exception:
            self.db.rollback()
            raise
