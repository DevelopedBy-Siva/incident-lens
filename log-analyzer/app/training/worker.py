import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.control.models import Project
from app.control.repositories import ProjectRepository
from app.shared.database import SessionLocal
from app.shared.model_config import configured_base_model
from app.shared.observability import trace_operation
from app.training.artifact_metadata import (
    ArtifactMetadataAlreadyExistsError,
    ArtifactMetadataWriter,
    configured_artifact_metadata_writer,
)
from app.training.dataset_serializer import JsonLinesDatasetSerializer
from app.training.dataset_storage import DatasetStorage, configured_dataset_storage
from app.training.ec2_orchestrator import (
    EC2TrainingOrchestrator,
    configured_ec2_training_config,
)
from app.training.evaluation import (
    BasicEvaluationService,
    EvaluationResult,
    EvaluationService,
)
from app.training.job_config import create_training_job_config
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
    """Execute queued training jobs locally or on temporary EC2 GPU instances.
    
    Supports two execution modes:
    1. Local/Synchronous: Runs training on the current application server (development)
    2. EC2 Remote: Launches temporary GPU instances and delegates training (production)
    
    The execution mode is determined by TRAINING_EC2_ENABLED environment variable.
    When disabled, behaves as before (local synchronous execution).
    When enabled, launches a temporary g6.xlarge instance and returns immediately with RUNNING status.
    """

    ARTIFACT_VERSION_RESERVATION_ATTEMPTS = 3

    def __init__(
        self,
        db: Session,
        dataset_storage: DatasetStorage | None = None,
        engine: TrainingEngine | None = None,
        evaluator: EvaluationService | None = None,
        artifact_writer: ArtifactMetadataWriter | None = None,
        training_profile: LoraTrainingProfile | None = None,
        ec2_orchestrator: EC2TrainingOrchestrator | None = None,
    ):
        self.db = db
        self.dataset_storage = dataset_storage or configured_dataset_storage()
        self.engine = engine or configured_training_engine()
        self.evaluator = evaluator or BasicEvaluationService()
        self.artifact_writer = artifact_writer or configured_artifact_metadata_writer()
        self.training_profile = training_profile or configured_training_profile()
        self.dataset_serializer = JsonLinesDatasetSerializer()
        self.projects = ProjectRepository(db)
        self.datasets = DatasetRepository(db)
        self.jobs = TrainingJobRepository(db)
        self.artifacts = ModelArtifactRepository(db)
        
        # EC2 orchestrator setup
        ec2_config = configured_ec2_training_config()
        if ec2_config and ec2_orchestrator is None:
            self.ec2_orchestrator = EC2TrainingOrchestrator(ec2_config)
        else:
            self.ec2_orchestrator = ec2_orchestrator
        self.use_ec2_remote = ec2_config is not None

    def run_next(self, project_id: str | None = None) -> TrainingJob | None:
        job = self.jobs.next_queued(project_id)
        if not job:
            return None
        return self.run(job.id, job.project_id)

    def run(self, job_id: str, project_id: str) -> TrainingJob:
        with trace_operation(
            "training_job_lifecycle",
            plane="training",
            metadata={
                "project_id": project_id,
                "training_job_id": job_id,
                "execution_mode": "ec2_remote" if self.use_ec2_remote else "local",
            },
        ) as span:
            if self.use_ec2_remote:
                job = self._run_ec2_remote(job_id, project_id)
            else:
                job = self._run_local(job_id, project_id)
            span.tags(
                {
                    "dataset_id": job.dataset_id,
                    "artifact_id": job.artifact_id,
                    "status": job.status.value,
                    "result": job.status.value,
                }
            )
            return job

    def _run_ec2_remote(self, job_id: str, project_id: str) -> TrainingJob:
        """Launch a temporary EC2 GPU instance for training and return with RUNNING status.
        
        This method:
        1. Validates the job and dataset
        2. Prepares training configuration
        3. Launches a temporary g6.xlarge instance with User Data
        4. Returns the job with RUNNING status (training proceeds asynchronously)
        
        The EC2 instance will:
        - Download the dataset from S3
        - Run fine-tuning
        - Upload artifacts to S3
        - Update the job status via API
        - Terminate automatically on completion/failure
        
        Args:
            job_id: Training job ID
            project_id: Project ID
        
        Returns:
            TrainingJob with RUNNING status and instance_id metadata
        
        Raises:
            TrainingJobNotFoundError: If job not found
            TrainingJobStateError: If job not in QUEUED state
            TrainingPipelineError: If configuration/launch fails
        """
        job = self.jobs.get_for_project(job_id, project_id)
        if not job:
            raise TrainingJobNotFoundError("Training job not found")

        # Atomically transition from QUEUED to RUNNING
        job = self.jobs.reserve(job_id, project_id, datetime.utcnow())
        if not job:
            raise TrainingJobStateError("Only QUEUED training jobs can be run")
        self._commit(job)

        try:
            project = self._require_project(project_id)
            dataset = self._require_trainable_dataset(project_id, job.dataset_id)
            base_model = configured_base_model()
            
            # Update project base model if needed
            if project.base_model != base_model:
                project.base_model = base_model
                self._commit(project)

            # Create training configuration for EC2 instance
            s3_bucket = os.getenv("S3_BUCKET", "").strip()
            if not s3_bucket:
                raise TrainingPipelineError("S3_BUCKET not configured for EC2 training")

            # Get Git repository URL and current commit SHA
            git_repository_url = os.getenv(
                "GIT_REPOSITORY_URL",
                "https://github.com/DevelopedBy-Siva/incident-lens.git"
            ).strip()
            
            git_commit_sha = self._get_current_git_commit()

            training_config = create_training_job_config(
                job_id=job_id,
                project_id=project_id,
                dataset_id=dataset.id,
                dataset_storage_key=dataset.storage_key,
                s3_bucket=s3_bucket,
                database_url=os.getenv("DATABASE_URL", "").strip(),
                base_model=base_model,
                git_repository_url=git_repository_url,
                git_commit_sha=git_commit_sha,
                training_profile=self.training_profile,
                selected_record_indices=job.selected_record_indices
                or dataset.selected_record_indices,
                huggingface_token=os.getenv("HF_TOKEN", "").strip() or None,
            )

            # Launch temporary EC2 instance
            instance_id = self.ec2_orchestrator.launch_training_instance(
                job_id=job_id,
                project_id=project_id,
                dataset_id=dataset.id,
                dataset_storage_key=dataset.storage_key,
                database_url=os.getenv("DATABASE_URL", "").strip(),
                training_config_json=training_config.to_json(),
                s3_bucket=s3_bucket,
                git_repository_url=git_repository_url,
                git_commit_sha=git_commit_sha,
            )

            # Store instance ID in job metadata for monitoring
            job.ec2_instance_id = instance_id
            self._commit(job)

            logger.info(
                f"EC2 training instance {instance_id} launched for job {job_id} "
                f"(project={project_id}). Job status: RUNNING"
            )
            return job

        except Exception as exc:
            logger.error(f"Failed to launch EC2 training instance for job {job_id}: {exc}")
            # Transition job back to QUEUED or mark as FAILED
            self._fail(job_id, project_id, None)
            raise

    def _get_current_git_commit(self) -> str:
        """Get the current Git commit SHA of the running application.
        
        Returns:
            Git commit SHA (40-character hex string)
            
        Raises:
            TrainingPipelineError: If unable to determine Git commit
        """
        import subprocess
        
        # First, check environment variable (production deployment)
        commit_sha = os.getenv("GIT_COMMIT_SHA", "").strip()
        if commit_sha:
            if len(commit_sha) == 40:
                logger.info(f"Using GIT_COMMIT_SHA environment variable: {commit_sha}")
                return commit_sha
            else:
                logger.warning(
                    f"GIT_COMMIT_SHA environment variable has invalid length "
                    f"({len(commit_sha)} chars, expected 40): {commit_sha}"
                )
        
        # Fallback: try git command (local development)
        try:
            # Get current working directory (should be the log-analyzer directory)
            cwd = Path(__file__).parent.parent.parent
            
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                check=True,
            )
            
            commit_sha = result.stdout.strip()
            
            if not commit_sha or len(commit_sha) != 40:
                raise TrainingPipelineError(
                    f"Invalid Git commit SHA from git command: {commit_sha}"
                )
            
            logger.info(f"Using git command for commit SHA: {commit_sha}")
            return commit_sha
            
        except FileNotFoundError as exc:
            raise TrainingPipelineError(
                "Unable to determine Git commit SHA. Neither GIT_COMMIT_SHA "
                "environment variable nor git command is available. "
                "Ensure GIT_COMMIT_SHA is set in production deployments."
            ) from exc
        
        except subprocess.CalledProcessError as exc:
            raise TrainingPipelineError(
                f"Unable to determine Git commit SHA. GIT_COMMIT_SHA environment "
                f"variable is not set and git command failed. "
                f"Error: {exc.stderr if exc.stderr else str(exc)}"
            ) from exc

    def _run_local(self, job_id: str, project_id: str) -> TrainingJob:
        """Run training synchronously on the local application server (development mode).
        
        This is the original execution path that trains models locally before returning.
        """
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
            dataset = self._require_trainable_dataset(project_id, job.dataset_id)
            dataset_content = self.dataset_storage.load(dataset.storage_key)
            expected_record_count = dataset.record_count
            selected_indices = (
                job.selected_record_indices
                if job.selected_record_indices is not None
                else dataset.selected_record_indices
            )
            if selected_indices is not None:
                records = self.dataset_serializer.deserialize(dataset_content)
                selected = [records[index] for index in selected_indices]
                dataset_content = self.dataset_serializer.serialize(selected)
                expected_record_count = len(selected)
                dataset._evaluation_record_count = expected_record_count
            base_model = configured_base_model()
            if project.base_model != base_model:
                project.base_model = base_model
                self._commit(project)
            artifact_version, adapter_path = self._prepare_artifact_output(project.id)
            with trace_operation(
                "lora_training",
                plane="training",
                metadata={
                    "project_id": project_id,
                    "training_job_id": job.id,
                    "dataset_id": dataset.id,
                    "dataset_version": dataset.dataset_version,
                    "artifact_version": artifact_version,
                    "base_model": base_model,
                },
            ) as training_span:
                training_result = self.engine.train(
                    TrainingRequest(
                        project_id=project_id,
                        dataset_id=dataset.id,
                        dataset_version=dataset.dataset_version,
                        base_model=base_model,
                        dataset_content=dataset_content,
                        expected_record_count=expected_record_count,
                        adapter_output_path=adapter_path,
                        profile=self.training_profile,
                        progress_callback=lambda current, total: self._save_progress(
                            job, current, total
                        ),
                    )
                )
                training_span.tags(
                    {
                        "training_engine": training_result.engine,
                        "result": (
                            "completed" if training_result.succeeded else "failed"
                        ),
                    }
                )
                training_span.metrics(
                    {
                        "training_duration_seconds": training_result.metrics.get(
                            "duration_seconds"
                        )
                    }
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
            with trace_operation(
                "evaluation",
                plane="training",
                metadata={
                    "project_id": project_id,
                    "training_job_id": job.id,
                    "dataset_id": dataset.id,
                    "dataset_version": dataset.dataset_version,
                    "artifact_version": artifact_version,
                    "base_model": base_model,
                },
            ) as evaluation_span:
                evaluation = self.evaluator.evaluate(
                    dataset, dataset_content, training_result
                )
                evaluation_span.tag(
                    "result", "passed" if evaluation.passed else "failed"
                )
                evaluation_span.metrics({"evaluation_score": evaluation.score})
            if not evaluation.passed:
                raise TrainingPipelineError("Training evaluation did not pass")

            artifact = self._register_artifact(
                project,
                dataset,
                artifact_version,
                adapter_path,
                evaluation,
            )
            with trace_operation(
                "artifact_creation",
                plane="training",
                metadata={
                    "project_id": project.id,
                    "training_job_id": job.id,
                    "dataset_id": dataset.id,
                    "dataset_version": dataset.dataset_version,
                    "artifact_id": artifact.id,
                    "artifact_version": artifact.artifact_version,
                    "base_model": base_model,
                },
            ) as artifact_span:
                self.artifact_writer.write(
                    project.id,
                    artifact.artifact_version,
                    self._artifact_metadata(
                        project, dataset, artifact, training_result, evaluation
                    ),
                )
                artifact_span.tag("result", "created")
            return self._activate_and_complete(project, dataset, job, artifact)
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
        dataset: Dataset,
        job: TrainingJob,
        artifact: ModelArtifact,
    ) -> TrainingJob:
        with trace_operation(
            "artifact_activation",
            plane="training",
            metadata={
                "project_id": project.id,
                "training_job_id": job.id,
                "dataset_id": job.dataset_id,
                "artifact_id": artifact.id,
                "artifact_version": artifact.artifact_version,
                "base_model": artifact.base_model,
            },
        ) as span:
            self.projects.set_active_artifact(project, artifact.id)
            self.datasets.set_status(dataset, DatasetStatus.TRAINED)
            self.jobs.transition(
                job,
                TrainingJobStatus.PASSED,
                artifact_id=artifact.id,
                finished_at=datetime.utcnow(),
            )
            try:
                self.db.commit()
                span.tag("result", "activated")
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

    def _save_progress(
        self, job: TrainingJob, current_step: int, total_steps: int
    ) -> None:
        update_interval = max(1, total_steps // 200)
        if current_step not in {0, total_steps} and current_step % update_interval:
            return
        try:
            self.jobs.set_progress(
                job,
                current_step,
                total_steps,
                datetime.utcnow(),
            )
            self.db.commit()
        except Exception:  # noqa: BLE001 - telemetry must not fail training
            self.db.rollback()
            logger.warning("Could not persist progress for training job %s", job.id)

    def _require_trainable_dataset(self, project_id: str, dataset_id: str) -> Dataset:
        dataset = self.datasets.get_for_project(dataset_id, project_id)
        if not dataset or dataset.status not in {
            DatasetStatus.READY,
            DatasetStatus.TRAINED,
        }:
            raise TrainingPipelineError("Training job dataset is not trainable")
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
