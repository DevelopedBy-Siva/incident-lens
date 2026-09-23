#!/usr/bin/env python3
"""
EC2 Training Bootstrap Script
Runs on temporary GPU EC2 instances to orchestrate the complete training pipeline.

Entry point for the training process on GPU instances. This script:
1. Loads configuration from EC2 User Data
2. Downloads dataset from S3
3. Runs QLoRA fine-tuning using local training engine
4. Uploads artifacts to S3
5. Updates training job status via API
6. Terminates the instance (or signals orchestrator)
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import requests

# Configure logging to go to syslog and stdout
logging.basicConfig(
    level=logging.INFO,
    format="[TRAINING] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/var/log/incident-lens-training.log"),
    ],
)
logger = logging.getLogger(__name__)


class TrainingBootstrapError(Exception):
    """Base exception for bootstrap failures."""
    pass


class ConfigurationError(TrainingBootstrapError):
    """Configuration is invalid or incomplete."""
    pass


class DatasetError(TrainingBootstrapError):
    """Dataset download or processing failed."""
    pass


class TrainingExecutionError(TrainingBootstrapError):
    """Training execution failed."""
    pass


class ArtifactUploadError(TrainingBootstrapError):
    """Artifact upload to S3 failed."""
    pass


class JobStatusUpdateError(TrainingBootstrapError):
    """Failed to update job status via API."""
    pass


class TrainingBootstrap:
    """Orchestrates the training pipeline on EC2 GPU instance."""

    def __init__(self):
        """Initialize bootstrap by loading configuration."""
        self.config = self._load_config()
        self.s3_client = None
        self.db_session = None
        self.working_dir = None

    def _load_config(self) -> dict[str, Any]:
        """Load training configuration from EC2 User Data file.
        
        Configuration is written by the orchestrator to:
        /tmp/incident-lens-training-config.json
        
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigurationError: If config file not found or invalid
        """
        config_path = Path("/tmp/incident-lens-training-config.json")
        
        if not config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_path}"
            )

        try:
            with config_path.open() as f:
                config = json.load(f)
            logger.info("Configuration loaded successfully")
            return config
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid JSON in config file: {exc}") from exc
        except Exception as exc:
            raise ConfigurationError(f"Failed to read config file: {exc}") from exc

    def run(self) -> int:
        """Execute the complete training pipeline.
        
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        try:
            logger.info(
                f"Starting training for job {self.config['job_id']} "
                f"(project={self.config['project_id']})"
            )

            # Setup working directory
            self._setup_working_directory()

            # Download dataset from S3
            dataset_path = self._download_dataset()

            # Run training using existing training engine
            training_result = self._run_training(dataset_path)

            # Upload artifacts to S3
            self._upload_artifacts(training_result["adapter_path"])

            # Update job status to PASSED
            self._update_job_status("PASSED", training_result)

            logger.info(
                f"Training completed successfully for job {self.config['job_id']}"
            )
            return 0

        except TrainingBootstrapError as exc:
            logger.error(f"Training failed: {exc}")
            try:
                self._update_job_status("FAILED", {"error": str(exc)})
            except Exception as update_exc:
                logger.error(f"Failed to update job status: {update_exc}")
            return 1

        except Exception as exc:
            logger.error(f"Unexpected error during training: {exc}", exc_info=True)
            return 1

        finally:
            self._cleanup()

    def _setup_working_directory(self) -> None:
        """Create temporary working directory for training artifacts."""
        self.working_dir = Path(tempfile.mkdtemp(prefix="incident-lens-training-"))
        logger.info(f"Working directory: {self.working_dir}")

    def _download_dataset(self) -> Path:
        """Download training dataset from S3.
        
        Returns:
            Path to downloaded dataset file
            
        Raises:
            DatasetError: If download fails
        """
        try:
            import boto3
        except ImportError as exc:
            raise DatasetError("boto3 not installed") from exc

        try:
            s3_client = boto3.client("s3")
            dataset_path = self.working_dir / "dataset.jsonl"

            storage_key = self.config["dataset_storage_key"]
            s3_bucket = self.config["s3_bucket"]

            logger.info(
                f"Downloading dataset from s3://{s3_bucket}/{storage_key}"
            )

            s3_client.download_file(
                Bucket=s3_bucket,
                Key=storage_key,
                Filename=str(dataset_path),
            )

            file_size = dataset_path.stat().st_size
            logger.info(
                f"Dataset downloaded successfully ({file_size / 1024 / 1024:.2f} MB)"
            )
            return dataset_path

        except Exception as exc:
            raise DatasetError(f"Failed to download dataset: {exc}") from exc

    def _run_training(self, dataset_path: Path) -> dict[str, Any]:
        """Run QLoRA fine-tuning using the training engine.
        
        Args:
            dataset_path: Path to downloaded dataset file
            
        Returns:
            Training result dictionary with metrics and artifact path
            
        Raises:
            TrainingExecutionError: If training fails
        """
        try:
            # Import training engine components
            from app.training.lora_trainer import TransformersPeftTrainingEngine
            from app.training.training_engine import TrainingRequest
            from app.training.training_profile import LoraTrainingProfile

            logger.info("Initializing training engine")

            # Create training profile from config
            profile = LoraTrainingProfile(
                rank=self.config["lora_rank"],
                alpha=self.config["lora_alpha"],
                dropout=self.config["lora_dropout"],
                epochs=self.config["lora_epochs"],
                learning_rate=self.config["lora_learning_rate"],
                batch_size=self.config["lora_batch_size"],
                gradient_accumulation_steps=self.config["lora_gradient_accumulation_steps"],
                max_sequence_length=self.config["lora_max_sequence_length"],
                validation_fraction=self.config["lora_validation_fraction"],
                seed=self.config["lora_seed"],
                target_modules=set(self.config["lora_target_modules"]),
            )

            # Prepare adapter output path
            adapter_output_path = self.working_dir / "adapter"
            adapter_output_path.mkdir()

            # Load dataset content
            with dataset_path.open("rb") as f:
                dataset_content = f.read()

            logger.info("Starting QLoRA fine-tuning")

            # Run training
            engine = TransformersPeftTrainingEngine()
            training_request = TrainingRequest(
                project_id=self.config["project_id"],
                dataset_id=self.config["dataset_id"],
                dataset_version="training",  # Temporary training session
                base_model=self.config["base_model"],
                dataset_content=dataset_content,
                expected_record_count=self._count_dataset_records(dataset_content),
                adapter_output_path=str(adapter_output_path),
                profile=profile,
                progress_callback=self._report_progress,
            )

            result = engine.train(training_request)

            if not result.succeeded:
                raise TrainingExecutionError(
                    result.error or "Training engine reported failure"
                )

            logger.info(
                f"Training completed: {result.metrics.get('training_loss', 'unknown')} loss"
            )
            return {
                "adapter_path": result.adapter_path,
                "metrics": result.metrics,
                "engine": result.engine,
                "framework_versions": result.framework_versions,
                "artifact_files": result.artifact_files,
            }

        except ImportError as exc:
            raise TrainingExecutionError(
                "Training dependencies not installed on instance"
            ) from exc
        except Exception as exc:
            raise TrainingExecutionError(f"Training failed: {exc}") from exc

    def _count_dataset_records(self, content: bytes) -> int:
        """Count records in JSONL dataset.
        
        Args:
            content: Raw dataset bytes
            
        Returns:
            Number of records
        """
        try:
            text = content.decode("utf-8")
            return len([line for line in text.splitlines() if line.strip()])
        except Exception:
            return 0

    def _report_progress(self, current: int, total: int) -> None:
        """Report training progress (stub for local execution).
        
        In EC2 mode, progress updates are logged but not sent back to orchestrator.
        Full progress tracking happens via job status API once training completes.
        
        Args:
            current: Current training step
            total: Total training steps
        """
        if total > 0:
            percent = (current / total) * 100
            if current % max(1, total // 20) == 0:  # Log 20 times
                logger.info(f"Training progress: {current}/{total} ({percent:.1f}%)")

    def _upload_artifacts(self, adapter_path: str) -> None:
        """Upload trained adapter artifacts to S3.
        
        Args:
            adapter_path: Path to adapter directory
            
        Raises:
            ArtifactUploadError: If upload fails
        """
        try:
            import boto3
        except ImportError as exc:
            raise ArtifactUploadError("boto3 not installed") from exc

        try:
            s3_client = boto3.client("s3")
            adapter_dir = Path(adapter_path)
            s3_bucket = self.config["s3_bucket"]
            s3_prefix = os.getenv("S3_ARTIFACT_PREFIX", "artifacts")

            logger.info(f"Uploading artifacts to S3: s3://{s3_bucket}/{s3_prefix}/...")

            artifact_key_prefix = (
                f"{s3_prefix}/"
                f"{self.config['project_id']}/"
                f"adapter-v1"  # Version managed on application side
            )

            # Upload all files in adapter directory
            file_count = 0
            total_size = 0
            for file_path in adapter_dir.rglob("*"):
                if not file_path.is_file():
                    continue

                relative_path = file_path.relative_to(adapter_dir)
                s3_key = f"{artifact_key_prefix}/{relative_path.as_posix()}"
                file_size = file_path.stat().st_size
                total_size += file_size

                logger.info(
                    f"Uploading {file_path.name} "
                    f"({file_size / 1024:.1f} KB)"
                )

                s3_client.upload_file(
                    Filename=str(file_path),
                    Bucket=s3_bucket,
                    Key=s3_key,
                )
                file_count += 1

            logger.info(
                f"Uploaded {file_count} artifact files "
                f"({total_size / 1024 / 1024:.2f} MB)"
            )

        except Exception as exc:
            raise ArtifactUploadError(f"Failed to upload artifacts: {exc}") from exc

    def _update_job_status(
        self, status: str, result: dict[str, Any]
    ) -> None:
        """Update training job status via API.
        
        Args:
            status: New job status (PASSED, FAILED, etc.)
            result: Training result or error dictionary
            
        Raises:
            JobStatusUpdateError: If API call fails
        """
        try:
            # Construct API URL to application instance
            # In production, this would be the private IP of the application server
            api_url = os.getenv(
                "INCIDENT_LENS_API_URL",
                "http://localhost:8000"
            ).strip()

            job_id = self.config["job_id"]
            project_id = self.config["project_id"]

            # Prepare status update payload
            payload = {
                "status": status,
                "result": result,
            }

            # Use internal database update instead of HTTP API for reliability
            # (avoids network dependency and authentication complexity)
            self._update_job_status_direct(job_id, project_id, status, result)

        except Exception as exc:
            raise JobStatusUpdateError(f"Failed to update job status: {exc}") from exc

    def _update_job_status_direct(
        self, job_id: str, project_id: str, status: str, result: dict[str, Any]
    ) -> None:
        """Update training job status directly in database.
        
        Uses SQLAlchemy to connect directly to the application database
        and update job status. This avoids HTTP API complexity and ensures
        reliable status updates even if application server is busy.
        
        Args:
            job_id: Training job ID
            project_id: Project ID
            status: New job status
            result: Training result or error dictionary
        """
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker

            from app.training.models import TrainingJob, TrainingJobStatus
            from app.training.repositories import TrainingJobRepository

            # Create database connection
            database_url = self.config["database_url"]
            engine = create_engine(database_url)
            Session = sessionmaker(bind=engine)
            db = Session()

            try:
                # Get job and update status
                job_repo = TrainingJobRepository(db)
                job = job_repo.get_for_project(job_id, project_id)

                if not job:
                    raise JobStatusUpdateError(f"Job {job_id} not found")

                # Transition to final status
                if status == "PASSED":
                    job_repo.transition(
                        job,
                        TrainingJobStatus.PASSED,
                        finished_at=None,  # Will be set by transition
                    )
                elif status == "FAILED":
                    job_repo.transition(
                        job,
                        TrainingJobStatus.FAILED,
                        finished_at=None,  # Will be set by transition
                    )

                db.commit()
                logger.info(f"Job status updated to {status}")

            finally:
                db.close()

        except Exception as exc:
            logger.error(f"Direct database update failed: {exc}")
            raise

    def _cleanup(self) -> None:
        """Clean up temporary files and resources."""
        try:
            if self.working_dir and self.working_dir.exists():
                import shutil
                shutil.rmtree(self.working_dir)
                logger.info("Temporary working directory cleaned up")
        except Exception as exc:
            logger.warning(f"Cleanup failed: {exc}")


def main() -> int:
    """Main entry point."""
    try:
        bootstrap = TrainingBootstrap()
        return bootstrap.run()
    except Exception as exc:
        logger.error(f"Bootstrap failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
