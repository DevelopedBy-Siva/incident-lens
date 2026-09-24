#!/usr/bin/env python3
"""
EC2 Training Bootstrap Script
Runs on temporary GPU EC2 instances to orchestrate the complete training pipeline.

Entry point for the training process on GPU instances. This script:
1. Clones the repository and checks out the specified Git commit
2. Sets up Python path and activates the training environment
3. Loads configuration from EC2 User Data
4. Downloads dataset from S3
5. Runs QLoRA fine-tuning using local training engine
6. Uploads artifacts to S3
7. Updates training job status via API
8. Terminates the instance (or signals orchestrator)

The training AMI contains stable GPU/Python dependencies in /home/ubuntu/training-env,
while application code is cloned fresh at runtime and pinned to a specific Git commit.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

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
        # Extract training_config from nested structure if present (backward compatibility)
        if "training_config" in self.config:
            training_config = self.config["training_config"]
            # Merge training_config into root for backward compatibility
            self.config.update(training_config)
        self.s3_client = None
        self.db_session = None
        self.working_dir = None
        self.repo_dir = None
        self.datadog_logger = None
        self.training_start_time = None

    def _load_config(self) -> dict[str, Any]:
        """Load training configuration from S3.
        
        The orchestrator uploads the complete training configuration to S3
        before launching the instance. This avoids the 25,600-byte User Data limit.
        
        Configuration location is specified via environment variables:
        - INCIDENT_LENS_S3_BUCKET: S3 bucket name
        - INCIDENT_LENS_CONFIG_S3_KEY: S3 object key
        
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigurationError: If config cannot be downloaded or is invalid
        """
        # Get S3 location from environment
        s3_bucket = os.environ.get("INCIDENT_LENS_S3_BUCKET", "").strip()
        config_s3_key = os.environ.get("INCIDENT_LENS_CONFIG_S3_KEY", "").strip()
        
        if not s3_bucket or not config_s3_key:
            raise ConfigurationError(
                "INCIDENT_LENS_S3_BUCKET and INCIDENT_LENS_CONFIG_S3_KEY "
                "environment variables are required"
            )
        
        try:
            import boto3
        except ImportError as exc:
            raise ConfigurationError("boto3 not installed") from exc

        try:
            logger.info(
                f"Downloading training config from s3://{s3_bucket}/{config_s3_key}"
            )
            
            s3_client = boto3.client("s3")
            response = s3_client.get_object(Bucket=s3_bucket, Key=config_s3_key)
            config_json = response["Body"].read().decode("utf-8")
            
            config = json.loads(config_json)
            logger.info("Training configuration loaded successfully from S3")
            
            # Add S3 bucket to config for dataset/artifact operations
            config["s3_bucket"] = s3_bucket
            
            return config
            
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid JSON in config file: {exc}") from exc
        except Exception as exc:
            raise ConfigurationError(
                f"Failed to download config from S3: {exc}"
            ) from exc

    def _clone_repository(self) -> None:
        """Clone the application repository and checkout the specified Git commit.
        
        Clones from the configured repository URL and checks out the Git commit
        specified in the configuration. This ensures training uses code that matches
        the application version that launched the job.
        
        Repository URL and commit SHA are loaded from environment variables set by User Data:
        - INCIDENT_LENS_GIT_REPOSITORY_URL
        - INCIDENT_LENS_GIT_COMMIT_SHA
        
        Raises:
            ConfigurationError: If repository URL or Git commit not configured
            TrainingBootstrapError: If git operations fail
        """
        # Load from environment variables (set by User Data)
        repo_url = os.environ.get("INCIDENT_LENS_GIT_REPOSITORY_URL", "").strip()
        git_commit = os.environ.get("INCIDENT_LENS_GIT_COMMIT_SHA", "").strip()

        if not repo_url:
            raise ConfigurationError("INCIDENT_LENS_GIT_REPOSITORY_URL not configured")
        
        if not git_commit:
            raise ConfigurationError("INCIDENT_LENS_GIT_COMMIT_SHA not configured")

        # Create temporary directory for repository
        self.repo_dir = Path(tempfile.mkdtemp(prefix="incident-lens-repo-"))
        
        logger.info(f"Cloning repository: {repo_url}")
        
        try:
            # Clone repository
            subprocess.run(
                ["git", "clone", "--quiet", repo_url, str(self.repo_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            
            logger.info(f"Checking out commit: {git_commit}")
            
            # Checkout specific commit
            subprocess.run(
                ["git", "checkout", "--quiet", git_commit],
                cwd=str(self.repo_dir),
                check=True,
                capture_output=True,
                text=True,
            )
            
            logger.info(f"Repository ready at {self.repo_dir}")
            
        except subprocess.CalledProcessError as exc:
            error_msg = exc.stderr if exc.stderr else str(exc)
            raise TrainingBootstrapError(
                f"Failed to clone/checkout repository: {error_msg}"
            ) from exc
        except Exception as exc:
            raise TrainingBootstrapError(
                f"Unexpected error during repository setup: {exc}"
            ) from exc

    def _setup_python_path(self) -> None:
        """Configure Python path to import modules from cloned repository.
        
        Adds the cloned log-analyzer directory to sys.path and PYTHONPATH
        so training engine modules can be imported.
        """
        if not self.repo_dir:
            raise TrainingBootstrapError("Repository not cloned")
        
        log_analyzer_path = self.repo_dir / "log-analyzer"
        
        if not log_analyzer_path.exists():
            raise TrainingBootstrapError(
                f"log-analyzer directory not found in repository at {log_analyzer_path}"
            )
        
        # Add to sys.path for current process
        sys.path.insert(0, str(log_analyzer_path))
        
        # Also set PYTHONPATH for subprocess compatibility
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        if current_pythonpath:
            os.environ["PYTHONPATH"] = f"{log_analyzer_path}:{current_pythonpath}"
        else:
            os.environ["PYTHONPATH"] = str(log_analyzer_path)
        
        logger.info(f"Python path configured: {log_analyzer_path}")
    
    def _setup_datadog_logger(self) -> None:
        """Initialize Datadog logger for training telemetry.
        
        Creates a DatadogLogger instance using credentials from the training config.
        Failures in Datadog setup are logged but do not prevent training.
        """
        try:
            # Import after Python path is configured
            from app.training.datadog_logger import create_training_logger
            
            # Get EC2 instance ID if available
            ec2_instance_id = None
            try:
                import requests
                response = requests.get(
                    "http://169.254.169.254/latest/meta-data/instance-id",
                    timeout=2,
                )
                if response.status_code == 200:
                    ec2_instance_id = response.text.strip()
            except Exception:
                pass  # Not running on EC2 or metadata service unavailable
            
            # Create Datadog logger with project credentials
            self.datadog_logger = create_training_logger(
                api_key=self.config.get("datadog_api_key"),
                site=self.config.get("datadog_site"),
                project_id=self.config["project_id"],
                training_job_id=self.config["job_id"],
                dataset_id=self.config["dataset_id"],
                ec2_instance_id=ec2_instance_id,
            )
            
            if self.datadog_logger.enabled:
                logger.info("Datadog training telemetry enabled")
            else:
                logger.info("Datadog training telemetry disabled (credentials not configured)")
                
        except Exception as exc:
            logger.warning(
                f"Failed to initialize Datadog logger: {exc}. Training will continue without telemetry."
            )
            # Create a disabled logger as fallback
            try:
                from app.training.datadog_logger import DatadogLogger, TrainingLogContext
                context = TrainingLogContext(
                    project_id=self.config["project_id"],
                    training_job_id=self.config["job_id"],
                    dataset_id=self.config["dataset_id"],
                )
                self.datadog_logger = DatadogLogger(None, None, context, enabled=False)
            except Exception:
                self.datadog_logger = None

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
            
            self.training_start_time = time.time()

            # Clone repository and checkout specified commit
            logger.info("Cloning repository")
            self._clone_repository()
            if self.datadog_logger:
                self.datadog_logger.info("repository_clone_completed", {
                    "git_commit_sha": self.config.get("git_commit_sha", "unknown")
                })

            # Setup Python path for cloned repository
            self._setup_python_path()
            logger.info("Checking out specified commit")
            if self.datadog_logger:
                self.datadog_logger.info("repository_checkout_completed")

            # Initialize Datadog logger after Python path is set
            self._setup_datadog_logger()
            
            # Emit worker started event
            if self.datadog_logger:
                self.datadog_logger.info("training_worker_started", {
                    "base_model": self.config.get("base_model", "unknown"),
                })

            # Setup working directory
            self._setup_working_directory()

            # Download dataset from S3
            logger.info("Downloading dataset from S3")
            if self.datadog_logger:
                self.datadog_logger.info("dataset_download_started")
            
            dataset_path = self._download_dataset()
            
            if self.datadog_logger:
                file_size_mb = dataset_path.stat().st_size / 1024 / 1024
                self.datadog_logger.info("dataset_download_completed", {
                    "dataset_size_mb": round(file_size_mb, 2),
                })

            # Run training using existing training engine
            logger.info("Starting model loading and training")
            if self.datadog_logger:
                self.datadog_logger.info("model_loading_started")
            
            training_result = self._run_training(dataset_path)
            
            if self.datadog_logger:
                elapsed = time.time() - self.training_start_time
                self.datadog_logger.info("training_completed", {
                    "total_elapsed_seconds": round(elapsed, 1),
                    "final_loss": training_result.get("metrics", {}).get("training_loss"),
                })

            # Upload artifacts to S3
            logger.info("Uploading artifacts to S3")
            if self.datadog_logger:
                self.datadog_logger.info("artifact_upload_started")
            
            self._upload_artifacts(training_result["adapter_path"])
            
            if self.datadog_logger:
                self.datadog_logger.info("artifact_upload_completed")

            # Update job status to PASSED
            if self.datadog_logger:
                self.datadog_logger.info("job_status_update_started", {
                    "new_status": "PASSED"
                })
            
            self._update_job_status("PASSED", training_result)
            
            if self.datadog_logger:
                self.datadog_logger.info("job_status_updated", {
                    "status": "PASSED"
                })

            logger.info(
                f"Training completed successfully for job {self.config['job_id']}"
            )
            return 0

        except TrainingBootstrapError as exc:
            logger.error(f"Training failed: {exc}")
            
            if self.datadog_logger:
                self.datadog_logger.error("training_failed", {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:200],  # Truncate to avoid leaking sensitive data
                })
            
            try:
                self._update_job_status("FAILED", {"error": str(exc)})
            except Exception as update_exc:
                logger.error(f"Failed to update job status: {update_exc}")
                if self.datadog_logger:
                    self.datadog_logger.error("job_status_update_failed", {
                        "error_type": type(update_exc).__name__,
                    })
            return 1

        except Exception as exc:
            logger.error(f"Unexpected error during training: {exc}", exc_info=True)
            
            if self.datadog_logger:
                self.datadog_logger.error("training_failed", {
                    "error_type": "UnexpectedError",
                    "error_class": type(exc).__name__,
                })
            
            return 1

        finally:
            self._cleanup()
            if self.datadog_logger:
                self.datadog_logger.close()

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
            
            if self.datadog_logger:
                self.datadog_logger.info("model_loading_completed")
                self.datadog_logger.info("training_started", {
                    "lora_rank": profile.rank,
                    "lora_alpha": profile.alpha,
                    "epochs": profile.epochs,
                    "batch_size": profile.batch_size,
                })

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
        """Report training progress to local logs and Datadog.
        
        Emits progress updates at regular intervals (approximately 20 times during training)
        to avoid flooding Datadog with excessive log entries.
        
        Args:
            current: Current training step
            total: Total training steps
        """
        if total > 0:
            percent = (current / total) * 100
            
            # Calculate reporting interval (emit ~20 progress logs during training)
            report_interval = max(1, total // 20)
            
            # Log progress locally at regular intervals
            if current % report_interval == 0 or current == total:
                elapsed = None
                if self.training_start_time:
                    elapsed = time.time() - self.training_start_time
                
                logger.info(
                    f"Training progress: {current}/{total} ({percent:.1f}%)"
                    + (f" - {elapsed:.1f}s elapsed" if elapsed else "")
                )
                
                # Send to Datadog at the same interval
                if self.datadog_logger:
                    self.datadog_logger.progress(
                        current_step=current,
                        total_steps=total,
                        training_loss=None,  # Loss not available in progress callback
                        elapsed_seconds=elapsed,
                    )

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
            logger.warning(f"Cleanup of working directory failed: {exc}")
        
        try:
            if self.repo_dir and self.repo_dir.exists():
                import shutil
                shutil.rmtree(self.repo_dir)
                logger.info("Repository directory cleaned up")
        except Exception as exc:
            logger.warning(f"Cleanup of repository directory failed: {exc}")


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
