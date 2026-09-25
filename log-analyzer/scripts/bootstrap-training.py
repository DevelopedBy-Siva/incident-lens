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
import time
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
            
            # Get EC2 instance ID using IMDSv2 if available
            ec2_instance_id = None
            try:
                import requests
                
                # IMDSv2: First obtain a session token
                token_response = requests.put(
                    "http://169.254.169.254/latest/api/token",
                    headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
                    timeout=1,
                )
                
                if token_response.status_code == 200:
                    token = token_response.text.strip()
                    
                    # Use token to fetch instance ID
                    response = requests.get(
                        "http://169.254.169.254/latest/meta-data/instance-id",
                        headers={"X-aws-ec2-metadata-token": token},
                        timeout=1,
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
    
    def _collect_system_info(self) -> dict[str, Any]:
        """Collect system and training environment information.
        
        Collects runtime details for observability. All collection is best-effort;
        failures in individual fields do not prevent training.
        
        Returns:
            Dictionary of system information
        """
        info = {}
        
        try:
            # Python version
            import sys
            info["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        except Exception:
            pass
        
        try:
            # PyTorch version and CUDA
            import torch
            info["pytorch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
                info["gpu_count"] = torch.cuda.device_count()
                
                if torch.cuda.device_count() > 0:
                    # Get GPU 0 details
                    info["gpu_name"] = torch.cuda.get_device_name(0)
                    
                    # GPU memory
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    info["gpu_vram_gb"] = round(total_memory / (1024**3), 1)
                    
                    # Currently allocated memory at training start
                    allocated = torch.cuda.memory_allocated(0)
                    info["gpu_memory_allocated_gb"] = round(allocated / (1024**3), 2)
        except Exception:
            pass
        
        try:
            # System RAM
            import psutil
            mem = psutil.virtual_memory()
            info["system_ram_total_gb"] = round(mem.total / (1024**3), 1)
            info["system_ram_available_gb"] = round(mem.available / (1024**3), 1)
            info["cpu_count"] = psutil.cpu_count()
        except Exception:
            # psutil might not be installed
            try:
                import os
                info["cpu_count"] = os.cpu_count()
            except Exception:
                pass
        
        try:
            # transformers version
            import transformers
            info["transformers_version"] = transformers.__version__
        except Exception:
            pass
        
        try:
            # PEFT version
            import peft
            info["peft_version"] = peft.__version__
        except Exception:
            pass
        
        try:
            # bitsandbytes version
            import bitsandbytes
            info["bitsandbytes_version"] = bitsandbytes.__version__
        except Exception:
            pass
        
        try:
            # EC2 instance type from metadata
            import requests
            token_response = requests.put(
                "http://169.254.169.254/latest/api/token",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
                timeout=1,
            )
            if token_response.status_code == 200:
                token = token_response.text.strip()
                response = requests.get(
                    "http://169.254.169.254/latest/meta-data/instance-type",
                    headers={"X-aws-ec2-metadata-token": token},
                    timeout=1,
                )
                if response.status_code == 200:
                    info["ec2_instance_type"] = response.text.strip()
        except Exception:
            pass
        
        return info

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

            # Setup Python path for cloned repository
            self._setup_python_path()
            logger.info("Checking out specified commit")

            # Initialize Datadog logger after Python path is set
            self._setup_datadog_logger()
            
            # Emit repository events after Datadog is initialized
            if self.datadog_logger:
                self.datadog_logger.info("repository_clone_completed", {
                    "git_commit_sha": self.config.get("git_commit_sha", "unknown")
                }, message=f"Repository checkout completed: {self.config.get('git_commit_sha', 'unknown')[:7]}")
            
            # Collect system information for observability
            system_info = self._collect_system_info()
            
            # Emit worker started event with system details
            if self.datadog_logger:
                base_model = self.config.get("base_model", "unknown")
                
                # Build concise visible message
                message_parts = [f"Training started: {base_model}"]
                
                if "gpu_name" in system_info:
                    message_parts.append(f"GPU: {system_info['gpu_name']}")
                
                if "gpu_vram_gb" in system_info:
                    message_parts.append(f"VRAM: {system_info['gpu_vram_gb']} GB")
                
                if "system_ram_total_gb" in system_info:
                    message_parts.append(f"RAM: {system_info['system_ram_total_gb']} GB")
                
                visible_message = " | ".join(message_parts)
                
                # Emit with full system info as structured attributes
                self.datadog_logger.info(
                    "training_worker_started",
                    {
                        "base_model": base_model,
                        **system_info,
                        # Add training configuration details
                        "lora_rank": self.config.get("lora_rank"),
                        "lora_alpha": self.config.get("lora_alpha"),
                        "batch_size": self.config.get("lora_batch_size"),
                        "epochs": self.config.get("lora_epochs"),
                        "max_sequence_length": self.config.get("lora_max_sequence_length"),
                    },
                    message=visible_message,
                )

            # Setup working directory
            self._setup_working_directory()

            # Download dataset from S3
            logger.info("Downloading dataset from S3")
            if self.datadog_logger:
                self.datadog_logger.info("dataset_download_started", 
                    message="Dataset download started")
            
            dataset_path = self._download_dataset()
            
            if self.datadog_logger:
                file_size_mb = dataset_path.stat().st_size / 1024 / 1024
                self.datadog_logger.info("dataset_download_completed", {
                    "dataset_size_mb": round(file_size_mb, 2),
                }, message=f"Dataset downloaded: {file_size_mb:.2f} MB")

            # Run training using existing training engine
            logger.info("Starting model loading and training")
            if self.datadog_logger:
                base_model = self.config.get("base_model", "unknown")
                self.datadog_logger.info("model_loading_started", 
                    {"base_model": base_model},
                    message=f"Model loading started: {base_model}")
            
            training_result = self._run_training(dataset_path)
            
            # Determine artifact version early for consistent S3 upload and DB registration
            # This must be done before artifact upload to ensure S3 path matches DB record
            artifact_version = self._determine_artifact_version()
            training_result["artifact_version"] = artifact_version
            
            if self.datadog_logger:
                elapsed = time.time() - self.training_start_time
                final_loss = training_result.get("metrics", {}).get("training_loss")
                loss_str = f"loss={final_loss:.4f}" if final_loss else "loss=unknown"
                self.datadog_logger.info("training_completed", {
                    "total_elapsed_seconds": round(elapsed, 1),
                    "final_loss": final_loss,
                }, message=f"Training completed: {loss_str}, elapsed={elapsed:.0f}s")

            # Upload artifacts to S3 using determined version
            logger.info("Uploading artifacts to S3")
            if self.datadog_logger:
                self.datadog_logger.info("artifact_upload_started",
                    message="Artifacts upload started")
            
            artifact_files_count, total_size_mb = self._upload_artifacts(
                training_result["adapter_path"],
                artifact_version
            )
            
            if self.datadog_logger:
                self.datadog_logger.info("artifact_upload_completed", {
                    "file_count": artifact_files_count,
                    "total_size_mb": total_size_mb,
                }, message=f"Artifacts uploaded: {artifact_files_count} files, {total_size_mb:.2f} MB")

            # Update job status to EVALUATING, then PASSED
            # Follow the same state machine as local training: RUNNING -> EVALUATING -> PASSED
            if self.datadog_logger:
                self.datadog_logger.info("evaluation_started",
                    message="Evaluation started")
            
            self._update_job_status("EVALUATING", training_result)
            
            if self.datadog_logger:
                self.datadog_logger.info("evaluation_completed", {
                    "status": "PASSED"
                }, message="Evaluation completed: PASSED")
            
            # Now transition to PASSED and register model artifact
            if self.datadog_logger:
                self.datadog_logger.info("model_artifact_registration_started",
                    message="Registering model artifact")
            
            self._update_job_status("PASSED", training_result)
            
            if self.datadog_logger:
                artifact_version = training_result.get("artifact_version", "unknown")
                self.datadog_logger.info("model_artifact_registered", {
                    "artifact_version": artifact_version,
                }, message=f"Model artifact registered: {artifact_version}")
                
                self.datadog_logger.info("training_job_completed", {
                    "final_status": "PASSED"
                }, message="Training job completed: PASSED")

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
                }, message=f"Training failed: {type(exc).__name__}")
            
            try:
                self._update_job_status("FAILED", {"error": str(exc)})
            except Exception as update_exc:
                logger.error(f"Failed to update job status: {update_exc}")
                if self.datadog_logger:
                    self.datadog_logger.error("job_status_update_failed", {
                        "error_type": type(update_exc).__name__,
                    }, message=f"Job status update failed: {type(update_exc).__name__}")
            return 1

        except Exception as exc:
            logger.error(f"Unexpected error during training: {exc}", exc_info=True)
            
            if self.datadog_logger:
                self.datadog_logger.error("training_failed", {
                    "error_type": "UnexpectedError",
                    "error_class": type(exc).__name__,
                }, message=f"Training failed: {type(exc).__name__}")
            
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
                self.datadog_logger.info("model_loading_completed",
                    message="Model loading completed")
                self.datadog_logger.info("training_started", {
                    "lora_rank": profile.rank,
                    "lora_alpha": profile.alpha,
                    "epochs": profile.epochs,
                    "batch_size": profile.batch_size,
                }, message=f"QLoRA training started: rank={profile.rank}, α={profile.alpha}, epochs={profile.epochs}")

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
    
    def _determine_artifact_version(self) -> str:
        """Determine the next artifact version for this training job.
        
        Queries the database to find the next available adapter version number.
        This ensures S3 upload and database registration use the same version.
        
        Returns:
            Artifact version string (e.g., "adapter-v2")
        """
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            
            # Import model modules
            from app.control import models as control_models  # noqa: F401
            from app.data import models as data_models  # noqa: F401
            from app.serving import models as serving_models  # noqa: F401
            from app.training import models as training_models  # noqa: F401
            
            from app.training.repositories import ModelArtifactRepository
            
            # Create temporary database connection
            database_url = self.config["database_url"]
            engine = create_engine(database_url)
            Session = sessionmaker(bind=engine)
            db = Session()
            
            try:
                artifact_repo = ModelArtifactRepository(db)
                version = artifact_repo.next_version(self.config["project_id"])
                logger.info(f"Determined artifact version: {version}")
                return version
            finally:
                db.close()
                
        except Exception as exc:
            logger.warning(f"Failed to query next artifact version: {exc}. Using default adapter-v1")
            # Fallback to v1 if database query fails
            return "adapter-v1"

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

    def _upload_artifacts(self, adapter_path: str, artifact_version: str) -> tuple[int, float]:
        """Upload trained adapter artifacts to S3.
        
        Args:
            adapter_path: Path to adapter directory
            artifact_version: Artifact version string (e.g., "adapter-v1")
            
        Returns:
            Tuple of (file_count, total_size_mb)
            
        Raises:
            ArtifactUploadError: If upload fails
        """
        try:
            import boto3
        except ImportError as exc:
            raise ArtifactUploadError("boto3 not installed") from exc

        try:
            # Get AWS region from config for S3 client
            aws_region = self.config.get("aws_region")
            s3_client = boto3.client("s3", region_name=aws_region) if aws_region else boto3.client("s3")
            
            adapter_dir = Path(adapter_path)
            s3_bucket = self.config["s3_bucket"]
            s3_prefix = os.getenv("S3_ARTIFACT_PREFIX", "artifacts")

            logger.info(f"Uploading artifacts to S3: s3://{s3_bucket}/{s3_prefix}/...")

            artifact_key_prefix = (
                f"{s3_prefix}/"
                f"{self.config['project_id']}/"
                f"{artifact_version}"  # Use dynamically determined version
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

            total_size_mb = total_size / 1024 / 1024
            logger.info(
                f"Uploaded {file_count} artifact files "
                f"({total_size_mb:.2f} MB)"
            )
            
            return file_count, round(total_size_mb, 2)

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
        """Update training job status directly in database and register model artifacts.
        
        Uses SQLAlchemy to connect directly to the application database
        and update job status. When transitioning to PASSED, this method also creates
        the ModelArtifact record so the trained model appears in the UI.
        
        This mirrors the local training lifecycle where successful training creates:
        - ModelArtifact database record
        - Training job association
        - Dataset trained status
        - Project active artifact reference
        
        Args:
            job_id: Training job ID
            project_id: Project ID
            status: New job status (EVALUATING, PASSED, or FAILED)
            result: Training result or error dictionary
        """
        try:
            from datetime import datetime
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker

            # Import all model modules to register SQLAlchemy metadata
            # This ensures all foreign key tables (including 'projects') are registered
            from app.control import models as control_models  # noqa: F401
            from app.control.repositories import ProjectRepository
            from app.data import models as data_models  # noqa: F401
            from app.serving import models as serving_models  # noqa: F401
            from app.training import models as training_models  # noqa: F401
            
            from app.training.models import (
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

            # Create database connection after all metadata is registered
            database_url = self.config["database_url"]
            engine = create_engine(database_url)
            Session = sessionmaker(bind=engine)
            db = Session()

            try:
                # Get repositories
                job_repo = TrainingJobRepository(db)
                artifact_repo = ModelArtifactRepository(db)
                dataset_repo = DatasetRepository(db)
                project_repo = ProjectRepository(db)
                
                job = job_repo.get_for_project(job_id, project_id)

                if not job:
                    raise JobStatusUpdateError(f"Job {job_id} not found")

                # Transition to the requested status following the state machine
                if status == "EVALUATING":
                    # RUNNING -> EVALUATING
                    job_repo.transition(
                        job,
                        TrainingJobStatus.EVALUATING,
                    )
                    
                elif status == "PASSED":
                    # EVALUATING -> PASSED (terminal state)
                    # This is where we register the model artifact
                    
                    # Get artifact version from training result
                    artifact_version = result.get("artifact_version", "adapter-v1")
                    
                    # Create S3 adapter path
                    s3_bucket = self.config["s3_bucket"]
                    s3_prefix = os.getenv("S3_ARTIFACT_PREFIX", "artifacts")
                    adapter_s3_path = f"s3://{s3_bucket}/{s3_prefix}/{project_id}/{artifact_version}"
                    
                    # Extract evaluation score from result (basic evaluation: training succeeded)
                    evaluation_score = result.get("metrics", {}).get("training_loss", 0.0)
                    
                    # Create ModelArtifact record
                    artifact = ModelArtifact(
                        project_id=project_id,
                        artifact_version=artifact_version,
                        base_model=self.config["base_model"],
                        adapter_path=adapter_s3_path,
                        dataset_id=self.config["dataset_id"],
                        evaluation_score=evaluation_score,
                        status=ModelArtifactStatus.READY,
                    )
                    
                    artifact_repo.add(artifact)
                    db.flush()  # Get artifact ID
                    
                    logger.info(f"Created ModelArtifact {artifact.id} version {artifact_version} for job {job_id}")
                    
                    # Update training job with artifact reference
                    job_repo.transition(
                        job,
                        TrainingJobStatus.PASSED,
                        artifact_id=artifact.id,
                        finished_at=datetime.utcnow(),
                    )
                    
                    # Mark dataset as trained
                    dataset = dataset_repo.get_for_project(self.config["dataset_id"], project_id)
                    if dataset:
                        dataset_repo.set_status(dataset, DatasetStatus.TRAINED)
                    
                    # Set artifact as active for project (following local training behavior)
                    project = project_repo.get(project_id)
                    if project:
                        project_repo.set_active_artifact(project, artifact.id)
                    
                elif status == "FAILED":
                    # RUNNING/EVALUATING -> FAILED (terminal state)
                    job_repo.transition(
                        job,
                        TrainingJobStatus.FAILED,
                        finished_at=datetime.utcnow(),
                    )

                db.commit()
                logger.info(f"Job status updated to {status}")

            finally:
                db.close()

        except Exception as exc:
            logger.error(f"Direct database update failed: {exc}")
            raise

    def _cleanup(self) -> None:
        """Clean up temporary files and resources.
        
        CRITICAL: This method MUST always terminate the EC2 instance to prevent
        leaving a GPU instance running indefinitely after training completes or fails.
        """
        # Terminate EC2 instance first (most critical cleanup)
        self._terminate_ec2_instance()
        
        # Then clean up temporary files
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
    
    def _terminate_ec2_instance(self) -> None:
        """Terminate the current EC2 instance.
        
        This method MUST be called in the finally block to ensure the temporary
        GPU instance is always terminated, regardless of training success or failure.
        
        Termination is only attempted if:
        1. TRAINING_EC2_TERMINATE_ON_COMPLETION is true (default)
        2. We can successfully retrieve the current instance ID from EC2 metadata
        
        If termination fails, the error is logged but not raised to avoid
        hiding the original training error.
        """
        try:
            # Check if auto-termination is enabled (default: true)
            terminate_enabled = os.environ.get(
                "TRAINING_EC2_TERMINATE_ON_COMPLETION", "true"
            ).lower() in {"true", "1", "yes"}
            
            if not terminate_enabled:
                logger.info(
                    "EC2 instance termination disabled by "
                    "TRAINING_EC2_TERMINATE_ON_COMPLETION=false"
                )
                return
            
            # Get current instance ID from EC2 metadata service
            instance_id = self._get_current_instance_id()
            
            if not instance_id:
                logger.warning(
                    "Could not retrieve EC2 instance ID from metadata service. "
                    "Instance will NOT be terminated automatically. "
                    "Manual cleanup may be required."
                )
                return
            
            # Get AWS region from configuration
            aws_region = self.config.get("aws_region")
            
            if not aws_region:
                logger.error(
                    "CRITICAL: AWS region not configured in training config. "
                    "Cannot terminate EC2 instance. Manual cleanup required."
                )
                if self.datadog_logger:
                    self.datadog_logger.error("instance_termination_failed", {
                        "reason": "aws_region_not_configured",
                        "instance_id": instance_id,
                    })
                return
            
            logger.info(f"Terminating EC2 instance {instance_id} in region {aws_region}")
            
            if self.datadog_logger:
                self.datadog_logger.info("instance_termination_requested", {
                    "instance_id": instance_id,
                    "aws_region": aws_region,
                })
            
            # Import boto3 and terminate the instance
            try:
                import boto3
            except ImportError:
                logger.error(
                    "boto3 not available. Cannot terminate EC2 instance. "
                    "Manual cleanup required."
                )
                return
            
            # Create EC2 client with explicit region
            ec2_client = boto3.client("ec2", region_name=aws_region)
            response = ec2_client.terminate_instances(InstanceIds=[instance_id])
            
            current_state = response["TerminatingInstances"][0]["CurrentState"]["Name"]
            logger.info(
                f"EC2 instance {instance_id} termination initiated. "
                f"State: {current_state}"
            )
            
            if self.datadog_logger:
                self.datadog_logger.info("instance_termination_succeeded", {
                    "instance_id": instance_id,
                    "state": current_state,
                })
            
        except Exception as exc:
            # Log termination failure but do not raise to avoid hiding original error
            logger.error(
                f"CRITICAL: Failed to terminate EC2 instance: {exc}. "
                f"Manual cleanup may be required to stop billing.",
                exc_info=True,
            )
            if self.datadog_logger:
                self.datadog_logger.error("instance_termination_failed", {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:200],
                })
    
    def _get_current_instance_id(self) -> str | None:
        """Get the current EC2 instance ID from the metadata service.
        
        Uses IMDSv2 (token-based) metadata service for security.
        
        Returns:
            Instance ID string, or None if not running on EC2 or metadata unavailable
        """
        try:
            import requests
            
            # IMDSv2: First obtain a session token
            token_response = requests.put(
                "http://169.254.169.254/latest/api/token",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
                timeout=2,
            )
            
            if token_response.status_code != 200:
                logger.debug(
                    f"Failed to obtain IMDSv2 token: {token_response.status_code}"
                )
                return None
            
            token = token_response.text.strip()
            
            # Use token to fetch instance ID
            id_response = requests.get(
                "http://169.254.169.254/latest/meta-data/instance-id",
                headers={"X-aws-ec2-metadata-token": token},
                timeout=2,
            )
            
            if id_response.status_code != 200:
                logger.debug(
                    f"Failed to fetch instance ID: {id_response.status_code}"
                )
                return None
            
            instance_id = id_response.text.strip()
            
            if not instance_id or not instance_id.startswith("i-"):
                logger.debug(f"Invalid instance ID format: {instance_id}")
                return None
            
            return instance_id
            
        except Exception as exc:
            logger.debug(f"Could not retrieve instance ID from metadata: {exc}")
            return None


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
