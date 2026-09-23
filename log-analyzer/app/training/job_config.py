"""
Training Job Configuration Serialization
Serializes training job metadata for EC2 User Data encoding.
"""

import json
from dataclasses import asdict, dataclass
from typing import Any

from app.training.training_profile import LoraTrainingProfile


@dataclass(frozen=True)
class TrainingJobConfig:
    """Serializable training job configuration for EC2 execution.
    
    This configuration is encoded in EC2 User Data and passed to the training bootstrap
    script running on the temporary GPU instance.
    """

    # Job and project identifiers
    job_id: str
    project_id: str
    dataset_id: str
    
    # Dataset storage location
    dataset_storage_key: str
    s3_bucket: str
    
    # Database and API configuration
    database_url: str
    
    # Base model configuration
    base_model: str
    
    # Git repository configuration (for runtime code cloning)
    git_repository_url: str
    git_commit_sha: str
    
    # LoRA training hyperparameters
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_epochs: int
    lora_learning_rate: float
    lora_batch_size: int
    lora_gradient_accumulation_steps: int
    lora_max_sequence_length: int
    lora_validation_fraction: float
    lora_seed: int
    lora_target_modules: list[str]
    
    # Optional: selected record indices for subset training
    selected_record_indices: list[int] | None = None
    
    # Optional: HuggingFace token for model access
    huggingface_token: str | None = None

    def to_json(self) -> str:
        """Serialize to JSON string for User Data encoding."""
        data = asdict(self)
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "TrainingJobConfig":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for template expansion."""
        return asdict(self)


def create_training_job_config(
    job_id: str,
    project_id: str,
    dataset_id: str,
    dataset_storage_key: str,
    s3_bucket: str,
    database_url: str,
    base_model: str,
    git_repository_url: str,
    git_commit_sha: str,
    training_profile: LoraTrainingProfile,
    selected_record_indices: list[int] | None = None,
    huggingface_token: str | None = None,
) -> TrainingJobConfig:
    """Create a training job configuration from parameters.
    
    Args:
        job_id: Training job ID
        project_id: Project ID
        dataset_id: Dataset ID
        dataset_storage_key: S3 storage key for dataset
        s3_bucket: S3 bucket name
        database_url: Database connection URL
        base_model: Base model name/path
        git_repository_url: Git repository URL to clone
        git_commit_sha: Git commit SHA to checkout
        training_profile: LoRA training profile with hyperparameters
        selected_record_indices: Optional list of record indices to train on
        huggingface_token: Optional HuggingFace API token
    
    Returns:
        TrainingJobConfig instance ready for serialization
    """
    return TrainingJobConfig(
        job_id=job_id,
        project_id=project_id,
        dataset_id=dataset_id,
        dataset_storage_key=dataset_storage_key,
        s3_bucket=s3_bucket,
        database_url=database_url,
        base_model=base_model,
        git_repository_url=git_repository_url,
        git_commit_sha=git_commit_sha,
        lora_rank=training_profile.rank,
        lora_alpha=training_profile.alpha,
        lora_dropout=training_profile.dropout,
        lora_epochs=training_profile.epochs,
        lora_learning_rate=training_profile.learning_rate,
        lora_batch_size=training_profile.batch_size,
        lora_gradient_accumulation_steps=training_profile.gradient_accumulation_steps,
        lora_max_sequence_length=training_profile.max_sequence_length,
        lora_validation_fraction=training_profile.validation_fraction,
        lora_seed=training_profile.seed,
        lora_target_modules=list(training_profile.target_modules),
        selected_record_indices=selected_record_indices,
        huggingface_token=huggingface_token,
    )
