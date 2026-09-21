from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.training.training_profile import LoraTrainingProfile


@dataclass(frozen=True)
class TrainingRequest:
    project_id: str
    dataset_id: str
    dataset_version: str
    base_model: str
    dataset_content: bytes
    expected_record_count: int
    adapter_output_path: str
    profile: LoraTrainingProfile


@dataclass(frozen=True)
class TrainingResult:
    succeeded: bool
    engine: str
    adapter_path: str
    metrics: dict[str, Any]
    framework_versions: dict[str, str]
    artifact_files: tuple[dict[str, Any], ...]
    error: str | None = None


class TrainingEngine(ABC):
    """Replaceable boundary for local or provider-hosted adapter training."""

    @abstractmethod
    def train(self, request: TrainingRequest) -> TrainingResult:
        raise NotImplementedError


def configured_training_engine() -> TrainingEngine:
    from app.training.lora_trainer import TransformersPeftTrainingEngine

    return TransformersPeftTrainingEngine()
