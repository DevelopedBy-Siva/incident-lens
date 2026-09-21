from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class TrainingRequest:
    project_id: str
    dataset_id: str
    dataset_version: str
    base_model: str
    dataset_content: bytes
    expected_record_count: int


@dataclass(frozen=True)
class TrainingResult:
    succeeded: bool
    engine: str
    simulated: bool
    metrics: dict[str, Any]
    error: str | None = None


class TrainingEngine(ABC):
    """Replaceable boundary for local or provider-hosted adapter training."""

    @abstractmethod
    def train(self, request: TrainingRequest) -> TrainingResult:
        raise NotImplementedError


class PlaceholderTrainingEngine(TrainingEngine):
    """Exercise the training lifecycle without creating model weights."""

    name = "placeholder-v1"

    def train(self, request: TrainingRequest) -> TrainingResult:
        started = perf_counter()
        serialized_records = sum(
            1 for line in request.dataset_content.splitlines() if line.strip()
        )
        duration = perf_counter() - started
        return TrainingResult(
            succeeded=True,
            engine=self.name,
            simulated=True,
            metrics={
                "training_mode": "simulation",
                "base_model": request.base_model,
                "dataset_version": request.dataset_version,
                "expected_records": request.expected_record_count,
                "records_seen": serialized_records,
                "duration_seconds": round(duration, 6),
                "weights_created": False,
            },
        )
