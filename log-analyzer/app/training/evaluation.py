from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.training.models import Dataset
from app.training.training_engine import TrainingResult


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    passed: bool
    metrics: dict[str, Any]


class EvaluationService(ABC):
    @abstractmethod
    def evaluate(
        self,
        dataset: Dataset,
        dataset_content: bytes,
        training_result: TrainingResult,
    ) -> EvaluationResult:
        raise NotImplementedError


class BasicEvaluationService(EvaluationService):
    """Evaluate availability, dataset size, and simulated training execution."""

    def evaluate(
        self,
        dataset: Dataset,
        dataset_content: bytes,
        training_result: TrainingResult,
    ) -> EvaluationResult:
        serialized_record_count = sum(
            1 for line in dataset_content.splitlines() if line.strip()
        )
        dataset_available = bool(dataset_content)
        dataset_non_empty = dataset.record_count > 0 and serialized_record_count > 0
        record_count_matches = dataset.record_count == serialized_record_count
        dataset_size_valid = dataset_non_empty and record_count_matches
        training_succeeded = training_result.succeeded

        checks = [dataset_available, dataset_size_valid, training_succeeded]
        score = round(sum(1 for check in checks if check) / len(checks), 4)
        passed = all(checks) and record_count_matches
        return EvaluationResult(
            score=score,
            passed=passed,
            metrics={
                "dataset_available": dataset_available,
                "dataset_non_empty": dataset_non_empty,
                "metadata_record_count": dataset.record_count,
                "serialized_record_count": serialized_record_count,
                "record_count_matches": record_count_matches,
                "dataset_size_valid": dataset_size_valid,
                "training_succeeded": training_succeeded,
                "training_engine": training_result.engine,
                "training_simulated": training_result.simulated,
            },
        )
