import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
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
    """Evaluate dataset, loss, completion, and persisted adapter integrity."""

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
        training_completed = training_result.metrics.get("training_completed") is True
        training_loss = training_result.metrics.get("training_loss")
        training_loss_valid = self._finite(training_loss)
        validation_loss = training_result.metrics.get("validation_loss")
        validation_loss_valid = validation_loss is None or self._finite(validation_loss)
        artifact_integrity, integrity_metrics = self._artifact_integrity(
            training_result
        )

        checks = [
            dataset_available,
            dataset_size_valid,
            training_succeeded,
            training_completed,
            training_loss_valid,
            validation_loss_valid,
            artifact_integrity,
        ]
        score = round(sum(1 for check in checks if check) / len(checks), 4)
        passed = all(checks)
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
                "training_completed": training_completed,
                "training_loss": training_loss,
                "training_loss_valid": training_loss_valid,
                "validation_loss": validation_loss,
                "validation_loss_available": validation_loss is not None,
                "validation_loss_valid": validation_loss_valid,
                "artifact_integrity": artifact_integrity,
                **integrity_metrics,
            },
        )

    @staticmethod
    def _finite(value) -> bool:
        if value is None or isinstance(value, bool):
            return False
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _artifact_integrity(
        training_result: TrainingResult,
    ) -> tuple[bool, dict[str, Any]]:
        adapter_path = Path(training_result.adapter_path)
        config_path = adapter_path / "adapter_config.json"
        weights_path = adapter_path / "adapter_model.safetensors"
        config_valid = False
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            expected_base_model = training_result.metrics.get("base_model")
            config_valid = (
                isinstance(config, dict)
                and bool(config.get("base_model_name_or_path"))
                and str(config.get("peft_type", "")).upper() == "LORA"
                and (
                    expected_base_model is None
                    or config.get("base_model_name_or_path") == expected_base_model
                )
            )
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            config_valid = False

        weights_present = weights_path.is_file() and weights_path.stat().st_size > 0
        manifest = {
            entry.get("name"): entry
            for entry in training_result.artifact_files
            if isinstance(entry, dict) and entry.get("name")
        }
        manifest_valid = {
            "adapter_config.json",
            "adapter_model.safetensors",
        }.issubset(manifest) and all(
            BasicEvaluationService._manifest_entry_valid(adapter_path, entry)
            for entry in manifest.values()
        )
        integrity = (
            adapter_path.is_dir()
            and config_valid
            and weights_present
            and manifest_valid
        )
        return integrity, {
            "adapter_config_valid": config_valid,
            "adapter_weights_present": weights_present,
            "artifact_manifest_valid": manifest_valid,
        }

    @staticmethod
    def _manifest_entry_valid(adapter_path: Path, entry: dict[str, Any]) -> bool:
        path = adapter_path / str(entry["name"])
        if not path.is_file():
            return False
        content = path.read_bytes()
        return (
            entry.get("size_bytes") == len(content)
            and entry.get("sha256") == hashlib.sha256(content).hexdigest()
        )
