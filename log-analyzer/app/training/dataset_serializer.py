"""Strict JSONL serialization for the same contract used at inference time."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from app.shared.incident_policy import validate_output_schema


class DatasetValidationError(ValueError):
    pass


class JsonLinesDatasetSerializer:
    format_version = "jsonl-v2"
    file_extension = "jsonl"

    def serialize(self, examples: Sequence[dict[str, Any]]) -> bytes:
        self.validate(examples)
        lines = [json.dumps(example, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for example in examples]
        return ("\n".join(lines) + "\n").encode("utf-8")

    def normalize(self, examples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for example in examples:
            item = dict(example)
            if "expected_output" not in item and isinstance(item.get("output"), dict):
                item["expected_output"] = item.pop("output")
            normalized.append(item)
        return normalized

    def validate(self, examples: Sequence[dict[str, Any]]) -> None:
        if not examples:
            raise DatasetValidationError("Dataset must contain at least one record")
        for index, example in enumerate(examples, start=1):
            if not isinstance(example, dict) or set(example) != {"input", "expected_output"}:
                raise DatasetValidationError(f"Record {index} must contain only input and expected_output")
            input_data = example["input"]
            if not isinstance(input_data, dict):
                raise DatasetValidationError(f"Record {index} input must be an object")
            logs = input_data.get("logs")
            if not isinstance(logs, list) or not logs or not all(isinstance(line, str) and line.strip() for line in logs):
                raise DatasetValidationError(f"Record {index} requires non-empty input.logs")
            for field in ("service", "environment"):
                if not isinstance(input_data.get(field), str) or not input_data[field].strip():
                    raise DatasetValidationError(f"Record {index} requires input.{field}")
            if not isinstance(input_data.get("metadata"), dict):
                raise DatasetValidationError(f"Record {index} requires input.metadata")
            valid, reason = validate_output_schema(example["expected_output"])
            if not valid:
                raise DatasetValidationError(f"Record {index}: {reason}")

    def deserialize(self, content: bytes) -> list[dict[str, Any]]:
        try:
            records = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatasetValidationError("Dataset is not valid UTF-8 JSONL") from exc
        self.validate(records)
        return records
