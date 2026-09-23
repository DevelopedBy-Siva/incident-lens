import json
from collections.abc import Sequence
from typing import Any


class DatasetValidationError(ValueError):
    pass


class JsonLinesDatasetSerializer:
    """Serialize training examples as deterministic UTF-8 JSON Lines."""

    format_version = "jsonl-v1"
    file_extension = "jsonl"

    def serialize(self, examples: Sequence[dict[str, Any]]) -> bytes:
        lines = [
            json.dumps(
                example,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for example in examples
        ]
        return ("\n".join(lines) + "\n").encode("utf-8")

    def normalize(self, examples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert supported legacy records to the current training schema."""
        normalized = []
        for example in examples:
            if (
                isinstance(example, dict)
                and "expected_output" not in example
                and isinstance(example.get("output"), dict)
            ):
                output = example["output"]
                example = {
                    key: value for key, value in example.items() if key != "output"
                }
                example["expected_output"] = output
            normalized.append(example)
        return normalized

    def validate(self, examples: Sequence[dict[str, Any]]) -> None:
        if not examples:
            raise DatasetValidationError("Dataset must contain at least one record")

        for index, example in enumerate(examples, start=1):
            if not isinstance(example, dict):
                raise DatasetValidationError(f"Record {index} must be an object")
            input_data = example.get("input")
            output = example.get("expected_output")
            if not isinstance(input_data, dict) or not isinstance(output, dict):
                raise DatasetValidationError(
                    f"Record {index} requires input and expected_output objects"
                )

            logs = input_data.get("logs")
            incident = input_data.get("incident")
            if incident is not None:
                if not isinstance(incident, dict):
                    raise DatasetValidationError(
                        f"Record {index} requires an input.incident object"
                    )
                for field in ("id", "source", "signature"):
                    if not self._non_empty_string(incident.get(field)):
                        raise DatasetValidationError(
                            f"Record {index} requires input.incident.{field}"
                        )
                if not isinstance(input_data.get("evidence"), dict):
                    raise DatasetValidationError(
                        f"Record {index} requires an input.evidence object"
                    )
            else:
                for field in ("service", "environment"):
                    if not self._non_empty_string(input_data.get(field)):
                        raise DatasetValidationError(
                            f"Record {index} requires input.{field}"
                        )
            if not isinstance(logs, list) or not all(
                isinstance(line, str) for line in logs
            ):
                raise DatasetValidationError(
                    f"Record {index} requires input.logs as a list of strings"
                )
            if not isinstance(input_data.get("metadata"), dict):
                raise DatasetValidationError(
                    f"Record {index} requires an input.metadata object"
                )

            for field in ("severity", "disposition", "summary"):
                if not self._non_empty_string(output.get(field)):
                    raise DatasetValidationError(
                        f"Record {index} requires expected_output.{field}"
                    )
            actions = output.get("recommended_actions")
            if not isinstance(actions, list) or not all(
                isinstance(action, str) for action in actions
            ):
                raise DatasetValidationError(
                    f"Record {index} requires expected_output.recommended_actions "
                    "as a list of strings"
                )
            if output.get("root_cause") is not None and not isinstance(
                output["root_cause"], dict
            ):
                raise DatasetValidationError(
                    f"Record {index} expected_output.root_cause must be an object or null"
                )

    @staticmethod
    def _non_empty_string(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def deserialize(self, content: bytes) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            content.decode("utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            example = json.loads(line)
            if not isinstance(example, dict):
                raise ValueError(
                    f"Dataset record on line {line_number} must be an object"
                )
            examples.append(example)
        return examples
