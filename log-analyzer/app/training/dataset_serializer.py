import json
from collections.abc import Sequence
from typing import Any


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
