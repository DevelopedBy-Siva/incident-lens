"""Stable, human-debuggable incident signatures."""

from __future__ import annotations

import hashlib
import re


_DYNAMIC_VALUES = (
    (r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", "<id>"),
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<ip>"),
    (r"\b(?:trace|request|span|user_id|order_id)=[^\s,]+", "<value>"),
    (r"\b\d+(?:\.\d+)?(?:ms|s|mb|gb|%)\b", "<metric>"),
    (r"\b\d+\b", "<n>"),
)


def normalize_message(message: str) -> str:
    normalized = message.lower()
    for pattern, replacement in _DYNAMIC_VALUES:
        normalized = re.sub(pattern, replacement, normalized, flags=re.I)
    return re.sub(r"\s+", " ", normalized).strip()[:500]


def generate_signature(source: str, parsed_log) -> str:
    """Group the same failure shape while retaining level and service boundaries."""
    text = "|".join(
        (str(source).strip().lower(), parsed_log.level, parsed_log.exception_type or "", normalize_message(parsed_log.message))
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
