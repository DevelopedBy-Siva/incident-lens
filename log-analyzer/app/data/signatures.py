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
    """Create a short-lived incident-window key, not a per-line error key.

    Different symptoms from the same pod/service (for example OOMKilled followed
    by SIGKILL) must reach analysis together.  A concrete pod/host entity gives
    the narrowest window; otherwise the service is the correlation boundary.
    """
    entity = getattr(parsed_log, "entity", None) or "service"
    text = "|".join((str(source).strip().lower(), str(entity).strip().lower()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
