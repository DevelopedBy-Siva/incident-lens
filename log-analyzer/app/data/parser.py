"""Parse the plain-text log format used by Datadog and the demo streamer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


_LINE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\s+"
    r"(?P<level>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL|FATAL)\s+"
    r"(?P<message>.*)$",
    re.IGNORECASE,
)
_BRACKET_TIME = re.compile(r"\[(?P<timestamp>\d{4}-\d{2}-\d{2}T[^\]]+)\]")
_LEVEL = re.compile(r"\b(?P<level>CRITICAL|FATAL|ERROR|WARN|WARNING|INFO|DEBUG|TRACE)\b", re.I)
_EXCEPTION = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Exception|Error))\b")
_BRACKET_COMPONENT = re.compile(r"\[([^\]]+)\]")
_K8S_CONTAINER = re.compile(r"\bcontainer\s+([A-Za-z0-9][A-Za-z0-9_.-]*)", re.I)
_SERVICE_FIELD = re.compile(r"\b(?:service(?:\.name)?|component)=([A-Za-z0-9][A-Za-z0-9_.-]*)", re.I)
_POD_FIELD = re.compile(r"\bPod\s+([A-Za-z0-9][A-Za-z0-9_.-]*)", re.I)
_HOST_FIELD = re.compile(r"\bhost=([A-Za-z0-9][A-Za-z0-9_.-]*)", re.I)


@dataclass(frozen=True)
class ParsedLog:
    raw: str
    timestamp: datetime
    level: str
    message: str
    exception_type: str | None
    component: str | None
    entity: str | None
    attributes: dict[str, Any]

    def __init__(
        self,
        raw: str,
        observed_at: datetime | None = None,
        attributes: dict[str, Any] | None = None,
    ):
        raw = str(raw or "").strip()
        match = _LINE.match(raw)
        if match:
            timestamp = _parse_timestamp(match.group("timestamp"))
            level = _normalise_level(match.group("level"))
            message = match.group("message").strip()
        else:
            bracket = _BRACKET_TIME.search(raw)
            timestamp = _parse_timestamp(bracket.group("timestamp")) if bracket else (observed_at or datetime.now(timezone.utc))
            level_match = _LEVEL.search(raw)
            level = _normalise_level(level_match.group("level")) if level_match else "INFO"
            message = raw[level_match.end() :].lstrip(" :|-[]") if level_match else raw

        exception = _EXCEPTION.search(message)
        component = _component(message)
        object.__setattr__(self, "raw", raw)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "exception_type", exception.group(1) if exception else None)
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "entity", _entity(message))
        object.__setattr__(self, "attributes", dict(attributes or {}))


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def _normalise_level(value: str) -> str:
    value = value.upper()
    return "WARN" if value == "WARNING" else ("CRITICAL" if value == "FATAL" else value)


def resolve_source(parsed_log: ParsedLog, fallback: str) -> str:
    """Prefer an observed service/component over an integration-wide fallback."""
    return parsed_log.component or str(fallback or "unknown")


def _component(message: str) -> str | None:
    for pattern in (_BRACKET_COMPONENT, _K8S_CONTAINER, _SERVICE_FIELD):
        match = pattern.search(message)
        if match:
            return match.group(1)
    return None


def _entity(message: str) -> str | None:
    for pattern in (_POD_FIELD, _HOST_FIELD):
        match = pattern.search(message)
        if match:
            return match.group(1)
    return None
