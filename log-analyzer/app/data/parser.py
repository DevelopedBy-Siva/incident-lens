"""Parse the plain-text log format used by Datadog and the demo streamer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone


_LINE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\s+"
    r"(?P<level>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL|FATAL)\s+"
    r"(?P<message>.*)$",
    re.IGNORECASE,
)
_BRACKET_TIME = re.compile(r"\[(?P<timestamp>\d{4}-\d{2}-\d{2}T[^\]]+)\]")
_LEVEL = re.compile(r"\b(?P<level>CRITICAL|FATAL|ERROR|WARN|WARNING|INFO|DEBUG|TRACE)\b", re.I)
_EXCEPTION = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Exception|Error))\b")


@dataclass(frozen=True)
class ParsedLog:
    raw: str
    timestamp: datetime
    level: str
    message: str
    exception_type: str | None

    def __init__(self, raw: str):
        raw = str(raw or "").strip()
        match = _LINE.match(raw)
        if match:
            timestamp = _parse_timestamp(match.group("timestamp"))
            level = _normalise_level(match.group("level"))
            message = match.group("message").strip()
        else:
            bracket = _BRACKET_TIME.search(raw)
            timestamp = _parse_timestamp(bracket.group("timestamp")) if bracket else datetime.now(timezone.utc)
            level_match = _LEVEL.search(raw)
            level = _normalise_level(level_match.group("level")) if level_match else "INFO"
            message = raw[level_match.end() :].lstrip(" :|-[]") if level_match else raw

        exception = _EXCEPTION.search(message)
        object.__setattr__(self, "raw", raw)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "exception_type", exception.group(1) if exception else None)


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def _normalise_level(value: str) -> str:
    value = value.upper()
    return "WARN" if value == "WARNING" else ("CRITICAL" if value == "FATAL" else value)
