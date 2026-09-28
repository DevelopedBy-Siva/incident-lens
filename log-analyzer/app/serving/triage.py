"""Conservative evidence baseline for incident analysis.

This module classifies urgency from observed log evidence.  It intentionally does
not encode incident-specific diagnoses or runbooks: those belong to the trained
model and project data, not application code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
DISPOSITION_RANK = {"NO_ACTION": 0, "OBSERVE": 1, "NEEDS_DEV": 2, "NEEDS_ONCALL": 3, "ESCALATE": 4}

CRITICAL_SIGNALS = (
    "outofmemory", "oomkilled", "heap space", "data corruption", "data loss",
    "security breach", "sql injection", "regional outage", "primary down",
)
HIGH_SIGNALS = (
    "connectionpoolexhausted", "connection pool exhausted", "databaseconnection",
    "connection refused", "certificate expired", "tls certificate", "5xx spike",
    "paymentgateway", "queue depth", "consumer lag", "diskspacecritical",
    "disk full", "crashloop", "panic", "deadlock",
)
MEDIUM_SIGNALS = (
    "timeout", "timed out", "latency", "degraded", "rate limit", "4xx",
    "5xx", "exception", "failed", "error", "unavailable", "retry",
)


@dataclass(frozen=True)
class TriageResult:
    severity: str
    disposition: str
    confidence: float
    summary: str
    root_cause: str | None
    next_steps: list[str]
    ticket_title: str = ""


def classify(incident, evidence=None) -> TriageResult:
    lines = list(getattr(incident, "sample_lines", None) or [])
    if evidence is not None:
        lines.extend(getattr(evidence, "sample_lines", []) or [])
    text = "\n".join(lines).lower()
    count = int(getattr(incident, "count", 1) or 1)

    if _contains(text, CRITICAL_SIGNALS):
        severity, disposition = "critical", "ESCALATE"
    elif _contains(text, HIGH_SIGNALS) or (count >= 5 and _contains(text, MEDIUM_SIGNALS)):
        severity, disposition = "high", "NEEDS_ONCALL"
    elif _contains(text, MEDIUM_SIGNALS) or count >= 2:
        severity, disposition = "medium", "NEEDS_DEV"
    else:
        severity, disposition = "low", "OBSERVE"

    source = _component(lines, getattr(incident, "source", "service"))
    event = _event_excerpt(lines)
    signal = _first_signal(text) or "an anomalous event"
    event_count = f"{count} observed event{'s' if count != 1 else ''}"
    return TriageResult(
        severity=severity,
        disposition=disposition,
        confidence=_confidence(severity, count),
        summary=f"{source}: {event} ({event_count}).",
        root_cause=(
            f"The event establishes {signal}, but the supplied logs do not establish a root cause."
            if severity != "low" else None
        ),
        next_steps=_baseline_steps(source, severity),
        ticket_title=f"{source}: {severity} incident",
    )


def merge_with_baseline(candidate, baseline: TriageResult):
    """Never permit an optional model to downgrade deterministic incident evidence."""
    severity = candidate.severity.lower()
    disposition = candidate.disposition.upper()
    if severity not in SEVERITY_RANK or disposition not in DISPOSITION_RANK:
        return baseline
    if SEVERITY_RANK[severity] < SEVERITY_RANK[baseline.severity]:
        return baseline
    if DISPOSITION_RANK[disposition] < DISPOSITION_RANK[baseline.disposition]:
        return baseline
    return candidate


def _baseline_steps(source: str, severity: str) -> list[str]:
    if severity == "low":
        return [f"Monitor {source} for the same event signature before taking action."]
    return [
        f"Search {source} logs for the same signature around this timestamp to establish recurrence and scope.",
        "Correlate the event with available request, peer, dependency, and deployment metadata.",
        "Verify whether the event recurs before selecting a remediation.",
    ]


def _confidence(severity: str, count: int) -> float:
    if severity == "critical":
        return 0.95
    if severity == "high":
        return 0.9
    if severity == "medium":
        return 0.8 if count >= 2 else 0.7
    return 0.7


def _component(lines: list[str], fallback: str) -> str:
    for line in lines:
        match = re.search(r"\[([^\]]+)\]", line)
        if match:
            return match.group(1)
    return str(fallback or "service")


def _event_excerpt(lines: list[str]) -> str:
    line = next((item.strip() for item in lines if item.strip()), "log event")
    line = re.sub(r"^\S+\s+(?:TRACE|DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL)\s+", "", line, flags=re.IGNORECASE)
    line = re.sub(r"^\[[^\]]+\]\s*", "", line)
    return line[:300].rstrip(".")


def _contains(text: str, signals: tuple[str, ...]) -> bool:
    return any(signal in text for signal in signals)


def _first_signal(text: str) -> str | None:
    for signal in CRITICAL_SIGNALS + HIGH_SIGNALS + MEDIUM_SIGNALS:
        if signal in text:
            return signal.replace("connectionpoolexhausted", "connection pool exhaustion").replace("databaseconnection", "database connection failure")
    return None
