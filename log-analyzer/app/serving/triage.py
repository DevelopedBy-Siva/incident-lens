"""Deterministic incident classification used before any optional model enrichment."""

from __future__ import annotations

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

    signal = _first_signal(text) or "repeated warning or error"
    summary = f"{getattr(incident, 'source', 'service')} shows {signal} ({count} observed event{'s' if count != 1 else ''})."
    root_cause = f"The logs indicate {signal}; confirm the affected dependency and recent changes." if severity != "low" else None
    next_steps = _steps(severity, signal)
    return TriageResult(severity, disposition, 0.9 if severity != "low" else 0.7, summary, root_cause, next_steps)


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


def _contains(text: str, signals: tuple[str, ...]) -> bool:
    return any(signal in text for signal in signals)


def _first_signal(text: str) -> str | None:
    for signal in CRITICAL_SIGNALS + HIGH_SIGNALS + MEDIUM_SIGNALS:
        if signal in text:
            return signal.replace("connectionpoolexhausted", "connection pool exhaustion").replace("databaseconnection", "database connection failure")
    return None


def _steps(severity: str, signal: str) -> list[str]:
    if severity == "critical":
        return ["Page the incident owner and assess the active impact.", "Stabilize the affected service before applying a permanent fix.", f"Inspect logs and metrics related to {signal}."]
    if severity == "high":
        return [f"Inspect logs and metrics related to {signal}.", "Verify whether the failure is still occurring.", "Assign operational ownership and mitigate if it persists."]
    if severity == "medium":
        return [f"Investigate the source of {signal}.", "Check recent deploys and dependency health.", "Monitor recurrence after the fix."]
    return ["Monitor for recurrence before taking action."]
