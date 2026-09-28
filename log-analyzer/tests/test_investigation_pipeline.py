from types import SimpleNamespace

from app.serving.investigator import InvestigationLoop
from app.serving.triage import classify


def incident(lines, count=1):
    return SimpleNamespace(id="incident-1", source="checkout", environment="prod", signature="sig", count=count, sample_lines=lines)


def test_critical_evidence_cannot_be_mapped_to_low():
    result = classify(incident(["2026-01-01T00:00:00Z ERROR OutOfMemoryError: Java heap space"], 1))
    assert (result.severity, result.disposition) == ("critical", "ESCALATE")


def test_repeated_timeout_is_high_and_oncall():
    result = classify(incident(["2026-01-01T00:00:00Z ERROR dependency timeout"], 5))
    assert (result.severity, result.disposition) == ("high", "NEEDS_ONCALL")


def test_missing_model_uses_reliable_rule_based_result():
    project = SimpleNamespace(id="project-1")
    result = InvestigationLoop().investigate(incident(["2026-01-01T00:00:00Z ERROR ConnectionPoolExhaustedError"], 2), project)
    assert result.severity == "high"
    assert result.disposition == "NEEDS_ONCALL"
    assert result.ticket_title
