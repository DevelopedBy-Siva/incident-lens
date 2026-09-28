from types import SimpleNamespace

from app.serving.triage import classify


def test_high_impact_single_event_is_not_suppressed_by_count():
    result = classify(SimpleNamespace(source="db", count=1, sample_lines=["ERROR OutOfMemoryError heap space exhausted"]))
    assert result.severity == "critical"
    assert result.disposition == "ESCALATE"


def test_healthcheck_warning_remains_observable_not_auto_suppressed():
    result = classify(SimpleNamespace(source="api", count=1, sample_lines=["WARN healthcheck timeout"]))
    assert result.severity == "medium"
    assert result.disposition == "NEEDS_DEV"
