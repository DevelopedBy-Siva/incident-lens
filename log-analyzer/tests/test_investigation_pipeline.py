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


def test_fallback_preserves_observed_tls_event_without_embedded_playbooks():
    result = InvestigationLoop().investigate(
        incident(["2026-09-22T14:08:56.045Z ERROR [webhook-dispatcher] TLS handshake error from 41.199.8.101:59398: remote error: tls: bad certificate"]),
        SimpleNamespace(id="project-1"),
    )

    assert result.severity == "medium"
    assert result.confidence == 0.7
    assert "webhook-dispatcher" in result.summary
    assert "41.199.8.101:59398" in result.summary
    assert "TLS handshake error" in result.summary
    assert "root cause" in result.suspected_root_cause
    assert "certificate" not in result.ticket_title.lower()
    assert all("CA trust chain" not in step for step in result.next_steps)
