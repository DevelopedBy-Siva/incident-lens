from types import SimpleNamespace

from app.serving.decision_engine import DecisionEngine
from app.serving.triage import TriageResult, merge_with_baseline


def test_compatibility_engine_uses_deterministic_contract():
    incident = SimpleNamespace(source="payments", count=1, sample_lines=["ERROR PaymentGatewayTimeout" ])
    result = DecisionEngine().analyze_incident(incident)
    assert result.severity == "high"
    assert result.disposition == "NEEDS_ONCALL"
    assert result.model_dump().keys() == {"severity", "disposition", "confidence", "summary", "suspected_root_cause", "next_steps", "ticket_title", "ticket_body"}


def test_model_cannot_downgrade_observed_critical_evidence():
    baseline = TriageResult("critical", "ESCALATE", 0.9, "critical", None, [])
    model = SimpleNamespace(severity="low", disposition="NO_ACTION")
    assert merge_with_baseline(model, baseline) is baseline
