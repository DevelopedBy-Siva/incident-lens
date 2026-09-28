from types import SimpleNamespace

from app.serving.policy import evaluate


def incident(status="open"):
    return SimpleNamespace(status=status)


def analysis(severity="high", disposition="NEEDS_ONCALL", confidence=0.9):
    return SimpleNamespace(severity=severity, disposition=disposition, confidence=confidence)


def test_policy_approves_only_known_safe_actions():
    decision = evaluate(incident(), analysis())
    assert decision.allow
    assert decision.allowed_actions == ["auto_enrich", "notify_oncall"]


def test_policy_never_suppresses_non_low_or_low_confidence_results():
    high = evaluate(incident(), analysis("high", "NO_ACTION", 0.99))
    uncertain = evaluate(incident(), analysis("low", "NO_ACTION", 0.7))
    assert not high.allow
    assert not uncertain.allow


def test_policy_blocks_closed_incidents():
    assert not evaluate(incident("closed"), analysis()).allow
