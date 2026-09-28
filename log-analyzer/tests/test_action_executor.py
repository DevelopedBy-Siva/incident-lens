from types import SimpleNamespace
from unittest.mock import patch

from app.serving.action_executor import execute_actions
from app.serving.policy import PolicyDecision


def test_executor_only_runs_explicit_safe_actions():
    incident = SimpleNamespace(id="i-1")
    project = SimpleNamespace(id="p-1")
    analysis = SimpleNamespace(summary="summary", severity="high", confidence=0.9)
    decision = PolicyDecision(True, "approved", "NEEDS_ONCALL", ["restart_service", "auto_enrich"], ["restart_service", "auto_enrich"], [])
    with patch("app.serving.action_executor._auto_enrich", return_value=True) as enrich, patch("app.serving.action_executor._log_action"):
        executed = execute_actions(incident, analysis, decision, project)
    assert executed == ["auto_enrich"]
    enrich.assert_called_once()
