import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.action_executor import execute_actions
from app.core.policy import PolicyDecision


def incident(**overrides):
    defaults = {"id": "incident-1"}
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def analysis(**overrides):
    defaults = {
        "severity": "low",
        "disposition": "NO_ACTION",
        "confidence": 0.90,
        "ticket_body": "Ticket body",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def project(**overrides):
    defaults = {"id": "project-1"}
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class ActionExecutorTests(unittest.TestCase):
    def test_execute_actions_only_runs_allowed_actions(self):
        calls = []
        logs = []

        def fake_auto_enrich(*_):
            calls.append("auto_enrich")
            return True

        def fake_auto_suppress(*_):
            calls.append("auto_suppress")
            return True

        def fake_log_action(**kwargs):
            logs.append(kwargs)

        decision = PolicyDecision(
            allow=True,
            reason="partial",
            effective_disposition="NO_ACTION",
            requested_actions=["auto_enrich", "auto_suppress"],
            allowed_actions=["auto_enrich"],
            blocked_actions=["auto_suppress"],
        )

        with patch("app.core.action_executor._auto_enrich", fake_auto_enrich), patch(
            "app.core.action_executor._auto_suppress", fake_auto_suppress
        ), patch("app.core.action_executor._log_action", fake_log_action):
            executed = execute_actions(incident(), analysis(), decision, project())

        self.assertEqual(executed, ["auto_enrich"])
        self.assertEqual(calls, ["auto_enrich"])
        self.assertEqual(logs[0]["actions_taken"], ["auto_enrich"])
        self.assertIs(logs[0]["policy_decision"], decision)

    def test_execute_actions_never_runs_blocked_tier_actions(self):
        calls = []
        logs = []

        def fake_auto_enrich(*_):
            calls.append("auto_enrich")
            return True

        def fake_log_action(**kwargs):
            logs.append(kwargs)

        decision = PolicyDecision(
            allow=True,
            reason="bad input",
            effective_disposition="ESCALATE",
            requested_actions=["restart_service", "auto_enrich"],
            allowed_actions=["restart_service", "auto_enrich"],
            blocked_actions=[],
        )

        with patch("app.core.action_executor._auto_enrich", fake_auto_enrich), patch(
            "app.core.action_executor._log_action", fake_log_action
        ):
            executed = execute_actions(incident(), analysis(), decision, project())

        self.assertEqual(executed, ["auto_enrich"])
        self.assertEqual(calls, ["auto_enrich"])
        self.assertEqual(logs[0]["actions_taken"], ["auto_enrich"])


if __name__ == "__main__":
    unittest.main()
