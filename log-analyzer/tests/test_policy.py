import unittest
import sys
import types
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from app.core import policy as policy_module


def incident(**overrides):
    defaults = {
        "id": "incident-1",
        "status": "open",
        "count": 3,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def analysis(**overrides):
    defaults = {
        "severity": "medium",
        "disposition": "OBSERVE",
        "confidence": 0.90,
        "analysis_source": "llm",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.cooldown_patch = patch.object(
            policy_module, "_was_recently_acted_on", return_value=False
        )
        self.cooldown_patch.start()

    def tearDown(self):
        self.cooldown_patch.stop()

    def test_high_confidence_noise_suppression_allowed(self):
        result = policy_module.evaluate(
            incident(count=10),
            analysis(
                severity="low",
                disposition="NO_ACTION",
                confidence=0.90,
                proposed_actions=["auto_enrich", "auto_suppress"],
            ),
        )

        self.assertTrue(result.allow)
        self.assertEqual(result.requested_actions, ["auto_enrich", "auto_suppress"])
        self.assertIn("auto_enrich", result.allowed_actions)
        self.assertIn("auto_suppress", result.allowed_actions)
        self.assertEqual(result.blocked_actions, [])

    def test_low_confidence_llm_action_blocked(self):
        result = policy_module.evaluate(
            incident(),
            analysis(
                analysis_source="llm",
                confidence=0.42,
                disposition="ESCALATE",
                severity="high",
            ),
        )

        self.assertFalse(result.allow)
        self.assertIn("confidence", result.reason.lower())
        self.assertEqual(result.allowed_actions, [])
        self.assertIn("blocked:low_confidence", result.tags)

    def test_runbook_can_pass_lower_confidence_than_llm(self):
        runbook_result = policy_module.evaluate(
            incident(),
            analysis(analysis_source="runbook", confidence=0.45),
        )
        llm_result = policy_module.evaluate(
            incident(),
            analysis(analysis_source="llm", confidence=0.45),
        )

        self.assertTrue(runbook_result.allow)
        self.assertFalse(llm_result.allow)
        self.assertIn("blocked:low_confidence", llm_result.tags)

    def test_dangerous_llm_actions_blocked_with_safe_actions_allowed(self):
        result = policy_module.evaluate(
            incident(count=3),
            analysis(
                severity="critical",
                disposition="ESCALATE",
                confidence=0.88,
                proposed_actions=[
                    "notify_oncall",
                    "create_incident_summary",
                    "restart_service",
                    "modify_database_config",
                ],
            ),
        )

        self.assertTrue(result.allow)
        self.assertIn("notify_oncall", result.allowed_actions)
        self.assertIn("create_incident_summary", result.allowed_actions)
        self.assertIn("restart_service", result.blocked_actions)
        self.assertIn("modify_database_config", result.blocked_actions)
        self.assertIn("blocked:dangerous_action:restart_service", result.tags)
        self.assertIn("blocked:dangerous_action:modify_database_config", result.tags)

    def test_cooldown_blocks_repeated_action(self):
        with patch.object(policy_module, "_was_recently_acted_on", return_value=True):
            result = policy_module.evaluate(
                incident(),
                analysis(confidence=0.90, disposition="OBSERVE", severity="medium"),
            )

        self.assertFalse(result.allow)
        self.assertIn("blocked:cooldown", result.tags)

    def test_observe_is_valid(self):
        result = policy_module.evaluate(
            incident(),
            analysis(severity="medium", disposition="OBSERVE", confidence=0.90),
        )

        self.assertTrue(result.allow)
        self.assertEqual(result.effective_disposition, "OBSERVE")
        self.assertNotIn("normalized:unknown_disposition", result.tags)

    def test_action_names_are_normalized(self):
        cases = [
            ("Restart Service", "restart_service"),
            ("modifyDatabaseConfig", "modify_database_config"),
            ("send-discord-notification", "send_discord_notification"),
        ]
        for raw, normalized in cases:
            with self.subTest(raw=raw):
                result = policy_module.get_requested_actions(
                    analysis(proposed_actions=[raw, raw]),
                    "ESCALATE",
                )
                self.assertEqual(result, [normalized])


class ActionLogCooldownTests(unittest.TestCase):
    def test_action_log_cooldown_checks_latest_row(self):
        class FakeQuery:
            def filter(self, *_):
                return self

            def order_by(self, *_):
                return self

            def first(self):
                return (datetime.utcnow() - timedelta(minutes=1),)

        class FakeSession:
            def query(self, *_):
                return FakeQuery()

            def close(self):
                pass

        class FakeColumn:
            def __eq__(self, _):
                return True

            def desc(self):
                return self

        fake_storage = types.ModuleType("app.services.storage")
        fake_storage.SessionLocal = lambda: FakeSession()
        fake_storage.ActionLog = SimpleNamespace(
            actioned_at=FakeColumn(),
            incident_id=FakeColumn(),
        )

        with patch.dict(sys.modules, {"app.services.storage": fake_storage}):
            self.assertTrue(policy_module._was_recently_acted_on(incident(), 20))


if __name__ == "__main__":
    unittest.main()
