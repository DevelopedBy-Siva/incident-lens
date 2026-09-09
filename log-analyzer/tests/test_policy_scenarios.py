import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core import policy as policy_module


def incident(**overrides):
    defaults = {
        "id": "scenario-incident",
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
        "analysis_source": "runbook",
        "ticket_body": "bounded evidence",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class PolicyScenarioTests(unittest.TestCase):
    def setUp(self):
        self.cooldown_patch = patch.object(
            policy_module, "_was_recently_acted_on", return_value=False
        )
        self.cooldown_patch.start()

    def tearDown(self):
        self.cooldown_patch.stop()

    def test_noisy_healthcheck_allows_high_confidence_suppression_only(self):
        result = policy_module.evaluate(
            incident(count=12),
            analysis(
                severity="low",
                disposition="NO_ACTION",
                confidence=0.92,
                proposed_actions=[
                    "auto_enrich",
                    "auto_suppress",
                    "restart_service",
                ],
            ),
        )

        self.assertEqual(result.effective_disposition, "NO_ACTION")
        self.assertEqual(result.requested_actions, ["auto_enrich", "auto_suppress", "restart_service"])
        self.assertIn("auto_enrich", result.allowed_actions)
        self.assertIn("auto_suppress", result.allowed_actions)
        self.assertIn("restart_service", result.blocked_actions)

    def test_db_pool_exhaustion_escalates_but_blocks_remediation(self):
        result = policy_module.evaluate(
            incident(count=8),
            analysis(
                severity="high",
                disposition="NEEDS_ONCALL",
                confidence=0.91,
                proposed_actions=[
                    "notify_oncall",
                    "create_incident_summary",
                    "restart_service",
                    "modify_database_config",
                ],
            ),
        )

        self.assertEqual(result.effective_disposition, "NEEDS_ONCALL")
        self.assertIn("notify_oncall", result.allowed_actions)
        self.assertIn("create_incident_summary", result.allowed_actions)
        self.assertIn("restart_service", result.blocked_actions)
        self.assertIn("modify_database_config", result.blocked_actions)

    def test_false_suppression_trap_blocks_auto_suppress_for_customer_impact(self):
        result = policy_module.evaluate(
            incident(count=25),
            analysis(
                severity="high",
                disposition="NEEDS_ONCALL",
                confidence=0.93,
                proposed_actions=["auto_enrich", "auto_suppress", "notify_oncall"],
            ),
        )

        self.assertEqual(result.effective_disposition, "NEEDS_ONCALL")
        self.assertIn("notify_oncall", result.allowed_actions)
        self.assertIn("auto_suppress", result.blocked_actions)

    def test_low_frequency_high_impact_failure_is_not_suppressed_by_count(self):
        result = policy_module.evaluate(
            incident(count=1),
            analysis(
                severity="critical",
                disposition="ESCALATE",
                confidence=0.89,
                proposed_actions=["notify_oncall", "auto_suppress"],
            ),
        )

        self.assertEqual(result.effective_disposition, "ESCALATE")
        self.assertIn("notify_oncall", result.allowed_actions)
        self.assertIn("auto_suppress", result.blocked_actions)
        self.assertNotIn("downgraded:count_too_low", result.tags)


if __name__ == "__main__":
    unittest.main()
