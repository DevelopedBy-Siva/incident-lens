import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "metrics_report.py"
spec = importlib.util.spec_from_file_location("metrics_report", SCRIPT_PATH)
metrics_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_report)


class EvalMetricTests(unittest.TestCase):
    def test_action_policy_fixture_metrics_are_computed_from_optional_fields(self):
        cases = metrics_report._load_json(
            Path(__file__).resolve().parents[1] / "evals" / "action_policy_cases.json"
        )
        scored = [
            (case, metrics_report._analyze_case(case, project=None))
            for case in cases
        ]

        self.assertTrue(all(not result.skipped_reason for _, result in scored))
        self.assertTrue(all(metrics_report._is_triage_correct(case, result) for case, result in scored))

        action_metrics = metrics_report._action_policy_metrics(scored)

        self.assertEqual(action_metrics["expected_blocked_total"], 2)
        self.assertEqual(action_metrics["expected_blocked_correct"], 2)
        self.assertEqual(action_metrics["expected_allowed_total"], 2)
        self.assertEqual(action_metrics["expected_allowed_correct"], 2)
        self.assertEqual(action_metrics["dangerous_expected"], 3)
        self.assertEqual(action_metrics["dangerous_blocked"], 3)

    def test_blocked_dangerous_action_is_not_unsafe_automation(self):
        case = {
            "id": "case-1",
            "name": "blocked remediation",
            "expected": {
                "severity": "high",
                "disposition": "NEEDS_ONCALL",
                "runbook_id": "db_connection_pool_exhausted",
            },
        }
        result = SimpleNamespace(
            effective_disposition="NEEDS_ONCALL",
            confidence=0.90,
            requested_actions=["notify_oncall", "restart_service"],
            allowed_actions=["notify_oncall"],
            blocked_actions=["restart_service"],
            actions_taken=[],
            would_auto_suppress=False,
        )

        categories = metrics_report._unsafe_categories(case, result)

        self.assertFalse(metrics_report._is_unsafe_automation(case, result))
        self.assertEqual(categories["dangerous_action_proposed_but_blocked"], ["restart_service"])

    def test_disposition_mismatch_is_reported_separately_from_unsafe_automation(self):
        case = {
            "id": "case-2",
            "name": "over-escalated dev incident",
            "expected": {
                "severity": "medium",
                "disposition": "NEEDS_DEV",
                "root_cause_keywords": ["dns"],
            },
        }
        result = SimpleNamespace(
            effective_disposition="NEEDS_ONCALL",
            confidence=0.90,
            requested_actions=["notify_oncall"],
            allowed_actions=["notify_oncall"],
            blocked_actions=[],
            actions_taken=[],
            would_auto_suppress=False,
        )

        categories = metrics_report._unsafe_categories(case, result)

        self.assertFalse(metrics_report._is_unsafe_automation(case, result))
        self.assertTrue(categories["unsafe_disposition"])


if __name__ == "__main__":
    unittest.main()
