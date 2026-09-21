import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "metrics_report.py"
spec = importlib.util.spec_from_file_location("metrics_report", SCRIPT_PATH)
metrics_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_report)


class EvalMetricTests(unittest.TestCase):
    def test_llm_eval_uses_model_runtime_provider(self):
        class RuntimeSession:
            provider_available = True

            def __init__(self):
                self.calls = []

            def complete_with_tools(self, **request):
                self.calls.append(request)
                return SimpleNamespace(
                    content=json.dumps(
                        {
                            "severity": "medium",
                            "disposition": "NEEDS_DEV",
                            "confidence": 0.8,
                            "suspected_root_cause": "Application error",
                            "summary": "Checkout failed.",
                            "next_steps": ["Inspect logs"],
                            "ticket_title": "Checkout failure",
                            "ticket_body": "Inspect the checkout service.",
                        }
                    ),
                    tool_calls=(),
                )

        runtime_session = RuntimeSession()
        runtime = SimpleNamespace(
            resolve_project_model=lambda project_id, project=None: runtime_session
        )
        incident = SimpleNamespace(
            id="incident-1",
            source="checkout",
            environment="production",
            count=2,
            first_seen="2026-01-01T12:00:00Z",
            last_seen="2026-01-01T12:01:00Z",
            sample_lines=["checkout request failed"],
        )

        with patch(
            "app.serving.model_runtime.get_model_runtime",
            return_value=runtime,
        ):
            result, error = metrics_report._llm_analysis(incident, project=None)

        self.assertIsNone(error)
        self.assertEqual(result.analysis_source, "llm-agent")
        self.assertEqual(result.disposition, "NEEDS_DEV")
        self.assertEqual(len(runtime_session.calls), 1)
        self.assertEqual(runtime_session.calls[0]["temperature"], 0.1)

    def test_action_policy_fixture_metrics_are_computed_from_optional_fields(self):
        cases = metrics_report._load_json(
            Path(__file__).resolve().parents[1] / "evals" / "action_policy_cases.json"
        )
        scored = [
            (case, metrics_report._analyze_case(case, project=None)) for case in cases
        ]

        self.assertTrue(all(not result.skipped_reason for _, result in scored))
        self.assertTrue(
            all(
                metrics_report._is_triage_correct(case, result)
                for case, result in scored
            )
        )

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
        self.assertEqual(
            categories["dangerous_action_proposed_but_blocked"], ["restart_service"]
        )

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
