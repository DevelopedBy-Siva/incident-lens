import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = ROOT / "log-server" / "server.py"
ANALYZER_PATH = ROOT / "log-analyzer"
sys.path.insert(0, str(ANALYZER_PATH))

spec = importlib.util.spec_from_file_location("log_server_module", SERVER_PATH)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)

from app.core.runbook_matcher import get_runbook_candidates  # noqa: E402


REQUIRED_SCENARIOS = {
    "healthcheck_timeout_noise",
    "db_pool_exhaustion",
    "payment_gateway_degraded",
    "api_gateway_5xx_spike",
    "memory_pressure_or_oom",
    "auth_failure_cascade",
    "deployment_regression",
    "queue_backlog",
    "vendor_api_timeout",
    "false_suppression_trap",
    "low_frequency_high_impact",
    "ambiguous_cascade",
}


class ScenarioRegistryTests(unittest.TestCase):
    def test_registry_includes_required_scenarios(self):
        self.assertTrue(REQUIRED_SCENARIOS.issubset(server.SCENARIOS.keys()))

    def test_generated_scenario_logs_contain_required_fields(self):
        for scenario_name, scenario in server.SCENARIOS.items():
            with self.subTest(scenario=scenario_name):
                self.assertGreater(len(scenario["steps"]), 0)
                for idx, step in enumerate(scenario["steps"], start=1):
                    log_line = server.format_scenario_log(step, scenario_name, 1, idx)
                    for field in server.REQUIRED_SCENARIO_FIELDS:
                        self.assertIn(f"{field}=", log_line)
                    self.assertIn(step["message"], log_line)

    def test_scenarios_have_expected_runbook_metadata(self):
        for scenario_name, scenario in server.SCENARIOS.items():
            with self.subTest(scenario=scenario_name):
                self.assertIn("description", scenario)
                self.assertIn("services", scenario)
                self.assertIn("expected_runbook", scenario)
                self.assertIn("expected_severity", scenario)
                self.assertIn("expected_disposition", scenario)
                self.assertIn("expected_allowed_actions", scenario)
                self.assertIn("expected_blocked_actions", scenario)

    def test_scenario_logs_match_expected_runbooks(self):
        for scenario_name, scenario in server.SCENARIOS.items():
            with self.subTest(scenario=scenario_name):
                incident = SimpleNamespace(
                    signature=scenario_name,
                    sample_lines=[
                        server.format_scenario_log(step, scenario_name, 1, idx)
                        for idx, step in enumerate(scenario["steps"], start=1)
                    ],
                    count=len(scenario["steps"]),
                )
                candidates = get_runbook_candidates(incident, limit=5)
                candidate_ids = [runbook.id for runbook, _ in candidates]
                self.assertIn(scenario["expected_runbook"], candidate_ids)


class ScenarioExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_scenario_pushes_to_loki_without_live_loki(self):
        pushed = []

        async def fake_push(lines, extra_labels=None):
            pushed.append((lines, extra_labels))
            return True

        with patch.object(server, "push_to_loki", new=AsyncMock(side_effect=fake_push)):
            await server._run_scenario("healthcheck_timeout_noise", repeat=1, speed=1000)

        self.assertEqual(len(pushed), len(server.SCENARIOS["healthcheck_timeout_noise"]["steps"]))
        for lines, labels in pushed:
            self.assertEqual(len(lines), 1)
            self.assertEqual(labels["scenario"], "healthcheck_timeout_noise")
            self.assertIn("timestamp=", lines[0])
            self.assertIn("message=", lines[0])

    async def test_legacy_scenario_aliases_still_execute(self):
        with patch.object(server, "push_to_loki", new=AsyncMock(return_value=True)) as push:
            await server._run_scenario("db_cascade", repeat=1, speed=1000)

        self.assertEqual(push.await_count, len(server.SCENARIOS["ambiguous_cascade"]["steps"]))


if __name__ == "__main__":
    unittest.main()
