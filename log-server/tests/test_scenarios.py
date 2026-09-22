import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = ROOT / "log-server" / "server.py"
ANALYZER_PATH = ROOT / "log-analyzer"
sys.path.insert(0, str(ANALYZER_PATH))

spec = importlib.util.spec_from_file_location("log_server_module", SERVER_PATH)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)

from app.serving.runbook_matcher import get_runbook_candidates  # noqa: E402

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
    def test_swagger_ui_is_exposed(self):
        self.assertEqual(server.app.docs_url, "/docs")
        self.assertEqual(server.app.openapi_url, "/openapi.json")

    def test_only_start_and_stop_business_endpoints_are_exposed(self):
        paths = set(server.app.openapi()["paths"])
        self.assertEqual(paths, {"/api/start", "/api/stop"})

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
    async def test_datadog_intake_uses_api_key_and_provider_fields(self):
        captured = {}

        def handler(request):
            captured["request"] = request
            return httpx.Response(202, request=request)

        transport = httpx.MockTransport(handler)
        config = server.DatadogWriteConfig(
            api_key="api-secret",
            site="datadoghq.eu",
            service="checkout",
        )
        async with httpx.AsyncClient(transport=transport) as client:
            result = await server.push_to_datadog(
                ["ERROR checkout failed"],
                config,
                extra_tags={"service": "ignored", "scenario": "payment"},
                client=client,
            )

        self.assertTrue(result)
        request = captured["request"]
        self.assertEqual(
            str(request.url),
            "https://http-intake.logs.datadoghq.eu/api/v2/logs",
        )
        self.assertEqual(request.headers["DD-API-KEY"], "api-secret")
        event = json.loads(request.content)[0]
        self.assertEqual(event["service"], "checkout")
        self.assertEqual(event["status"], "error")
        self.assertIn("env:prod", event["ddtags"])
        self.assertIn("scenario:payment", event["ddtags"])

    async def test_start_builds_datadog_config_from_headers(self):
        generator = SimpleNamespace(
            start=AsyncMock(return_value=(True, "started")), running=True
        )
        with patch.object(server, "log_generator", generator):
            response = await server.start_generation(
                duration=60,
                interval_seconds=1,
                batch_size=1,
                error_rate=0.5,
                slow_rate=0.1,
                datadog_api_key="api-secret",
                datadog_site="datadoghq.com",
                datadog_service="project-api",
            )

        config = generator.start.await_args.kwargs["datadog_config"]
        self.assertEqual(config.api_key, "api-secret")
        self.assertEqual(config.site, "datadoghq.com")
        self.assertEqual(config.service, "project-api")
        self.assertEqual(response["status"], "running")


if __name__ == "__main__":
    unittest.main()
