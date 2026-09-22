import json
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

from app.serving.decision_engine import DecisionEngine
from app.serving.investigator import InvestigationLoop
from app.serving.model_provider import ProviderResponse
from app.serving.runbook_loader import Runbook
from app.serving.runbook_matcher import _call_llm_tiebreaker


def incident():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    return SimpleNamespace(
        id="incident-1",
        source="checkout",
        environment="production",
        count=4,
        first_seen=now,
        last_seen=now,
        signature="checkout request failed",
        sample_lines=["checkout request failed for order 123"],
    )


def project():
    return SimpleNamespace(
        id="project-1",
        name="project",
    )


class RecordingRuntimeSession:
    provider_available = True
    model_candidates = ("local-qwen-model",)
    default_model = "local-qwen-model"

    def __init__(self, content):
        self.content = content
        self.calls = []

    def complete(self, **request):
        self.calls.append(("complete", request))
        return ProviderResponse(content=self.content)

    def complete_with_tools(self, **request):
        self.calls.append(("complete_with_tools", request))
        return ProviderResponse(content=self.content)


class RecordingRuntime:
    def __init__(self, session):
        self.session = session
        self.resolutions = []

    def resolve_project_model(self, project_id, *, project=None):
        self.resolutions.append((project_id, project))
        return self.session


class ServingRuntimeIntegrationTests(unittest.TestCase):
    def test_decision_engine_uses_runtime_and_preserves_analysis_contract(self):
        content = json.dumps(
            {
                "severity": "medium",
                "disposition": "NEEDS_DEV",
                "confidence": 0.82,
                "summary": "Checkout requests are failing.",
                "suspected_root_cause": "Application error",
                "next_steps": ["Inspect checkout logs"],
                "ticket_title": "Checkout requests failing",
                "ticket_body": "Investigate checkout request failures.",
            }
        )
        session = RecordingRuntimeSession(content)
        runtime = RecordingRuntime(session)
        active_project = project()

        with patch(
            "app.serving.decision_engine.get_model_runtime",
            return_value=runtime,
        ):
            result = DecisionEngine().analyze_incident(
                incident(),
                project=active_project,
            )

        self.assertEqual(result.severity, "medium")
        self.assertEqual(result.disposition, "NEEDS_DEV")
        self.assertEqual(runtime.resolutions, [(active_project.id, active_project)])
        method, request = session.calls[0]
        self.assertEqual(method, "complete")
        self.assertEqual(request["model"], "local-qwen-model")
        self.assertEqual(request["temperature"], 0.3)

    def test_investigator_uses_runtime_tool_interface_and_preserves_output(self):
        content = json.dumps(
            {
                "severity": "medium",
                "disposition": "NEEDS_DEV",
                "confidence": 0.8,
                "summary": "Checkout requests are failing.",
                "suspected_root_cause": "Application error",
                "next_steps": ["Inspect checkout logs"],
                "ticket_title": "Checkout requests failing",
                "ticket_body": "Investigate checkout request failures.",
            }
        )
        session = RecordingRuntimeSession(content)
        runtime = RecordingRuntime(session)
        active_project = project()

        with patch(
            "app.serving.investigator.get_model_runtime",
            return_value=runtime,
        ):
            loop = InvestigationLoop()
            result = loop.investigate(incident(), active_project)

        self.assertEqual(result.severity, "medium")
        self.assertEqual(result.disposition, "NEEDS_DEV")
        self.assertFalse(loop._last_fallback)
        method, request = session.calls[0]
        self.assertEqual(method, "complete_with_tools")
        self.assertEqual(request["model"], "local-qwen-model")
        self.assertEqual(request["tool_choice"], "auto")
        self.assertEqual(request["temperature"], 0.2)
        self.assertEqual(request["max_tokens"], 1500)

    def test_runbook_tiebreaker_uses_local_runtime(self):
        content = json.dumps(
            {
                "selected_runbook_id": "checkout_failure",
                "confidence": 0.75,
                "reason": "The checkout evidence is specific.",
            }
        )
        session = RecordingRuntimeSession(content)
        runtime = RecordingRuntime(session)
        candidate = Runbook(
            {
                "id": "checkout_failure",
                "name": "Checkout Failure",
                "description": "Checkout requests are failing",
                "patterns": ["checkout request failed"],
                "steps": ["Inspect logs"],
            }
        )
        active_project = project()

        with patch(
            "app.serving.model_runtime.get_model_runtime",
            return_value=runtime,
        ):
            result = _call_llm_tiebreaker(
                incident(),
                None,
                [(candidate, 0.6)],
                active_project,
            )

        self.assertEqual(result["selected_runbook_id"], "checkout_failure")
        method, request = session.calls[0]
        self.assertEqual(method, "complete")
        self.assertEqual(request["model"], session.default_model)
        self.assertEqual(request["temperature"], 0.3)


if __name__ == "__main__":
    unittest.main()
