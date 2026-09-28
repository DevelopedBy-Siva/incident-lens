"""A small, reliable incident investigator.

Classification is deterministic.  A trained adapter may enrich the result, but it
cannot lower a severity or disposition established by observed evidence.
"""

from __future__ import annotations

import json
import logging

from app.serving.decision_engine import IncidentAnalysis, validate_analysis
from app.serving.triage import classify, merge_with_baseline
from app.shared.incident_policy import SYSTEM_PROMPT, build_inference_messages, validate_output_schema


logger = logging.getLogger(__name__)


class InvestigationLoop:
    _last_tool_calls: list = []
    _last_iterations: int = 0
    _last_fallback: bool = False

    def investigate(self, incident, project, evidence=None) -> IncidentAnalysis:
        baseline = classify(incident, evidence)
        result = _from_baseline(incident, baseline)
        self._last_tool_calls = []
        self._last_iterations = 1
        self._last_fallback = True

        candidate = self._model_candidate(incident, project, evidence)
        if candidate is None:
            return result

        merged = merge_with_baseline(candidate, baseline)
        if merged is baseline:
            return result
        self._last_fallback = False
        return validate_analysis(merged, incident)

    @staticmethod
    def _model_candidate(incident, project, evidence):
        """Use the adapter only when it is fully available and returns canonical JSON."""
        try:
            from app.serving.model_runtime import get_model_runtime

            session = get_model_runtime().resolve_project_model(project.id, project=project)
            if not session.provider_available:
                return None
            logs = list(getattr(evidence, "sample_lines", None) or incident.sample_lines or [])
            response = session.complete(
                model=session.default_model,
                messages=build_inference_messages(logs),
                temperature=0.0,
            )
            payload = json.loads(response.content)
            valid, _ = validate_output_schema(payload)
            if not valid:
                return None
            return IncidentAnalysis(**payload)
        except Exception as exc:  # A model outage must not suppress incident triage.
            logger.info("Model enrichment unavailable for incident %s: %s", incident.id, exc)
            return None


def _from_baseline(incident, baseline) -> IncidentAnalysis:
    return IncidentAnalysis(
        severity=baseline.severity,
        disposition=baseline.disposition,
        confidence=baseline.confidence,
        summary=baseline.summary,
        suspected_root_cause=baseline.root_cause,
        next_steps=baseline.next_steps,
        ticket_title=baseline.ticket_title,
        ticket_body=f"{baseline.summary}\n\nRecommended next steps: " + "; ".join(baseline.next_steps),
    )


_loop: InvestigationLoop | None = None


def get_investigation_loop() -> InvestigationLoop:
    global _loop
    if _loop is None:
        _loop = InvestigationLoop()
    return _loop
