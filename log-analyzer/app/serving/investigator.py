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
        baseline, result = self._baseline(incident, evidence)

        candidate = self._model_candidate(incident, project, evidence)
        if candidate is None:
            return result

        merged = merge_with_baseline(candidate, baseline)
        if merged is baseline:
            logger.warning(
                "[ANALYSIS] incident=%s model output rejected because it downgraded observed evidence",
                incident.id,
            )
            return result
        _complete_candidate(merged, baseline)
        self._last_fallback = False
        return validate_analysis(merged, incident)

    def baseline(self, incident, evidence=None) -> IncidentAnalysis:
        """Produce a durable, deterministic assessment before model enrichment.

        This is intentionally public because persistence must happen before a
        local native model is invoked.  A process-level model failure must not
        erase the incident's evidence-backed assessment.
        """
        _baseline, result = self._baseline(incident, evidence)
        return result

    def _baseline(self, incident, evidence):
        baseline = classify(incident, evidence)
        result = _from_baseline(incident, baseline)
        self._last_tool_calls = []
        self._last_iterations = 1
        self._last_fallback = True
        return baseline, result

    @staticmethod
    def _model_candidate(incident, project, evidence):
        """Use the adapter only when it is fully available and returns canonical JSON."""
        try:
            from app.serving.model_runtime import get_model_runtime

            session = get_model_runtime().resolve_project_model(project.id, project=project)
            if not session.provider_available:
                logger.info(
                    "[ANALYSIS] incident=%s model enrichment skipped: no ready active adapter",
                    incident.id,
                )
                return None
            incident_input = _runtime_incident_input(incident, evidence)
            response = session.complete(
                model=session.default_model,
                messages=build_inference_messages(incident_input),
                temperature=0.0,
            )
            payload = json.loads(response.content)
            valid, _ = validate_output_schema(payload)
            if not valid:
                logger.warning(
                    "[ANALYSIS] incident=%s model enrichment returned an invalid schema",
                    incident.id,
                )
                return None
            logger.info("[ANALYSIS] incident=%s model enrichment completed", incident.id)
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


def _complete_candidate(candidate: IncidentAnalysis, baseline) -> None:
    """Prevent incomplete model JSON from becoming a persisted analysis."""
    if not candidate.summary.strip():
        candidate.summary = baseline.summary
    if not candidate.ticket_title.strip():
        candidate.ticket_title = baseline.ticket_title
    if not candidate.ticket_body.strip():
        candidate.ticket_body = (
            f"{candidate.summary}\n\nRecommended next steps: "
            + "; ".join(candidate.next_steps or baseline.next_steps)
        )
    if not candidate.next_steps and baseline.severity != "low":
        candidate.next_steps = baseline.next_steps
    if not candidate.suspected_root_cause and baseline.root_cause:
        candidate.suspected_root_cause = baseline.root_cause


def _runtime_incident_input(incident, evidence) -> dict:
    """Build the same evidence object used by the training prompt."""
    related = []
    for item in getattr(evidence, "related_incidents", []) or []:
        related.append(
            {
                "source": item.source,
                "count": item.count,
                "severity": item.severity,
                "disposition": item.disposition,
                "summary": item.ticket_title,
                "first_seen": item.first_seen.isoformat(),
            }
        )
    return {
        "logs": list(getattr(evidence, "sample_lines", None) or incident.sample_lines or []),
        "service": incident.source,
        "environment": incident.environment,
        "count": incident.count,
        "metadata": dict(getattr(incident, "auto_tags", None) or {}),
        "related_incidents": related,
    }


_loop: InvestigationLoop | None = None


def get_investigation_loop() -> InvestigationLoop:
    global _loop
    if _loop is None:
        _loop = InvestigationLoop()
    return _loop
