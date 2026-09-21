"""Prepare data-plane evidence for serving by attaching runbook context."""

import logging

from app.data.evidence import EvidenceBundle, build_evidence
from app.serving.runbook_matcher import select_runbook_for_incident


logger = logging.getLogger(__name__)


def build_serving_evidence(incident, project) -> EvidenceBundle:
    bundle = build_evidence(incident, project)

    try:
        selection = select_runbook_for_incident(
            incident,
            evidence=bundle,
            project=project,
        )
        runbook = selection.runbook
        score = selection.score
        bundle.runbook_selection_source = selection.source
        bundle.candidate_runbooks = list(selection.candidate_runbooks)
        if runbook and score > 0:
            bundle.runbook_id = runbook.id
            bundle.runbook_name = runbook.name
            bundle.runbook_steps = list(runbook.steps or [])
            bundle.runbook_score = score
    except Exception as e:
        logger.warning("[EVIDENCE] Failed to match runbook: %s", e)

    logger.info(
        "[EVIDENCE] Prepared serving bundle for %s — runbook=%s",
        incident.id,
        bundle.runbook_name or "none",
    )
    return bundle
