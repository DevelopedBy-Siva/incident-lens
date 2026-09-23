"""Prepare data-plane evidence for serving."""

import logging

from app.data.evidence import EvidenceBundle, build_evidence


logger = logging.getLogger(__name__)


def build_serving_evidence(incident, project) -> EvidenceBundle:
    bundle = build_evidence(incident, project)

    logger.info(
        "[EVIDENCE] Prepared serving bundle for %s",
        incident.id,
    )
    return bundle
