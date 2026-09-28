"""Persist one clear incident-analysis lifecycle."""

from __future__ import annotations

import logging
from datetime import datetime

from app.shared.database import SessionLocal


logger = logging.getLogger(__name__)


def analyze_incident(incident, project, force: bool = False):
    from app.serving.action_executor import execute_actions
    from app.serving.evidence import build_serving_evidence
    from app.serving.investigator import get_investigation_loop
    from app.serving.models import Analysis, InvestigationRun
    from app.serving.notifications import get_notification_service
    from app.serving.policy import evaluate

    db = SessionLocal()
    try:
        existing = db.query(Analysis).filter(Analysis.incident_id == incident.id).first()
        if existing and not force:
            return existing

        started = datetime.utcnow()
        evidence = build_serving_evidence(incident, project)
        loop = get_investigation_loop()
        result = loop.investigate(incident, project, evidence)

        if existing:
            analysis = existing
        else:
            analysis = Analysis(incident_id=incident.id)
            db.add(analysis)
        analysis.severity = result.severity
        analysis.disposition = result.disposition
        analysis.confidence = result.confidence
        analysis.summary = result.summary
        analysis.next_steps = result.next_steps
        analysis.ticket_title = result.ticket_title
        analysis.ticket_body = result.ticket_body
        analysis.analysis_source = "model" if not loop._last_fallback else "rules"
        analysis.created_at = datetime.utcnow()
        db.commit()
        db.refresh(analysis)

        policy = evaluate(incident, analysis)
        actions = execute_actions(incident, analysis, policy, project) if policy.allow else []
        _notify(incident, analysis, policy, project)
        db.add(InvestigationRun(
            incident_id=incident.id,
            project_id=project.id,
            started_at=started,
            finished_at=datetime.utcnow(),
            evidence_samples=len(evidence.sample_lines),
            evidence_related_count=len(evidence.related_incidents),
            evidence_snapshot=evidence.as_prompt_context(),
            tool_calls=[],
            iterations=loop._last_iterations,
            fallback_used=loop._last_fallback,
            analysis_source=analysis.analysis_source,
            policy_allowed=policy.allow,
            policy_reason=policy.reason,
            policy_tags=policy.tags,
            effective_disposition=policy.effective_disposition,
            actions_taken=actions,
            verifier_outcome="pending" if any(a in actions for a in ("notify_oncall", "send_discord_notification")) else "not_required",
            final_severity=analysis.severity,
            final_disposition=analysis.disposition,
            final_confidence=analysis.confidence,
            final_summary=analysis.summary,
        ))
        db.commit()
        return analysis
    except Exception:
        db.rollback()
        logger.exception("Analysis failed for incident %s", incident.id)
        raise
    finally:
        db.close()


def _notify(incident, analysis, policy, project) -> None:
    if not policy.allow:
        return
    if not {"notify_oncall", "send_discord_notification"}.intersection(policy.allowed_actions):
        return
    try:
        from app.serving.notifications import get_notification_service
        get_notification_service(project).route_notification(incident, analysis)
    except Exception:
        logger.exception("Notification failed for incident %s", incident.id)


def run_root_cause_chaining(*_args, **_kwargs) -> None:
    """Root-cause links need explicit evidence; automatic speculative chaining is removed."""
    return None
