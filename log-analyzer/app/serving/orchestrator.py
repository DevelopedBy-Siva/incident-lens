"""Persist one clear incident-analysis lifecycle."""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy.exc import OperationalError

from app.shared.database import SessionLocal, engine


logger = logging.getLogger(__name__)


def analyze_incident(incident, project, force: bool = False):
    from app.serving.action_executor import execute_actions
    from app.serving.evidence import build_serving_evidence
    from app.serving.investigator import get_investigation_loop
    from app.serving.models import Analysis, InvestigationRun
    from app.serving.notifications import get_notification_service
    from app.serving.policy import evaluate

    existing = _load_analysis(Analysis, incident.id)
    if existing and not force:
        return existing

    try:
        started = datetime.utcnow()
        evidence = build_serving_evidence(incident, project)
        loop = get_investigation_loop()
        # Save an evidence-backed result before entering PyTorch/Transformers.
        # Native inference failures can terminate a Python process without
        # raising an exception, so persisting only after generation turns one
        # model fault into lost incident analysis.
        baseline = loop.baseline(incident, evidence)
        analysis = _persist_analysis(
            Analysis,
            incident.id,
            baseline,
            "rules",
        )
        logger.info(
            "[ANALYSIS] incident=%s baseline persisted source=rules evidence_samples=%s",
            incident.id,
            len(evidence.sample_lines),
        )

        result = loop.investigate(incident, project, evidence)
        if not loop._last_fallback:
            analysis = _persist_analysis(Analysis, incident.id, result, "model")
            logger.info("[ANALYSIS] incident=%s model enrichment persisted", incident.id)
        else:
            logger.info("[ANALYSIS] incident=%s using persisted rule analysis", incident.id)

        policy = evaluate(incident, analysis)
        actions = execute_actions(incident, analysis, policy, project) if policy.allow else []
        _notify(incident, analysis, policy, project)
        _persist_investigation(
            InvestigationRun,
            incident=incident,
            project=project,
            analysis=analysis,
            started=started,
            evidence=evidence,
            loop=loop,
            policy=policy,
            actions=actions,
        )
        logger.info(
            "[ANALYSIS] incident=%s completed source=%s policy_allowed=%s actions=%s",
            incident.id,
            analysis.analysis_source,
            policy.allow,
            len(actions),
        )
        return analysis
    except Exception:
        logger.exception("Analysis failed for incident %s", incident.id)
        raise


def _load_analysis(analysis_model, incident_id):
    db = SessionLocal()
    try:
        analysis = (
            db.query(analysis_model)
            .filter(analysis_model.incident_id == incident_id)
            .order_by(analysis_model.created_at.desc())
            .first()
        )
        if analysis:
            db.expunge(analysis)
        return analysis
    finally:
        db.close()


def _persist_analysis(analysis_model, incident_id, result, analysis_source):
    """Persist an already-generated result with a fresh connection on retry.

    ``pool_pre_ping`` can discard a stale connection before checkout, but a
    managed Postgres provider can still close a connection during a commit. In
    that case the old transaction is unusable. Retrying with a new session is
    safe: if the first commit reached Postgres, the retry updates that row;
    otherwise it inserts it.
    """
    for attempt in range(2):
        db = SessionLocal()
        try:
            analysis = (
                db.query(analysis_model)
                .filter(analysis_model.incident_id == incident_id)
                .order_by(analysis_model.created_at.desc())
                .first()
            )
            if analysis is None:
                analysis = analysis_model(incident_id=incident_id)
                db.add(analysis)
            analysis.severity = result.severity
            analysis.disposition = result.disposition
            analysis.confidence = result.confidence
            analysis.summary = result.summary
            analysis.suspected_root_cause = result.suspected_root_cause
            analysis.next_steps = result.next_steps
            analysis.ticket_title = result.ticket_title
            analysis.ticket_body = result.ticket_body
            analysis.analysis_source = analysis_source
            analysis.created_at = datetime.utcnow()
            db.commit()
            db.refresh(analysis)
            db.expunge(analysis)
            return analysis
        except OperationalError:
            db.rollback()
            engine.dispose()
            if attempt:
                raise
            logger.warning(
                "Database connection closed while saving analysis for incident %s; retrying",
                incident_id,
            )
        finally:
            db.close()


def _persist_investigation(
    investigation_model,
    *,
    incident,
    project,
    analysis,
    started,
    evidence,
    loop,
    policy,
    actions,
):
    """Best-effort audit persistence; analysis remains available if this fails."""
    db = SessionLocal()
    try:
        db.add(investigation_model(
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
    except Exception:
        db.rollback()
        logger.exception("Could not persist investigation audit for incident %s", incident.id)
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
