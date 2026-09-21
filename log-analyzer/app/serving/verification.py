import logging
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

VERIFY_INTERVAL = 60
VERIFY_AFTER_MINUTES = 15
STILL_FIRING_THRESHOLD = 3
MAX_VERIFY_PER_SWEEP = 20


def run():
    logger.info(
        "[VERIFIER] Starting — interval=%ds, verify_after=%dmin",
        VERIFY_INTERVAL,
        VERIFY_AFTER_MINUTES,
    )
    while True:
        try:
            _sweep()
        except Exception as e:
            logger.error("[VERIFIER] Sweep failed: %s", e)
        time.sleep(VERIFY_INTERVAL)


def _sweep():
    from app.data.models import Incident
    from app.serving.models import ActionLog, Analysis
    from app.shared.database import SessionLocal

    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(minutes=VERIFY_AFTER_MINUTES)

        pending = (
            db.query(ActionLog)
            .filter(
                ActionLog.outcome == "pending",
                ActionLog.actioned_at <= cutoff,
            )
            .order_by(ActionLog.actioned_at.asc())
            .limit(MAX_VERIFY_PER_SWEEP)
            .all()
        )

        if not pending:
            logger.debug("[VERIFIER] Nothing to verify")
            return

        logger.info("[VERIFIER] Verifying %d action(s)", len(pending))

        for log_entry in pending:
            try:
                _verify_one(log_entry, db)
            except Exception as e:
                logger.warning(
                    "[VERIFIER] Failed to verify action %s: %s", log_entry.id, e
                )
                continue

        db.commit()

    finally:
        db.close()


def _verify_one(log_entry, db):
    from app.data.models import Incident

    incident = db.query(Incident).filter(Incident.id == log_entry.incident_id).first()
    if not incident:
        log_entry.outcome = "incident_not_found"
        _update_investigation_verifier_state(
            incident_id=log_entry.incident_id,
            outcome=log_entry.outcome,
            checked_at=datetime.utcnow(),
            db=db,
        )
        return

    count_at_action = _estimate_count_at_action(log_entry, incident)
    count_grew = (incident.count - count_at_action) >= STILL_FIRING_THRESHOLD
    is_closed = incident.status in ("closed", "ignored")
    went_quiet = not count_grew or is_closed

    if went_quiet:
        log_entry.outcome = "resolved"
        log_entry.resolved_at = datetime.utcnow()
        logger.info(
            "[VERIFIER] Resolved: incident=%s disposition=%s count_delta=%d",
            incident.id,
            log_entry.disposition,
            incident.count - count_at_action,
        )
    else:
        log_entry.outcome = "still_firing"
        logger.info(
            "[VERIFIER] Still firing: incident=%s disposition=%s count_delta=%d",
            incident.id,
            log_entry.disposition,
            incident.count - count_at_action,
        )
        _maybe_reescalate(incident, log_entry)

    _update_investigation_verifier_state(
        incident_id=incident.id,
        outcome=log_entry.outcome,
        checked_at=datetime.utcnow(),
        db=db,
    )
    _maybe_tune_runbook(incident, log_entry, went_quiet, db)


def _estimate_count_at_action(log_entry, incident) -> int:
    """
    Rough estimate of incident count when the action was taken.
    We don't store it explicitly — use current count minus recent growth
    as a proxy. Good enough for threshold checks.
    """
    age_minutes = (datetime.utcnow() - incident.first_seen).total_seconds() / 60
    if age_minutes < VERIFY_AFTER_MINUTES + 2:
        return max(1, incident.count - STILL_FIRING_THRESHOLD - 1)
    return max(1, incident.count // 2)


def _maybe_reescalate(incident, log_entry):
    """
    If the incident is still firing and was under-triaged, re-analyze with force=True.
    Only re-escalates if current disposition is OBSERVE or lower.
    """
    low_dispositions = {"OBSERVE", "NO_ACTION"}
    if log_entry.disposition.upper() not in low_dispositions:
        logger.debug(
            "[VERIFIER] Already high disposition — no re-escalation for %s", incident.id
        )
        return

    logger.info("[VERIFIER] Re-escalating under-triaged incident %s", incident.id)

    try:
        from app.control.models import Project
        from app.shared.database import SessionLocal
        from app.serving.orchestrator import analyze_incident

        db2 = SessionLocal()
        try:
            project = (
                db2.query(Project).filter(Project.id == incident.project_id).first()
            )
            if project:
                db2.expunge(project)
        finally:
            db2.close()

        if project:
            analyze_incident(incident, project=project, force=True)
            logger.info("[VERIFIER] Re-analysis triggered for %s", incident.id)
    except Exception as e:
        logger.warning("[VERIFIER] Re-escalation failed for %s: %s", incident.id, e)


def _maybe_tune_runbook(incident, log_entry, went_quiet: bool, db):
    """
    If a runbook-sourced analysis said OBSERVE but the incident kept firing,
    the observe_threshold in the runbook is too high. Log it.
    (Full YAML rewrite is a manual step — we just emit a tuning hint here.)
    """
    from app.serving.models import Analysis

    analysis = (
        db.query(Analysis)
        .filter(Analysis.incident_id == incident.id)
        .order_by(Analysis.created_at.desc())
        .first()
    )
    if not analysis or analysis.analysis_source != "runbook":
        return

    if not went_quiet and analysis.disposition == "OBSERVE":
        logger.warning(
            "[VERIFIER] Tuning hint: runbook '%s' used OBSERVE for incident %s "
            "but it kept firing (count=%d). Consider lowering observe_threshold.count.",
            analysis.matched_runbook_id,
            incident.id,
            incident.count,
        )


def _update_investigation_verifier_state(incident_id: str, outcome: str, checked_at, db):
    from app.serving.models import InvestigationRun

    run = (
        db.query(InvestigationRun)
        .filter(InvestigationRun.incident_id == incident_id)
        .order_by(InvestigationRun.started_at.desc())
        .first()
    )
    if not run:
        return

    run.verifier_outcome = outcome
    run.verifier_checked_at = checked_at
