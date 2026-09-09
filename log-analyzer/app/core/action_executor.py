import logging
from datetime import datetime

logger = logging.getLogger(__name__)

AUTO_SUPPRESS_MIN_CONFIDENCE = 0.80
AUTO_ENRICH_MIN_CONFIDENCE = 0.55

NOTIFICATION_ACTIONS = {
    "send_discord_notification",
    "send_email_notification",
    "notify_oncall",
}


def execute_actions(incident, analysis, policy_decision, project) -> list[str]:
    """
    Execute all applicable autonomous actions for this incident.

    Returns list of action names that were executed (for logging/Langfuse).
    Never raises — all errors are caught and logged.
    """
    executed = []

    try:
        from app.core.policy import BLOCKED_ACTIONS

        allowed_actions = [
            action
            for action in getattr(policy_decision, "allowed_actions", []) or []
            if action not in BLOCKED_ACTIONS
        ]

        if "auto_enrich" in allowed_actions and _should_enrich(analysis):
            if _auto_enrich(incident, analysis):
                executed.append("auto_enrich")

        if "auto_suppress" in allowed_actions and _should_suppress(
            analysis, policy_decision
        ):
            if _auto_suppress(incident, analysis):
                executed.append("auto_suppress")

        for action in allowed_actions:
            if action in NOTIFICATION_ACTIONS:
                executed.append(action)

        _log_action(
            incident_id=incident.id,
            project_id=project.id,
            actions_taken=executed,
            disposition=policy_decision.effective_disposition or analysis.disposition,
            severity=analysis.severity,
            confidence=analysis.confidence,
            policy_decision=policy_decision,
            policy_tags=policy_decision.tags,
        )

    except Exception as e:
        logger.error("[ACTION] execute_actions failed for %s: %s", incident.id, e)

    if executed:
        logger.info("[ACTION] Executed for %s: %s", incident.id, executed)
    else:
        logger.debug("[ACTION] No automated actions for %s", incident.id)

    return executed


def _auto_enrich(incident, analysis) -> bool:
    """
    Write ticket_body and cause_explanation back to the incident row.
    Gives the dashboard rich context without any external call.
    """
    from app.services.storage import Incident, SessionLocal

    db = SessionLocal()
    try:
        row = db.query(Incident).filter(Incident.id == incident.id).first()
        if not row:
            return False

        updated = False
        if analysis.ticket_body and not row.cause_explanation:
            row.cause_explanation = (
                f"[Auto-enriched {datetime.utcnow().strftime('%H:%M:%S')}]\n"
                f"{analysis.summary}\n\n"
                f"Ticket: {analysis.ticket_title}"
            )
            updated = True

        if updated:
            db.commit()
            logger.info("[ACTION] auto_enrich: enriched incident %s", incident.id)
        return updated

    except Exception as e:
        db.rollback()
        logger.warning("[ACTION] auto_enrich failed for %s: %s", incident.id, e)
        return False
    finally:
        db.close()


def _auto_suppress(incident, analysis) -> bool:
    """
    Automatically close incidents the LLM is very confident are noise.
    Only fires for NO_ACTION at high confidence.
    """
    from app.services.storage import Incident, SessionLocal

    db = SessionLocal()
    try:
        row = db.query(Incident).filter(Incident.id == incident.id).first()
        if not row or row.status != "open":
            return False

        row.status = "ignored"
        row.cause_explanation = (
            f"[Auto-suppressed {datetime.utcnow().strftime('%H:%M:%S')}] "
            f"Confidence {analysis.confidence:.0%} — {analysis.summary}"
        )
        db.commit()
        logger.info(
            "[ACTION] auto_suppress: ignored incident %s (confidence=%.2f)",
            incident.id,
            analysis.confidence,
        )
        return True

    except Exception as e:
        db.rollback()
        logger.warning("[ACTION] auto_suppress failed for %s: %s", incident.id, e)
        return False
    finally:
        db.close()


def _should_enrich(analysis) -> bool:
    conf = float(analysis.confidence or 0)
    return conf >= AUTO_ENRICH_MIN_CONFIDENCE and bool(
        getattr(analysis, "ticket_body", None)
    )


def _should_suppress(analysis, policy_decision) -> bool:
    disposition = (
        policy_decision.effective_disposition or analysis.disposition or ""
    ).upper()
    conf = float(analysis.confidence or 0)
    return disposition == "NO_ACTION" and conf >= AUTO_SUPPRESS_MIN_CONFIDENCE


def _log_action(
    incident_id: str,
    project_id: str,
    actions_taken: list[str],
    disposition: str,
    severity: str,
    confidence: float,
    policy_decision,
    policy_tags: list[str],
):
    """Write an ActionLog row for Phase 5 verification."""
    from app.services.storage import ActionLog, SessionLocal

    db = SessionLocal()
    try:
        entry = ActionLog(
            incident_id=incident_id,
            project_id=project_id,
            requested_actions=getattr(policy_decision, "requested_actions", []) or [],
            allowed_actions=getattr(policy_decision, "allowed_actions", []) or [],
            blocked_actions=getattr(policy_decision, "blocked_actions", []) or [],
            actions_taken=actions_taken,
            disposition=disposition,
            severity=severity,
            confidence=confidence,
            policy_reason=getattr(policy_decision, "reason", None),
            policy_tags=policy_tags,
            actioned_at=datetime.utcnow(),
            outcome="completed" if actions_taken else "skipped",
        )
        db.add(entry)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.warning("[ACTION] Failed to write ActionLog for %s: %s", incident_id, e)
    finally:
        db.close()
