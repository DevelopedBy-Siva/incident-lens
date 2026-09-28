"""Execute only local, reversible actions and record a single audit event."""

from __future__ import annotations

from datetime import datetime


def execute_actions(incident, analysis, policy_decision, project) -> list[str]:
    executed: list[str] = []
    allowed = set(policy_decision.allowed_actions)
    if "auto_enrich" in allowed:
        if _auto_enrich(incident, analysis):
            executed.append("auto_enrich")
    if "auto_suppress" in allowed:
        if _auto_suppress(incident, analysis):
            executed.append("auto_suppress")
    executed.extend(action for action in ("notify_oncall", "send_discord_notification") if action in allowed)
    _log_action(incident, project, analysis, policy_decision, executed)
    return executed


def _auto_enrich(incident, analysis) -> bool:
    from app.data.models import Incident
    from app.shared.database import SessionLocal

    db = SessionLocal()
    try:
        row = db.query(Incident).filter(Incident.id == incident.id).first()
        if not row:
            return False
        row.cause_explanation = analysis.summary
        db.commit()
        return True
    finally:
        db.close()


def _auto_suppress(incident, analysis) -> bool:
    from app.data.models import Incident
    from app.shared.database import SessionLocal

    db = SessionLocal()
    try:
        row = db.query(Incident).filter(Incident.id == incident.id).first()
        if not row or row.status != "open":
            return False
        row.status = "ignored"
        row.cause_explanation = analysis.summary
        db.commit()
        return True
    finally:
        db.close()


def _log_action(incident, project, analysis, policy, executed: list[str]) -> None:
    from app.serving.models import ActionLog
    from app.shared.database import SessionLocal

    db = SessionLocal()
    try:
        external = any(action in executed for action in ("notify_oncall", "send_discord_notification"))
        db.add(ActionLog(
            incident_id=incident.id,
            project_id=project.id,
            requested_actions=policy.requested_actions,
            allowed_actions=policy.allowed_actions,
            blocked_actions=policy.blocked_actions,
            actions_taken=executed,
            disposition=policy.effective_disposition,
            severity=analysis.severity,
            confidence=analysis.confidence,
            policy_reason=policy.reason,
            policy_tags=policy.tags,
            actioned_at=datetime.utcnow(),
            outcome="pending" if external else "completed",
        ))
        db.commit()
    finally:
        db.close()
