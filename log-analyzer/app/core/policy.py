from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import logging
import re

logger = logging.getLogger(__name__)


MIN_CONFIDENCE_TO_ACT = 0.55

MIN_CONFIDENCE_RUNBOOK = 0.40

AUTO_SUPPRESS_MIN_CONFIDENCE = 0.80

COOLDOWN_MINUTES = 20

ALLOWED_AUTO_DISPOSITIONS = {
    "ESCALATE",
    "NEEDS_ONCALL",
    "NEEDS_DEV",
    "OBSERVE",
    "NO_ACTION",
}

MIN_COUNT_TO_ESCALATE = 3

SAFE_ACTIONS = {
    "auto_enrich",
    "create_incident_summary",
    "attach_evidence_bundle",
    "send_discord_notification",
    "send_email_notification",
    "notify_oncall",
    "run_verification_check",
}

CONDITIONAL_ACTIONS = {
    "auto_suppress",
    "mark_duplicate_incident",
    "suppress_duplicate_notifications",
}

BLOCKED_ACTIONS = {
    "restart_service",
    "modify_database_config",
    "deploy_code",
    "rollback_release",
    "change_infrastructure",
    "delete_data",
    "rotate_secrets",
    "scale_cluster",
}

DEFAULT_ACTIONS_BY_DISPOSITION = {
    "NO_ACTION": ["auto_enrich", "auto_suppress"],
    "OBSERVE": ["auto_enrich"],
    "NEEDS_DEV": [
        "auto_enrich",
        "create_incident_summary",
        "send_discord_notification",
    ],
    "NEEDS_ONCALL": [
        "auto_enrich",
        "create_incident_summary",
        "notify_oncall",
        "send_discord_notification",
    ],
    "ESCALATE": [
        "auto_enrich",
        "create_incident_summary",
        "attach_evidence_bundle",
        "notify_oncall",
        "send_discord_notification",
    ],
}

SEVERITY_DISPOSITION_FLOOR = {
    "critical": "NEEDS_ONCALL",
    "high": "NEEDS_DEV",
    "medium": "OBSERVE",
    "low": "OBSERVE",
}

DISPOSITION_RANK = {
    "NO_ACTION": 0,
    "OBSERVE": 1,
    "NEEDS_DEV": 2,
    "NEEDS_ONCALL": 3,
    "ESCALATE": 4,
}


@dataclass
class PolicyDecision:
    allow: bool
    reason: str
    effective_disposition: Optional[str] = None
    requested_actions: list[str] = field(default_factory=list)
    allowed_actions: list[str] = field(default_factory=list)
    blocked_actions: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


def _rank(disposition: str) -> int:
    return DISPOSITION_RANK.get(disposition.upper(), 0)


def _floor_disposition(severity: str, proposed: str) -> str:
    """Ensure disposition is at least the floor for this severity."""
    if severity.lower() == "low" and proposed == "NO_ACTION":
        return proposed

    floor = SEVERITY_DISPOSITION_FLOOR.get(severity.lower(), "OBSERVE")
    if _rank(proposed) < _rank(floor):
        return floor
    return proposed


def normalize_action_name(action: str) -> str:
    """Convert user/LLM action labels to lowercase snake_case names."""
    value = str(action or "").strip()
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value.lower()


def _dedupe_preserve_order(actions: list[str]) -> list[str]:
    seen = set()
    result = []
    for action in actions:
        normalized = normalize_action_name(action)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def get_requested_actions(analysis, effective_disposition: str) -> list[str]:
    for attr in ("proposed_actions", "requested_actions", "actions"):
        actions = getattr(analysis, attr, None)
        if actions:
            if isinstance(actions, str):
                actions = [actions]
            return _dedupe_preserve_order(list(actions))

    defaults = DEFAULT_ACTIONS_BY_DISPOSITION.get(
        (effective_disposition or "OBSERVE").upper(),
        DEFAULT_ACTIONS_BY_DISPOSITION["OBSERVE"],
    )
    return _dedupe_preserve_order(defaults)


def _was_recently_acted_on(incident, cooldown_minutes: int) -> bool:
    """
    Check whether this incident already had an action fired recently.
    Uses ActionLog as the source of truth. Falls back open on query failures.
    """
    try:
        from app.services.storage import ActionLog, SessionLocal

        db = SessionLocal()
        try:
            latest = (
                db.query(ActionLog.actioned_at)
                .filter(ActionLog.incident_id == incident.id)
                .order_by(ActionLog.actioned_at.desc())
                .first()
            )
            acted_at: Optional[datetime] = latest[0] if latest else None
        finally:
            db.close()
    except Exception as e:
        logger.warning(
            "[POLICY] ActionLog cooldown check failed for incident %s: %s",
            getattr(incident, "id", "<unknown>"),
            e,
        )
        return False

    if acted_at is None:
        return False

    cutoff = datetime.utcnow() - timedelta(minutes=cooldown_minutes)
    return acted_at > cutoff


def _evaluate_requested_actions(
    requested_actions: list[str],
    effective_disposition: str,
    severity: str,
    confidence: float,
    tags: list[str],
) -> tuple[list[str], list[str]]:
    allowed_actions: list[str] = []
    blocked_actions: list[str] = []

    for action in requested_actions:
        if action in BLOCKED_ACTIONS:
            blocked_actions.append(action)
            tags.append(f"blocked:dangerous_action:{action}")
        elif action in SAFE_ACTIONS:
            allowed_actions.append(action)
        elif action == "auto_suppress":
            if (
                effective_disposition == "NO_ACTION"
                and confidence >= AUTO_SUPPRESS_MIN_CONFIDENCE
                and severity == "low"
            ):
                allowed_actions.append(action)
            else:
                blocked_actions.append(action)
                tags.append(f"blocked:conditional_action:{action}")
        elif action == "suppress_duplicate_notifications":
            if severity in ("low", "medium") and confidence >= 0.80:
                allowed_actions.append(action)
            else:
                blocked_actions.append(action)
                tags.append(f"blocked:conditional_action:{action}")
        elif action == "mark_duplicate_incident":
            if confidence >= 0.75:
                allowed_actions.append(action)
            else:
                blocked_actions.append(action)
                tags.append(f"blocked:conditional_action:{action}")
        else:
            blocked_actions.append(action)
            tags.append(f"blocked:unknown_action:{action}")

    return allowed_actions, blocked_actions


def evaluate(incident, analysis) -> PolicyDecision:
    """
    Evaluate whether the analysis warrants immediate action.

    Args:
        incident:  Incident ORM object (needs .count, .status, optionally .last_actioned_at)
        analysis:  Analysis ORM object OR IncidentAnalysis pydantic model
                   (needs .severity, .disposition, .confidence, .analysis_source)

    Returns:
        PolicyDecision — always returned, never raises.
    """
    tags: list[str] = []

    severity = (analysis.severity or "low").lower().strip()
    disposition = (analysis.disposition or "OBSERVE").upper().strip()
    confidence = float(analysis.confidence or 0.0)
    source = str(getattr(analysis, "analysis_source", "llm") or "llm").lower()

    if disposition not in DISPOSITION_RANK:
        tags.append("normalized:unknown_disposition")
        disposition = "OBSERVE"

    if incident.status in ("closed", "ignored"):
        requested_actions = get_requested_actions(analysis, disposition)
        return PolicyDecision(
            allow=False,
            reason=f"Incident is {incident.status} — no action",
            effective_disposition=disposition,
            requested_actions=requested_actions,
            blocked_actions=requested_actions,
            tags=["blocked:status"],
        )

    min_conf = MIN_CONFIDENCE_RUNBOOK if source == "runbook" else MIN_CONFIDENCE_TO_ACT
    if confidence < min_conf:
        tags.append("blocked:low_confidence")
        requested_actions = get_requested_actions(analysis, "OBSERVE")
        logger.info(
            "[POLICY] Blocked — confidence %.2f < %.2f for incident %s",
            confidence,
            min_conf,
            incident.id,
        )
        return PolicyDecision(
            allow=False,
            reason=f"Confidence {confidence:.2f} below threshold {min_conf:.2f} ({source})",
            effective_disposition="OBSERVE",
            requested_actions=requested_actions,
            blocked_actions=requested_actions,
            tags=tags,
        )

    if _was_recently_acted_on(incident, COOLDOWN_MINUTES):
        tags.append("blocked:cooldown")
        requested_actions = get_requested_actions(analysis, disposition)
        return PolicyDecision(
            allow=False,
            reason=f"Incident actioned within last {COOLDOWN_MINUTES}min — cooldown active",
            effective_disposition=disposition,
            requested_actions=requested_actions,
            blocked_actions=requested_actions,
            tags=tags,
        )

    if (
        disposition == "ESCALATE"
        and severity != "critical"
        and incident.count < MIN_COUNT_TO_ESCALATE
    ):
        tags.append("downgraded:count_too_low")
        disposition = "NEEDS_DEV"
        logger.info(
            "[POLICY] Downgraded ESCALATE → NEEDS_DEV — count %d < %d for incident %s",
            incident.count,
            MIN_COUNT_TO_ESCALATE,
            incident.id,
        )

    floored = _floor_disposition(severity, disposition)
    if floored != disposition:
        tags.append(f"floored:{disposition}→{floored}")
        disposition = floored

    if disposition not in ALLOWED_AUTO_DISPOSITIONS:
        tags.append("normalized:unknown_disposition")
        disposition = "OBSERVE"

    requested_actions = get_requested_actions(analysis, disposition)
    allowed_actions, blocked_actions = _evaluate_requested_actions(
        requested_actions=requested_actions,
        effective_disposition=disposition,
        severity=severity,
        confidence=confidence,
        tags=tags,
    )

    allow = bool(allowed_actions)
    if not allow:
        reason = "All requested actions blocked"
    elif blocked_actions:
        reason = "Policy checks passed with blocked actions"
    else:
        reason = "All policy checks passed"

    logger.info(
        "[POLICY] %s — %s/%s conf=%.2f src=%s incident=%s allowed=%s blocked=%s",
        "Allowed" if allow else "Blocked",
        severity,
        disposition,
        confidence,
        source,
        incident.id,
        allowed_actions,
        blocked_actions,
    )

    return PolicyDecision(
        allow=allow,
        reason=reason,
        effective_disposition=disposition,
        requested_actions=requested_actions,
        allowed_actions=allowed_actions,
        blocked_actions=blocked_actions,
        tags=tags,
    )
