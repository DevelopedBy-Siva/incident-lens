"""Small, explicit policy between a diagnosis and any side effect."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.shared.observability import trace_operation


MIN_CONFIDENCE = 0.55
DISPOSITIONS = {"NO_ACTION", "OBSERVE", "NEEDS_DEV", "NEEDS_ONCALL", "ESCALATE"}
DEFAULT_ACTIONS = {
    "NO_ACTION": ["auto_suppress"],
    "OBSERVE": ["auto_enrich"],
    "NEEDS_DEV": ["auto_enrich", "send_discord_notification"],
    "NEEDS_ONCALL": ["auto_enrich", "notify_oncall"],
    "ESCALATE": ["auto_enrich", "notify_oncall", "send_discord_notification"],
}
SAFE_ACTIONS = {"auto_enrich", "auto_suppress", "send_discord_notification", "notify_oncall"}
BLOCKED_ACTIONS = {
    "restart_service", "modify_database_config", "deploy_code", "rollback_release",
    "change_infrastructure", "delete_data", "rotate_secrets", "scale_cluster",
}


@dataclass
class PolicyDecision:
    allow: bool
    reason: str
    effective_disposition: str
    requested_actions: list[str] = field(default_factory=list)
    allowed_actions: list[str] = field(default_factory=list)
    blocked_actions: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


def evaluate(incident, analysis) -> PolicyDecision:
    with trace_operation("policy_evaluation", plane="serving", metadata={"project_id": getattr(incident, "project_id", None), "incident_id": str(getattr(incident, "id", "unknown"))}) as span:
        decision = _evaluate(incident, analysis)
        span.tag("result", "approved" if decision.allow else "blocked")
        return decision


def _evaluate(incident, analysis) -> PolicyDecision:
    disposition = str(analysis.disposition or "OBSERVE").upper()
    confidence = float(analysis.confidence or 0.0)
    if getattr(incident, "status", "open") != "open":
        return PolicyDecision(False, "Incident is closed or ignored", disposition, tags=["status_closed"])
    if disposition not in DISPOSITIONS:
        return PolicyDecision(False, "Invalid disposition", "OBSERVE", tags=["invalid_disposition"])
    if confidence < MIN_CONFIDENCE:
        return PolicyDecision(False, f"Confidence {confidence:.2f} is below {MIN_CONFIDENCE:.2f}", "OBSERVE", tags=["low_confidence"])

    requested = list(DEFAULT_ACTIONS[disposition])
    if disposition == "NO_ACTION" and (analysis.severity != "low" or confidence < 0.9):
        return PolicyDecision(False, "Automatic suppression requires low severity and high confidence", disposition, requested, tags=["suppression_not_safe"])
    return PolicyDecision(True, "Policy approved", disposition, requested, requested, [])
