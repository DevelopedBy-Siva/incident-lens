"""
app/api/routes_investigation.py

New endpoints for IncidentLens agent visibility:

  GET  /api/incidents/{id}/evidence      — evidence bundle used for this incident
  GET  /api/incidents/{id}/actions       — action log + verifier outcome
  GET  /api/incidents/{id}/investigation — full InvestigationRun audit trail
  POST /api/log-server/scenario/{name}   — trigger a demo scenario (test projects)
  GET  /api/log-server/scenarios         — list available scenarios

Register in app/main.py:
    from app.api import routes_investigation
    app.include_router(routes_investigation.router, prefix="/api", tags=["investigation"])
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional

from app.services.storage import get_db, Incident, Analysis, Project
from app.api.routes_auth import get_current_project

router = APIRouter()


# ---------------------------------------------------------------------------
# /incidents/{id}/evidence
# ---------------------------------------------------------------------------


@router.get("/incidents/{incident_id}/evidence")
def get_evidence(
    incident_id: str,
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    """
    Returns the evidence bundle snapshot used during investigation.
    Falls back to building it live if no InvestigationRun exists yet.
    """
    incident = _get_incident_or_404(incident_id, project.id, db)

    # Try to return stored snapshot from InvestigationRun
    try:
        from app.services.storage import InvestigationRun

        run = (
            db.query(InvestigationRun)
            .filter(InvestigationRun.incident_id == incident_id)
            .order_by(InvestigationRun.started_at.desc())
            .first()
        )
        if run and run.evidence_snapshot:
            return {
                "incident_id": incident_id,
                "source": "investigation_run",
                "gathered_at": run.started_at.isoformat(),
                "evidence_samples": run.evidence_samples,
                "evidence_related_count": run.evidence_related_count,
                "evidence_runbook": run.evidence_runbook,
                "snapshot": run.evidence_snapshot,
                "tool_calls": run.tool_calls or [],
                "iterations": run.iterations,
                "fallback_used": run.fallback_used,
            }
    except Exception:
        pass

    # Live build
    from app.core.evidence import build_evidence

    evidence = build_evidence(incident, project)

    return {
        "incident_id": incident_id,
        "source": "live",
        "gathered_at": evidence.gathered_at.isoformat(),
        "evidence_samples": len(evidence.sample_lines),
        "evidence_related_count": len(evidence.related_incidents),
        "evidence_runbook": evidence.runbook_name,
        "snapshot": evidence.as_prompt_context(),
        "tool_calls": [],
        "iterations": 0,
        "fallback_used": False,
        "sample_lines": evidence.sample_lines,
        "related_incidents": [
            {
                "id": r.id,
                "source": r.source,
                "signature": r.signature,
                "count": r.count,
                "severity": r.severity,
                "disposition": r.disposition,
                "first_seen": r.first_seen.isoformat(),
            }
            for r in evidence.related_incidents
        ],
        "runbook": (
            {
                "id": evidence.runbook_id,
                "name": evidence.runbook_name,
                "score": evidence.runbook_score,
                "steps": evidence.runbook_steps,
            }
            if evidence.runbook_id
            else None
        ),
        "root_cause": (
            {
                "incident_id": evidence.root_cause_id,
                "signature": evidence.root_cause_signature,
                "explanation": evidence.root_cause_explanation,
            }
            if evidence.root_cause_id
            else None
        ),
    }


# ---------------------------------------------------------------------------
# /incidents/{id}/actions
# ---------------------------------------------------------------------------


@router.get("/incidents/{incident_id}/actions")
def get_actions(
    incident_id: str,
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    """
    Returns the action log for this incident — what the agent did and why,
    plus the verifier outcome.
    """
    _get_incident_or_404(incident_id, project.id, db)

    try:
        from app.services.storage import ActionLog

        logs = (
            db.query(ActionLog)
            .filter(ActionLog.incident_id == incident_id)
            .order_by(ActionLog.actioned_at.desc())
            .all()
        )
        return {
            "incident_id": incident_id,
            "action_count": len(logs),
            "actions": [
                {
                    "id": log.id,
                    "actioned_at": log.actioned_at.isoformat(),
                    "requested_actions": getattr(log, "requested_actions", None) or [],
                    "allowed_actions": getattr(log, "allowed_actions", None) or [],
                    "blocked_actions": getattr(log, "blocked_actions", None) or [],
                    "actions_taken": log.actions_taken or [],
                    "disposition": log.disposition,
                    "severity": log.severity,
                    "confidence": log.confidence,
                    "policy_reason": getattr(log, "policy_reason", None),
                    "policy_tags": log.policy_tags or [],
                    "outcome": log.outcome,
                    "resolved_at": (
                        log.resolved_at.isoformat() if log.resolved_at else None
                    ),
                }
                for log in logs
            ],
        }
    except Exception as e:
        return {
            "incident_id": incident_id,
            "action_count": 0,
            "actions": [],
            "note": f"ActionLog table not yet migrated: {e}",
        }


# ---------------------------------------------------------------------------
# /incidents/{id}/investigation  — full audit trail
# ---------------------------------------------------------------------------


@router.get("/incidents/{incident_id}/investigation")
def get_investigation(
    incident_id: str,
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    """
    Full InvestigationRun audit trail — the complete "why did the agent do this?"
    record including evidence, tool calls, policy decision, actions, and verifier result.
    """
    incident = _get_incident_or_404(incident_id, project.id, db)

    # Fetch latest analysis
    analysis = (
        db.query(Analysis)
        .filter(Analysis.incident_id == incident_id)
        .order_by(Analysis.created_at.desc())
        .first()
    )

    result = {
        "incident_id": incident_id,
        "signature": incident.signature,
        "status": incident.status,
        "count": incident.count,
    }

    if analysis:
        result["analysis"] = {
            "severity": analysis.severity,
            "disposition": analysis.disposition,
            "confidence": analysis.confidence,
            "summary": analysis.summary,
            "next_steps": analysis.next_steps,
            "ticket_title": analysis.ticket_title,
            "ticket_body": analysis.ticket_body,
            "analysis_source": analysis.analysis_source,
            "matched_runbook_id": analysis.matched_runbook_id,
        }

    try:
        from app.services.storage import InvestigationRun, ActionLog

        run = (
            db.query(InvestigationRun)
            .filter(InvestigationRun.incident_id == incident_id)
            .order_by(InvestigationRun.started_at.desc())
            .first()
        )
        if run:
            result["investigation"] = {
                "started_at": run.started_at.isoformat(),
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                "evidence": {
                    "samples": run.evidence_samples,
                    "related_incidents": run.evidence_related_count,
                    "runbook_matched": run.evidence_runbook,
                },
                "agent": {
                    "iterations": run.iterations,
                    "tool_calls": run.tool_calls or [],
                    "fallback_used": run.fallback_used,
                    "analysis_source": run.analysis_source,
                },
                "policy": {
                    "allowed": run.policy_allowed,
                    "reason": run.policy_reason,
                    "tags": run.policy_tags or [],
                    "effective_disposition": run.effective_disposition,
                },
                "actions": run.actions_taken or [],
                "verifier": {
                    "outcome": run.verifier_outcome,
                    "checked_at": (
                        run.verifier_checked_at.isoformat()
                        if run.verifier_checked_at
                        else None
                    ),
                },
                "result": {
                    "severity": run.final_severity,
                    "disposition": run.final_disposition,
                    "confidence": run.final_confidence,
                    "summary": run.final_summary,
                },
            }
        else:
            result["investigation"] = None

        action_logs = (
            db.query(ActionLog)
            .filter(ActionLog.incident_id == incident_id)
            .order_by(ActionLog.actioned_at.desc())
            .limit(5)
            .all()
        )
        result["action_log"] = [
            {
                "actioned_at": al.actioned_at.isoformat(),
                "requested_actions": getattr(al, "requested_actions", None) or [],
                "allowed_actions": getattr(al, "allowed_actions", None) or [],
                "blocked_actions": getattr(al, "blocked_actions", None) or [],
                "actions_taken": al.actions_taken or [],
                "outcome": al.outcome,
                "policy_reason": getattr(al, "policy_reason", None),
                "policy_tags": al.policy_tags or [],
            }
            for al in action_logs
        ]

    except Exception as e:
        result["investigation"] = None
        result["action_log"] = []
        result["note"] = f"InvestigationRun table not yet migrated: {e}"

    return result


# ---------------------------------------------------------------------------
# Log server scenario routes (moved here from routes_log_server.py)
# ---------------------------------------------------------------------------


@router.post("/log-server/scenario/{scenario_name}")
def run_scenario(
    scenario_name: str,
    project: Project = Depends(get_current_project),
):
    """Trigger a named demo scenario against the log server (test projects only)."""
    import os, requests

    if not project.is_test:
        raise HTTPException(
            status_code=403, detail="Scenarios only available for demo project"
        )

    log_server_url = os.getenv("LOG_SERVER_URL", "http://localhost:5001").rstrip("/")
    try:
        resp = requests.post(
            f"{log_server_url}/api/scenario/{scenario_name}", timeout=10
        )
        if resp.status_code == 404:
            raise HTTPException(
                status_code=404, detail=f"Unknown scenario: {scenario_name}"
            )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Log server unreachable: {e}")


@router.get("/log-server/scenarios")
def list_scenarios(project: Project = Depends(get_current_project)):
    """List available demo scenarios."""
    import os, requests

    if not project.is_test:
        raise HTTPException(
            status_code=403, detail="Scenarios only available for demo project"
        )

    log_server_url = os.getenv("LOG_SERVER_URL", "http://localhost:5001").rstrip("/")
    try:
        resp = requests.get(f"{log_server_url}/api/scenario", timeout=10)
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Log server unreachable: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_incident_or_404(incident_id: str, project_id: str, db: Session) -> Incident:
    incident = (
        db.query(Incident)
        .filter(Incident.id == incident_id, Incident.project_id == project_id)
        .first()
    )
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident
