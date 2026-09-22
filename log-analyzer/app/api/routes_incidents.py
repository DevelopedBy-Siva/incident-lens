from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.routes_auth import get_current_project
from app.control.maintenance import cleanup_project_data
from app.control.models import Project
from app.data.models import Incident
from app.serving.models import Analysis
from app.shared.database import get_db

router = APIRouter()


@router.delete("/incidents")
def clear_project_incident_data(
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    """Clear incident-processing data for the authenticated project."""
    try:
        deleted = cleanup_project_data(db, project.id)
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500, detail="Failed to clear project incident data"
        ) from exc
    return {"status": "cleared", "project_id": project.id, "deleted": deleted}


def _incident_to_dict(inc: Incident, analysis: Optional[Analysis]) -> dict:
    """Shared serializer used by list and detail endpoints."""
    result = {
        "id": inc.id,
        "source": inc.source,
        "environment": inc.environment,
        "signature": inc.signature,
        "first_seen": inc.first_seen.isoformat(),
        "last_seen": inc.last_seen.isoformat(),
        "count": inc.count,
        "status": inc.status,
        "sample_lines": inc.sample_lines,
        "root_cause_incident_id": inc.root_cause_incident_id,
        "cause_explanation": inc.cause_explanation,
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
        }

    return result


@router.get("/incidents")
def list_incidents(
    status: Optional[str] = None,
    severity: Optional[str] = None,
    ticket_title: Optional[str] = None,
    limit: int = 50,
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    """List incidents with optional filters."""
    query = db.query(Incident).filter(Incident.project_id == project.id)

    if status:
        status = status.strip().lower()
        query = query.filter(func.lower(Incident.status) == status)

    if severity or ticket_title:
        latest_analysis = db.query(
            Analysis.incident_id,
            Analysis.id.label("analysis_id"),
            func.row_number()
            .over(
                partition_by=Analysis.incident_id, order_by=Analysis.created_at.desc()
            )
            .label("rn"),
        ).subquery()

        query = query.join(
            latest_analysis, Incident.id == latest_analysis.c.incident_id
        ).filter(latest_analysis.c.rn == 1)

        query = query.join(Analysis, Analysis.id == latest_analysis.c.analysis_id)

        if severity:
            severity = severity.strip().lower()
            query = query.filter(func.lower(Analysis.severity) == severity)

        if ticket_title:
            ticket_title = ticket_title.strip()
            query = query.filter(
                func.lower(Analysis.ticket_title).contains(ticket_title.lower())
            )

    incidents = query.order_by(Incident.last_seen.desc()).limit(limit).all()

    result = []
    for inc in incidents:
        analysis = (
            db.query(Analysis)
            .filter(Analysis.incident_id == inc.id)
            .order_by(Analysis.created_at.desc())
            .first()
        )
        result.append(_incident_to_dict(inc, analysis))

    return result


@router.get("/incidents/{incident_id}")
def get_incident(
    incident_id: str,
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    """Get single incident by ID."""
    incident = (
        db.query(Incident)
        .filter(Incident.id == incident_id, Incident.project_id == project.id)
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    analysis = (
        db.query(Analysis)
        .filter(Analysis.incident_id == incident_id)
        .order_by(Analysis.created_at.desc())
        .first()
    )

    result = _incident_to_dict(incident, analysis)

    if incident.root_cause_incident_id:
        cause = (
            db.query(Incident)
            .filter(Incident.id == incident.root_cause_incident_id)
            .first()
        )
        if cause:
            cause_analysis = (
                db.query(Analysis)
                .filter(Analysis.incident_id == cause.id)
                .order_by(Analysis.created_at.desc())
                .first()
            )
            result["root_cause_incident"] = {
                "id": cause.id,
                "signature": cause.signature,
                "first_seen": cause.first_seen.isoformat(),
                "ticket_title": cause_analysis.ticket_title if cause_analysis else None,
                "severity": cause_analysis.severity if cause_analysis else None,
            }

    return result


@router.post("/incidents/{incident_id}/close")
def close_incident(
    incident_id: str,
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    """Mark incident as closed."""
    incident = (
        db.query(Incident)
        .filter(Incident.id == incident_id, Incident.project_id == project.id)
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    incident.status = "closed"
    db.commit()
    return {"status": "closed"}


@router.post("/incidents/{incident_id}/ignore")
def ignore_incident(
    incident_id: str,
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    """Mark incident as ignored."""
    incident = (
        db.query(Incident)
        .filter(Incident.id == incident_id, Incident.project_id == project.id)
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    incident.status = "ignored"
    db.commit()
    return {"status": "ignored"}
