"""The single synchronous path from normalized logs to incident analyses."""

from app.control.models import Project
from app.data.clustering import cluster_log_db
from app.data.parser import ParsedLog
from app.data.signatures import generate_signature
from app.shared.database import SessionLocal


ACTIONABLE_LEVELS = {"WARN", "ERROR", "CRITICAL"}
REANALYSE_COUNTS = {2, 5, 10, 20, 50}


def process_log_batch(payload: dict) -> dict[str, int]:
    from app.serving.orchestrator import analyze_incident

    project = payload.get("_project") or _project(payload["project_id"])
    if project is None:
        return {"incidents_created": 0, "incidents_updated": 0, "failed": 1}

    totals = {"incidents_created": 0, "incidents_updated": 0, "failed": 0}
    for raw in payload.get("logs", []):
        try:
            parsed = ParsedLog(raw)
            if parsed.level not in ACTIONABLE_LEVELS:
                continue
            incident, created = cluster_log_db(
                project_id=project.id,
                source=payload["source"],
                environment=payload.get("environment", "prod"),
                parsed_log=parsed,
                signature=generate_signature(payload["source"], parsed),
            )
            totals["incidents_created" if created else "incidents_updated"] += 1
            if created or incident.count in REANALYSE_COUNTS:
                analyze_incident(incident, project=project, force=not created)
        except Exception:
            totals["failed"] += 1
    return totals


def _project(project_id: str):
    db = SessionLocal()
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            db.expunge(project)
        return project
    finally:
        db.close()
