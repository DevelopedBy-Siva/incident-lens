"""Synchronous log-processing pipeline used by Loki and evaluation ingestion."""

from app.control.models import Project
from app.data.clustering import cluster_log_db
from app.data.parser import ParsedLog
from app.data.signatures import generate_signature
from app.shared.database import SessionLocal


def process_log_batch(payload: dict):
    # Imported lazily to keep the data modules independently importable while the
    # existing synchronous pipeline still hands new incidents to the serving plane.
    from app.serving.orchestrator import analyze_incident, run_root_cause_chaining

    project_id = payload.get("project_id")
    source = payload["source"]
    environment = payload["environment"]
    logs = payload["logs"]
    project = payload.get("_project")

    if project is None:
        db = SessionLocal()
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                db.expunge(project)
            else:
                print(f"[WORKER] Project {project_id} not found")
                return
        finally:
            db.close()

    print(f"[WORKER] {len(logs)} logs for '{project.name}'")
    created = updated = failed = 0

    for log_line in logs:
        try:
            parsed = ParsedLog(log_line)
            if parsed.level not in ["ERROR", "WARN", "WARNING", "CRITICAL"]:
                continue

            sig = generate_signature(source, parsed)
            incident, is_new = cluster_log_db(
                project_id=project_id,
                source=source,
                environment=environment,
                parsed_log=parsed,
                signature=sig,
            )

            if is_new:
                analyze_incident(incident, project=project, force=False)
                run_root_cause_chaining(incident, project_id, project=project)
                created += 1
            else:
                if incident.count in {5, 10, 20}:
                    analyze_incident(incident, project=project, force=True)
                updated += 1

        except Exception as e:
            print(f"[WORKER] Failed: {e} | {log_line[:80]}")
            failed += 1

    print(f"[WORKER] created={created} updated={updated} failed={failed}")
    if failed > 0 and created == 0 and updated == 0:
        raise RuntimeError(f"Batch entirely failed — {failed} errors")

    return {
        "incidents_created": created,
        "incidents_updated": updated,
        "failed": failed,
    }
