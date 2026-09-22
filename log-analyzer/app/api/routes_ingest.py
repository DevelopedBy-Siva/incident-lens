import os
import requests
from fastapi import APIRouter, Depends, HTTPException
from app.control.models import Project
from app.api.routes_auth import get_current_project

router = APIRouter()

LOG_SERVER_URL = os.getenv("LOG_SERVER_URL", "http://localhost:5001").rstrip("/")


def _url(path: str) -> str:
    return f"{LOG_SERVER_URL}{path}"


@router.post("/log-server/start")
def start_log_server(project: Project = Depends(get_current_project)):
    if not all(
        [project.datadog_api_key, project.datadog_site, project.datadog_service]
    ):
        raise HTTPException(
            status_code=422,
            detail="Configure the project's Datadog API key, site, and service first",
        )
    try:
        resp = requests.post(
            _url("/api/start"),
            headers={
                "X-Datadog-API-Key": project.datadog_api_key,
                "X-Datadog-Site": project.datadog_site,
                "X-Datadog-Service": project.datadog_service,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Log server unreachable: {e}")


@router.post("/log-server/stop")
def stop_log_server(project: Project = Depends(get_current_project)):
    try:
        resp = requests.post(_url("/api/stop"), timeout=10)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Log server unreachable: {e}")


@router.post("/eval/ingest")
def eval_ingest(
    data: dict,
    project: Project = Depends(get_current_project),
):
    if not project.is_test:
        raise HTTPException(
            status_code=403, detail="Eval ingest is only available for the demo project"
        )
    from app.data.pipeline import process_log_batch

    result = process_log_batch(
        {
            "project_id": str(project.id),
            "source": data.get("source", "eval"),
            "environment": data.get("environment", "eval"),
            "logs": data.get("logs", []),
            "_project": project,
        }
    )
    return result or {"incidents_created": 0, "incidents_updated": 0, "failed": 0}
