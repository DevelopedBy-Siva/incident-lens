import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, validator
from sqlalchemy.orm import Session

from app.control.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.control.models import Project
from app.control.maintenance import delete_project
from app.data.ingestion.datadog_connector import (
    SUPPORTED_DATADOG_SITES,
    DatadogLogConnector,
    DatadogLogSourceConfig,
)
from app.data.ingestion.log_source import (
    LogSourceAuthenticationError,
    LogSourceError,
)
from app.shared.database import get_db
from app.shared.observability import trace_operation

router = APIRouter()
security = HTTPBearer()

HIDDEN = "HIDDEN CREDENTIAL"


class ProjectRegister(BaseModel):
    name: str
    password: str

    @validator("name")
    def validate_name(cls, v):
        if len(v) < 3:
            raise ValueError("Name must be at least 3 characters")
        if len(v) > 50:
            raise ValueError("Name must be less than 50 characters")
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Name can only contain letters, numbers, hyphens, underscores"
            )
        return v

    @validator("password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class ProjectLogin(BaseModel):
    name: str
    password: str


class ProjectSettings(BaseModel):
    # Datadog Logs
    datadog_api_key: Optional[str] = None
    datadog_app_key: Optional[str] = None
    datadog_site: Optional[str] = None
    datadog_query: Optional[str] = None
    datadog_service: Optional[str] = None
    # Notifications
    user_email: Optional[str] = None
    discord_webhook_escalate: Optional[str] = None
    discord_webhook_dev: Optional[str] = None
    # Password change
    password: Optional[str] = None

    @validator("datadog_site")
    def validate_datadog_site(cls, v):
        if v is None:
            return v
        site = v.strip().lower().removeprefix("https://").rstrip("/")
        if site not in SUPPORTED_DATADOG_SITES:
            raise ValueError("Unsupported Datadog site")
        return site

    @validator("password")
    def validate_password(cls, v):
        if v is not None and len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class DatadogVerificationRequest(BaseModel):
    datadog_api_key: Optional[str] = None
    datadog_app_key: Optional[str] = None
    datadog_site: Optional[str] = None
    datadog_query: Optional[str] = None
    datadog_service: Optional[str] = None

    @validator("datadog_site")
    def validate_datadog_site(cls, v):
        return ProjectSettings.validate_datadog_site(v)


def _mask(value: Optional[str], is_test: bool = False) -> Optional[str]:
    """Return a masked or hidden value for API keys."""
    if not value:
        return None
    if is_test:
        return HIDDEN
    return "••••••"


def _enabled(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _project_to_dict(project: Project) -> dict:
    t = project.is_test
    datadog_api_key = project.datadog_api_key
    datadog_app_key = project.datadog_app_key
    datadog_configured = all(
        [
            datadog_api_key,
            datadog_app_key,
            project.datadog_site,
            project.datadog_query,
            project.datadog_service,
        ]
    )
    return {
        "id": project.id,
        "name": project.name,
        "base_model": project.base_model,
        "active_artifact_id": project.active_artifact_id,
        "created_at": project.created_at.isoformat(),
        "is_test": t,
        # Datadog Logs
        "datadog_api_key": _mask(datadog_api_key, t),
        "datadog_app_key": _mask(datadog_app_key, t),
        "datadog_site": project.datadog_site,
        "datadog_query": project.datadog_query,
        "datadog_service": project.datadog_service,
        "observability": {
            "platform": "Datadog",
            "llm_observability": _enabled("DD_LLMOBS_ENABLED"),
            "logs": datadog_configured,
            "tracing": _enabled("DD_TRACE_ENABLED"),
        },
        # Notifications
        "user_email": HIDDEN if t else project.user_email,
        "discord_webhook_escalate": HIDDEN if t else project.discord_webhook_escalate,
        "discord_webhook_dev": HIDDEN if t else project.discord_webhook_dev,
        # Setup status
        "setup_complete": all(
            [
                datadog_configured,
                project.active_artifact_id,
            ]
        ),
        "setup_status": {
            "datadog": datadog_configured,
            "llm": bool(project.active_artifact_id),
            "observability": _enabled("DD_LLMOBS_ENABLED"),
            "notifications": any(
                [
                    project.user_email,
                    project.discord_webhook_escalate,
                    project.discord_webhook_dev,
                ]
            ),
        },
    }


def get_current_project(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> Project:
    payload = decode_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    project = db.query(Project).filter(Project.id == payload.get("project_id")).first()
    if not project:
        raise HTTPException(status_code=401, detail="Project not found")
    return project


@router.post("/register")
def register_project(data: ProjectRegister, db: Session = Depends(get_db)):
    """Name + password only. All credentials configured in Settings."""
    if db.query(Project).filter(Project.name == data.name).first():
        raise HTTPException(status_code=400, detail="Project name already exists")

    with trace_operation(
        "project_creation",
        plane="control",
        metadata={"result": "started"},
    ) as span:
        project = Project(
            name=data.name,
            password_hash=hash_password(data.password),
        )
        try:
            db.add(project)
            db.commit()
            db.refresh(project)
            span.tags({"project_id": project.id, "result": "created"})
            token = create_access_token(
                {"project_id": project.id, "project_name": project.name}
            )
            return {
                "access_token": token,
                "token_type": "bearer",
                "project": _project_to_dict(project),
                "message": "Project created. Configure your credentials in Settings to start monitoring.",
            }
        except Exception:
            db.rollback()
            raise HTTPException(status_code=500, detail="Failed to create project")


@router.post("/login")
def login_project(credentials: ProjectLogin, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.name == credentials.name).first()
    if not project or not verify_password(credentials.password, project.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect name or password")
    if not project.is_active:
        raise HTTPException(status_code=403, detail="Project is inactive")
    token = create_access_token(
        {"project_id": project.id, "project_name": project.name}
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "project": _project_to_dict(project),
    }


@router.get("/me")
def get_me(project: Project = Depends(get_current_project)):
    return _project_to_dict(project)


@router.put("/settings")
def update_settings(
    settings: ProjectSettings,
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    if project.is_test:
        raise HTTPException(
            status_code=403, detail="Test project settings are read-only"
        )

    fields = [
        "datadog_api_key",
        "datadog_app_key",
        "datadog_site",
        "datadog_query",
        "datadog_service",
        "user_email",
        "discord_webhook_escalate",
        "discord_webhook_dev",
    ]
    for field in fields:
        value = getattr(settings, field)
        if value is not None:
            setattr(project, field, value)

    if settings.password:
        project.password_hash = hash_password(settings.password)

    try:
        db.commit()
        db.refresh(project)
        return {
            "message": "Settings updated successfully",
            "project": _project_to_dict(project),
        }
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update settings")


def _verification_secret(candidate: Optional[str], stored: Optional[str]) -> str:
    if candidate and candidate not in {"••••••", HIDDEN}:
        return candidate
    return stored or ""


@router.post("/settings/datadog/verify")
def verify_datadog_settings(
    settings: DatadogVerificationRequest,
    project: Project = Depends(get_current_project),
):
    """Verify the proposed Datadog configuration without saving it."""
    try:
        config = DatadogLogSourceConfig(
            api_key=_verification_secret(
                settings.datadog_api_key, project.datadog_api_key
            ),
            app_key=_verification_secret(
                settings.datadog_app_key, project.datadog_app_key
            ),
            site=settings.datadog_site or "",
            query=settings.datadog_query or "",
            service_filter=settings.datadog_service or "",
        )
        end = datetime.now(timezone.utc)
        with DatadogLogConnector(config, page_size=1) as connector:
            connector.verify_access(end - timedelta(minutes=5), end)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LogSourceAuthenticationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LogSourceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "valid": True,
        "message": "Datadog credentials and logs-read access verified",
    }


@router.delete("/project")
def remove_project(
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    """Permanently delete the authenticated project and all of its data."""
    project_id = project.id
    try:
        deleted = delete_project(db, project_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete project") from exc
    return {"status": "deleted", "project_id": project_id, "deleted": deleted}


@router.get("/settings/status")
def settings_status(project: Project = Depends(get_current_project)):
    datadog_configured = all(
        [
            project.datadog_api_key,
            project.datadog_app_key,
            project.datadog_site,
            project.datadog_query,
            project.datadog_service,
        ]
    )
    return {
        "datadog": {"configured": datadog_configured},
        "llm": {"configured": bool(project.active_artifact_id)},
        "observability": {
            "configured": _enabled("DD_LLMOBS_ENABLED"),
            "platform": "Datadog",
            "llm_observability": _enabled("DD_LLMOBS_ENABLED"),
            "logs": datadog_configured,
            "tracing": _enabled("DD_TRACE_ENABLED"),
        },
        "notifications": {
            "configured": any(
                [
                    project.user_email,
                    project.discord_webhook_escalate,
                    project.discord_webhook_dev,
                ]
            )
        },
        "setup_complete": all(
            [
                datadog_configured,
                project.active_artifact_id,
            ]
        ),
    }
