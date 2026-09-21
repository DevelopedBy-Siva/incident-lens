import os
import re
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
from app.data.ingestion.datadog_connector import SUPPORTED_DATADOG_SITES
from app.shared.database import get_db

router = APIRouter()
security = HTTPBearer()

HIDDEN = "HIDDEN: TEST CREDENTIAL"


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
    datadog_environment: Optional[str] = None
    datadog_service: Optional[str] = None
    # Observability
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: Optional[str] = None
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


def _mask(value: Optional[str], is_test: bool = False) -> Optional[str]:
    """Return masked value for API keys, HIDDEN for test projects."""
    if not value:
        return None
    if is_test:
        return HIDDEN
    return "••••••"


def _project_to_dict(project: Project) -> dict:
    t = project.is_test
    datadog_api_key = project.datadog_api_key or os.getenv("DATADOG_API_KEY")
    datadog_app_key = project.datadog_app_key or os.getenv("DATADOG_APP_KEY")
    datadog_configured = bool(datadog_api_key and datadog_app_key)
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
        "datadog_site": project.datadog_site
        or os.getenv("DATADOG_SITE", "datadoghq.com"),
        "datadog_query": project.datadog_query
        or os.getenv("DATADOG_QUERY", "status:(error OR warn OR critical)"),
        "datadog_environment": project.datadog_environment
        or os.getenv("DATADOG_ENVIRONMENT", "prod"),
        "datadog_service": project.datadog_service or os.getenv("DATADOG_SERVICE"),
        # Observability
        "langfuse_public_key": _mask(project.langfuse_public_key, t),
        "langfuse_secret_key": _mask(project.langfuse_secret_key, t),
        "langfuse_host": HIDDEN if t else project.langfuse_host,
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
            "observability": all(
                [project.langfuse_public_key, project.langfuse_secret_key]
            ),
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

    project = Project(
        name=data.name,
        password_hash=hash_password(data.password),
    )
    try:
        db.add(project)
        db.commit()
        db.refresh(project)
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
        "datadog_environment",
        "datadog_service",
        "langfuse_public_key",
        "langfuse_secret_key",
        "langfuse_host",
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


@router.get("/settings/status")
def settings_status(project: Project = Depends(get_current_project)):
    datadog_configured = bool(
        (project.datadog_api_key or os.getenv("DATADOG_API_KEY"))
        and (project.datadog_app_key or os.getenv("DATADOG_APP_KEY"))
    )
    return {
        "datadog": {"configured": datadog_configured},
        "llm": {"configured": bool(project.active_artifact_id)},
        "observability": {
            "configured": all(
                [project.langfuse_public_key, project.langfuse_secret_key]
            )
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
