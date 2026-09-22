import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, String

from app.shared.database import Base
from app.shared.model_config import DEFAULT_BASE_MODEL, configured_base_model


class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    base_model = Column(
        String,
        nullable=False,
        default=configured_base_model,
        server_default=DEFAULT_BASE_MODEL,
    )
    active_artifact_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    is_test = Column(Boolean, default=False)
    datadog_api_key = Column(String, nullable=True)
    datadog_app_key = Column(String, nullable=True)
    datadog_site = Column(String, nullable=True)
    datadog_query = Column(String, nullable=True)
    datadog_environment = Column(String, nullable=True)
    datadog_service = Column(String, nullable=True)
    user_email = Column(String, nullable=True)
    discord_webhook_escalate = Column(String, nullable=True)
    discord_webhook_dev = Column(String, nullable=True)
