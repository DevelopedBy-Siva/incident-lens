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
    loki_url = Column(String, nullable=True)
    loki_username = Column(String, nullable=True)
    loki_api_key = Column(String, nullable=True)
    loki_service = Column(String, nullable=True)
    groq_api_key = Column(String, nullable=True)
    langfuse_public_key = Column(String, nullable=True)
    langfuse_secret_key = Column(String, nullable=True)
    langfuse_host = Column(String, nullable=True, default="https://cloud.langfuse.com")
    user_email = Column(String, nullable=True)
    discord_webhook_escalate = Column(String, nullable=True)
    discord_webhook_dev = Column(String, nullable=True)
