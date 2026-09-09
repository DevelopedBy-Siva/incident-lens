from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    DateTime,
    JSON,
    Float,
    Boolean,
    Text,
    inspect,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid, os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://log_user:password@localhost:5432/log_analyzer"
)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Project(Base):
    __tablename__ = "projects"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
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


class Incident(Base):
    __tablename__ = "incidents"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, nullable=False, index=True)
    source = Column(String, nullable=False, index=True)
    environment = Column(String, default="prod")
    signature = Column(String, nullable=False, index=True)
    first_seen = Column(DateTime, default=datetime.utcnow, index=True)
    last_seen = Column(DateTime, default=datetime.utcnow, index=True)
    count = Column(Integer, default=1)
    sample_lines = Column(JSON)
    status = Column(String, default="open", index=True)
    root_cause_incident_id = Column(String, nullable=True, index=True)
    cause_explanation = Column(Text, nullable=True)
    last_actioned_at = Column(DateTime, nullable=True)
    auto_tags = Column(JSON, nullable=True)


class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    incident_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    severity = Column(String)
    disposition = Column(String)
    confidence = Column(Float)
    summary = Column(String)
    next_steps = Column(JSON)
    matched_runbook_id = Column(String)
    runbook_match_score = Column(Float)
    ticket_title = Column(String)
    ticket_body = Column(String)
    analysis_source = Column(String)


class ActionLog(Base):
    __tablename__ = "action_logs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    incident_id = Column(String, nullable=False, index=True)
    project_id = Column(String, nullable=False, index=True)
    actioned_at = Column(DateTime, default=datetime.utcnow, index=True)
    requested_actions = Column(JSON, nullable=True)
    allowed_actions = Column(JSON, nullable=True)
    blocked_actions = Column(JSON, nullable=True)
    actions_taken = Column(JSON, nullable=True)
    disposition = Column(String, nullable=True)
    severity = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    policy_reason = Column(String, nullable=True)
    policy_tags = Column(JSON, nullable=True)
    outcome = Column(String, default="pending", index=True)
    resolved_at = Column(DateTime, nullable=True)


class InvestigationRun(Base):
    __tablename__ = "investigation_runs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    incident_id = Column(String, nullable=False, index=True)
    project_id = Column(String, nullable=False, index=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    evidence_samples = Column(Integer, default=0)
    evidence_related_count = Column(Integer, default=0)
    evidence_runbook = Column(String, nullable=True)
    evidence_snapshot = Column(Text, nullable=True)
    tool_calls = Column(JSON, nullable=True)
    iterations = Column(Integer, default=0)
    fallback_used = Column(Boolean, default=False)
    analysis_source = Column(String, nullable=True)
    policy_allowed = Column(Boolean, nullable=True)
    policy_reason = Column(String, nullable=True)
    policy_tags = Column(JSON, nullable=True)
    effective_disposition = Column(String, nullable=True)
    actions_taken = Column(JSON, nullable=True)
    verifier_outcome = Column(String, nullable=True)
    verifier_checked_at = Column(DateTime, nullable=True)
    final_severity = Column(String, nullable=True)
    final_disposition = Column(String, nullable=True)
    final_confidence = Column(Float, nullable=True)
    final_summary = Column(Text, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)
    _ensure_action_log_columns()
    print("[DB] Tables initialised")


def _ensure_action_log_columns():
    """Best-effort additive migration for deployments using create_all."""
    try:
        inspector = inspect(engine)
        if not inspector.has_table("action_logs"):
            return

        existing = {column["name"] for column in inspector.get_columns("action_logs")}
        missing = {
            "requested_actions": "JSON",
            "allowed_actions": "JSON",
            "blocked_actions": "JSON",
            "policy_reason": "VARCHAR",
        }

        with engine.begin() as conn:
            for column_name, column_type in missing.items():
                if column_name not in existing:
                    conn.execute(
                        text(
                            f"ALTER TABLE action_logs ADD COLUMN {column_name} {column_type}"
                        )
                    )
    except Exception as e:
        print(f"[DB] ActionLog column check skipped: {e}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
