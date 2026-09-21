from datetime import datetime
import uuid

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String, Text

from app.shared.database import Base


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
