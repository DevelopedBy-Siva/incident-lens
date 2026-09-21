from datetime import datetime
import uuid

from sqlalchemy import Column, DateTime, Integer, JSON, String, Text

from app.shared.database import Base


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
