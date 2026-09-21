import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import (
    Column,
    DateTime,
    Enum as SQLAlchemyEnum,
    Float,
    ForeignKey,
    String,
    UniqueConstraint,
)

from app.shared.database import Base


class DatasetStatus(str, Enum):
    VALIDATING = "VALIDATING"
    READY = "READY"
    FAILED = "FAILED"


class TrainingJobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    EVALUATING = "EVALUATING"
    PASSED = "PASSED"
    FAILED = "FAILED"


class ModelArtifactStatus(str, Enum):
    READY = "READY"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (
        UniqueConstraint(
            "project_id", "dataset_version", name="uq_dataset_project_version"
        ),
    )

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"), nullable=False, index=True)
    dataset_version = Column(String, nullable=False)
    storage_key = Column(String, nullable=False)
    status = Column(
        SQLAlchemyEnum(
            DatasetStatus,
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            name="dataset_status",
        ),
        nullable=False,
        default=DatasetStatus.VALIDATING,
        server_default=DatasetStatus.VALIDATING.value,
    )
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "project_id", "artifact_version", name="uq_artifact_project_version"
        ),
    )

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"), nullable=False, index=True)
    artifact_version = Column(String, nullable=False)
    base_model = Column(String, nullable=False)
    adapter_path = Column(String, nullable=True)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False, index=True)
    evaluation_score = Column(Float, nullable=True)
    status = Column(
        SQLAlchemyEnum(
            ModelArtifactStatus,
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            name="model_artifact_status",
        ),
        nullable=False,
        default=ModelArtifactStatus.READY,
        server_default=ModelArtifactStatus.READY.value,
    )
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"), nullable=False, index=True)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False, index=True)
    status = Column(
        SQLAlchemyEnum(
            TrainingJobStatus,
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            name="training_job_status",
        ),
        nullable=False,
        default=TrainingJobStatus.QUEUED,
        server_default=TrainingJobStatus.QUEUED.value,
    )
    artifact_id = Column(
        String, ForeignKey("model_artifacts.id"), nullable=True, index=True
    )
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
