from sqlalchemy import inspect, text

from app.control.models import Project
from app.training.models import Dataset, ModelArtifact, TrainingJob

VERSION = "0002_model_lifecycle"


def upgrade(connection) -> None:
    """Add the model-lifecycle schema without changing incident data."""
    inspector = inspect(connection)

    if not inspector.has_table(Project.__tablename__):
        Project.__table__.create(connection, checkfirst=True)
    else:
        project_columns = {
            column["name"] for column in inspector.get_columns(Project.__tablename__)
        }
        if "base_model" not in project_columns:
            connection.execute(
                text(
                    "ALTER TABLE projects ADD COLUMN base_model "
                    "VARCHAR NOT NULL DEFAULT 'qwen-v2'"
                )
            )
        if "active_artifact_id" not in project_columns:
            connection.execute(
                text("ALTER TABLE projects ADD COLUMN active_artifact_id VARCHAR")
            )

    Dataset.__table__.create(connection, checkfirst=True)
    ModelArtifact.__table__.create(connection, checkfirst=True)
    TrainingJob.__table__.create(connection, checkfirst=True)
