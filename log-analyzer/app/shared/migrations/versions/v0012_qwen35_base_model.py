from sqlalchemy import inspect, text

from app.shared.model_config import DEFAULT_BASE_MODEL

VERSION = "0012_qwen35_base_model"


def upgrade(connection) -> None:
    """Switch projects to Qwen3.5 and detach incompatible active adapters."""
    inspector = inspect(connection)
    if not inspector.has_table("projects"):
        return

    columns = {column["name"] for column in inspector.get_columns("projects")}
    if "base_model" not in columns:
        return

    if "active_artifact_id" in columns and inspector.has_table("model_artifacts"):
        connection.execute(
            text(
                "UPDATE projects SET active_artifact_id = NULL "
                "WHERE active_artifact_id IS NOT NULL AND NOT EXISTS ("
                "SELECT 1 FROM model_artifacts "
                "WHERE model_artifacts.id = projects.active_artifact_id "
                "AND model_artifacts.project_id = projects.id "
                "AND model_artifacts.base_model = :base_model"
                ")"
            ),
            {"base_model": DEFAULT_BASE_MODEL},
        )

    connection.execute(
        text("UPDATE projects SET base_model = :base_model"),
        {"base_model": DEFAULT_BASE_MODEL},
    )

    if connection.dialect.name == "postgresql":
        connection.execute(
            text(
                "ALTER TABLE projects ALTER COLUMN base_model "
                f"SET DEFAULT '{DEFAULT_BASE_MODEL}'"
            )
        )
