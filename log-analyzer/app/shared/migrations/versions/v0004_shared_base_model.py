from sqlalchemy import inspect, text

from app.shared.model_config import DEFAULT_BASE_MODEL

VERSION = "0004_shared_base_model"


def upgrade(connection) -> None:
    """Replace the legacy base-model alias with the shared Qwen identifier."""
    inspector = inspect(connection)
    if not inspector.has_table("projects"):
        return

    columns = {column["name"] for column in inspector.get_columns("projects")}
    if "base_model" not in columns:
        return

    connection.execute(
        text(
            "UPDATE projects SET base_model = :base_model "
            "WHERE base_model IS NULL OR base_model = '' OR base_model = 'qwen-v2'"
        ),
        {"base_model": DEFAULT_BASE_MODEL},
    )
