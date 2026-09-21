from sqlalchemy import inspect, text

VERSION = "0005_remove_remote_model_config"


def upgrade(connection) -> None:
    """Remove the retired per-project remote-provider credential."""
    inspector = inspect(connection)
    if not inspector.has_table("projects"):
        return

    columns = {column["name"] for column in inspector.get_columns("projects")}
    legacy_column = "groq_api_key"
    if legacy_column in columns:
        connection.execute(text(f"ALTER TABLE projects DROP COLUMN {legacy_column}"))
