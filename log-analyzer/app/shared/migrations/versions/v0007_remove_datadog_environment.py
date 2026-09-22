from sqlalchemy import inspect, text

VERSION = "0007_remove_datadog_environment"


def upgrade(connection) -> None:
    """Environment is intentionally fixed to prod for every project."""
    inspector = inspect(connection)
    if not inspector.has_table("projects"):
        return

    columns = {column["name"] for column in inspector.get_columns("projects")}
    if "datadog_environment" in columns:
        connection.execute(text("ALTER TABLE projects DROP COLUMN datadog_environment"))
