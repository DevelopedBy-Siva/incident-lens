from sqlalchemy import inspect, text

VERSION = "0006_datadog_log_source"

DATADOG_COLUMNS = {
    "datadog_api_key": "VARCHAR",
    "datadog_app_key": "VARCHAR",
    "datadog_site": "VARCHAR",
    "datadog_query": "VARCHAR",
    "datadog_service": "VARCHAR",
}

RETIRED_COLUMNS = (
    "loki_url",
    "loki_username",
    "loki_api_key",
    "loki_service",
)


def upgrade(connection) -> None:
    """Replace the retired log-provider configuration with Datadog settings."""
    inspector = inspect(connection)
    if not inspector.has_table("projects"):
        return

    columns = {column["name"] for column in inspector.get_columns("projects")}
    for name, definition in DATADOG_COLUMNS.items():
        if name not in columns:
            connection.execute(
                text(f"ALTER TABLE projects ADD COLUMN {name} {definition}")
            )

    columns = {column["name"] for column in inspect(connection).get_columns("projects")}
    for name in RETIRED_COLUMNS:
        if name in columns:
            connection.execute(text(f"ALTER TABLE projects DROP COLUMN {name}"))
