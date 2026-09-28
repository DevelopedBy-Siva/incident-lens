"""Persist the grounded root-cause hypothesis produced by analysis."""

from sqlalchemy import inspect, text


VERSION = "0014_analysis_root_cause"


def upgrade(connection) -> None:
    inspector = inspect(connection)
    if not inspector.has_table("analyses"):
        return
    columns = {column["name"] for column in inspector.get_columns("analyses")}
    if "suspected_root_cause" not in columns:
        connection.execute(
            text("ALTER TABLE analyses ADD COLUMN suspected_root_cause TEXT")
        )
