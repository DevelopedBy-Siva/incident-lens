from sqlalchemy import inspect, text

VERSION = "0003_dataset_record_count"


def upgrade(connection) -> None:
    """Add the stored record count to existing dataset metadata."""
    inspector = inspect(connection)
    if not inspector.has_table("datasets"):
        return

    columns = {column["name"] for column in inspector.get_columns("datasets")}
    if "record_count" not in columns:
        connection.execute(
            text(
                "ALTER TABLE datasets ADD COLUMN record_count "
                "INTEGER NOT NULL DEFAULT 0"
            )
        )
