from sqlalchemy import inspect, text

VERSION = "0009_dataset_record_selection"


def upgrade(connection) -> None:
    """Persist the approved record selection on each dataset."""
    inspector = inspect(connection)
    if not inspector.has_table("datasets"):
        return

    columns = {column["name"] for column in inspector.get_columns("datasets")}
    if "selected_record_indices" not in columns:
        connection.execute(
            text("ALTER TABLE datasets ADD COLUMN selected_record_indices JSON")
        )
