from sqlalchemy import inspect, text

VERSION = "0008_training_job_record_selection"


def upgrade(connection) -> None:
    """Persist the dataset records selected for each training job."""
    inspector = inspect(connection)
    if not inspector.has_table("training_jobs"):
        return

    columns = {column["name"] for column in inspector.get_columns("training_jobs")}
    if "selected_record_indices" not in columns:
        connection.execute(
            text("ALTER TABLE training_jobs ADD COLUMN selected_record_indices JSON")
        )
