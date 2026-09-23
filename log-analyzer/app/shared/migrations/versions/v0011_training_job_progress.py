from sqlalchemy import inspect, text

VERSION = "0011_training_job_progress"


def upgrade(connection) -> None:
    """Persist trainer step progress for polling clients."""
    inspector = inspect(connection)
    if not inspector.has_table("training_jobs"):
        return

    columns = {column["name"] for column in inspector.get_columns("training_jobs")}
    additions = {
        "progress_current_step": "INTEGER",
        "progress_total_steps": "INTEGER",
        "progress_updated_at": "TIMESTAMP",
    }
    for column_name, column_type in additions.items():
        if column_name not in columns:
            connection.execute(
                text(
                    f"ALTER TABLE training_jobs ADD COLUMN "
                    f"{column_name} {column_type}"
                )
            )
