from sqlalchemy import text

from app.shared.migrations.versions import (
    v0002_model_lifecycle,
    v0003_dataset_record_count,
    v0004_shared_base_model,
    v0005_remove_remote_model_config,
    v0006_datadog_log_source,
    v0007_remove_datadog_environment,
    v0008_training_job_record_selection,
    v0009_dataset_record_selection,
    v0010_trained_dataset_status,
    v0011_training_job_progress,
)

MIGRATIONS = (
    v0002_model_lifecycle,
    v0003_dataset_record_count,
    v0004_shared_base_model,
    v0005_remove_remote_model_config,
    v0006_datadog_log_source,
    v0007_remove_datadog_environment,
    v0008_training_job_record_selection,
    v0009_dataset_record_selection,
    v0010_trained_dataset_status,
    v0011_training_job_progress,
)


def run_migrations(engine) -> None:
    """Apply pending IncidentLens migrations in version order."""
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version VARCHAR PRIMARY KEY, "
                "applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"
                ")"
            )
        )
        applied = {
            row[0]
            for row in connection.execute(text("SELECT version FROM schema_migrations"))
        }

        for migration in MIGRATIONS:
            if migration.VERSION in applied:
                continue
            migration.upgrade(connection)
            connection.execute(
                text("INSERT INTO schema_migrations (version) VALUES (:version)"),
                {"version": migration.VERSION},
            )
