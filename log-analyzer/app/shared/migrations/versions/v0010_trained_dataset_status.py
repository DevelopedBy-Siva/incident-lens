from sqlalchemy import inspect, text

VERSION = "0010_trained_dataset_status"


def upgrade(connection) -> None:
    """Add TRAINED and backfill datasets used by successful training jobs."""
    inspector = inspect(connection)
    if not inspector.has_table("datasets"):
        return

    if connection.dialect.name == "postgresql":
        connection.execute(
            text("ALTER TABLE datasets DROP CONSTRAINT IF EXISTS dataset_status")
        )
        connection.execute(
            text(
                "ALTER TABLE datasets ADD CONSTRAINT dataset_status "
                "CHECK (status IN ('VALIDATING', 'READY', 'TRAINED', 'FAILED'))"
            )
        )
    elif connection.dialect.name == "sqlite":
        table_sql = connection.execute(
            text(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'table' AND name = 'datasets'"
            )
        ).scalar_one_or_none()
        if table_sql and "'TRAINED'" not in table_sql:
            connection.execute(
                text(
                    "CREATE TABLE datasets_new ("
                    "id VARCHAR NOT NULL PRIMARY KEY, "
                    "project_id VARCHAR NOT NULL REFERENCES projects (id), "
                    "dataset_version VARCHAR NOT NULL, "
                    "storage_key VARCHAR NOT NULL, "
                    "record_count INTEGER NOT NULL DEFAULT 0, "
                    "selected_record_indices JSON, "
                    "status VARCHAR(10) NOT NULL DEFAULT 'VALIDATING', "
                    "created_at DATETIME NOT NULL, "
                    "CONSTRAINT uq_dataset_project_version "
                    "UNIQUE (project_id, dataset_version), "
                    "CONSTRAINT dataset_status CHECK "
                    "(status IN ('VALIDATING', 'READY', 'TRAINED', 'FAILED'))"
                    ")"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO datasets_new "
                    "(id, project_id, dataset_version, storage_key, record_count, "
                    "selected_record_indices, status, created_at) "
                    "SELECT id, project_id, dataset_version, storage_key, record_count, "
                    "selected_record_indices, status, created_at FROM datasets"
                )
            )
            connection.execute(text("DROP TABLE datasets"))
            connection.execute(text("ALTER TABLE datasets_new RENAME TO datasets"))
            connection.execute(
                text("CREATE INDEX ix_datasets_project_id ON datasets (project_id)")
            )

    if inspector.has_table("training_jobs"):
        connection.execute(
            text(
                "UPDATE datasets SET status = 'TRAINED' "
                "WHERE status = 'READY' AND id IN ("
                "SELECT dataset_id FROM training_jobs WHERE status = 'PASSED'"
                ")"
            )
        )
