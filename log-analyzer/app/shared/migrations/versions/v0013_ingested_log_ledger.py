"""Persist provider log identifiers so polling/restarts are idempotent."""

from sqlalchemy import text


VERSION = "0013_ingested_log_ledger"


def upgrade(connection) -> None:
    connection.execute(
        text(
            "CREATE TABLE IF NOT EXISTS ingested_logs ("
            "id VARCHAR PRIMARY KEY, project_id VARCHAR NOT NULL, "
            "provider_id VARCHAR NOT NULL, observed_at TIMESTAMP NOT NULL, "
            "created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, "
            "CONSTRAINT uq_ingested_log_provider UNIQUE (project_id, provider_id)"
            ")"
        )
    )
    connection.execute(text("CREATE INDEX IF NOT EXISTS ix_ingested_logs_project_id ON ingested_logs (project_id)"))
    connection.execute(text("CREATE INDEX IF NOT EXISTS ix_ingested_logs_observed_at ON ingested_logs (observed_at)"))
