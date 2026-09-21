import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from app.data.ingestion.datadog_connector import (
    DatadogLogConnector,
    DatadogLogSourceConfig,
)
from app.data.ingestion.log_source import LogEnvelope, LogSourceError

load_dotenv()

POLL_INTERVAL = max(1, int(os.getenv("POLL_INTERVAL", "30")))
LOOKBACK_SECONDS = max(1, int(os.getenv("DATADOG_LOOKBACK_SECONDS", "30")))


def _load_active_projects():
    from app.control.models import Project
    from app.shared.database import SessionLocal

    db = SessionLocal()
    try:
        global_api_key = os.getenv("DATADOG_API_KEY")
        global_app_key = os.getenv("DATADOG_APP_KEY")
        active_projects = db.query(Project).filter(Project.is_active.is_(True)).all()
        projects = [
            project
            for project in active_projects
            if (project.datadog_api_key or global_api_key)
            and (project.datadog_app_key or global_app_key)
        ]
        for project in projects:
            db.expunge(project)
        return projects
    finally:
        db.close()


def fetch_logs_for_project(
    project,
    start: datetime,
    end: datetime,
    *,
    connector_factory=DatadogLogConnector,
) -> list[LogEnvelope]:
    config = DatadogLogSourceConfig.from_project(project)
    with connector_factory(config) as connector:
        return connector.fetch(start, end)


def poll_project(
    project,
    last_queried: datetime | None,
    end: datetime,
    *,
    connector_factory=DatadogLogConnector,
) -> tuple[list[LogEnvelope], datetime]:
    """Fetch exactly one contiguous interval and return its new cursor time."""
    start = last_queried or end - timedelta(seconds=LOOKBACK_SECONDS)
    envelopes = fetch_logs_for_project(
        project,
        start,
        end,
        connector_factory=connector_factory,
    )
    return envelopes, end


def process_envelopes(project, envelopes: list[LogEnvelope]) -> dict[str, int]:
    """Adapt neutral envelopes to the unchanged Data Plane batch contract."""
    from app.data.pipeline import process_log_batch

    totals = {"incidents_created": 0, "incidents_updated": 0, "failed": 0}
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for envelope in envelopes:
        groups[(envelope.source, envelope.environment)].append(envelope.message)

    for (source, environment), messages in groups.items():
        result = process_log_batch(
            {
                "project_id": str(project.id),
                "source": source,
                "environment": environment,
                "logs": messages,
                "_project": project,
            }
        )
        if result:
            for key in totals:
                totals[key] += result.get(key, 0)
    return totals


def run():
    print(f"[LOG-SOURCE] Starting Datadog polling — interval={POLL_INTERVAL}s")
    print(f"[LOG-SOURCE] Initial lookback={LOOKBACK_SECONDS}s")
    last_queried: dict[str, datetime] = {}

    while True:
        try:
            projects = _load_active_projects()
            if not projects:
                print("[LOG-SOURCE] No projects with Datadog credentials — waiting")

            for project in projects:
                end = datetime.now(timezone.utc)
                previous_cursor = last_queried.get(project.id)
                start = previous_cursor or end - timedelta(seconds=LOOKBACK_SECONDS)
                try:
                    envelopes, next_cursor = poll_project(
                        project,
                        previous_cursor,
                        end,
                    )
                    if envelopes:
                        result = process_envelopes(project, envelopes)
                        print(
                            f"[LOG-SOURCE] [{project.name}] logs={len(envelopes)} "
                            f"created={result['incidents_created']} "
                            f"updated={result['incidents_updated']} "
                            f"failed={result['failed']}"
                        )
                    else:
                        print(
                            f"[LOG-SOURCE] [{project.name}] No new logs since "
                            f"{start.strftime('%H:%M:%S')}"
                        )
                    last_queried[project.id] = next_cursor
                except (LogSourceError, ValueError) as exc:
                    print(f"[LOG-SOURCE] [{project.name}] Datadog poll failed: {exc}")
                except Exception as exc:
                    print(f"[LOG-SOURCE] [{project.name}] ingestion failed: {exc}")

        except KeyboardInterrupt:
            print("[LOG-SOURCE] Shutting down")
            break
        except Exception as exc:
            print(
                f"[LOG-SOURCE] Unexpected error: {exc} — retrying in {POLL_INTERVAL}s"
            )

        time.sleep(POLL_INTERVAL)
