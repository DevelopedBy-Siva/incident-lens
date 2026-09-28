import os
import time
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from app.data.ingestion.datadog_connector import (
    DatadogLogConnector,
    DatadogLogSourceConfig,
)
from app.data.ingestion.log_source import LogEnvelope, LogSourceError
from app.shared.observability import trace_operation

load_dotenv()

POLL_INTERVAL = max(1, int(os.getenv("POLL_INTERVAL", "30")))
LOOKBACK_SECONDS = max(1, int(os.getenv("DATADOG_LOOKBACK_SECONDS", "30")))
INITIAL_LOOKBACK_SECONDS = max(
    LOOKBACK_SECONDS,
    int(os.getenv("DATADOG_INITIAL_LOOKBACK_SECONDS", "600")),
)


logger = logging.getLogger(__name__)


def _load_active_projects():
    from app.control.models import Project
    from app.shared.database import SessionLocal

    db = SessionLocal()
    try:
        active_projects = db.query(Project).filter(Project.is_active.is_(True)).all()
        projects = [
            project
            for project in active_projects
            if project.datadog_api_key
            and project.datadog_app_key
            and project.datadog_site
            and project.datadog_query
            and project.datadog_service
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
    with trace_operation(
        "log_ingestion",
        plane="data",
        metadata={
            "project_id": str(project.id),
            "source": "datadog",
            "environment": "prod",
        },
    ) as span:
        config = DatadogLogSourceConfig.from_project(project)
        with connector_factory(config) as connector:
            envelopes = connector.fetch(start, end)
        span.metrics({"log_count": len(envelopes)})
        span.tag("result", "completed")
        return envelopes


def poll_project(
    project,
    last_queried: datetime | None,
    end: datetime,
    *,
    connector_factory=DatadogLogConnector,
) -> tuple[list[LogEnvelope], datetime]:
    """Fetch exactly one contiguous interval and return its new cursor time."""
    start = last_queried or end - timedelta(seconds=INITIAL_LOOKBACK_SECONDS)
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
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with trace_operation(
        "normalization",
        plane="data",
        metadata={
            "project_id": str(project.id),
            "source": "datadog",
            "log_count": len(envelopes),
        },
    ):
        for envelope in envelopes:
            groups[(envelope.source, envelope.environment)].append(
                {
                    "provider_id": envelope.provider_id,
                    "message": envelope.message,
                    "timestamp": envelope.timestamp.isoformat(),
                    "attributes": envelope.attributes,
                }
            )

    for (source, environment), events in groups.items():
        result = process_log_batch(
            {
                "project_id": str(project.id),
                "source": source,
                "environment": environment,
                "events": events,
                "_project": project,
            }
        )
        if result:
            for key in totals:
                totals[key] += result.get(key, 0)
    return totals


def recover_pending_incidents(project) -> dict[str, int]:
    """Finish any incident left without an analysis by an interrupted worker."""
    from app.data.pipeline import process_pending_incidents

    return process_pending_incidents(project)


def run():
    logger.info(
        "[LOG-SOURCE] Starting Datadog polling interval=%ss initial_lookback=%ss",
        POLL_INTERVAL,
        INITIAL_LOOKBACK_SECONDS,
    )
    last_queried: dict[str, datetime] = {}

    while True:
        try:
            projects = _load_active_projects()
            if not projects:
                logger.warning(
                    "[LOG-SOURCE] No active projects have complete Datadog credentials; waiting"
                )

            for project in projects:
                end = datetime.now(timezone.utc)
                previous_cursor = last_queried.get(project.id)
                start = previous_cursor or end - timedelta(seconds=INITIAL_LOOKBACK_SECONDS)
                try:
                    envelopes, next_cursor = poll_project(
                        project,
                        previous_cursor,
                        end,
                    )
                    if envelopes:
                        result = process_envelopes(project, envelopes)
                        logger.info(
                            "[LOG-SOURCE] project=%s window=%s..%s logs=%s created=%s updated=%s failed=%s",
                            project.name,
                            start.isoformat(),
                            end.isoformat(),
                            len(envelopes),
                            result["incidents_created"],
                            result["incidents_updated"],
                            result["failed"],
                        )
                        if result["failed"]:
                            raise LogSourceError(
                                f"{result['failed']} events failed processing; cursor will retry this window"
                            )
                    else:
                        logger.info(
                            "[LOG-SOURCE] project=%s window=%s..%s logs=0",
                            project.name,
                            start.isoformat(),
                            end.isoformat(),
                        )
                    recovery = recover_pending_incidents(project)
                    if recovery["recovered"] or recovery["failed"]:
                        logger.info(
                            "[LOG-SOURCE] project=%s recovery completed=%s failed=%s",
                            project.name,
                            recovery["recovered"],
                            recovery["failed"],
                        )
                    last_queried[project.id] = next_cursor
                except (LogSourceError, ValueError) as exc:
                    logger.warning(
                        "[LOG-SOURCE] project=%s Datadog poll failed: %s",
                        project.name,
                        exc,
                    )
                except Exception as exc:
                    logger.exception(
                        "[LOG-SOURCE] project=%s ingestion failed: %s",
                        project.name,
                        exc,
                    )

        except KeyboardInterrupt:
            logger.info("[LOG-SOURCE] Shutting down")
            break
        except Exception as exc:
            logger.exception(
                "[LOG-SOURCE] Unexpected error: %s; retrying in %ss",
                exc,
                POLL_INTERVAL,
            )

        time.sleep(POLL_INTERVAL)
