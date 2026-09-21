import base64
import httpx
import os
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))
LOG_ENVIRONMENT = os.getenv("LOG_ENVIRONMENT", "prod")
MAX_LINES_PER_POLL = 500


def _auth_header(username: str, api_key: str) -> str:
    token = base64.b64encode(f"{username}:{api_key}".encode()).decode()
    return f"Basic {token}"


def _to_loki_ns(dt: datetime) -> str:
    return str(int(dt.timestamp() * 1_000_000_000))


def fetch_logs_for_project(project, start: datetime, end: datetime) -> list[str]:
    loki_url = project.loki_url
    loki_username = project.loki_username
    loki_api_key = project.loki_api_key
    loki_service = project.loki_service or "log-server"

    if not all([loki_url, loki_username, loki_api_key]):
        return []

    query = f'{{service="{loki_service}"}} |~ "ERROR|WARN|CRITICAL"'
    params = {
        "query": query,
        "start": _to_loki_ns(start),
        "end": _to_loki_ns(end),
        "limit": MAX_LINES_PER_POLL,
        "direction": "forward",
    }

    try:
        response = httpx.get(
            f"{loki_url}/loki/api/v1/query_range",
            headers={"Authorization": _auth_header(loki_username, loki_api_key)},
            params=params,
            timeout=15,
        )
        if response.status_code != 200:
            print(
                f"[LOKI-WATCHER] [{project.name}] Query failed: {response.status_code} — {response.text[:200]}"
            )
            return []

        streams = response.json().get("data", {}).get("result", [])
        lines = []
        for stream in streams:
            for _ts, line in stream.get("values", []):
                lines.append(line)
        return lines

    except Exception as e:
        print(f"[LOKI-WATCHER] [{project.name}] fetch_logs exception: {e}")
        return []


def _load_active_projects():
    from app.control.models import Project
    from app.shared.database import SessionLocal

    db = SessionLocal()
    try:
        projects = (
            db.query(Project)
            .filter(
                Project.is_active == True,
                Project.loki_url != None,
                Project.loki_username != None,
                Project.loki_api_key != None,
            )
            .all()
        )
        for p in projects:
            db.expunge(p)
        return projects
    finally:
        db.close()


def run():
    print(f"[LOKI-WATCHER] Starting — poll_interval={POLL_INTERVAL}s")
    print("[LOKI-WATCHER] Credentials loaded per-project from DB")

    from app.data.pipeline import process_log_batch

    last_queried: dict[str, datetime] = {}

    while True:
        try:
            projects = _load_active_projects()

            if not projects:
                print(
                    "[LOKI-WATCHER] No projects with Loki credentials configured — waiting"
                )
            else:
                print(f"[LOKI-WATCHER] Polling {len(projects)} project(s)")

            for project in projects:
                project_last = last_queried.get(
                    project.id,
                    datetime.now(timezone.utc) - timedelta(seconds=POLL_INTERVAL),
                )
                poll_start = datetime.now(timezone.utc)
                lines = fetch_logs_for_project(project, project_last, poll_start)

                if lines:
                    print(
                        f"[LOKI-WATCHER] [{project.name}] Fetched {len(lines)} log line(s)"
                    )
                    payload = {
                        "project_id": str(project.id),
                        "source": project.loki_service or "log-server",
                        "environment": LOG_ENVIRONMENT,
                        "logs": lines,
                        "_project": project,
                    }
                    try:
                        result = process_log_batch(payload)
                        if result:
                            print(
                                f"[LOKI-WATCHER] [{project.name}] "
                                f"created={result.get('incidents_created', 0)} "
                                f"updated={result.get('incidents_updated', 0)} "
                                f"failed={result.get('failed', 0)}"
                            )
                    except Exception as e:
                        print(
                            f"[LOKI-WATCHER] [{project.name}] process_log_batch error: {e}"
                        )
                        continue
                else:
                    print(
                        f"[LOKI-WATCHER] [{project.name}] No new logs since {project_last.strftime('%H:%M:%S')}"
                    )

                last_queried[project.id] = poll_start

        except KeyboardInterrupt:
            print("[LOKI-WATCHER] Shutting down")
            break
        except Exception as e:
            print(
                f"[LOKI-WATCHER] Unexpected error: {e} — retrying in {POLL_INTERVAL}s"
            )

        time.sleep(POLL_INTERVAL)
