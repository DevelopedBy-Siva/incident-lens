from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from app.data.ingestion.log_source import (
    LogEnvelope,
    LogSourceAuthenticationError,
    LogSourceConnector,
    LogSourceError,
)

DATADOG_ENVIRONMENT = "prod"
SUPPORTED_DATADOG_SITES = {
    "datadoghq.com",
    "us3.datadoghq.com",
    "us5.datadoghq.com",
    "datadoghq.eu",
    "ap1.datadoghq.com",
    "ap2.datadoghq.com",
    "uk1.datadoghq.com",
    "ddog-gov.com",
    "us2.ddog-gov.com",
}


@dataclass(frozen=True)
class DatadogLogSourceConfig:
    api_key: str
    app_key: str
    site: str
    query: str
    service_filter: str

    def __post_init__(self) -> None:
        required = {
            "API key": self.api_key,
            "application key": self.app_key,
            "site": self.site,
            "query": self.query,
            "service": self.service_filter,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(f"Datadog {', '.join(missing)} required")
        site = self.site.strip().lower().removeprefix("https://").rstrip("/")
        if site not in SUPPORTED_DATADOG_SITES:
            supported = ", ".join(sorted(SUPPORTED_DATADOG_SITES))
            raise ValueError(
                f"Unsupported Datadog site {site!r}; expected one of {supported}"
            )
        object.__setattr__(self, "site", site)
        object.__setattr__(self, "api_key", self.api_key.strip())
        object.__setattr__(self, "app_key", self.app_key.strip())
        object.__setattr__(self, "query", self.query.strip())
        object.__setattr__(self, "service_filter", self.service_filter.strip())

    @classmethod
    def from_project(cls, project) -> "DatadogLogSourceConfig":
        return cls(
            api_key=getattr(project, "datadog_api_key", None) or "",
            app_key=getattr(project, "datadog_app_key", None) or "",
            site=getattr(project, "datadog_site", None) or "",
            query=getattr(project, "datadog_query", None) or "",
            service_filter=getattr(project, "datadog_service", None) or "",
        )

    @property
    def search_url(self) -> str:
        return f"https://api.{self.site}/api/v2/logs/events/search"

    @property
    def effective_query(self) -> str:
        filters = [
            f"({self.query})",
            f"env:{_quoted(DATADOG_ENVIRONMENT)}",
            f"service:{_quoted(self.service_filter)}",
        ]
        return " AND ".join(filters)


class DatadogLogConnector(LogSourceConnector):
    """Query Datadog Logs and normalize each result into a LogEnvelope."""

    def __init__(
        self,
        config: DatadogLogSourceConfig,
        *,
        client: httpx.Client | None = None,
        page_size: int = 1000,
    ):
        if page_size <= 0:
            raise ValueError("Datadog page size must be positive")
        self.config = config
        self.page_size = page_size
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=15)

    def fetch(self, start: datetime, end: datetime) -> list[LogEnvelope]:
        if start >= end:
            return []

        cursor = None
        seen_cursors = set()
        envelopes = []
        while True:
            body: dict[str, Any] = {
                "filter": {
                    "from": _iso_utc(start),
                    "to": _iso_utc(end),
                    "query": self.config.effective_query,
                },
                "sort": "timestamp",
                "page": {"limit": self.page_size},
            }
            if cursor:
                body["page"]["cursor"] = cursor

            try:
                response = self.client.post(
                    self.config.search_url,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        "DD-API-KEY": self.config.api_key,
                        "DD-APPLICATION-KEY": self.config.app_key,
                    },
                    json=body,
                )
            except httpx.HTTPError as exc:
                raise LogSourceError(
                    f"Datadog log search request failed: {exc}"
                ) from exc

            if response.status_code in {401, 403}:
                raise LogSourceAuthenticationError(
                    "Datadog rejected the API or application key"
                )
            if response.status_code != 200:
                raise LogSourceError(
                    f"Datadog log search failed with HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise LogSourceError(
                    "Datadog log search returned invalid JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise LogSourceError("Datadog log search returned an invalid response")
            data = payload.get("data") or []
            if not isinstance(data, list):
                raise LogSourceError("Datadog log search returned invalid log data")
            envelopes.extend(
                self._to_envelope(item, end)
                for item in data
                if isinstance(item, dict) and _message(item)
            )
            next_cursor = payload.get("meta", {}).get("page", {}).get("after")
            if not next_cursor or next_cursor == cursor or next_cursor in seen_cursors:
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor

        return sorted(
            envelopes,
            key=lambda envelope: (envelope.timestamp, envelope.provider_id),
        )

    def verify_access(self, start: datetime, end: datetime) -> None:
        """Verify credentials and logs-read access without persisting anything."""
        body = {
            "filter": {
                "from": _iso_utc(start),
                "to": _iso_utc(end),
                "query": self.config.effective_query,
            },
            "sort": "-timestamp",
            "page": {"limit": 1},
        }
        try:
            response = self.client.post(
                self.config.search_url,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "DD-API-KEY": self.config.api_key,
                    "DD-APPLICATION-KEY": self.config.app_key,
                },
                json=body,
            )
        except httpx.HTTPError as exc:
            raise LogSourceError(f"Datadog access verification failed: {exc}") from exc

        if response.status_code in {401, 403}:
            raise LogSourceAuthenticationError(
                "Datadog rejected the credentials or logs_read_data access"
            )
        if response.status_code != 200:
            raise LogSourceError(
                f"Datadog verification failed with HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def _to_envelope(
        self, item: dict[str, Any], fallback_time: datetime
    ) -> LogEnvelope:
        attributes = item.get("attributes") or {}
        custom = attributes.get("attributes") or {}
        tags = attributes.get("tags") or []
        environment = (
            custom.get("env")
            or custom.get("environment")
            or _tag_value(tags, "env")
            or DATADOG_ENVIRONMENT
        )
        source = (
            attributes.get("service")
            or custom.get("service")
            or self.config.service_filter
            or "datadog"
        )
        return LogEnvelope(
            provider_id=str(item.get("id") or ""),
            timestamp=_parse_timestamp(attributes.get("timestamp"), fallback_time),
            message=_message(item),
            source=str(source),
            environment=str(environment),
            attributes=custom if isinstance(custom, dict) else {},
        )


def _quoted(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: Any, fallback: datetime) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
    if fallback.tzinfo is None:
        fallback = fallback.replace(tzinfo=timezone.utc)
    return fallback.astimezone(timezone.utc)


def _message(item: dict[str, Any]) -> str:
    attributes = item.get("attributes") or {}
    value = attributes.get("message")
    return value if isinstance(value, str) else ""


def _tag_value(tags: Any, name: str) -> str | None:
    if not isinstance(tags, list):
        return None
    prefix = f"{name}:"
    for tag in tags:
        if isinstance(tag, str) and tag.startswith(prefix):
            return tag[len(prefix) :]
    return None
