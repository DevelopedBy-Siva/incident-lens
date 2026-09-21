from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class LogEnvelope:
    """Provider-neutral log record at the Data Plane ingestion boundary."""

    provider_id: str
    timestamp: datetime
    message: str
    source: str
    environment: str
    attributes: dict[str, Any] = field(default_factory=dict)


class LogSourceError(RuntimeError):
    pass


class LogSourceAuthenticationError(LogSourceError):
    pass


class LogSourceConnector(ABC):
    """Read logs without exposing provider response shapes downstream."""

    @abstractmethod
    def fetch(self, start: datetime, end: datetime) -> list[LogEnvelope]:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
