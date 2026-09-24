"""
Datadog HTTP Log Ingestion for Training Telemetry

Best-effort structured logging to Datadog using the HTTP Logs API.
Failures in log delivery must never cause training to fail.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Datadog HTTP Logs API endpoints by site
DATADOG_LOG_INTAKE_URLS = {
    "datadoghq.com": "https://http-intake.logs.datadoghq.com/api/v2/logs",
    "datadoghq.eu": "https://http-intake.logs.datadoghq.eu/api/v2/logs",
    "us3.datadoghq.com": "https://http-intake.logs.us3.datadoghq.com/api/v2/logs",
    "us5.datadoghq.com": "https://http-intake.logs.us5.datadoghq.com/api/v2/logs",
    "ap1.datadoghq.com": "https://http-intake.logs.ap1.datadoghq.com/api/v2/logs",
    "ddog-gov.com": "https://http-intake.logs.ddog-gov.com/api/v2/logs",
}

# Service and source identifiers for training logs
SERVICE_NAME = "incidentlens-training"
SOURCE_NAME = "incidentlens"

# Connection and timeout settings (keep aggressive to avoid blocking training)
REQUEST_TIMEOUT_SECONDS = 5
MAX_RETRIES = 0  # No retries - fail fast


@dataclass
class TrainingLogContext:
    """Correlation context for training logs."""
    
    project_id: str
    training_job_id: str
    dataset_id: str
    ec2_instance_id: str | None = None
    environment: str = "production"
    
    def to_tags(self) -> dict[str, str]:
        """Convert context to Datadog tags."""
        tags = {
            "project_id": self.project_id,
            "training_job_id": self.training_job_id,
            "dataset_id": self.dataset_id,
            "environment": self.environment,
        }
        if self.ec2_instance_id:
            tags["ec2_instance_id"] = self.ec2_instance_id
        return tags


class DatadogLogger:
    """Best-effort structured logging to Datadog for training telemetry.
    
    This logger sends structured training events to Datadog's HTTP Logs API.
    All operations are best-effort with aggressive timeouts. Failures are logged
    locally but never propagated to prevent training failures.
    
    Usage:
        logger = DatadogLogger(api_key, site, context)
        logger.info("training_started", {"model": "Qwen3.5-4B"})
        logger.error("training_failed", {"error": "OOM"})
    """
    
    def __init__(
        self,
        api_key: str | None,
        site: str | None,
        context: TrainingLogContext,
        *,
        enabled: bool = True,
    ):
        """Initialize Datadog logger.
        
        Args:
            api_key: Datadog API key (not application key - logs API only needs API key)
            site: Datadog site (e.g., "datadoghq.com", "datadoghq.eu")
            context: Training context for correlation
            enabled: Whether to actually send logs (default: True)
        """
        self.api_key = api_key
        self.site = site
        self.context = context
        self.enabled = enabled
        self._intake_url = None
        self._session = None
        
        # Validate and prepare
        if self.enabled and api_key and site:
            normalized_site = site.strip().lower().removeprefix("https://").rstrip("/")
            self._intake_url = DATADOG_LOG_INTAKE_URLS.get(normalized_site)
            
            if not self._intake_url:
                logger.warning(
                    f"Unsupported Datadog site '{site}'. Training logs will not be sent to Datadog. "
                    f"Supported sites: {', '.join(DATADOG_LOG_INTAKE_URLS.keys())}"
                )
                self.enabled = False
            else:
                # Create persistent session for connection reuse
                self._session = requests.Session()
                self._session.headers.update({
                    "DD-API-KEY": api_key,
                    "Content-Type": "application/json",
                })
                logger.info(
                    f"Datadog training telemetry enabled: "
                    f"service={SERVICE_NAME}, site={normalized_site}"
                )
        else:
            if not api_key or not site:
                logger.info(
                    "Datadog training telemetry disabled: credentials not configured"
                )
            self.enabled = False
    
    def info(self, event: str, attributes: dict[str, Any] | None = None) -> None:
        """Log an info-level training event.
        
        Args:
            event: Event name (e.g., "training_started", "dataset_downloaded")
            attributes: Additional structured attributes
        """
        self._log("info", event, attributes)
    
    def warning(self, event: str, attributes: dict[str, Any] | None = None) -> None:
        """Log a warning-level training event.
        
        Args:
            event: Event name
            attributes: Additional structured attributes
        """
        self._log("warn", event, attributes)
    
    def error(self, event: str, attributes: dict[str, Any] | None = None) -> None:
        """Log an error-level training event.
        
        Args:
            event: Event name (e.g., "training_failed")
            attributes: Additional structured attributes (never include full error messages with secrets)
        """
        self._log("error", event, attributes)
    
    def progress(
        self,
        current_step: int,
        total_steps: int,
        *,
        training_loss: float | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        """Log training progress.
        
        Args:
            current_step: Current training step
            total_steps: Total training steps
            training_loss: Current training loss (optional)
            elapsed_seconds: Elapsed time since training start (optional)
        """
        progress_percent = (current_step / total_steps * 100) if total_steps > 0 else 0
        
        attributes = {
            "current_step": current_step,
            "total_steps": total_steps,
            "progress_percent": round(progress_percent, 1),
        }
        
        if training_loss is not None:
            attributes["training_loss"] = round(training_loss, 6)
        
        if elapsed_seconds is not None:
            attributes["elapsed_seconds"] = round(elapsed_seconds, 1)
        
        self._log("info", "training_progress", attributes)
    
    def _log(
        self,
        level: str,
        event: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Internal log method that sends to Datadog.
        
        Args:
            level: Log level (info, warn, error)
            event: Event name
            attributes: Additional attributes
        """
        if not self.enabled or not self._session or not self._intake_url:
            return
        
        try:
            # Build structured log entry
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Merge context tags with event-specific attributes
            log_attributes = {
                **self.context.to_tags(),
                "event": event,
            }
            
            if attributes:
                # Filter out None values and ensure serializable types
                for key, value in attributes.items():
                    if value is not None and isinstance(value, (str, int, float, bool)):
                        log_attributes[key] = value
            
            log_entry = {
                "ddsource": SOURCE_NAME,
                "service": SERVICE_NAME,
                "hostname": log_attributes.get("ec2_instance_id", "training-worker"),
                "message": f"Training event: {event}",
                "timestamp": timestamp,
                "level": level,
                **log_attributes,
            }
            
            # Send to Datadog with aggressive timeout
            self._session.post(
                self._intake_url,
                json=[log_entry],  # API accepts array of log entries
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            
            # Success - no need to check response (best-effort)
            
        except requests.Timeout:
            logger.warning(
                f"Datadog log delivery timed out for event '{event}' "
                f"(timeout={REQUEST_TIMEOUT_SECONDS}s). Training continues."
            )
        except requests.RequestException as exc:
            logger.warning(
                f"Datadog log delivery failed for event '{event}': {type(exc).__name__}. "
                "Training continues."
            )
        except Exception as exc:
            # Catch any unexpected errors to prevent training failure
            logger.warning(
                f"Unexpected error sending log to Datadog for event '{event}': "
                f"{type(exc).__name__}. Training continues."
            )
    
    def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            try:
                self._session.close()
            except Exception:
                pass  # Best-effort cleanup
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False  # Don't suppress exceptions


def create_training_logger(
    api_key: str | None,
    site: str | None,
    project_id: str,
    training_job_id: str,
    dataset_id: str,
    ec2_instance_id: str | None = None,
) -> DatadogLogger:
    """Create a Datadog logger for training telemetry.
    
    Args:
        api_key: Datadog API key (optional - training continues without it)
        site: Datadog site (optional)
        project_id: Project ID
        training_job_id: Training job ID
        dataset_id: Dataset ID
        ec2_instance_id: EC2 instance ID (optional)
    
    Returns:
        DatadogLogger instance (may be disabled if credentials absent)
    """
    context = TrainingLogContext(
        project_id=project_id,
        training_job_id=training_job_id,
        dataset_id=dataset_id,
        ec2_instance_id=ec2_instance_id,
        environment="production",
    )
    
    return DatadogLogger(
        api_key=api_key,
        site=site,
        context=context,
        enabled=bool(api_key and site),
    )
