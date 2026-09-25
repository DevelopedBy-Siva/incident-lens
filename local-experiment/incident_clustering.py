"""
Signature-based incident clustering for local evaluation.

This module implements production-compatible clustering logic that groups logs
by signature (error pattern) within time windows, NOT by service.

Key differences from production:
- Uses in-memory data structures instead of PostgreSQL
- Reads from log files instead of Datadog API
- Otherwise matches production clustering behavior exactly
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import uuid

from incident_signatures import generate_signature, extract_exception_type


# Production constants (must match log-analyzer/app/data/clustering.py)
CLUSTER_WINDOW_MINUTES = 2
MAX_SAMPLES = 10

# Evidence constants (must match log-analyzer/app/data/evidence.py)
MAX_INCIDENT_SAMPLES = 8


@dataclass
class ParsedLog:
    """Structured representation of a parsed log line."""
    timestamp: datetime
    level: str
    service: str
    message: str
    exception_type: Optional[str]
    raw: str
    signature: str = ""  # Computed after creation
    
    def __post_init__(self):
        """Generate signature after initialization."""
        if not self.signature:
            self.signature = generate_signature(
                service=self.service,
                level=self.level,
                message=self.message,
                exception_type=self.exception_type
            )


@dataclass
class Incident:
    """
    Incident representation matching production schema.
    
    Fields match log-analyzer/app/data/models.py::Incident
    """
    id: str
    signature: str
    service: str
    environment: str
    first_seen: datetime
    last_seen: datetime
    count: int
    sample_lines: list[str] = field(default_factory=list)
    status: str = "open"
    
    # Metadata for evaluation
    log_indices: list[int] = field(default_factory=list)  # Original log line numbers
    
    def add_log(self, parsed_log: ParsedLog, log_index: int):
        """Add a log to this incident (production-compatible logic)."""
        self.count += 1
        self.last_seen = parsed_log.timestamp
        
        # Store up to MAX_SAMPLES (production behavior)
        if len(self.sample_lines) < MAX_SAMPLES:
            self.sample_lines.append(parsed_log.raw)
        
        self.log_indices.append(log_index)
    
    def get_evidence_samples(self) -> list[str]:
        """Get samples for evidence bundle (up to MAX_INCIDENT_SAMPLES)."""
        return self.sample_lines[:MAX_INCIDENT_SAMPLES]


class IncidentClusterer:
    """
    Production-compatible incident clustering for local evaluation.
    
    Implements the same clustering logic as production:
    - Signature-based grouping (NOT service-based)
    - Time-window constraints
    - Multiple incidents per service support
    - Sample line limits
    """
    
    def __init__(self, time_window_minutes: int = CLUSTER_WINDOW_MINUTES):
        """
        Initialize clusterer.
        
        Args:
            time_window_minutes: Time window for incident grouping (default: 2)
        """
        self.time_window_minutes = time_window_minutes
        self.incidents: dict[str, Incident] = {}  # signature -> Incident
        self.closed_incidents: list[Incident] = []
    
    def process_log(self, parsed_log: ParsedLog, log_index: int) -> tuple[Incident, bool]:
        """
        Process a log line and cluster it into an incident.
        
        Returns:
            (incident, is_new) tuple
            - incident: The incident this log belongs to
            - is_new: True if a new incident was created
        """
        signature = parsed_log.signature
        
        # Check if there's an open incident with this signature
        if signature in self.incidents:
            existing = self.incidents[signature]
            
            # Check if still within time window
            time_gap = parsed_log.timestamp - existing.last_seen
            
            if time_gap <= timedelta(minutes=self.time_window_minutes):
                # Add to existing incident
                existing.add_log(parsed_log, log_index)
                return existing, False
            else:
                # Time window expired - close old incident
                self.closed_incidents.append(existing)
                del self.incidents[signature]
                # Fall through to create new incident
        
        # Create new incident
        new_incident = Incident(
            id=str(uuid.uuid4()),
            signature=signature,
            service=parsed_log.service,
            environment="prod",  # Assume prod for evaluation
            first_seen=parsed_log.timestamp,
            last_seen=parsed_log.timestamp,
            count=1,
            sample_lines=[parsed_log.raw],
            status="open",
            log_indices=[log_index]
        )
        
        self.incidents[signature] = new_incident
        return new_incident, True
    
    def close_all(self):
        """Close all remaining open incidents."""
        for incident in self.incidents.values():
            self.closed_incidents.append(incident)
        self.incidents.clear()
    
    def get_all_incidents(self) -> list[Incident]:
        """Get all incidents (open + closed)."""
        return list(self.incidents.values()) + self.closed_incidents
    
    def get_statistics(self) -> dict:
        """Get clustering statistics."""
        all_incidents = self.get_all_incidents()
        
        services = set(inc.service for inc in all_incidents)
        
        incidents_per_service = defaultdict(int)
        for inc in all_incidents:
            incidents_per_service[inc.service] += 1
        
        return {
            "total_incidents": len(all_incidents),
            "unique_services": len(services),
            "unique_signatures": len(set(inc.signature for inc in all_incidents)),
            "incidents_per_service": dict(incidents_per_service),
            "multi_incident_services": [
                service for service, count in incidents_per_service.items() if count > 1
            ]
        }


def cluster_logs(parsed_logs: list[ParsedLog], time_window_minutes: int = CLUSTER_WINDOW_MINUTES) -> list[Incident]:
    """
    Cluster parsed logs into incidents using production-compatible logic.
    
    Args:
        parsed_logs: List of parsed log entries (must be time-ordered)
        time_window_minutes: Time window for incident grouping
    
    Returns:
        List of Incident objects
    """
    clusterer = IncidentClusterer(time_window_minutes=time_window_minutes)
    
    for idx, parsed_log in enumerate(parsed_logs):
        clusterer.process_log(parsed_log, log_index=idx)
    
    clusterer.close_all()
    
    return clusterer.get_all_incidents()
