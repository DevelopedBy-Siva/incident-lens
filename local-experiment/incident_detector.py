"""
Incident detection using production clustering logic.

Reuses production signature generation and clustering to create incident candidates
from evaluation logs.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import hashlib

# Import production modules
sys.path.insert(0, str(Path(__file__).parent.parent / "log-analyzer"))
from app.data.parser import ParsedLog
from app.data.signatures import generate_signature, normalize_message


# Production clustering parameters
CLUSTER_WINDOW_MINUTES = 2
MAX_SAMPLES = 10


@dataclass
class IncidentCandidate:
    """
    Represents a candidate incident detected from logs.
    
    This is what gets sent to the model for analysis.
    """
    incident_id: str
    service: str
    environment: str
    signature: str
    first_seen: datetime
    last_seen: datetime
    count: int
    sample_lines: List[str] = field(default_factory=list)
    
    # Metadata for debugging
    grouping_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        from datetime import timezone
        
        # Ensure timezone-aware timestamps
        first_seen_aware = self.first_seen
        last_seen_aware = self.last_seen
        
        if first_seen_aware and first_seen_aware.tzinfo is None:
            first_seen_aware = first_seen_aware.replace(tzinfo=timezone.utc)
        
        if last_seen_aware and last_seen_aware.tzinfo is None:
            last_seen_aware = last_seen_aware.replace(tzinfo=timezone.utc)
        
        return {
            "incident_id": self.incident_id,
            "service": self.service,
            "environment": self.environment,
            "signature": self.signature,
            "first_seen": first_seen_aware.isoformat() if first_seen_aware else None,
            "last_seen": last_seen_aware.isoformat() if last_seen_aware else None,
            "count": self.count,
            "sample_lines": self.sample_lines,
            "grouping_metadata": self.grouping_metadata,
        }
    
    def get_log_summary(self) -> str:
        """Get first few logs for display."""
        return "\n".join(self.sample_lines[:3])


def detect_incidents_in_service_logs(
    service: str,
    log_entries: List[Dict],
    environment: str = "prod"
) -> List[IncidentCandidate]:
    """
    Detect incident candidates from a service's logs using production clustering.
    
    Args:
        service: Service name
        log_entries: List of dicts with 'raw', 'parsed', 'timestamp' keys
        environment: Environment name
    
    Returns:
        List of IncidentCandidate objects
    """
    # Filter to ERROR/WARN/CRITICAL logs only (matching production)
    filtered = []
    for entry in log_entries:
        parsed = entry["parsed"]
        if parsed.level in ["ERROR", "WARN", "WARNING", "CRITICAL"]:
            filtered.append(entry)
    
    if not filtered:
        return []
    
    # Sort by timestamp
    filtered.sort(key=lambda e: e["timestamp"])
    
    # Cluster using production logic
    incidents: Dict[str, IncidentCandidate] = {}
    
    for entry in filtered:
        parsed = entry["parsed"]
        raw = entry["raw"]
        timestamp = entry["timestamp"]
        
        # Generate signature using production logic
        sig = generate_signature(service, parsed)
        
        # Find existing incident within time window (production clustering)
        matching_incident = None
        for incident_id, incident in incidents.items():
            if incident.signature == sig:
                # Check if within time window
                time_diff = (timestamp - incident.last_seen).total_seconds() / 60
                if time_diff <= CLUSTER_WINDOW_MINUTES:
                    matching_incident = incident
                    break
        
        if matching_incident:
            # Update existing incident
            matching_incident.count += 1
            matching_incident.last_seen = timestamp
            if len(matching_incident.sample_lines) < MAX_SAMPLES:
                matching_incident.sample_lines.append(raw)
        else:
            # Create new incident
            incident_id = f"{service}-{sig[:8]}-{len(incidents)}"
            new_incident = IncidentCandidate(
                incident_id=incident_id,
                service=service,
                environment=environment,
                signature=sig,
                first_seen=timestamp,
                last_seen=timestamp,
                count=1,
                sample_lines=[raw],
                grouping_metadata={
                    "normalized_message": normalize_message(parsed.message),
                    "level": parsed.level,
                    "exception_type": parsed.exception_type,
                }
            )
            incidents[incident_id] = new_incident
    
    return list(incidents.values())


def detect_all_incidents(
    service_logs: Dict[str, List[Dict]],
    environment: str = "prod"
) -> List[IncidentCandidate]:
    """
    Detect incidents across all services.
    
    Args:
        service_logs: Dict mapping service name -> list of log entries
        environment: Environment name
    
    Returns:
        List of all incident candidates across all services
    """
    all_incidents = []
    
    for service, logs in service_logs.items():
        incidents = detect_incidents_in_service_logs(service, logs, environment)
        all_incidents.extend(incidents)
    
    # Sort by first_seen
    all_incidents.sort(key=lambda i: i.first_seen)
    
    return all_incidents


def generate_incident_id(service: str, signature: str, index: int) -> str:
    """Generate a unique incident ID."""
    return f"{service}-{signature[:12]}-{index:03d}"


def filter_benign_logs(candidates: List[IncidentCandidate]) -> List[IncidentCandidate]:
    """
    Filter out obvious benign logs (INFO level that shouldn't be incidents).
    
    This is a safety check - production already filters to ERROR/WARN/CRITICAL,
    but we double-check here.
    """
    filtered = []
    for candidate in candidates:
        # Check if any sample contains INFO (shouldn't happen but be safe)
        has_error_warn = any(
            level in log.upper() for log in candidate.sample_lines
            for level in ["ERROR", "WARN", "CRITICAL"]
        )
        
        if has_error_warn or candidate.count >= 3:
            filtered.append(candidate)
    
    return filtered


def summarize_candidates(candidates: List[IncidentCandidate]) -> Dict[str, Any]:
    """Generate summary statistics for incident candidates."""
    services = set(c.service for c in candidates)
    
    total_logs = sum(c.count for c in candidates)
    
    return {
        "total_candidates": len(candidates),
        "unique_services": len(services),
        "services": sorted(services),
        "total_log_count": total_logs,
        "candidates_by_service": {
            service: len([c for c in candidates if c.service == service])
            for service in services
        },
    }
