"""
Log parser - imports production parsing logic.

Reuses the production parser to ensure consistency.
"""

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

# Import production parser
sys.path.insert(0, str(Path(__file__).parent.parent / "log-analyzer"))
from app.data.parser import ParsedLog


def parse_log_line(raw_line: str) -> Optional[ParsedLog]:
    """
    Parse a single log line using production parser.
    Returns None if the line is a comment or empty.
    """
    line = raw_line.strip()
    
    # Skip comments and empty lines
    if not line or line.startswith("#"):
        return None
    
    try:
        return ParsedLog(line)
    except Exception:
        return None


def extract_service_from_log(raw_line: str) -> Optional[str]:
    """Extract service name from log line."""
    # Look for [service-name] pattern
    match = re.search(r"\[([a-z0-9\-_]+)\]", raw_line, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def load_evaluation_log(log_path: Path) -> List[str]:
    """Load raw log lines from evaluation file."""
    lines = []
    with open(log_path, "r") as f:
        for line in f:
            stripped = line.strip()
            # Skip comments and empty lines
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)
    return lines


def group_logs_by_service(log_lines: List[str]) -> Dict[str, List[str]]:
    """
    Group raw log lines by service name.
    
    This is the first step in evaluation - separate the stream by service
    so we can analyze each service independently.
    """
    service_logs = {}
    
    for line in log_lines:
        service = extract_service_from_log(line)
        if service:
            if service not in service_logs:
                service_logs[service] = []
            service_logs[service].append(line)
    
    return service_logs


def group_logs_by_service_with_metadata(log_lines: List[str]) -> Dict[str, List[Dict]]:
    """
    Group logs by service with parsed metadata.
    
    Returns dict mapping service -> list of dicts with:
        - raw: original log line
        - parsed: ParsedLog object
        - timestamp: extracted timestamp
    """
    service_logs = {}
    
    for line in log_lines:
        service = extract_service_from_log(line)
        if not service:
            continue
            
        parsed = parse_log_line(line)
        if not parsed:
            continue
        
        if service not in service_logs:
            service_logs[service] = []
        
        service_logs[service].append({
            "raw": line,
            "parsed": parsed,
            "timestamp": parsed.timestamp,
        })
    
    return service_logs
