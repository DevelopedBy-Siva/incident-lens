"""
Structured output parser for incident analysis responses.

This module implements production-compatible JSON parsing with:
- Markdown code fence removal
- Multiple JSON object detection
- Partial output handling
- Comprehensive error tracking
- Field completion metrics
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ParseStatus(Enum):
    """Parse result status."""
    SUCCESS = "success"
    PARTIAL = "partial"  # Some fields missing
    MALFORMED_JSON = "malformed_json"
    NO_JSON_FOUND = "no_json"
    MULTIPLE_JSON_OBJECTS = "multiple_json"  # Concatenated objects
    EMPTY_RESPONSE = "empty"


@dataclass
class ParsedOutput:
    """
    Structured representation of parsed model output.
    
    This matches the production IncidentAnalysis schema.
    """
    # Core fields (required)
    severity: Optional[str] = None
    disposition: Optional[str] = None
    summary: Optional[str] = None
    
    # Quality fields
    confidence: Optional[float] = None
    suspected_root_cause: Optional[str] = None
    
    # Action fields
    next_steps: list[str] = field(default_factory=list)
    ticket_title: Optional[str] = None
    ticket_body: Optional[str] = None
    
    # Parse metadata
    parse_status: ParseStatus = ParseStatus.SUCCESS
    parse_error: Optional[str] = None
    raw_response: str = ""
    extracted_json: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if parse was successful with required fields."""
        return (
            self.parse_status == ParseStatus.SUCCESS
            and self.severity is not None
            and self.disposition is not None
            and self.summary is not None
        )
    
    def is_complete(self) -> bool:
        """Check if all expected fields are present."""
        return (
            self.is_valid()
            and self.confidence is not None
            and self.suspected_root_cause is not None
            and len(self.next_steps) > 0
            and self.ticket_title is not None
            and self.ticket_body is not None
        )
    
    def get_completion_score(self) -> float:
        """Calculate field completion percentage (0.0-1.0)."""
        expected_fields = [
            self.severity is not None,
            self.disposition is not None,
            self.summary is not None and len(self.summary) > 0,
            self.confidence is not None,
            self.suspected_root_cause is not None,
            len(self.next_steps) > 0,
            self.ticket_title is not None and len(self.ticket_title) > 0,
            self.ticket_body is not None and len(self.ticket_body) > 0,
        ]
        return sum(expected_fields) / len(expected_fields)


def parse_model_output(raw_response: str) -> ParsedOutput:
    """
    Parse model output into structured format.
    
    This implements production-compatible parsing logic from
    log-analyzer/app/serving/investigator.py::_parse_final()
    
    Args:
        raw_response: Raw text output from model
    
    Returns:
        ParsedOutput with parse status and extracted data
    """
    if not raw_response or not raw_response.strip():
        return ParsedOutput(
            parse_status=ParseStatus.EMPTY_RESPONSE,
            parse_error="Empty model response",
            raw_response=raw_response
        )
    
    # Step 1: Remove markdown code fences
    clean = re.sub(r"```(?:json)?", "", raw_response).strip()
    
    # Step 2: Check for multiple JSON objects (common failure mode)
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = list(re.finditer(json_pattern, clean, re.DOTALL))
    
    if len(matches) == 0:
        return ParsedOutput(
            parse_status=ParseStatus.NO_JSON_FOUND,
            parse_error="No JSON object found in response",
            raw_response=raw_response
        )
    
    if len(matches) > 1:
        # Multiple JSON objects detected (e.g., {"confidence":0.8}{"ticket_body":"..."})
        return ParsedOutput(
            parse_status=ParseStatus.MULTIPLE_JSON_OBJECTS,
            parse_error=f"Found {len(matches)} JSON objects (expected 1)",
            raw_response=raw_response,
            extracted_json=f"[{', '.join(m.group() for m in matches)}]"
        )
    
    # Step 3: Parse the single JSON object
    json_text = matches[0].group()
    
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        return ParsedOutput(
            parse_status=ParseStatus.MALFORMED_JSON,
            parse_error=f"JSON decode error: {e}",
            raw_response=raw_response,
            extracted_json=json_text
        )
    
    # Step 4: Extract fields
    parsed = ParsedOutput(
        severity=data.get("severity"),
        disposition=data.get("disposition"),
        summary=data.get("summary"),
        confidence=_parse_float(data.get("confidence")),
        suspected_root_cause=data.get("suspected_root_cause"),
        next_steps=_parse_list(data.get("next_steps")),
        ticket_title=data.get("ticket_title"),
        ticket_body=data.get("ticket_body"),
        raw_response=raw_response,
        extracted_json=json_text
    )
    
    # Step 5: Validate required fields
    required_fields = ["severity", "disposition", "summary"]
    missing_required = [f for f in required_fields if not data.get(f)]
    
    if missing_required:
        parsed.parse_status = ParseStatus.PARTIAL
        parsed.parse_error = f"Missing required fields: {', '.join(missing_required)}"
    else:
        parsed.parse_status = ParseStatus.SUCCESS
    
    return parsed


def _parse_float(value) -> Optional[float]:
    """Safely parse float value."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _parse_list(value) -> list[str]:
    """Safely parse list value."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    # Sometimes model returns a string instead of list
    if isinstance(value, str):
        # Try to parse as JSON array
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
        # Fall back to single-item list
        return [value]
    return []


def validate_severity(severity: Optional[str]) -> bool:
    """Validate severity value."""
    if severity is None:
        return False
    valid_severities = {"low", "medium", "high", "critical"}
    return severity.lower() in valid_severities


def validate_disposition(disposition: Optional[str]) -> bool:
    """Validate disposition value."""
    if disposition is None:
        return False
    valid_dispositions = {
        "NO_ACTION",
        "OBSERVE",
        "NEEDS_DEV",
        "NEEDS_ONCALL",
        "ESCALATE"
    }
    return disposition.upper() in valid_dispositions


def validate_confidence(confidence: Optional[float]) -> bool:
    """Validate confidence value."""
    if confidence is None:
        return False
    return 0.0 <= confidence <= 1.0


def get_parse_metrics(parsed_outputs: list[ParsedOutput]) -> dict:
    """
    Calculate parsing metrics across multiple outputs.
    
    Returns:
        Dictionary with parsing statistics
    """
    if not parsed_outputs:
        return {
            "total": 0,
            "valid_json_rate": 0.0,
            "parse_failure_rate": 0.0,
            "partial_rate": 0.0,
            "multiple_json_rate": 0.0,
            "mean_completion": 0.0,
        }
    
    total = len(parsed_outputs)
    
    status_counts = {
        ParseStatus.SUCCESS: 0,
        ParseStatus.PARTIAL: 0,
        ParseStatus.MALFORMED_JSON: 0,
        ParseStatus.NO_JSON_FOUND: 0,
        ParseStatus.MULTIPLE_JSON_OBJECTS: 0,
        ParseStatus.EMPTY_RESPONSE: 0,
    }
    
    for output in parsed_outputs:
        status_counts[output.parse_status] += 1
    
    valid_count = status_counts[ParseStatus.SUCCESS]
    parse_failures = (
        status_counts[ParseStatus.MALFORMED_JSON]
        + status_counts[ParseStatus.NO_JSON_FOUND]
        + status_counts[ParseStatus.EMPTY_RESPONSE]
    )
    
    completion_scores = [out.get_completion_score() for out in parsed_outputs]
    mean_completion = sum(completion_scores) / len(completion_scores) if completion_scores else 0.0
    
    return {
        "total": total,
        "valid_json_rate": valid_count / total if total > 0 else 0.0,
        "parse_failure_rate": parse_failures / total if total > 0 else 0.0,
        "partial_rate": status_counts[ParseStatus.PARTIAL] / total if total > 0 else 0.0,
        "multiple_json_rate": status_counts[ParseStatus.MULTIPLE_JSON_OBJECTS] / total if total > 0 else 0.0,
        "mean_completion": mean_completion,
        "status_breakdown": {
            status.value: count for status, count in status_counts.items()
        },
        "field_completion": {
            "severity": sum(1 for out in parsed_outputs if out.severity is not None) / total,
            "disposition": sum(1 for out in parsed_outputs if out.disposition is not None) / total,
            "summary": sum(1 for out in parsed_outputs if out.summary) / total,
            "confidence": sum(1 for out in parsed_outputs if out.confidence is not None) / total,
            "suspected_root_cause": sum(1 for out in parsed_outputs if out.suspected_root_cause) / total,
            "next_steps": sum(1 for out in parsed_outputs if len(out.next_steps) > 0) / total,
            "ticket_title": sum(1 for out in parsed_outputs if out.ticket_title) / total,
            "ticket_body": sum(1 for out in parsed_outputs if out.ticket_body) / total,
        }
    }
