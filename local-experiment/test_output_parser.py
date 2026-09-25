"""
Tests for structured output parser.

Validates handling of various model output formats including
edge cases discovered in previous evaluations.
"""

import pytest
import json
from output_parser import (
    parse_model_output,
    ParseStatus,
    ParsedOutput,
    validate_severity,
    validate_disposition,
    validate_confidence,
    get_parse_metrics,
)


class TestValidOutput:
    """Test parsing of valid, well-formed outputs."""
    
    def test_complete_valid_json(self):
        """Parse complete valid JSON response."""
        response = """{
            "severity": "high",
            "disposition": "NEEDS_ONCALL",
            "confidence": 0.85,
            "summary": "Database connection pool exhausted",
            "suspected_root_cause": "Connection leak in payment processing",
            "next_steps": ["Scale connection pool", "Investigate leak"],
            "ticket_title": "Payment API connection pool exhaustion",
            "ticket_body": "Detailed description here"
        }"""
        
        parsed = parse_model_output(response)
        
        assert parsed.parse_status == ParseStatus.SUCCESS
        assert parsed.is_valid()
        assert parsed.is_complete()
        assert parsed.severity == "high"
        assert parsed.disposition == "NEEDS_ONCALL"
        assert parsed.confidence == 0.85
        assert parsed.summary == "Database connection pool exhausted"
        assert parsed.suspected_root_cause == "Connection leak in payment processing"
        assert len(parsed.next_steps) == 2
        assert parsed.ticket_title == "Payment API connection pool exhaustion"
        assert parsed.ticket_body == "Detailed description here"
    
    def test_minimal_valid_json(self):
        """Parse minimal valid JSON (required fields only)."""
        response = """{
            "severity": "medium",
            "disposition": "OBSERVE",
            "summary": "Temporary lag spike"
        }"""
        
        parsed = parse_model_output(response)
        
        assert parsed.parse_status == ParseStatus.SUCCESS
        assert parsed.is_valid()
        assert not parsed.is_complete()  # Missing optional fields
        assert parsed.severity == "medium"
        assert parsed.disposition == "OBSERVE"
        assert parsed.summary == "Temporary lag spike"
        assert parsed.confidence is None
        assert parsed.suspected_root_cause is None
    
    def test_json_with_markdown_fences(self):
        """Parse JSON wrapped in markdown code fences."""
        response = """```json
{
    "severity": "critical",
    "disposition": "ESCALATE",
    "summary": "Service down"
}
```"""
        
        parsed = parse_model_output(response)
        
        assert parsed.parse_status == ParseStatus.SUCCESS
        assert parsed.is_valid()
        assert parsed.severity == "critical"
    
    def test_json_with_surrounding_text(self):
        """Parse JSON with explanatory text before/after."""
        response = """Here is my analysis:

{
    "severity": "low",
    "disposition": "NO_ACTION",
    "summary": "Normal operations"
}

This is normal activity."""
        
        parsed = parse_model_output(response)
        
        assert parsed.parse_status == ParseStatus.SUCCESS
        assert parsed.is_valid()
        assert parsed.severity == "low"


class TestMalformedOutput:
    """Test handling of malformed or invalid outputs."""
    
    def test_empty_response(self):
        """Handle empty response."""
        parsed = parse_model_output("")
        
        assert parsed.parse_status == ParseStatus.EMPTY_RESPONSE
        assert not parsed.is_valid()
        assert parsed.parse_error is not None
    
    def test_no_json_found(self):
        """Handle response with no JSON."""
        response = "This is just plain text with no JSON object."
        
        parsed = parse_model_output(response)
        
        assert parsed.parse_status == ParseStatus.NO_JSON_FOUND
        assert not parsed.is_valid()
        assert "No JSON object found" in parsed.parse_error
    
    def test_malformed_json(self):
        """Handle syntactically invalid JSON."""
        response = """{
            "severity": "high",
            "disposition": "NEEDS_ONCALL",
            "summary": "Missing closing brace"
        """
        
        parsed = parse_model_output(response)
        
        # Regex may not match unclosed braces, so either status is acceptable
        assert parsed.parse_status in (ParseStatus.MALFORMED_JSON, ParseStatus.NO_JSON_FOUND)
        assert not parsed.is_valid()
        assert parsed.parse_error is not None
    
    def test_multiple_json_objects(self):
        """Handle multiple concatenated JSON objects (observed failure mode)."""
        # This is the actual failure case from previous evaluation:
        # Model outputs two separate JSON objects instead of one
        response = """{"confidence": 0.8}{"ticket_body": "Details here", "ticket_title": "Issue"}"""
        
        parsed = parse_model_output(response)
        
        assert parsed.parse_status == ParseStatus.MULTIPLE_JSON_OBJECTS
        assert not parsed.is_valid()
        assert "2 JSON objects" in parsed.parse_error
        assert parsed.extracted_json is not None
    
    def test_missing_required_fields(self):
        """Handle JSON missing required fields."""
        response = """{
            "confidence": 0.9,
            "next_steps": ["Do something"]
        }"""
        
        parsed = parse_model_output(response)
        
        assert parsed.parse_status == ParseStatus.PARTIAL
        assert not parsed.is_valid()
        assert "Missing required fields" in parsed.parse_error


class TestFieldValidation:
    """Test field value validation."""
    
    def test_severity_validation(self):
        """Validate severity values."""
        assert validate_severity("low")
        assert validate_severity("medium")
        assert validate_severity("high")
        assert validate_severity("critical")
        assert validate_severity("LOW")  # Case insensitive
        assert not validate_severity("severe")
        assert not validate_severity(None)
        assert not validate_severity("")
    
    def test_disposition_validation(self):
        """Validate disposition values."""
        assert validate_disposition("NO_ACTION")
        assert validate_disposition("OBSERVE")
        assert validate_disposition("NEEDS_DEV")
        assert validate_disposition("NEEDS_ONCALL")
        assert validate_disposition("ESCALATE")
        assert validate_disposition("escalate")  # Case insensitive
        assert not validate_disposition("FIX_NOW")
        assert not validate_disposition(None)
    
    def test_confidence_validation(self):
        """Validate confidence values."""
        assert validate_confidence(0.0)
        assert validate_confidence(0.5)
        assert validate_confidence(1.0)
        assert not validate_confidence(-0.1)
        assert not validate_confidence(1.5)
        assert not validate_confidence(None)


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_nested_json(self):
        """Handle JSON with nested objects."""
        response = """{
            "severity": "high",
            "disposition": "NEEDS_DEV",
            "summary": "Issue detected",
            "metadata": {
                "nested": "value"
            }
        }"""
        
        parsed = parse_model_output(response)
        
        assert parsed.parse_status == ParseStatus.SUCCESS
        assert parsed.is_valid()
    
    def test_next_steps_as_string(self):
        """Handle next_steps as string instead of array (model error)."""
        response = """{
            "severity": "medium",
            "disposition": "OBSERVE",
            "summary": "Issue",
            "next_steps": "Monitor the situation"
        }"""
        
        parsed = parse_model_output(response)
        
        assert parsed.parse_status == ParseStatus.SUCCESS
        assert len(parsed.next_steps) == 1
        assert parsed.next_steps[0] == "Monitor the situation"
    
    def test_unicode_content(self):
        """Handle unicode in summary and other fields."""
        response = """{
            "severity": "low",
            "disposition": "NO_ACTION",
            "summary": "Connection timeout – retry succeeded ✓"
        }"""
        
        parsed = parse_model_output(response)
        
        assert parsed.parse_status == ParseStatus.SUCCESS
        assert "–" in parsed.summary
        assert "✓" in parsed.summary
    
    def test_null_values(self):
        """Handle explicit null values."""
        response = """{
            "severity": "medium",
            "disposition": "OBSERVE",
            "summary": "Issue detected",
            "suspected_root_cause": null,
            "ticket_title": null
        }"""
        
        parsed = parse_model_output(response)
        
        assert parsed.parse_status == ParseStatus.SUCCESS
        assert parsed.suspected_root_cause is None
        assert parsed.ticket_title is None


class TestMetrics:
    """Test parse metrics calculation."""
    
    def test_metrics_empty_list(self):
        """Handle empty output list."""
        metrics = get_parse_metrics([])
        
        assert metrics["total"] == 0
        assert metrics["valid_json_rate"] == 0.0
    
    def test_metrics_all_valid(self):
        """Calculate metrics for all valid outputs."""
        outputs = [
            ParsedOutput(
                severity="high",
                disposition="NEEDS_ONCALL",
                summary="Issue 1",
                parse_status=ParseStatus.SUCCESS
            ),
            ParsedOutput(
                severity="medium",
                disposition="OBSERVE",
                summary="Issue 2",
                parse_status=ParseStatus.SUCCESS
            ),
        ]
        
        metrics = get_parse_metrics(outputs)
        
        assert metrics["total"] == 2
        assert metrics["valid_json_rate"] == 1.0
        assert metrics["parse_failure_rate"] == 0.0
    
    def test_metrics_mixed_results(self):
        """Calculate metrics for mixed success/failure."""
        outputs = [
            ParsedOutput(
                severity="high",
                disposition="NEEDS_ONCALL",
                summary="Valid",
                parse_status=ParseStatus.SUCCESS
            ),
            ParsedOutput(
                parse_status=ParseStatus.MALFORMED_JSON,
                parse_error="Invalid JSON"
            ),
            ParsedOutput(
                parse_status=ParseStatus.MULTIPLE_JSON_OBJECTS,
                parse_error="Multiple objects"
            ),
        ]
        
        metrics = get_parse_metrics(outputs)
        
        assert metrics["total"] == 3
        assert metrics["valid_json_rate"] == 1/3
        assert metrics["parse_failure_rate"] == 1/3
        assert metrics["multiple_json_rate"] == 1/3
    
    def test_field_completion_metrics(self):
        """Calculate field completion percentages."""
        outputs = [
            ParsedOutput(
                severity="high",
                disposition="NEEDS_ONCALL",
                summary="Complete",
                confidence=0.9,
                suspected_root_cause="Root cause",
                next_steps=["Step 1"],
                ticket_title="Title",
                ticket_body="Body",
                parse_status=ParseStatus.SUCCESS
            ),
            ParsedOutput(
                severity="medium",
                disposition="OBSERVE",
                summary="Partial",
                parse_status=ParseStatus.PARTIAL
            ),
        ]
        
        metrics = get_parse_metrics(outputs)
        
        assert metrics["field_completion"]["severity"] == 1.0
        assert metrics["field_completion"]["disposition"] == 1.0
        assert metrics["field_completion"]["summary"] == 1.0
        assert metrics["field_completion"]["confidence"] == 0.5
        assert metrics["field_completion"]["ticket_title"] == 0.5


class TestCompletionScore:
    """Test completion score calculation."""
    
    def test_complete_output_score(self):
        """Complete output should score 1.0."""
        parsed = ParsedOutput(
            severity="high",
            disposition="NEEDS_ONCALL",
            summary="Complete incident analysis",
            confidence=0.9,
            suspected_root_cause="Root cause identified",
            next_steps=["Step 1", "Step 2"],
            ticket_title="Title",
            ticket_body="Body",
            parse_status=ParseStatus.SUCCESS
        )
        
        assert parsed.get_completion_score() == 1.0
    
    def test_minimal_output_score(self):
        """Minimal valid output should score 3/8."""
        parsed = ParsedOutput(
            severity="medium",
            disposition="OBSERVE",
            summary="Minimal",
            parse_status=ParseStatus.SUCCESS
        )
        
        score = parsed.get_completion_score()
        assert score == 3/8  # Only severity, disposition, summary
    
    def test_failed_parse_score(self):
        """Failed parse should score 0.0."""
        parsed = ParsedOutput(
            parse_status=ParseStatus.MALFORMED_JSON,
            parse_error="Invalid JSON"
        )
        
        assert parsed.get_completion_score() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
