"""
Integration test to verify the investigation pipeline works end-to-end
after restoring validate_analysis without severity/disposition overrides.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from app.serving.investigator import InvestigationLoop
from app.serving.decision_engine import get_decision_engine


def test_investigation_loop_calls_validate_analysis():
    """Verify that the investigation loop successfully calls validate_analysis."""
    loop = InvestigationLoop()
    
    # Create mock incident
    mock_incident = Mock()
    mock_incident.id = "test-incident-123"
    mock_incident.source = "api-service"
    mock_incident.environment = "production"
    mock_incident.signature = "Connection timeout error"
    mock_incident.count = 5
    mock_incident.first_seen = datetime(2025, 2, 15, 10, 0, 0)
    mock_incident.last_seen = datetime(2025, 2, 15, 10, 5, 0)
    mock_incident.sample_lines = [
        "2025-02-15T10:00:00Z ERROR Connection timeout after 30s",
        "2025-02-15T10:01:00Z ERROR Connection timeout after 30s",
    ]
    
    # Mock project
    mock_project = Mock()
    mock_project.id = "test-project"
    
    # Test that _parse_final successfully validates analysis
    test_response = """{
        "severity": "HIGH",
        "disposition": "needs_oncall",
        "confidence": 0.92,
        "summary": "API service experiencing connection timeouts",
        "suspected_root_cause": "Network connectivity issue",
        "next_steps": ["Check network latency", "Verify database connections"],
        "ticket_title": "API Service Connection Timeouts",
        "ticket_body": "The API service is experiencing repeated connection timeouts."
    }"""
    
    result = loop._parse_final(test_response, mock_incident)
    
    assert result is not None, "Investigation loop should successfully parse and validate"
    assert result.severity == "high", "Severity should be normalized to lowercase"
    assert result.disposition == "NEEDS_ONCALL", "Disposition should be normalized to uppercase"
    assert result.confidence == 0.92, "Confidence should be preserved"
    assert result.summary == "API service experiencing connection timeouts"
    print("✓ Investigation loop successfully validates analysis")


def test_validate_analysis_does_not_override_predictions():
    """Verify that validate_analysis does minimal normalization only when patterns don't match."""
    from app.serving.decision_engine import validate_analysis, IncidentAnalysis
    
    # Create mock incident WITHOUT critical patterns
    mock_incident = Mock()
    mock_incident.source = "database"
    mock_incident.signature = "Connection warning"  # Not a critical pattern
    mock_incident.sample_lines = [
        "Database query slow",
        "Connection latency detected",
    ]
    
    # Model predicts MEDIUM
    analysis = IncidentAnalysis(
        severity="medium",
        disposition="OBSERVE",
        confidence=0.7,
        summary="Memory usage spike detected",
        suspected_root_cause="Large batch processing job",
        next_steps=["Monitor heap usage", "Review batch size"],
        ticket_title="Memory Usage Spike",
        ticket_body="Heap usage increased during batch processing"
    )
    
    result = validate_analysis(analysis, mock_incident)
    
    # Verify predictions are NOT overridden when patterns don't match
    assert result.severity == "medium", "Should NOT override when patterns don't match"
    assert result.disposition == "OBSERVE", "Should NOT override disposition"
    assert result.confidence == 0.7, "Confidence should be preserved"
    print("✓ validate_analysis does NOT override model predictions")


def test_decision_engine_analyze_incident_integration():
    """Test that decision_engine.analyze_incident works with the fixed validate_analysis."""
    
    # Create mock incident
    mock_incident = Mock()
    mock_incident.id = "integration-test-1"
    mock_incident.source = "payment-service"
    mock_incident.signature = "Database connection pool exhausted"
    mock_incident.sample_lines = [
        "2025-02-15T10:00:00Z ERROR Could not get JDBC Connection",
        "2025-02-15T10:00:01Z ERROR Connection pool exhausted",
    ]
    
    # Mock the model runtime
    with patch('app.serving.decision_engine.get_model_runtime') as mock_runtime:
        mock_session = Mock()
        mock_session.default_model = "test-model"
        mock_session.base_model = "Qwen/Qwen3.5-4B"
        mock_session.active_artifact = None
        
        # Model returns valid JSON
        mock_response = Mock()
        mock_response.content = """{
            "severity": "high",
            "disposition": "NEEDS_ONCALL",
            "confidence": 0.88,
            "summary": "Database connection pool exhaustion in payment service",
            "suspected_root_cause": "Connection leak or insufficient pool size",
            "next_steps": [
                "Check connection pool configuration",
                "Review recent deployments",
                "Monitor active connections"
            ],
            "ticket_title": "Payment Service DB Pool Exhausted",
            "ticket_body": "The payment service database connection pool has been exhausted. This is causing service degradation."
        }"""
        
        mock_session.complete.return_value = mock_response
        mock_runtime.return_value.resolve_project_model.return_value = mock_session
        
        # Call analyze_incident
        engine = get_decision_engine()
        result = engine.analyze_incident(mock_incident, project=None)
        
        # Verify result
        assert result is not None, "Should return valid analysis"
        assert result.severity == "high"
        assert result.disposition == "NEEDS_ONCALL"
        assert result.confidence == 0.88
        assert len(result.next_steps) == 3
        print("✓ decision_engine.analyze_incident works correctly")


if __name__ == "__main__":
    test_investigation_loop_calls_validate_analysis()
    test_validate_analysis_does_not_override_predictions()
    test_decision_engine_analyze_incident_integration()
    print("\n✓ All integration tests passed!")
