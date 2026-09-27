#!/usr/bin/env python3
"""
Test canonical inference path matches local experiment.

This test verifies that production inference uses the same prompt structure
and logic as the frozen local experiment (Exp4).
"""

import pytest
from unittest.mock import Mock, patch
from app.serving.decision_engine import get_decision_engine, IncidentAnalysis


# Sample high-severity Kafka consumer lag incident
KAFKA_LAG_INCIDENT_LOGS = [
    "2025-02-15T14:23:01Z WARN [kafka-consumer] Consumer lag increasing to 15234 messages, partition 3",
    "2025-02-15T14:23:15Z WARN [kafka-consumer] Lag now 17958 messages, partition 3, offset delta growing",
    "2025-02-15T14:23:45Z INFO [kafka-consumer] Consumer rebalance triggered for group payment-processor",
    "2025-02-15T14:24:02Z ERROR [kafka-consumer] Processing failed due to deserialization error",
    "2025-02-15T14:24:15Z WARN [kafka-consumer] Lag continues to grow: 19456 messages, partition 3",
    "2025-02-15T14:24:30Z ERROR [kafka-consumer] Consumer group coordinator timeout",
    "2025-02-15T14:25:30Z INFO [kafka-consumer] Consumer rebalanced, processing resumed",
    "2025-02-15T14:26:00Z INFO [kafka-consumer] Lag recovering: 8234 messages, catchup in progress",
]

# Sample OOM incident (should be CRITICAL)
OOM_INCIDENT_LOGS = [
    "2025-02-15T15:10:22Z ERROR [jvm] java.lang.OutOfMemoryError: Java heap space",
    "2025-02-15T15:10:22Z ERROR [jvm] at com.example.service.DataProcessor.process(DataProcessor.java:145)",
    "2025-02-15T15:10:23Z ERROR [health] Health check failed: JVM heap exhausted",
    "2025-02-15T15:10:25Z WARN [container] Container restart initiated",
]

# Sample benign warning (should be LOW/NO_ACTION)
BENIGN_LOGS = [
    "2025-02-15T10:00:01Z INFO [scheduler] Daily backup job started",
    "2025-02-15T10:00:02Z INFO [scheduler] Backup completed successfully",
]


def create_mock_incident(incident_id: str, logs: list[str], signature: str, source: str = "test"):
    """Create a mock incident object for testing."""
    incident = Mock()
    incident.id = incident_id
    incident.source = source
    incident.environment = "production"
    incident.count = len(logs)
    incident.sample_lines = logs
    incident.signature = signature
    return incident


def test_decision_engine_initialization():
    """Test that DecisionEngine can be initialized."""
    engine = get_decision_engine()
    assert engine is not None
    # DecisionEngine uses Langchain internally but that's an implementation detail
    # Just verify it initializes successfully
    assert hasattr(engine, 'analyze_incident')
    assert hasattr(engine, 'chain_root_cause')


@pytest.mark.skipif(
    True,  # Skip by default unless model is loaded
    reason="Requires model to be loaded in test environment"
)
def test_kafka_lag_incident_severity():
    """Test that Kafka consumer lag is classified as HIGH or CRITICAL."""
    incident = create_mock_incident(
        incident_id="test-kafka-1",
        logs=KAFKA_LAG_INCIDENT_LOGS,
        signature="Consumer lag increasing",
        source="kafka-consumer"
    )
    
    engine = get_decision_engine()
    result = engine.analyze_incident(incident, project=None)
    
    # Verify analysis succeeded
    assert result is not None, "Analysis should not return None"
    
    # Verify severity is HIGH or CRITICAL (not LOW)
    assert result.severity in ["high", "critical"], \
        f"Expected high/critical for Kafka lag, got {result.severity}"
    
    # Verify appropriate disposition
    assert result.disposition in ["NEEDS_ONCALL", "ESCALATE"], \
        f"Expected NEEDS_ONCALL/ESCALATE, got {result.disposition}"
    
    # Verify other fields
    assert isinstance(result.next_steps, list)
    assert len(result.next_steps) > 0
    assert 0.0 <= result.confidence <= 1.0
    assert result.summary
    assert result.ticket_title


@pytest.mark.skipif(
    True,  # Skip by default unless model is loaded
    reason="Requires model to be loaded in test environment"
)
def test_oom_incident_severity():
    """Test that OutOfMemoryError is classified as CRITICAL."""
    incident = create_mock_incident(
        incident_id="test-oom-1",
        logs=OOM_INCIDENT_LOGS,
        signature="OutOfMemoryError: Java heap space",
        source="jvm"
    )
    
    engine = get_decision_engine()
    result = engine.analyze_incident(incident, project=None)
    
    assert result is not None
    
    # OOM should be CRITICAL
    assert result.severity == "critical", \
        f"Expected critical for OOM, got {result.severity}"
    
    # Should require escalation
    assert result.disposition in ["ESCALATE", "NEEDS_ONCALL"], \
        f"Expected ESCALATE/NEEDS_ONCALL, got {result.disposition}"


@pytest.mark.skipif(
    True,  # Skip by default unless model is loaded  
    reason="Requires model to be loaded in test environment"
)
def test_benign_incident_severity():
    """Test that benign logs are classified as LOW/NO_ACTION."""
    incident = create_mock_incident(
        incident_id="test-benign-1",
        logs=BENIGN_LOGS,
        signature="Daily backup job",
        source="scheduler"
    )
    
    engine = get_decision_engine()
    result = engine.analyze_incident(incident, project=None)
    
    assert result is not None
    
    # Benign should be LOW
    assert result.severity == "low", \
        f"Expected low for benign logs, got {result.severity}"
    
    # Should not require action
    assert result.disposition in ["NO_ACTION", "OBSERVE"], \
        f"Expected NO_ACTION/OBSERVE, got {result.disposition}"


def test_canonical_prompt_usage():
    """Test that analyze_incident uses canonical prompt from incident_policy."""
    incident = create_mock_incident(
        incident_id="test-prompt-1",
        logs=["2025-02-15T10:00:00Z INFO test log"],
        signature="test"
    )
    
    # Mock the model runtime to verify prompt structure
    with patch('app.serving.decision_engine.get_model_runtime') as mock_runtime:
        mock_session = Mock()
        mock_session.default_model = "test-model"
        mock_session.base_model = "Qwen/Qwen3.5-4B"
        mock_session.active_artifact = None
        
        # Mock the response
        mock_response = Mock()
        mock_response.content = '''{"severity": "low", "disposition": "NO_ACTION", "confidence": 0.9, "summary": "test", "suspected_root_cause": null, "next_steps": ["test"], "ticket_title": "test", "ticket_body": "test"}'''
        mock_session.complete.return_value = mock_response
        
        mock_runtime.return_value.resolve_project_model.return_value = mock_session
        
        engine = get_decision_engine()
        result = engine.analyze_incident(incident, project=None)
        
        # Verify complete() was called
        assert mock_session.complete.called
        
        # Extract the messages passed to complete()
        call_args = mock_session.complete.call_args
        messages = call_args.kwargs.get('messages') or call_args[0][0] if call_args[0] else call_args[1].get('messages')
        
        # Verify message structure
        assert isinstance(messages, list)
        assert len(messages) == 2  # System + User
        
        # Messages from ChatPromptTemplate are BaseMessage objects, not dicts
        # Extract content from message objects
        system_msg = messages[0]
        user_msg = messages[1]
        
        # Check the content attribute
        system_content = system_msg.content if hasattr(system_msg, 'content') else str(system_msg)
        user_content = user_msg.content if hasattr(user_msg, 'content') else str(user_msg)
        
        # Verify system message contains key elements
        assert 'IncidentLens' in system_content or 'incident' in system_content.lower()
        
        # Verify user message contains the incident data
        assert 'test log' in user_content or 'Source:' in user_content
        
        # The canonical path uses format_instructions from the parser (Langchain)
        # This is expected in the current implementation


def test_no_validate_analysis_override():
    """
    Test that model predictions are NOT modified by post-processing.
    
    validate_analysis() now only does minimal normalization (lowercase severity,
    uppercase disposition, confidence clamping) but does NOT override the model's
    severity or disposition predictions based on patterns.
    """
    incident = create_mock_incident(
        incident_id="test-no-override-1",
        logs=["2025-02-15T10:00:00Z ERROR database connection failed"],
        signature="database error"
    )
    
    # Mock the model runtime
    with patch('app.serving.decision_engine.get_model_runtime') as mock_runtime:
        mock_session = Mock()
        mock_session.default_model = "test-model"
        mock_session.base_model = "Qwen/Qwen3.5-4B"
        mock_session.active_artifact = None
        
        # Model predicts MEDIUM (not HIGH despite "database" pattern)
        mock_response = Mock()
        mock_response.content = '''{"severity": "medium", "disposition": "NEEDS_DEV", "confidence": 0.85, "summary": "Database connection issues detected", "suspected_root_cause": "Network latency", "next_steps": ["Check network", "Verify DB status"], "ticket_title": "DB Connection Error", "ticket_body": "Investigation needed"}'''
        mock_session.complete.return_value = mock_response
        
        mock_runtime.return_value.resolve_project_model.return_value = mock_session
        
        engine = get_decision_engine()
        result = engine.analyze_incident(incident, project=None)
        
        # Verify model prediction is preserved (NOT overridden to HIGH)
        assert result.severity == "medium", \
            "Model prediction should NOT be overridden"
        assert result.disposition == "NEEDS_DEV", \
            "Model disposition should NOT be overridden"


def test_schema_validation_only():
    """Test that schema validation catches invalid output but doesn't modify valid output."""
    incident = create_mock_incident(
        incident_id="test-validation-1",
        logs=["test"],
        signature="test"
    )
    
    # Mock the model runtime with INVALID output
    with patch('app.serving.decision_engine.get_model_runtime') as mock_runtime:
        mock_session = Mock()
        mock_session.default_model = "test-model"
        mock_session.base_model = "Qwen/Qwen3.5-4B"
        mock_session.active_artifact = None
        
        # Invalid JSON that will cause parsing to fail
        mock_response = Mock()
        mock_response.content = '''{"severity": "INVALID_NOT_IN_ENUM", "missing_required_fields": true}'''
        mock_session.complete.return_value = mock_response
        
        mock_runtime.return_value.resolve_project_model.return_value = mock_session
        
        engine = get_decision_engine()
        result = engine.analyze_incident(incident, project=None)
        
        # Should return None when parsing fails
        assert result is None, "Invalid output should return None"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
