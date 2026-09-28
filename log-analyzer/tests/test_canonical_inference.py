from app.serving.decision_engine import IncidentAnalysis, validate_analysis
from app.shared.incident_policy import REQUIRED_OUTPUT_FIELDS, SYSTEM_PROMPT, build_inference_messages, validate_output_schema


def test_training_and_inference_use_the_same_canonical_prompt():
    messages = build_inference_messages(["ERROR database connection failed"])
    assert messages[0]["content"] == SYSTEM_PROMPT
    for field in REQUIRED_OUTPUT_FIELDS:
        assert field in messages[1]["content"]


def test_canonical_prompt_includes_observed_context_not_training_labels():
    messages = build_inference_messages(
        {
            "logs": ["ERROR k8s event Pod reporting-123 container reporting OOMKilled"],
            "service": "reporting",
            "environment": "prod",
            "count": 2,
            "metadata": {"pod": "reporting-123", "incident_type": "memory_pressure_oom"},
            "related_incidents": [{"source": "reporting", "count": 1, "severity": "critical"}],
        }
    )
    prompt = messages[1]["content"]
    assert '"observed_event_count":2' in prompt
    assert '"pod":"reporting-123"' in prompt
    assert '"related_incidents"' in prompt
    assert "memory_pressure_oom" not in prompt


def test_output_schema_and_normalization_are_strict():
    analysis = IncidentAnalysis(severity="HIGH", disposition="needs_oncall", confidence=0.9, summary="Database is unavailable", suspected_root_cause=None, next_steps=["Inspect database"], ticket_title="Database failure", ticket_body="Investigate database")
    analysis = validate_analysis(analysis, object())
    valid, reason = validate_output_schema(analysis.model_dump())
    assert valid, reason
    assert analysis.severity == "high"
    assert analysis.disposition == "NEEDS_ONCALL"
