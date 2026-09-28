from app.serving.decision_engine import IncidentAnalysis, validate_analysis
from app.shared.incident_policy import REQUIRED_OUTPUT_FIELDS, SYSTEM_PROMPT, build_inference_messages, validate_output_schema


def test_training_and_inference_use_the_same_canonical_prompt():
    messages = build_inference_messages(["ERROR database connection failed"])
    assert messages[0]["content"] == SYSTEM_PROMPT
    for field in REQUIRED_OUTPUT_FIELDS:
        assert field in messages[1]["content"]


def test_output_schema_and_normalization_are_strict():
    analysis = IncidentAnalysis(severity="HIGH", disposition="needs_oncall", confidence=0.9, summary="Database is unavailable", suspected_root_cause=None, next_steps=["Inspect database"], ticket_title="Database failure", ticket_body="Investigate database")
    analysis = validate_analysis(analysis, object())
    valid, reason = validate_output_schema(analysis.model_dump())
    assert valid, reason
    assert analysis.severity == "high"
    assert analysis.disposition == "NEEDS_ONCALL"
