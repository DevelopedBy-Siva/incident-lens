"""
Test that training data schema matches inference expectations.
"""

import json
from pathlib import Path

import pytest


def test_training_output_schema_matches_inference():
    """Verify training data expected_output matches what inference requests."""
    # Load a sample from the training dataset
    dataset_path = Path(__file__).parent.parent.parent / "data" / "dataset_v2.jsonl"
    
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    
    with dataset_path.open() as f:
        first_line = f.readline()
        record = json.loads(first_line)
    
    # Expected fields by inference (from investigator.py)
    inference_fields = {
        "severity", "disposition", "confidence", "summary",
        "suspected_root_cause", "next_steps", "ticket_title", "ticket_body"
    }
    
    # Check training data has all fields
    assert "expected_output" in record, "Training record missing 'expected_output'"
    output_fields = set(record["expected_output"].keys())
    
    missing = inference_fields - output_fields
    assert not missing, f"Training data missing fields that inference expects: {missing}"
    
    # Extra fields are OK (they'll just be ignored by inference)
    # but log them for awareness
    extra = output_fields - inference_fields
    if extra:
        print(f"Note: Training data has extra fields not used by inference: {extra}")


def test_training_data_has_no_old_schema():
    """Verify training data doesn't use old 'output' key."""
    dataset_path = Path(__file__).parent.parent.parent / "data" / "dataset_v2.jsonl"
    
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    
    with dataset_path.open() as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            
            # Should use 'expected_output', not 'output'
            assert "expected_output" in record, f"Line {i}: missing 'expected_output'"
            assert "output" not in record, f"Line {i}: uses old 'output' key instead of 'expected_output'"
            
            if i >= 10:  # Check first 10 records
                break


def test_severity_values_are_valid():
    """Verify all severity values match the expected vocabulary.
    
    NOTE: 'benign' is NOT a valid severity value. Benign/no-incident examples
    should use severity='low' with disposition='NO_ACTION'.
    """
    dataset_path = Path(__file__).parent.parent.parent / "data" / "dataset_v2.jsonl"
    
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    
    valid_severities = {"low", "medium", "high", "critical"}
    
    with dataset_path.open() as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            severity = record["expected_output"]["severity"]
            
            # Explicitly reject 'benign' severity
            assert severity != "benign", (
                f"Line {i}: 'benign' is not a valid severity value. "
                f"Use severity='low' with disposition='NO_ACTION' for benign examples."
            )
            
            assert severity in valid_severities, (
                f"Line {i}: invalid severity '{severity}'. "
                f"Must be one of: {valid_severities}"
            )


def test_disposition_values_are_valid():
    """Verify all disposition values match the expected vocabulary."""
    dataset_path = Path(__file__).parent.parent.parent / "data" / "dataset_v2.jsonl"
    
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    
    valid_dispositions = {"NO_ACTION", "OBSERVE", "NEEDS_DEV", "NEEDS_ONCALL", "ESCALATE"}
    
    with dataset_path.open() as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            disposition = record["expected_output"]["disposition"]
            
            assert disposition in valid_dispositions, (
                f"Line {i}: invalid disposition '{disposition}'. "
                f"Must be one of: {valid_dispositions}"
            )


def test_confidence_is_valid_range():
    """Verify confidence values are in [0, 1] range."""
    dataset_path = Path(__file__).parent.parent.parent / "data" / "dataset_v2.jsonl"
    
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    
    with dataset_path.open() as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            confidence = record["expected_output"]["confidence"]
            
            assert isinstance(confidence, (int, float)), (
                f"Line {i}: confidence must be numeric, got {type(confidence)}"
            )
            assert 0.0 <= confidence <= 1.0, (
                f"Line {i}: confidence {confidence} out of range [0, 1]"
            )


def test_next_steps_is_list():
    """Verify next_steps is always a list."""
    dataset_path = Path(__file__).parent.parent.parent / "data" / "dataset_v2.jsonl"
    
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    
    with dataset_path.open() as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            next_steps = record["expected_output"]["next_steps"]
            
            assert isinstance(next_steps, list), (
                f"Line {i}: next_steps must be a list, got {type(next_steps)}"
            )
            
            for step in next_steps:
                assert isinstance(step, str), (
                    f"Line {i}: next_steps items must be strings"
                )


def test_required_fields_non_empty():
    """Verify required fields (severity, disposition, summary) are non-empty."""
    dataset_path = Path(__file__).parent.parent.parent / "data" / "dataset_v2.jsonl"
    
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    
    with dataset_path.open() as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            output = record["expected_output"]
            
            assert output["severity"], f"Line {i}: severity is empty"
            assert output["disposition"], f"Line {i}: disposition is empty"
            assert output["summary"], f"Line {i}: summary is empty"


def test_training_prompt_matches_inference_prompt():
    """Verify training system prompt teaches the same schema as inference expects."""
    from app.training.lora_trainer import TransformersPeftTrainingEngine
    from app.serving.investigator import SYSTEM_PROMPT
    
    training_prompt = TransformersPeftTrainingEngine.system_prompt
    inference_prompt = SYSTEM_PROMPT
    
    # Both should mention all 8 output fields
    required_fields = [
        "severity", "disposition", "confidence", "summary",
        "suspected_root_cause", "next_steps", "ticket_title", "ticket_body"
    ]
    
    for field in required_fields:
        assert field in training_prompt.lower(), (
            f"Training prompt doesn't mention '{field}'"
        )
        assert field in inference_prompt.lower(), (
            f"Inference prompt doesn't mention '{field}'"
        )


def test_no_incident_type_in_expected_output():
    """Verify training data doesn't include incident_type in expected_output."""
    dataset_path = Path(__file__).parent.parent.parent / "data" / "dataset_v2.jsonl"
    
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    
    with dataset_path.open() as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            output = record["expected_output"]
            
            # incident_type can be in metadata for tracking, but NOT in expected_output
            assert "incident_type" not in output, (
                f"Line {i}: incident_type should not be in expected_output "
                f"(inference doesn't request it)"
            )
            
            if i >= 10:
                break


def test_actionable_incidents_have_required_details():
    """Verify actionable incidents (not NO_ACTION) have non-empty detail fields."""
    dataset_path = Path(__file__).parent.parent.parent / "data" / "dataset_v2.jsonl"
    
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    
    actionable_dispositions = {"OBSERVE", "NEEDS_DEV", "NEEDS_ONCALL", "ESCALATE"}
    
    with dataset_path.open() as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            output = record["expected_output"]
            disposition = output["disposition"]
            
            # Only check actionable incidents
            if disposition not in actionable_dispositions:
                continue
            
            # For actionable incidents, require meaningful details
            assert output.get("summary"), (
                f"Line {i}: actionable incident ({disposition}) must have non-empty summary"
            )
            
            assert output.get("suspected_root_cause"), (
                f"Line {i}: actionable incident ({disposition}) should have suspected_root_cause"
            )
            
            assert output.get("next_steps") and len(output["next_steps"]) > 0, (
                f"Line {i}: actionable incident ({disposition}) must have next_steps"
            )
            
            # For incidents requiring tickets (NEEDS_DEV, NEEDS_ONCALL, ESCALATE), require ticket fields
            if disposition in {"NEEDS_DEV", "NEEDS_ONCALL", "ESCALATE"}:
                assert output.get("ticket_title"), (
                    f"Line {i}: incident requiring action ({disposition}) must have ticket_title"
                )
                
                assert output.get("ticket_body"), (
                    f"Line {i}: incident requiring action ({disposition}) must have ticket_body"
                )


def test_benign_no_action_can_have_empty_ticket_fields():
    """Verify benign/no-action examples can legitimately have empty ticket fields."""
    dataset_path = Path(__file__).parent.parent.parent / "data" / "dataset_v2.jsonl"
    
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    
    benign_count = 0
    benign_with_empty_tickets = 0
    
    with dataset_path.open() as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            output = record["expected_output"]
            
            # Identify benign/no-incident examples
            if output["severity"] == "low" and output["disposition"] == "NO_ACTION":
                benign_count += 1
                
                # These CAN have empty ticket fields
                if not output.get("ticket_title") or not output.get("ticket_body"):
                    benign_with_empty_tickets += 1
                
                # But they MUST still have summary
                assert output.get("summary"), (
                    f"Line {i}: even benign examples must have a summary"
                )
    
    # Just informational - benign examples CAN have empty tickets
    print(f"\nBenign/no-action examples: {benign_count}")
    print(f"  With empty ticket fields: {benign_with_empty_tickets} ({100*benign_with_empty_tickets/benign_count:.1f}%)")
