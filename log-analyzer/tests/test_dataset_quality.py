"""
Test training dataset quality and distribution.
"""

import json
from collections import Counter
from pathlib import Path

import pytest


def test_severity_distribution_is_balanced():
    """Verify training dataset has reasonable severity distribution."""
    dataset_path = Path(__file__).parent.parent.parent / "data" / "dataset_v2.jsonl"
    
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    
    severity_counts = Counter()
    
    with dataset_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            severity = record["expected_output"]["severity"]
            severity_counts[severity] += 1
    
    total = sum(severity_counts.values())
    assert total > 0, "Dataset is empty"
    
    # Check all severities are represented
    required_severities = {"benign", "low", "medium", "high", "critical"}
    missing = required_severities - set(severity_counts.keys())
    assert not missing, f"Dataset missing severities: {missing}"
    
    # Check no severity is underrepresented (at least 5% of dataset)
    for severity, count in severity_counts.items():
        pct = 100 * count / total
        assert pct >= 5.0, (
            f"Severity '{severity}' underrepresented: {count} ({pct:.1f}%) "
            f"Should be at least 5% of dataset"
        )
    
    print(f"\nSeverity distribution ({total} total):")
    for severity in ["benign", "low", "medium", "high", "critical"]:
        count = severity_counts[severity]
        pct = 100 * count / total
        print(f"  {severity:10s} {count:4d} ({pct:5.1f}%)")


def test_dataset_has_benign_examples():
    """Verify dataset includes benign/no-incident examples."""
    dataset_path = Path(__file__).parent.parent.parent / "data" / "dataset_v2.jsonl"
    
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    
    benign_count = 0
    total = 0
    
    with dataset_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)
            if record["expected_output"]["severity"] == "benign":
                benign_count += 1
    
    assert benign_count > 0, "Dataset has NO benign examples!"
    
    benign_pct = 100 * benign_count / total
    assert benign_pct >= 10.0, (
        f"Dataset has too few benign examples: {benign_count} ({benign_pct:.1f}%). "
        f"Should be at least 10%"
    )
    
    print(f"\nBenign examples: {benign_count}/{total} ({benign_pct:.1f}%)")


def test_dataset_minimum_size():
    """Verify dataset has minimum viable size for training."""
    dataset_path = Path(__file__).parent.parent.parent / "data" / "dataset_v2.jsonl"
    
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    
    count = 0
    with dataset_path.open() as f:
        for line in f:
            if line.strip():
                count += 1
    
    # Minimum 1000 examples for viable training
    assert count >= 1000, (
        f"Dataset too small: {count} examples. "
        f"Need at least 1000 for viable training"
    )
    
    print(f"\nDataset size: {count} examples")


def test_log_count_distribution():
    """Verify examples have reasonable number of log lines."""
    dataset_path = Path(__file__).parent.parent.parent / "data" / "dataset_v2.jsonl"
    
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    
    log_counts = []
    
    with dataset_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            logs = record["input"]["logs"]
            log_counts.append(len(logs))
    
    assert log_counts, "No examples found"
    
    min_logs = min(log_counts)
    max_logs = max(log_counts)
    avg_logs = sum(log_counts) / len(log_counts)
    
    # Each example should have at least 1 log
    assert min_logs >= 1, f"Found example with {min_logs} logs (need at least 1)"
    
    # Average should be reasonable (not all single-line)
    assert avg_logs >= 1.5, (
        f"Average logs per example too low: {avg_logs:.1f}. "
        f"Examples may lack context"
    )
    
    print(f"\nLogs per example: min={min_logs}, max={max_logs}, avg={avg_logs:.1f}")


def test_incident_type_diversity():
    """Verify dataset covers diverse incident types."""
    dataset_path = Path(__file__).parent.parent.parent / "data" / "dataset_v2.jsonl"
    
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    
    incident_types = set()
    
    with dataset_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            # incident_type is in metadata for tracking
            incident_type = record["input"]["metadata"].get("incident_type", "unknown")
            incident_types.add(incident_type)
    
    # Should have at least 20 different incident types for diversity
    assert len(incident_types) >= 20, (
        f"Dataset lacks incident diversity: only {len(incident_types)} types. "
        f"Need at least 20 for model to generalize"
    )
    
    print(f"\nIncident type diversity: {len(incident_types)} unique types")


def test_summaries_are_meaningful():
    """Verify summaries are not empty and have reasonable length."""
    dataset_path = Path(__file__).parent.parent.parent / "data" / "dataset_v2.jsonl"
    
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    
    summary_lengths = []
    empty_count = 0
    
    with dataset_path.open() as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            summary = record["expected_output"]["summary"]
            
            if not summary or not summary.strip():
                empty_count += 1
            else:
                summary_lengths.append(len(summary))
    
    assert empty_count == 0, f"Found {empty_count} examples with empty summaries"
    
    if summary_lengths:
        avg_length = sum(summary_lengths) / len(summary_lengths)
        min_length = min(summary_lengths)
        
        # Summaries should be at least 20 chars (more than trivial)
        assert min_length >= 20, (
            f"Found summary with only {min_length} chars. "
            f"Summaries should be meaningful (>20 chars)"
        )
        
        print(f"\nSummary lengths: min={min_length}, avg={avg_length:.0f} chars")


def test_next_steps_for_actionable_incidents():
    """Verify incidents requiring action have next_steps."""
    dataset_path = Path(__file__).parent.parent.parent / "data" / "dataset_v2.jsonl"
    
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    
    actionable_dispositions = {"NEEDS_DEV", "NEEDS_ONCALL", "ESCALATE"}
    missing_steps_count = 0
    actionable_count = 0
    
    with dataset_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            output = record["expected_output"]
            
            if output["disposition"] in actionable_dispositions:
                actionable_count += 1
                if not output["next_steps"] or len(output["next_steps"]) == 0:
                    missing_steps_count += 1
    
    # Most actionable incidents should have next_steps
    if actionable_count > 0:
        missing_pct = 100 * missing_steps_count / actionable_count
        assert missing_pct < 20, (
            f"{missing_steps_count}/{actionable_count} ({missing_pct:.1f}%) "
            f"actionable incidents lack next_steps. Should be <20%"
        )


def test_confidence_distribution():
    """Verify confidence values have reasonable distribution."""
    dataset_path = Path(__file__).parent.parent.parent / "data" / "dataset_v2.jsonl"
    
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    
    confidences = []
    
    with dataset_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            confidences.append(record["expected_output"]["confidence"])
    
    assert confidences, "No examples found"
    
    avg_confidence = sum(confidences) / len(confidences)
    min_confidence = min(confidences)
    max_confidence = max(confidences)
    
    # Average confidence should be reasonable (not all 1.0 or all 0.5)
    assert 0.6 <= avg_confidence <= 0.95, (
        f"Average confidence seems unrealistic: {avg_confidence:.2f}. "
        f"Expected range: 0.6-0.95"
    )
    
    # Should have some variation (not all the same value)
    unique_confidences = len(set(confidences))
    assert unique_confidences >= 5, (
        f"Confidence values lack variation: only {unique_confidences} unique values"
    )
    
    print(f"\nConfidence: min={min_confidence:.2f}, max={max_confidence:.2f}, avg={avg_confidence:.2f}")


def test_kafka_examples_exist():
    """Verify dataset includes Kafka/queue-related examples."""
    dataset_path = Path(__file__).parent.parent.parent / "data" / "dataset_v2.jsonl"
    
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    
    kafka_count = 0
    queue_count = 0
    
    with dataset_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            
            # Check logs or incident type for kafka/queue keywords
            logs_text = " ".join(record["input"]["logs"]).lower()
            incident_type = record["input"]["metadata"].get("incident_type", "").lower()
            
            if "kafka" in logs_text or "kafka" in incident_type:
                kafka_count += 1
            if "queue" in logs_text or "queue" in incident_type:
                queue_count += 1
    
    assert kafka_count > 0, "Dataset has NO Kafka examples!"
    assert queue_count > 0, "Dataset has NO queue examples!"
    
    print(f"\nKafka examples: {kafka_count}")
    print(f"Queue examples: {queue_count}")
