#!/usr/bin/env python3
"""
Test the local experiment harness to verify train/eval separation.

These are unit tests that verify the harness behavior WITHOUT
running expensive QLoRA training.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

def test_evaluate_rejects_training_data():
    """Test that evaluate.py refuses to use training data."""
    print("\nTest 1: evaluate.py rejects training data...")
    
    import subprocess
    
    # Try to evaluate with train.jsonl (should fail)
    result = subprocess.run(
        [
            sys.executable,
            "evaluate.py",
            "--adapter", "fake/adapter",
            "--logs", "data/train.jsonl",  # WRONG! This is training data
            "--output", "fake/output"
        ],
        capture_output=True,
        cwd=Path(__file__).parent
    )
    
    # Should exit with error
    assert result.returncode != 0, "evaluate.py should reject train.jsonl"
    assert b"train.jsonl" in result.stdout, "Should mention train.jsonl in error"
    print("  ✓ evaluate.py correctly rejects training data")


def test_run_experiment_rejects_same_files():
    """Test that run_experiment.py refuses same file for train and eval."""
    print("\nTest 2: run_experiment.py rejects same train/eval file...")
    
    import subprocess
    
    # Try to use same file for both (should fail)
    result = subprocess.run(
        [
            sys.executable,
            "run_experiment.py",
            "--dataset", "data/train.jsonl",
            "--evaluation-log", "data/train.jsonl",  # WRONG! Same as training
        ],
        capture_output=True,
        cwd=Path(__file__).parent
    )
    
    # Should exit with error
    assert result.returncode != 0, "run_experiment.py should reject same file"
    assert b"same" in result.stdout.lower() or b"same" in result.stderr.lower(), \
        "Should mention files are the same"
    print("  ✓ run_experiment.py correctly rejects same files")


def test_evaluate_parses_raw_logs():
    """Test that evaluate.py can parse raw log lines."""
    print("\nTest 3: evaluate.py parses raw log format...")
    
    # Import parse function
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate import parse_log_line
    
    # Test structured log
    line = "2026-09-25T10:30:00.123Z ERROR [payment-api] connection timeout"
    parsed = parse_log_line(line)
    
    assert parsed["level"] == "ERROR"
    assert parsed["service"] == "payment-api"
    assert "timeout" in parsed["message"]
    assert parsed["raw"] == line
    print("  ✓ Parses structured logs correctly")
    
    # Test unstructured log
    line2 = "Some unstructured log message"
    parsed2 = parse_log_line(line2)
    
    assert parsed2["message"] == line2
    assert parsed2["raw"] == line2
    print("  ✓ Handles unstructured logs gracefully")


def test_evaluate_clusters_logs():
    """Test that evaluate.py clusters logs by service."""
    print("\nTest 4: evaluate.py clusters logs by service...")
    
    from evaluate import cluster_logs_by_service, parse_log_line
    
    logs = [
        "2026-09-25T10:30:00Z ERROR [payment-api] error 1",
        "2026-09-25T10:30:01Z ERROR [payment-api] error 2",
        "2026-09-25T10:30:02Z ERROR [user-service] error 3",
    ]
    
    parsed = [parse_log_line(line) for line in logs]
    clusters = cluster_logs_by_service(parsed)
    
    assert "payment-api" in clusters
    assert "user-service" in clusters
    assert len(clusters["payment-api"]) == 2
    assert len(clusters["user-service"]) == 1
    print("  ✓ Clusters logs by service correctly")


def test_train_only_reads_training_data():
    """Test that train.py only references training data."""
    print("\nTest 5: train.py only references training data...")
    
    # Check that train.py doesn't mention evaluation
    train_code = (Path(__file__).parent / "train.py").read_text()
    
    # Should not contain "evaluation"
    assert "evaluation" not in train_code.lower(), \
        "train.py should not reference evaluation data"
    
    # Should reference training dataset
    assert "--dataset" in train_code
    print("  ✓ train.py only handles training data")


def test_cli_arguments_match_docs():
    """Test that CLI arguments match what's documented."""
    print("\nTest 6: CLI arguments match documentation...")
    
    import subprocess
    
    # Check train.py help
    result = subprocess.run(
        [sys.executable, "train.py", "--help"],
        capture_output=True,
        cwd=Path(__file__).parent
    )
    assert b"--dataset" in result.stdout
    assert b"--config" in result.stdout
    print("  ✓ train.py has correct CLI args")
    
    # Check evaluate.py help
    result = subprocess.run(
        [sys.executable, "evaluate.py", "--help"],
        capture_output=True,
        cwd=Path(__file__).parent
    )
    assert b"--logs" in result.stdout
    assert b"--adapter" in result.stdout
    assert b"--ground-truth" in result.stdout
    print("  ✓ evaluate.py has correct CLI args")
    
    # Check run_experiment.py help
    result = subprocess.run(
        [sys.executable, "run_experiment.py", "--help"],
        capture_output=True,
        cwd=Path(__file__).parent
    )
    assert b"--dataset" in result.stdout
    assert b"--evaluation-log" in result.stdout
    assert b"--ground-truth" in result.stdout
    print("  ✓ run_experiment.py has correct CLI args")


def test_no_aws_imports():
    """Test that evaluation code doesn't import AWS libraries."""
    print("\nTest 7: No AWS imports in local code...")
    
    for file in ["train.py", "evaluate.py", "validate_dataset.py", "metrics.py"]:
        code = (Path(__file__).parent / file).read_text()
        
        # Should not import AWS libraries
        forbidden = ["boto3", "botocore", "import s3", "import ec2"]
        for lib in forbidden:
            assert lib not in code.lower(), \
                f"{file} should not import {lib}"
    
    print("  ✓ No AWS imports found")


def test_no_database_imports():
    """Test that evaluation code doesn't connect to databases."""
    print("\nTest 8: No database connections in local code...")
    
    for file in ["evaluate.py", "metrics.py"]:
        code = (Path(__file__).parent / file).read_text()
        
        # Should not import database libraries
        forbidden = ["psycopg", "sqlalchemy", "import database"]
        for lib in forbidden:
            assert lib not in code.lower(), \
                f"{file} should not import {lib}"
    
    print("  ✓ No database imports found")


def main():
    print("=" * 80)
    print("LOCAL EXPERIMENT HARNESS TESTS")
    print("=" * 80)
    
    tests = [
        test_evaluate_rejects_training_data,
        test_run_experiment_rejects_same_files,
        test_evaluate_parses_raw_logs,
        test_evaluate_clusters_logs,
        test_train_only_reads_training_data,
        test_cli_arguments_match_docs,
        test_no_aws_imports,
        test_no_database_imports,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✅ All harness tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
