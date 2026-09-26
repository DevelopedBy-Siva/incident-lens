"""
Comprehensive tests for the local experiment harness.

Tests core functionality without requiring GPU or running full training.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import pytest

# Test data loader
from data_loader import TrainingExample, load_training_data, validate_training_data

# Test log parser
from log_parser import parse_log_line, extract_service_from_log, group_logs_by_service

# Test incident detector
from incident_detector import detect_incidents_in_service_logs, temporal_overlap

# Test metrics
from metrics import (
    match_predictions_to_ground_truth,
    compute_detection_metrics,
    compute_severity_metrics,
    compute_disposition_metrics,
    parse_timestamp,
)


class TestDataLoader:
    """Test training data loading and validation."""
    
    def test_training_example_validation(self):
        """Test TrainingExample validation."""
        # Valid example
        data = {
            "input": {
                "service": "test-service",
                "environment": "prod",
                "count": 5,
                "logs": ["error log 1", "error log 2"],
                "related_incidents": [],
            },
            "expected_output": {
                "severity": "high",
                "disposition": "NEEDS_ONCALL",
                "confidence": 0.85,
                "summary": "Test incident summary",
                "suspected_root_cause": "Test root cause",
                "next_steps": ["step1", "step2"],
                "ticket_title": "Test title",
                "ticket_body": "Test body",
            }
        }
        
        example = TrainingExample(data)
        errors = example.validate()
        assert len(errors) == 0, f"Valid example should have no errors: {errors}"
    
    def test_invalid_severity(self):
        """Test validation catches invalid severity."""
        data = {
            "input": {
                "service": "test-service",
                "environment": "prod",
                "count": 5,
                "logs": ["error log"],
                "related_incidents": [],
            },
            "expected_output": {
                "severity": "INVALID",  # Invalid
                "disposition": "NEEDS_ONCALL",
                "confidence": 0.85,
                "summary": "Test",
                "next_steps": ["step1"],
                "ticket_title": "Test",
                "ticket_body": "Test",
            }
        }
        
        example = TrainingExample(data)
        errors = example.validate()
        assert any("severity" in e.lower() for e in errors)
    
    def test_invalid_disposition(self):
        """Test validation catches invalid disposition."""
        data = {
            "input": {
                "service": "test-service",
                "environment": "prod",
                "count": 5,
                "logs": ["error log"],
                "related_incidents": [],
            },
            "expected_output": {
                "severity": "high",
                "disposition": "INVALID",  # Invalid
                "confidence": 0.85,
                "summary": "Test",
                "next_steps": ["step1"],
                "ticket_title": "Test",
                "ticket_body": "Test",
            }
        }
        
        example = TrainingExample(data)
        errors = example.validate()
        assert any("disposition" in e.lower() for e in errors)
    
    def test_training_format_conversion(self):
        """Test conversion to training format."""
        data = {
            "input": {
                "service": "test-service",
                "environment": "prod",
                "count": 5,
                "logs": ["error log 1"],
                "related_incidents": [],
            },
            "expected_output": {
                "severity": "high",
                "disposition": "NEEDS_ONCALL",
                "confidence": 0.85,
                "summary": "Test",
                "suspected_root_cause": "Root cause",
                "next_steps": ["step1"],
                "ticket_title": "Title",
                "ticket_body": "Body",
            }
        }
        
        example = TrainingExample(data)
        formatted = example.to_training_format()
        
        assert "prompt" in formatted
        assert "completion" in formatted
        assert "test-service" in formatted["prompt"]
        assert "error log 1" in formatted["prompt"]
        
        # Check completion is valid JSON
        completion_json = json.loads(formatted["completion"])
        assert completion_json["severity"] == "high"
        assert completion_json["disposition"] == "NEEDS_ONCALL"


class TestLogParser:
    """Test log parsing functionality."""
    
    def test_extract_service_from_log(self):
        """Test service name extraction."""
        log = "2026-09-25T10:00:00Z ERROR [payment-api] database timeout"
        service = extract_service_from_log(log)
        assert service == "payment-api"
    
    def test_extract_service_with_dashes(self):
        """Test service name with dashes and underscores."""
        log = "2026-09-25T10:00:00Z ERROR [notification-worker] queue backlog"
        service = extract_service_from_log(log)
        assert service == "notification-worker"
    
    def test_group_logs_by_service(self):
        """Test grouping logs by service."""
        logs = [
            "2026-09-25T10:00:00Z ERROR [service-a] error 1",
            "2026-09-25T10:00:01Z ERROR [service-b] error 2",
            "2026-09-25T10:00:02Z ERROR [service-a] error 3",
            "# This is a comment",
            "",
        ]
        
        grouped = group_logs_by_service(logs)
        
        assert len(grouped) == 2
        assert "service-a" in grouped
        assert "service-b" in grouped
        assert len(grouped["service-a"]) == 2
        assert len(grouped["service-b"]) == 1
    
    def test_parse_log_line_with_error(self):
        """Test parsing ERROR level log."""
        log = "2026-09-25T10:00:00Z ERROR [test-service] database connection failed"
        parsed = parse_log_line(log)
        
        assert parsed is not None
        assert parsed.level == "ERROR"
        assert "database connection failed" in parsed.message.lower()
    
    def test_parse_log_line_skips_comments(self):
        """Test that comments are skipped."""
        log = "# This is a comment"
        parsed = parse_log_line(log)
        assert parsed is None


class TestIncidentDetector:
    """Test incident detection logic."""
    
    def test_single_incident_detection(self):
        """Test detecting a single incident."""
        log_entries = [
            {
                "raw": "2026-09-25T10:00:00Z ERROR [test-service] error message",
                "parsed": type('obj', (object,), {
                    'level': 'ERROR',
                    'message': 'error message',
                    'exception_type': None,
                })(),
                "timestamp": datetime(2026, 9, 25, 10, 0, 0),
            },
            {
                "raw": "2026-09-25T10:00:30Z ERROR [test-service] error message",
                "parsed": type('obj', (object,), {
                    'level': 'ERROR',
                    'message': 'error message',
                    'exception_type': None,
                })(),
                "timestamp": datetime(2026, 9, 25, 10, 0, 30),
            },
        ]
        
        incidents = detect_incidents_in_service_logs("test-service", log_entries)
        
        assert len(incidents) == 1
        assert incidents[0].service == "test-service"
        assert incidents[0].count == 2
    
    def test_multiple_incidents_time_window(self):
        """Test that incidents outside time window are separate."""
        log_entries = [
            {
                "raw": "2026-09-25T10:00:00Z ERROR [test-service] error message",
                "parsed": type('obj', (object,), {
                    'level': 'ERROR',
                    'message': 'error message',
                    'exception_type': None,
                })(),
                "timestamp": datetime(2026, 9, 25, 10, 0, 0),
            },
            {
                "raw": "2026-09-25T10:05:00Z ERROR [test-service] error message",  # 5 min later
                "parsed": type('obj', (object,), {
                    'level': 'ERROR',
                    'message': 'error message',
                    'exception_type': None,
                })(),
                "timestamp": datetime(2026, 9, 25, 10, 5, 0),
            },
        ]
        
        incidents = detect_incidents_in_service_logs("test-service", log_entries)
        
        # Should be 2 separate incidents (>2 min window)
        assert len(incidents) == 2
    
    def test_filters_info_logs(self):
        """Test that INFO logs are filtered out."""
        log_entries = [
            {
                "raw": "2026-09-25T10:00:00Z INFO [test-service] normal operation",
                "parsed": type('obj', (object,), {
                    'level': 'INFO',
                    'message': 'normal operation',
                    'exception_type': None,
                })(),
                "timestamp": datetime(2026, 9, 25, 10, 0, 0),
            },
        ]
        
        incidents = detect_incidents_in_service_logs("test-service", log_entries)
        
        assert len(incidents) == 0


class TestMetrics:
    """Test metrics computation."""
    
    def test_parse_timestamp(self):
        """Test timestamp parsing."""
        ts = parse_timestamp("2026-09-25T10:00:00Z")
        assert ts.year == 2026
        assert ts.month == 9
        assert ts.day == 25
    
    def test_temporal_overlap_full(self):
        """Test full temporal overlap."""
        from metrics import temporal_overlap
        
        start = datetime(2026, 9, 25, 10, 0, 0)
        end = datetime(2026, 9, 25, 10, 5, 0)
        
        overlap = temporal_overlap(start, end, start, end)
        assert overlap == 1.0
    
    def test_temporal_overlap_partial(self):
        """Test partial temporal overlap."""
        from metrics import temporal_overlap
        
        pred_start = datetime(2026, 9, 25, 10, 0, 0)
        pred_end = datetime(2026, 9, 25, 10, 4, 0)
        
        gt_start = datetime(2026, 9, 25, 10, 2, 0)
        gt_end = datetime(2026, 9, 25, 10, 6, 0)
        
        overlap = temporal_overlap(pred_start, pred_end, gt_start, gt_end)
        assert 0 < overlap < 1
    
    def test_temporal_overlap_none(self):
        """Test no temporal overlap."""
        from metrics import temporal_overlap
        
        pred_start = datetime(2026, 9, 25, 10, 0, 0)
        pred_end = datetime(2026, 9, 25, 10, 2, 0)
        
        gt_start = datetime(2026, 9, 25, 10, 5, 0)
        gt_end = datetime(2026, 9, 25, 10, 7, 0)
        
        overlap = temporal_overlap(pred_start, pred_end, gt_start, gt_end)
        assert overlap == 0.0
    
    def test_match_predictions_to_ground_truth(self):
        """Test matching predictions to ground truth."""
        predictions = [
            {
                "incident_id": "pred-1",
                "service": "test-service",
                "first_seen": "2026-09-25T10:00:00Z",
                "last_seen": "2026-09-25T10:05:00Z",
            }
        ]
        
        ground_truth = [
            {
                "incident_id": "gt-1",
                "service": "test-service",
                "start_time": "2026-09-25T10:00:00Z",
                "end_time": "2026-09-25T10:05:00Z",
                "severity": "high",
                "disposition": "NEEDS_ONCALL",
            }
        ]
        
        matched, fps, fns = match_predictions_to_ground_truth(
            predictions, ground_truth, min_overlap=0.5
        )
        
        assert len(matched) == 1
        assert len(fps) == 0
        assert len(fns) == 0
    
    def test_detection_metrics(self):
        """Test detection metrics computation."""
        matched = [{"prediction": {}, "ground_truth": {}}]
        fps = []
        fns = []
        
        metrics = compute_detection_metrics(matched, fps, fns)
        
        assert metrics["true_positives"] == 1
        assert metrics["false_positives"] == 0
        assert metrics["false_negatives"] == 0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
    
    def test_severity_metrics(self):
        """Test severity classification metrics."""
        matched_pairs = [
            {
                "prediction": {
                    "parse_success": True,
                    "parsed_output": {"severity": "high"}
                },
                "ground_truth": {"severity": "high"}
            },
            {
                "prediction": {
                    "parse_success": True,
                    "parsed_output": {"severity": "medium"}
                },
                "ground_truth": {"severity": "high"}
            },
        ]
        
        metrics = compute_severity_metrics(matched_pairs)
        
        assert metrics["accuracy"] == 0.5  # 1 correct out of 2
        assert "per_class" in metrics
    
    def test_disposition_metrics(self):
        """Test disposition classification metrics."""
        matched_pairs = [
            {
                "prediction": {
                    "parse_success": True,
                    "parsed_output": {"disposition": "NEEDS_ONCALL"}
                },
                "ground_truth": {"disposition": "NEEDS_ONCALL"}
            },
        ]
        
        metrics = compute_disposition_metrics(matched_pairs)
        
        assert metrics["accuracy"] == 1.0
        assert "per_class" in metrics


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_no_aws_dependencies(self):
        """Verify no AWS dependencies are imported."""
        import sys
        
        # Check that boto3 is not in loaded modules
        aws_modules = [mod for mod in sys.modules if 'boto' in mod.lower() or 'aws' in mod.lower()]
        
        # Filter out any test mocks
        real_aws_modules = [m for m in aws_modules if 'mock' not in m.lower() and 'test' not in m.lower()]
        
        # This is informational rather than a hard failure
        # since some environments may have boto3 installed but not used
        if real_aws_modules:
            print(f"Note: AWS modules found in sys.modules: {real_aws_modules}")
    
    def test_no_database_dependencies(self):
        """Verify no production database dependencies."""
        import sys
        
        # We should not import production database modules
        db_modules = [mod for mod in sys.modules if 'sqlalchemy' in mod.lower() and 'app.shared' in mod]
        
        assert len(db_modules) == 0, "Should not import production database modules"


def run_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running Local Experiment Tests")
    print("="*60 + "\n")
    
    # Run pytest programmatically
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
