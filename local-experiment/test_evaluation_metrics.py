"""Tests for evaluation metrics."""

import pytest
from datetime import datetime, timedelta
from evaluation_metrics import (
    GroundTruthIncident,
    PredictedIncident,
    calculate_temporal_overlap,
    match_predictions_to_ground_truth,
    calculate_detection_metrics,
    calculate_severity_metrics,
    evaluate,
)
from output_parser import ParsedOutput, ParseStatus


class TestTemporalOverlap:
    """Test temporal overlap calculation."""
    
    def test_complete_overlap(self):
        """Identical time windows should have 100% overlap."""
        start = datetime(2026, 9, 25, 10, 0, 0)
        end = datetime(2026, 9, 25, 10, 5, 0)
        
        overlap = calculate_temporal_overlap(start, end, start, end)
        assert overlap == 1.0
    
    def test_no_overlap(self):
        """Non-overlapping windows should have 0% overlap."""
        start1 = datetime(2026, 9, 25, 10, 0, 0)
        end1 = datetime(2026, 9, 25, 10, 5, 0)
        
        start2 = datetime(2026, 9, 25, 10, 10, 0)
        end2 = datetime(2026, 9, 25, 10, 15, 0)
        
        overlap = calculate_temporal_overlap(start1, end1, start2, end2)
        assert overlap == 0.0
    
    def test_partial_overlap(self):
        """Partially overlapping windows."""
        start1 = datetime(2026, 9, 25, 10, 0, 0)
        end1 = datetime(2026, 9, 25, 10, 5, 0)  # 5 minutes
        
        start2 = datetime(2026, 9, 25, 10, 3, 0)
        end2 = datetime(2026, 9, 25, 10, 8, 0)  # 5 minutes
        
        # Intersection: 10:03-10:05 = 2 minutes
        # Union: 10:00-10:08 = 8 minutes
        # Overlap: 2/8 = 0.25
        overlap = calculate_temporal_overlap(start1, end1, start2, end2)
        assert abs(overlap - 0.25) < 0.01


class TestPredictionMatching:
    """Test prediction-to-ground-truth matching."""
    
    def create_gt(self, service: str, start_offset: int, duration: int, severity: str) -> GroundTruthIncident:
        """Helper to create ground truth incident."""
        base = datetime(2026, 9, 25, 10, 0, 0)
        start = base + timedelta(minutes=start_offset)
        end = start + timedelta(minutes=duration)
        
        return GroundTruthIncident(
            incident_id=f"gt-{service}-{start_offset}",
            incident_type="test",
            service=service,
            severity=severity,
            disposition="NEEDS_DEV",
            start_time=start,
            end_time=end,
            log_count=10,
            description="Test incident"
        )
    
    def create_pred(self, service: str, start_offset: int, duration: int, severity: str) -> PredictedIncident:
        """Helper to create predicted incident."""
        base = datetime(2026, 9, 25, 10, 0, 0)
        start = base + timedelta(minutes=start_offset)
        end = start + timedelta(minutes=duration)
        
        return PredictedIncident(
            incident_id=f"pred-{service}-{start_offset}",
            service=service,
            start_time=start,
            end_time=end,
            log_count=10,
            parsed_output=ParsedOutput(
                severity=severity,
                disposition="NEEDS_DEV",
                summary="Test",
                parse_status=ParseStatus.SUCCESS
            ),
            latency_ms=100.0
        )
    
    def test_perfect_match(self):
        """Perfect temporal and service match."""
        gt = [self.create_gt("payment-api", 0, 5, "high")]
        pred = [self.create_pred("payment-api", 0, 5, "high")]
        
        matches = match_predictions_to_ground_truth(pred, gt)
        
        assert len(matches) == 1
        assert matches[0].is_matched
        assert matches[0].match_score == 1.0
        assert matches[0].is_correct_severity()
    
    def test_service_mismatch(self):
        """Different services should not match."""
        gt = [self.create_gt("payment-api", 0, 5, "high")]
        pred = [self.create_pred("order-api", 0, 5, "high")]
        
        matches = match_predictions_to_ground_truth(pred, gt)
        
        assert len(matches) == 1
        assert not matches[0].is_matched
    
    def test_temporal_mismatch(self):
        """Non-overlapping time windows should not match."""
        gt = [self.create_gt("payment-api", 0, 5, "high")]
        pred = [self.create_pred("payment-api", 10, 5, "high")]
        
        matches = match_predictions_to_ground_truth(pred, gt)
        
        assert len(matches) == 1
        assert not matches[0].is_matched
    
    def test_multiple_predictions_best_match(self):
        """Should select best matching prediction (greedy)."""
        gt = [self.create_gt("payment-api", 0, 5, "high")]
        pred = [
            self.create_pred("payment-api", 0, 3, "high"),  # Partial overlap
            self.create_pred("payment-api", 0, 5, "high"),  # Perfect overlap
        ]
        
        matches = match_predictions_to_ground_truth(pred, gt)
        
        assert len(matches) == 1
        assert matches[0].is_matched
        assert matches[0].match_score == 1.0
        assert matches[0].prediction.incident_id == "pred-payment-api-0"


class TestDetectionMetrics:
    """Test detection metrics calculation."""
    
    def test_all_correct(self):
        """All predictions match ground truth."""
        gt = GroundTruthIncident(
            incident_id="gt-1",
            incident_type="test",
            service="svc",
            severity="high",
            disposition="NEEDS_DEV",
            start_time=datetime.now(),
            end_time=datetime.now(),
            log_count=1,
            description="test"
        )
        
        pred = PredictedIncident(
            incident_id="pred-1",
            service="svc",
            start_time=datetime.now(),
            end_time=datetime.now(),
            log_count=1,
            parsed_output=ParsedOutput(
                severity="high",
                disposition="NEEDS_DEV",
                summary="test",
                parse_status=ParseStatus.SUCCESS
            ),
            latency_ms=100.0
        )
        
        from evaluation_metrics import IncidentMatch
        matches = [IncidentMatch(gt, pred, 1.0, True)]
        unmatched = []
        
        metrics = calculate_detection_metrics(matches, unmatched)
        
        assert metrics.true_positives == 1
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
    
    def test_false_positive(self):
        """Prediction with no ground truth."""
        matches = []
        unmatched = [
            PredictedIncident(
                incident_id="pred-1",
                service="svc",
                start_time=datetime.now(),
                end_time=datetime.now(),
                log_count=1,
                parsed_output=ParsedOutput(
                    severity="high",
                    disposition="NEEDS_DEV",
                    summary="false alarm",
                    parse_status=ParseStatus.SUCCESS
                ),
                latency_ms=100.0
            )
        ]
        
        metrics = calculate_detection_metrics(matches, unmatched)
        
        assert metrics.true_positives == 0
        assert metrics.false_positives == 1
        assert metrics.precision == 0.0


class TestSeverityMetrics:
    """Test severity classification metrics."""
    
    def create_match(self, true_sev: str, pred_sev: str):
        """Helper to create a match with specified severities."""
        gt = GroundTruthIncident(
            incident_id="gt-1",
            incident_type="test",
            service="svc",
            severity=true_sev,
            disposition="NEEDS_DEV",
            start_time=datetime.now(),
            end_time=datetime.now(),
            log_count=1,
            description="test"
        )
        
        pred = PredictedIncident(
            incident_id="pred-1",
            service="svc",
            start_time=datetime.now(),
            end_time=datetime.now(),
            log_count=1,
            parsed_output=ParsedOutput(
                severity=pred_sev,
                disposition="NEEDS_DEV",
                summary="test",
                parse_status=ParseStatus.SUCCESS
            ),
            latency_ms=100.0
        )
        
        from evaluation_metrics import IncidentMatch
        return IncidentMatch(gt, pred, 1.0, True)
    
    def test_perfect_severity_accuracy(self):
        """All severities correct."""
        matches = [
            self.create_match("high", "high"),
            self.create_match("medium", "medium"),
        ]
        
        metrics = calculate_severity_metrics(matches)
        
        assert metrics.accuracy == 1.0
    
    def test_severity_confusion_matrix(self):
        """Verify confusion matrix calculation."""
        matches = [
            self.create_match("high", "high"),      # TP for high
            self.create_match("high", "medium"),    # FP for medium, FN for high
            self.create_match("medium", "medium"),  # TP for medium
        ]
        
        metrics = calculate_severity_metrics(matches)
        
        assert metrics.confusion_matrix["high"]["high"] == 1
        assert metrics.confusion_matrix["high"]["medium"] == 1
        assert metrics.confusion_matrix["medium"]["medium"] == 1
        assert metrics.accuracy == 2/3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
