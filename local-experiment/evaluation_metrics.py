"""
Comprehensive evaluation metrics for incident detection and classification.

This module implements:
- Prediction-to-ground-truth matching
- Detection metrics (TP/FP/FN, precision/recall/F1)
- Severity classification metrics
- Disposition evaluation
- Benign false positive detection
- Error analysis
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from collections import defaultdict

from output_parser import ParsedOutput, validate_severity, validate_disposition


@dataclass
class GroundTruthIncident:
    """Ground truth incident representation."""
    incident_id: str
    incident_type: str
    service: str
    severity: str
    disposition: str
    start_time: datetime
    end_time: datetime
    log_count: int
    description: str


@dataclass
class PredictedIncident:
    """Predicted incident from model."""
    incident_id: str  # Technical cluster ID
    service: str
    start_time: datetime
    end_time: datetime
    log_count: int
    parsed_output: ParsedOutput
    latency_ms: float


@dataclass
class IncidentMatch:
    """Represents a match between prediction and ground truth."""
    ground_truth: GroundTruthIncident
    prediction: Optional[PredictedIncident]
    match_score: float  # 0.0-1.0, based on temporal overlap
    is_matched: bool
    
    def is_correct_severity(self) -> bool:
        """Check if severity matches."""
        if not self.prediction or not self.prediction.parsed_output.severity:
            return False
        return (
            self.prediction.parsed_output.severity.lower()
            == self.ground_truth.severity.lower()
        )
    
    def is_correct_disposition(self) -> bool:
        """Check if disposition matches."""
        if not self.prediction or not self.prediction.parsed_output.disposition:
            return False
        return (
            self.prediction.parsed_output.disposition.upper()
            == self.ground_truth.disposition.upper()
        )


@dataclass
class DetectionMetrics:
    """Incident detection metrics."""
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    
    # Breakdown by severity
    benign_false_positives: int  # LOW/NO_ACTION predicted as HIGH/CRITICAL
    actionable_false_negatives: int  # HIGH/CRITICAL missed


@dataclass
class SeverityMetrics:
    """Severity classification metrics."""
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    
    # Per-class metrics
    per_class: dict[str, dict[str, float]]  # {severity: {precision, recall, f1}}
    
    # Confusion matrix
    confusion_matrix: dict[str, dict[str, int]]  # {true: {pred: count}}


@dataclass
class EvaluationResults:
    """Complete evaluation results."""
    detection: DetectionMetrics
    severity: SeverityMetrics
    
    # Incident matching
    matches: list[IncidentMatch]
    unmatched_predictions: list[PredictedIncident]
    
    # Parse metrics
    total_predictions: int
    valid_parse_count: int
    parse_failure_count: int
    parse_success_rate: float
    mean_completion_score: float
    
    # Latency
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Errors for analysis
    false_positives: list[tuple[PredictedIncident, str]]  # (prediction, reason)
    false_negatives: list[tuple[GroundTruthIncident, str]]  # (gt, reason)
    severity_mismatches: list[tuple[IncidentMatch, str, str]]  # (match, expected, actual)


def match_predictions_to_ground_truth(
    predictions: list[PredictedIncident],
    ground_truth: list[GroundTruthIncident],
    min_overlap: float = 0.3
) -> list[IncidentMatch]:
    """
    Match predictions to ground truth incidents.
    
    Matching rules:
    1. Service must match exactly
    2. Time windows must overlap
    3. Overlap score >= min_overlap threshold
    4. Greedy best-match assignment
    
    Args:
        predictions: List of predicted incidents
        ground_truth: List of ground truth incidents
        min_overlap: Minimum temporal overlap (0.0-1.0)
    
    Returns:
        List of matches (one per ground truth incident)
    """
    matches = []
    used_predictions = set()
    
    for gt in ground_truth:
        best_prediction = None
        best_score = 0.0
        
        # Find best matching prediction
        for pred in predictions:
            if pred.incident_id in used_predictions:
                continue
            
            # Check service match
            if pred.service != gt.service:
                continue
            
            # Calculate temporal overlap
            overlap = calculate_temporal_overlap(
                gt.start_time, gt.end_time,
                pred.start_time, pred.end_time
            )
            
            if overlap >= min_overlap and overlap > best_score:
                best_score = overlap
                best_prediction = pred
        
        # Create match
        if best_prediction:
            used_predictions.add(best_prediction.incident_id)
            matches.append(IncidentMatch(
                ground_truth=gt,
                prediction=best_prediction,
                match_score=best_score,
                is_matched=True
            ))
        else:
            # No matching prediction (false negative)
            matches.append(IncidentMatch(
                ground_truth=gt,
                prediction=None,
                match_score=0.0,
                is_matched=False
            ))
    
    return matches


def calculate_temporal_overlap(
    start1: datetime,
    end1: datetime,
    start2: datetime,
    end2: datetime
) -> float:
    """
    Calculate temporal overlap between two time windows.
    
    Returns:
        Overlap score (0.0-1.0), based on Jaccard similarity
    """
    # Calculate intersection
    latest_start = max(start1, start2)
    earliest_end = min(end1, end2)
    
    if latest_start >= earliest_end:
        return 0.0  # No overlap
    
    intersection = (earliest_end - latest_start).total_seconds()
    
    # Calculate union
    earliest_start = min(start1, start2)
    latest_end = max(end1, end2)
    union = (latest_end - earliest_start).total_seconds()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_detection_metrics(
    matches: list[IncidentMatch],
    unmatched_predictions: list[PredictedIncident]
) -> DetectionMetrics:
    """Calculate detection metrics (TP/FP/FN, precision/recall/F1)."""
    tp = sum(1 for m in matches if m.is_matched)
    fn = sum(1 for m in matches if not m.is_matched)
    fp = len(unmatched_predictions)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Count benign false positives (LOW predicted as actionable)
    benign_fp = 0
    for pred in unmatched_predictions:
        if pred.parsed_output.severity in ("high", "critical"):
            benign_fp += 1
    
    # Count actionable false negatives (HIGH/CRITICAL missed)
    actionable_fn = 0
    for match in matches:
        if not match.is_matched and match.ground_truth.severity in ("high", "critical"):
            actionable_fn += 1
    
    return DetectionMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        benign_false_positives=benign_fp,
        actionable_false_negatives=actionable_fn
    )


def calculate_severity_metrics(matches: list[IncidentMatch]) -> SeverityMetrics:
    """Calculate severity classification metrics."""
    # Only consider matched incidents with valid severity
    valid_matches = [
        m for m in matches
        if m.is_matched
        and m.prediction
        and m.prediction.parsed_output.severity
    ]
    
    if not valid_matches:
        return SeverityMetrics(
            accuracy=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            per_class={},
            confusion_matrix={}
        )
    
    # Build confusion matrix
    severities = ["low", "medium", "high", "critical"]
    confusion = {true: {pred: 0 for pred in severities} for true in severities}
    
    for match in valid_matches:
        true_sev = match.ground_truth.severity.lower()
        pred_sev = match.prediction.parsed_output.severity.lower()
        if true_sev in confusion and pred_sev in severities:
            confusion[true_sev][pred_sev] += 1
    
    # Calculate accuracy
    correct = sum(1 for m in valid_matches if m.is_correct_severity())
    accuracy = correct / len(valid_matches)
    
    # Calculate per-class metrics
    per_class = {}
    precisions = []
    recalls = []
    f1_scores = []
    
    for sev in severities:
        tp = confusion[sev][sev]
        fp = sum(confusion[other][sev] for other in severities if other != sev)
        fn = sum(confusion[sev][other] for other in severities if other != sev)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[sev] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[sev].values())
        }
        
        if per_class[sev]["support"] > 0:  # Only include in macro if present
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
    
    # Calculate macro averages
    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    return SeverityMetrics(
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion_matrix=confusion
    )


def calculate_latency_metrics(predictions: list[PredictedIncident]) -> dict:
    """Calculate latency metrics."""
    if not predictions:
        return {
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0
        }
    
    latencies = sorted([p.latency_ms for p in predictions])
    n = len(latencies)
    
    return {
        "mean_ms": sum(latencies) / n,
        "p50_ms": latencies[int(n * 0.50)],
        "p95_ms": latencies[int(n * 0.95)] if n > 1 else latencies[-1],
        "p99_ms": latencies[int(n * 0.99)] if n > 1 else latencies[-1],
    }


def identify_errors(
    matches: list[IncidentMatch],
    unmatched_predictions: list[PredictedIncident]
) -> tuple[list, list, list]:
    """
    Identify false positives, false negatives, and severity mismatches.
    
    Returns:
        (false_positives, false_negatives, severity_mismatches)
    """
    false_positives = []
    for pred in unmatched_predictions:
        reason = "No matching ground truth incident"
        if not pred.parsed_output.is_valid():
            reason += f" (parse status: {pred.parsed_output.parse_status.value})"
        false_positives.append((pred, reason))
    
    false_negatives = []
    for match in matches:
        if not match.is_matched:
            reason = "No prediction found"
            false_negatives.append((match.ground_truth, reason))
    
    severity_mismatches = []
    for match in matches:
        if match.is_matched and not match.is_correct_severity():
            expected = match.ground_truth.severity
            actual = match.prediction.parsed_output.severity or "None"
            severity_mismatches.append((match, expected, actual))
    
    return false_positives, false_negatives, severity_mismatches


def evaluate(
    predictions: list[PredictedIncident],
    ground_truth: list[GroundTruthIncident]
) -> EvaluationResults:
    """
    Perform comprehensive evaluation.
    
    Args:
        predictions: List of predicted incidents
        ground_truth: List of ground truth incidents
    
    Returns:
        EvaluationResults with all metrics
    """
    # Match predictions to ground truth
    matches = match_predictions_to_ground_truth(predictions, ground_truth)
    
    # Find unmatched predictions (false positives)
    matched_pred_ids = {
        m.prediction.incident_id for m in matches
        if m.prediction is not None
    }
    unmatched_predictions = [
        p for p in predictions
        if p.incident_id not in matched_pred_ids
    ]
    
    # Calculate detection metrics
    detection = calculate_detection_metrics(matches, unmatched_predictions)
    
    # Calculate severity metrics
    severity = calculate_severity_metrics(matches)
    
    # Calculate parse metrics
    total_preds = len(predictions)
    valid_parse = sum(1 for p in predictions if p.parsed_output.is_valid())
    parse_failures = sum(
        1 for p in predictions
        if not p.parsed_output.is_valid()
    )
    
    completion_scores = [p.parsed_output.get_completion_score() for p in predictions]
    mean_completion = sum(completion_scores) / len(completion_scores) if completion_scores else 0.0
    
    # Calculate latency metrics
    latency = calculate_latency_metrics(predictions)
    
    # Identify errors
    fps, fns, sev_mismatches = identify_errors(matches, unmatched_predictions)
    
    return EvaluationResults(
        detection=detection,
        severity=severity,
        matches=matches,
        unmatched_predictions=unmatched_predictions,
        total_predictions=total_preds,
        valid_parse_count=valid_parse,
        parse_failure_count=parse_failures,
        parse_success_rate=valid_parse / total_preds if total_preds > 0 else 0.0,
        mean_completion_score=mean_completion,
        mean_latency_ms=latency["mean_ms"],
        p50_latency_ms=latency["p50_ms"],
        p95_latency_ms=latency["p95_ms"],
        p99_latency_ms=latency["p99_ms"],
        false_positives=fps,
        false_negatives=fns,
        severity_mismatches=sev_mismatches
    )
