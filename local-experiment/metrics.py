"""
Metrics computation for incident analysis evaluation.

Provides stage-specific metrics:
- Candidate detection quality (did we find the right incident windows?)
- Model analysis quality (given correct window, did model classify correctly?)
- End-to-end performance
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict


def load_ground_truth(path: Path) -> Dict[str, Any]:
    """Load ground truth JSON."""
    with open(path, "r") as f:
        return json.load(f)


def parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO timestamp and ensure it's timezone-aware."""
    from datetime import timezone
    
    if not ts_str:
        return datetime.min.replace(tzinfo=timezone.utc)
    
    # Handle both with and without 'Z' suffix
    if ts_str.endswith('Z'):
        ts_str = ts_str[:-1] + '+00:00'
    
    dt = datetime.fromisoformat(ts_str)
    
    # If timezone-naive, assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    return dt


def temporal_overlap(
    pred_start: datetime,
    pred_end: datetime,
    gt_start: datetime,
    gt_end: datetime,
) -> float:
    """
    Compute temporal overlap ratio between prediction and ground truth.
    
    Returns value between 0.0 (no overlap) and 1.0 (complete overlap).
    """
    if pred_start > gt_end or pred_end < gt_start:
        return 0.0
    
    overlap_start = max(pred_start, gt_start)
    overlap_end = min(pred_end, gt_end)
    
    overlap_duration = (overlap_end - overlap_start).total_seconds()
    gt_duration = (gt_end - gt_start).total_seconds()
    
    if gt_duration == 0:
        return 1.0 if overlap_duration > 0 else 0.0
    
    return overlap_duration / gt_duration


def match_predictions_to_ground_truth(
    predictions: List[Dict],
    ground_truth_incidents: List[Dict],
    min_overlap: float = 0.5,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Match predictions to ground truth incidents.
    
    Matching rules:
    - Must match on service
    - Must have temporal overlap >= min_overlap
    - Each ground truth can match at most one prediction (best match)
    
    Returns:
        (matched_pairs, unmatched_predictions, unmatched_ground_truth)
        
        matched_pairs: list of {"prediction": ..., "ground_truth": ..., "overlap": ...}
        unmatched_predictions: predictions with no ground truth match (false positives)
        unmatched_ground_truth: ground truth with no prediction (false negatives)
    """
    # Parse timestamps
    for pred in predictions:
        if pred.get("first_seen"):
            pred["_first_seen_dt"] = parse_timestamp(pred["first_seen"])
        if pred.get("last_seen"):
            pred["_last_seen_dt"] = parse_timestamp(pred["last_seen"])
    
    for gt in ground_truth_incidents:
        if gt.get("start_time"):
            gt["_start_dt"] = parse_timestamp(gt["start_time"])
        if gt.get("end_time"):
            gt["_end_dt"] = parse_timestamp(gt["end_time"])
    
    # Find best matches
    matched_pairs = []
    matched_gt_ids = set()
    matched_pred_ids = set()
    
    # For each ground truth, find best matching prediction
    for gt in ground_truth_incidents:
        if "incident_id" not in gt:
            continue
        
        best_pred = None
        best_overlap = 0.0
        
        for pred in predictions:
            # Must match on service
            if pred.get("service") != gt.get("service"):
                continue
            
            # Skip if already matched
            if pred.get("incident_id") in matched_pred_ids:
                continue
            
            # Compute temporal overlap
            if "_first_seen_dt" in pred and "_start_dt" in gt:
                overlap = temporal_overlap(
                    pred["_first_seen_dt"],
                    pred["_last_seen_dt"],
                    gt["_start_dt"],
                    gt["_end_dt"],
                )
                
                if overlap >= min_overlap and overlap > best_overlap:
                    best_overlap = overlap
                    best_pred = pred
        
        if best_pred:
            matched_pairs.append({
                "prediction": best_pred,
                "ground_truth": gt,
                "overlap": best_overlap,
            })
            matched_gt_ids.add(gt["incident_id"])
            matched_pred_ids.add(best_pred["incident_id"])
    
    # Unmatched predictions (false positives)
    unmatched_predictions = [
        p for p in predictions
        if p.get("incident_id") not in matched_pred_ids
    ]
    
    # Unmatched ground truth (false negatives)
    unmatched_ground_truth = [
        gt for gt in ground_truth_incidents
        if gt.get("incident_id") not in matched_gt_ids
    ]
    
    return matched_pairs, unmatched_predictions, unmatched_ground_truth


def compute_detection_metrics(
    matched_pairs: List[Dict],
    false_positives: List[Dict],
    false_negatives: List[Dict],
) -> Dict[str, Any]:
    """Compute incident detection metrics."""
    tp = len(matched_pairs)
    fp = len(false_positives)
    fn = len(false_negatives)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_severity_metrics(matched_pairs: List[Dict]) -> Dict[str, Any]:
    """Compute severity classification metrics for matched incidents."""
    if not matched_pairs:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}}
    
    # Only compute for successfully parsed predictions
    valid_pairs = [
        p for p in matched_pairs
        if p["prediction"].get("parse_success") and p["prediction"].get("parsed_output")
    ]
    
    if not valid_pairs:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "note": "No valid predictions"}
    
    # Collect predictions and ground truth
    y_true = []
    y_pred = []
    
    for pair in valid_pairs:
        gt_sev = pair["ground_truth"].get("severity", "").lower()
        pred_sev = pair["prediction"]["parsed_output"].get("severity", "").lower()
        
        if gt_sev and pred_sev:
            y_true.append(gt_sev)
            y_pred.append(pred_sev)
    
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}}
    
    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true)
    
    # Per-class metrics
    severities = ["low", "medium", "high", "critical"]
    per_class = {}
    class_f1s = []
    
    for sev in severities:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == sev and p == sev)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != sev and p == sev)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == sev and p != sev)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[sev] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for t in y_true if t == sev),
        }
        
        if sum(1 for t in y_true if t == sev) > 0:  # Only include classes present in ground truth
            class_f1s.append(f1)
    
    macro_f1 = sum(class_f1s) / len(class_f1s) if class_f1s else 0.0
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": _build_confusion_matrix(y_true, y_pred, severities),
    }


def compute_disposition_metrics(matched_pairs: List[Dict]) -> Dict[str, Any]:
    """Compute disposition classification metrics for matched incidents."""
    if not matched_pairs:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}}
    
    valid_pairs = [
        p for p in matched_pairs
        if p["prediction"].get("parse_success") and p["prediction"].get("parsed_output")
    ]
    
    if not valid_pairs:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "note": "No valid predictions"}
    
    y_true = []
    y_pred = []
    
    for pair in valid_pairs:
        gt_disp = pair["ground_truth"].get("disposition", "").upper()
        pred_disp = pair["prediction"]["parsed_output"].get("disposition", "").upper()
        
        if gt_disp and pred_disp:
            y_true.append(gt_disp)
            y_pred.append(pred_disp)
    
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}}
    
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true)
    
    dispositions = ["NO_ACTION", "OBSERVE", "NEEDS_DEV", "NEEDS_ONCALL", "ESCALATE"]
    per_class = {}
    class_f1s = []
    
    for disp in dispositions:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == disp and p == disp)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != disp and p == disp)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == disp and p != disp)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[disp] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for t in y_true if t == disp),
        }
        
        if sum(1 for t in y_true if t == disp) > 0:
            class_f1s.append(f1)
    
    macro_f1 = sum(class_f1s) / len(class_f1s) if class_f1s else 0.0
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def compute_structured_output_metrics(predictions: List[Dict]) -> Dict[str, Any]:
    """Compute metrics about structured output quality."""
    total = len(predictions)
    if total == 0:
        return {}
    
    valid_json = sum(1 for p in predictions if p.get("parse_success"))
    
    # For successfully parsed outputs, check field completion
    parsed = [p for p in predictions if p.get("parse_success") and p.get("parsed_output")]
    
    field_completion = {}
    if parsed:
        fields = ["summary", "suspected_root_cause", "next_steps", "ticket_title", "ticket_body"]
        for field in fields:
            non_empty = sum(
                1 for p in parsed
                if p["parsed_output"].get(field) not in [None, "", []]
            )
            field_completion[field] = non_empty / len(parsed)
    
    return {
        "valid_json_rate": valid_json / total,
        "parse_failure_rate": (total - valid_json) / total,
        "field_completion": field_completion,
    }


def _build_confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    """Build confusion matrix."""
    matrix = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return dict(matrix)


def generate_error_analysis(
    matched_pairs: List[Dict],
    false_positives: List[Dict],
    false_negatives: List[Dict],
) -> Dict[str, Any]:
    """Generate detailed error analysis."""
    errors = {
        "false_positives": [],
        "false_negatives": [],
        "severity_mismatches": [],
        "disposition_mismatches": [],
        "parse_failures": [],
    }
    
    # False positives
    for fp in false_positives:
        errors["false_positives"].append({
            "incident_id": fp.get("incident_id"),
            "service": fp.get("service"),
            "first_seen": fp.get("first_seen"),
            "count": fp.get("count"),
            "sample_logs": fp.get("sample_logs", [])[:3],
            "predicted_severity": fp.get("parsed_output", {}).get("severity") if fp.get("parse_success") else None,
            "predicted_disposition": fp.get("parsed_output", {}).get("disposition") if fp.get("parse_success") else None,
        })
    
    # False negatives
    for fn in false_negatives:
        errors["false_negatives"].append({
            "incident_id": fn.get("incident_id"),
            "service": fn.get("service"),
            "expected_severity": fn.get("severity"),
            "expected_disposition": fn.get("disposition"),
            "description": fn.get("description"),
            "log_markers": fn.get("log_line_markers", []),
        })
    
    # Severity mismatches (in matched pairs)
    for pair in matched_pairs:
        if not pair["prediction"].get("parse_success"):
            continue
        
        pred_output = pair["prediction"].get("parsed_output", {})
        gt = pair["ground_truth"]
        
        pred_sev = pred_output.get("severity", "").lower()
        gt_sev = gt.get("severity", "").lower()
        
        if pred_sev and gt_sev and pred_sev != gt_sev:
            errors["severity_mismatches"].append({
                "incident_id": pair["prediction"].get("incident_id"),
                "service": pair["prediction"].get("service"),
                "predicted": pred_sev,
                "expected": gt_sev,
                "ground_truth_justification": gt.get("severity_justification"),
                "sample_logs": pair["prediction"].get("sample_logs", [])[:3],
            })
        
        # Disposition mismatches
        pred_disp = pred_output.get("disposition", "").upper()
        gt_disp = gt.get("disposition", "").upper()
        
        if pred_disp and gt_disp and pred_disp != gt_disp:
            errors["disposition_mismatches"].append({
                "incident_id": pair["prediction"].get("incident_id"),
                "service": pair["prediction"].get("service"),
                "predicted": pred_disp,
                "expected": gt_disp,
                "ground_truth_justification": gt.get("disposition_justification"),
            })
    
    # Parse failures
    for pred in [pair["prediction"] for pair in matched_pairs]:
        if not pred.get("parse_success"):
            errors["parse_failures"].append({
                "incident_id": pred.get("incident_id"),
                "service": pred.get("service"),
                "error": pred.get("parse_error"),
                "raw_output": pred.get("raw_output", "")[:500],  # First 500 chars
            })
    
    return errors


def compute_metrics(
    predictions: List[Dict],
    ground_truth: Dict[str, Any],
    candidates: List,
) -> Dict[str, Any]:
    """
    Compute all metrics.
    
    Returns comprehensive metrics including:
    - Detection metrics
    - Severity metrics
    - Disposition metrics
    - Structured output metrics
    - Error analysis
    """
    gt_incidents = ground_truth.get("incidents", [])
    
    # Match predictions to ground truth
    matched_pairs, false_positives, false_negatives = match_predictions_to_ground_truth(
        predictions,
        gt_incidents,
        min_overlap=0.5,
    )
    
    # Compute metrics
    detection = compute_detection_metrics(matched_pairs, false_positives, false_negatives)
    severity = compute_severity_metrics(matched_pairs)
    disposition = compute_disposition_metrics(matched_pairs)
    structured_output = compute_structured_output_metrics(predictions)
    error_analysis = generate_error_analysis(matched_pairs, false_positives, false_negatives)
    
    return {
        "detection": detection,
        "severity": severity,
        "disposition": disposition,
        "structured_output": structured_output,
        "error_analysis": error_analysis,
        "summary": {
            "total_ground_truth_incidents": len(gt_incidents),
            "total_predicted_incidents": len(predictions),
            "matched_incidents": len(matched_pairs),
            "detection_f1": detection["f1"],
            "severity_accuracy": severity.get("accuracy", 0.0),
            "disposition_accuracy": disposition.get("accuracy", 0.0),
            "valid_json_rate": structured_output.get("valid_json_rate", 0.0),
        },
    }
