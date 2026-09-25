#!/usr/bin/env python3
"""
Calculate evaluation metrics for incident detection and classification.
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DetectionMetrics:
    """Incident detection metrics (binary: incident vs no-incident)."""
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    
    @property
    def precision(self) -> Optional[float]:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else None
    
    @property
    def recall(self) -> Optional[float]:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else None
    
    @property
    def f1(self) -> Optional[float]:
        p = self.precision
        r = self.recall
        if p is not None and r is not None and (p + r) > 0:
            return 2 * p * r / (p + r)
        return None
    
    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.false_positives + self.false_negatives + self.true_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total


@dataclass
class ClassificationMetrics:
    """Multi-class classification metrics."""
    class_counts: dict[str, dict[str, int]]  # class -> {tp, fp, fn}
    
    def precision(self, cls: str) -> Optional[float]:
        tp = self.class_counts[cls]["tp"]
        fp = self.class_counts[cls]["fp"]
        denom = tp + fp
        return tp / denom if denom > 0 else None
    
    def recall(self, cls: str) -> Optional[float]:
        tp = self.class_counts[cls]["tp"]
        fn = self.class_counts[cls]["fn"]
        denom = tp + fn
        return tp / denom if denom > 0 else None
    
    def f1(self, cls: str) -> Optional[float]:
        p = self.precision(cls)
        r = self.recall(cls)
        if p is not None and r is not None and (p + r) > 0:
            return 2 * p * r / (p + r)
        return None
    
    @property
    def macro_precision(self) -> float:
        precisions = [p for cls in self.class_counts if (p := self.precision(cls)) is not None]
        return sum(precisions) / len(precisions) if precisions else 0.0
    
    @property
    def macro_recall(self) -> float:
        recalls = [r for cls in self.class_counts if (r := self.recall(cls)) is not None]
        return sum(recalls) / len(recalls) if recalls else 0.0
    
    @property
    def macro_f1(self) -> float:
        f1s = [f for cls in self.class_counts if (f := self.f1(cls)) is not None]
        return sum(f1s) / len(f1s) if f1s else 0.0
    
    @property
    def accuracy(self) -> float:
        total_tp = sum(counts["tp"] for counts in self.class_counts.values())
        total = sum(counts["tp"] + counts["fp"] for counts in self.class_counts.values())
        return total_tp / total if total > 0 else 0.0


def calculate_detection_metrics(ground_truth: list[dict], predictions: list[dict]) -> DetectionMetrics:
    """
    Calculate binary detection metrics (incident vs no-incident).
    
    Ground truth format: {"is_incident": bool, ...}
    Prediction format: {"severity": str, "disposition": str, ...}
    """
    tp = fp = fn = tn = 0
    
    for gt, pred in zip(ground_truth, predictions):
        gt_is_incident = gt.get("is_incident", True)
        
        # Treat low+NO_ACTION as no-incident
        pred_is_incident = not (
            pred.get("severity") == "low" and 
            pred.get("disposition") == "NO_ACTION"
        )
        
        if gt_is_incident and pred_is_incident:
            tp += 1
        elif not gt_is_incident and pred_is_incident:
            fp += 1
        elif gt_is_incident and not pred_is_incident:
            fn += 1
        else:
            tn += 1
    
    return DetectionMetrics(tp, fp, fn, tn)


def calculate_severity_metrics(ground_truth: list[dict], predictions: list[dict]) -> ClassificationMetrics:
    """Calculate multi-class severity classification metrics."""
    classes = {"low", "medium", "high", "critical"}
    class_counts = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in classes}
    
    for gt, pred in zip(ground_truth, predictions):
        gt_severity = gt.get("severity", "low")
        pred_severity = pred.get("severity", "low")
        
        for cls in classes:
            if gt_severity == cls and pred_severity == cls:
                class_counts[cls]["tp"] += 1
            elif gt_severity != cls and pred_severity == cls:
                class_counts[cls]["fp"] += 1
            elif gt_severity == cls and pred_severity != cls:
                class_counts[cls]["fn"] += 1
    
    return ClassificationMetrics(class_counts)


def calculate_disposition_metrics(ground_truth: list[dict], predictions: list[dict]) -> ClassificationMetrics:
    """Calculate multi-class disposition classification metrics."""
    classes = {"NO_ACTION", "OBSERVE", "NEEDS_DEV", "NEEDS_ONCALL", "ESCALATE"}
    class_counts = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in classes}
    
    for gt, pred in zip(ground_truth, predictions):
        gt_disp = gt.get("disposition", "OBSERVE")
        pred_disp = pred.get("disposition", "OBSERVE")
        
        for cls in classes:
            if gt_disp == cls and pred_disp == cls:
                class_counts[cls]["tp"] += 1
            elif gt_disp != cls and pred_disp == cls:
                class_counts[cls]["fp"] += 1
            elif gt_disp == cls and pred_disp != cls:
                class_counts[cls]["fn"] += 1
    
    return ClassificationMetrics(class_counts)


def build_confusion_matrix(ground_truth: list[dict], predictions: list[dict], field: str) -> dict:
    """Build confusion matrix for a field."""
    matrix = defaultdict(lambda: defaultdict(int))
    
    for gt, pred in zip(ground_truth, predictions):
        gt_val = gt.get(field, "unknown")
        pred_val = pred.get(field, "unknown")
        matrix[gt_val][pred_val] += 1
    
    return dict(matrix)


def calculate_field_completion_rates(predictions: list[dict]) -> dict:
    """Calculate completion rates for structured output fields."""
    total = len(predictions)
    if total == 0:
        return {}
    
    completion = {
        "valid_json": 0,
        "has_severity": 0,
        "has_disposition": 0,
        "has_confidence": 0,
        "has_summary": 0,
        "has_suspected_root_cause": 0,
        "has_next_steps": 0,
        "has_ticket_title": 0,
        "has_ticket_body": 0,
    }
    
    for pred in predictions:
        if pred is not None:
            completion["valid_json"] += 1
        
        if pred.get("severity"):
            completion["has_severity"] += 1
        if pred.get("disposition"):
            completion["has_disposition"] += 1
        if pred.get("confidence") is not None:
            completion["has_confidence"] += 1
        if pred.get("summary"):
            completion["has_summary"] += 1
        if pred.get("suspected_root_cause"):
            completion["has_suspected_root_cause"] += 1
        if pred.get("next_steps"):
            completion["has_next_steps"] += 1
        if pred.get("ticket_title"):
            completion["has_ticket_title"] += 1
        if pred.get("ticket_body"):
            completion["has_ticket_body"] += 1
    
    return {k: v / total for k, v in completion.items()}


def analyze_errors(ground_truth: list[dict], predictions: list[dict], log_sequences: list[list[str]]) -> dict:
    """Analyze false positives and false negatives."""
    false_positives = []
    false_negatives = []
    severity_errors = []
    
    for i, (gt, pred, logs) in enumerate(zip(ground_truth, predictions, log_sequences)):
        gt_is_incident = gt.get("is_incident", True)
        pred_is_incident = not (
            pred.get("severity") == "low" and 
            pred.get("disposition") == "NO_ACTION"
        )
        
        # Detection errors
        if not gt_is_incident and pred_is_incident:
            false_positives.append({
                "index": i,
                "logs": logs,
                "ground_truth": gt,
                "prediction": pred,
            })
        elif gt_is_incident and not pred_is_incident:
            false_negatives.append({
                "index": i,
                "logs": logs,
                "ground_truth": gt,
                "prediction": pred,
            })
        
        # Severity errors (for detected incidents only)
        if gt_is_incident and pred_is_incident:
            if gt.get("severity") != pred.get("severity"):
                severity_errors.append({
                    "index": i,
                    "logs": logs,
                    "expected_severity": gt.get("severity"),
                    "predicted_severity": pred.get("severity"),
                    "ground_truth": gt,
                    "prediction": pred,
                })
    
    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "severity_errors": severity_errors,
    }


def format_metrics_report(
    detection: DetectionMetrics,
    severity: ClassificationMetrics,
    disposition: ClassificationMetrics,
    completion: dict,
    latencies: list[float]
) -> str:
    """Format metrics as human-readable report."""
    lines = [
        "=" * 80,
        "EVALUATION METRICS REPORT",
        "=" * 80,
        "",
        "INCIDENT DETECTION (Binary: incident vs no-incident)",
        f"  Precision:  {detection.precision:.3f}" if detection.precision is not None else "  Precision:  N/A",
        f"  Recall:     {detection.recall:.3f}" if detection.recall is not None else "  Recall:     N/A",
        f"  F1:         {detection.f1:.3f}" if detection.f1 is not None else "  F1:         N/A",
        f"  Accuracy:   {detection.accuracy:.3f}",
        f"  TP: {detection.true_positives}, FP: {detection.false_positives}, "
        f"FN: {detection.false_negatives}, TN: {detection.true_negatives}",
        "",
        "SEVERITY CLASSIFICATION",
        f"  Accuracy:        {severity.accuracy:.3f}",
        f"  Macro Precision: {severity.macro_precision:.3f}",
        f"  Macro Recall:    {severity.macro_recall:.3f}",
        f"  Macro F1:        {severity.macro_f1:.3f}",
        "",
        "  Per-class metrics:",
    ]
    
    for cls in ["low", "medium", "high", "critical"]:
        p = severity.precision(cls)
        r = severity.recall(cls)
        f = severity.f1(cls)
        lines.append(
            f"    {cls:10s} P:{p:.3f} R:{r:.3f} F1:{f:.3f}" if p and r and f else f"    {cls:10s} N/A"
        )
    
    lines.extend([
        "",
        "DISPOSITION CLASSIFICATION",
        f"  Accuracy:        {disposition.accuracy:.3f}",
        f"  Macro F1:        {disposition.macro_f1:.3f}",
        "",
        "STRUCTURED OUTPUT COMPLETION",
        f"  Valid JSON:              {completion.get('valid_json', 0):.1%}",
        f"  Has severity:            {completion.get('has_severity', 0):.1%}",
        f"  Has disposition:         {completion.get('has_disposition', 0):.1%}",
        f"  Has confidence:          {completion.get('has_confidence', 0):.1%}",
        f"  Has summary:             {completion.get('has_summary', 0):.1%}",
        f"  Has root cause:          {completion.get('has_suspected_root_cause', 0):.1%}",
        f"  Has next_steps:          {completion.get('has_next_steps', 0):.1%}",
        f"  Has ticket_title:        {completion.get('has_ticket_title', 0):.1%}",
        f"  Has ticket_body:         {completion.get('has_ticket_body', 0):.1%}",
        "",
        "INFERENCE LATENCY",
        f"  Mean:  {sum(latencies) / len(latencies):.1f}ms" if latencies else "  Mean:  N/A",
        f"  P50:   {sorted(latencies)[len(latencies)//2]:.1f}ms" if latencies else "  P50:   N/A",
        f"  P95:   {sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms" if latencies else "  P95:   N/A",
        f"  P99:   {sorted(latencies)[int(len(latencies)*0.99)]:.1f}ms" if latencies else "  P99:   N/A",
        "",
        "=" * 80,
    ])
    
    return "\n".join(lines)
