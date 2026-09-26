"""
Compare two experiment results side-by-side.

Shows differences in metrics to help understand model improvements.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any


def load_metrics(experiment_dir: Path) -> Dict[str, Any]:
    """Load metrics from experiment directory."""
    metrics_file = experiment_dir / "metrics.json"
    if not metrics_file.exists():
        # Try subdirectory
        eval_dirs = sorted(experiment_dir.glob("evaluation/eval_*"))
        if eval_dirs:
            metrics_file = eval_dirs[-1] / "metrics.json"
    
    if not metrics_file.exists():
        raise FileNotFoundError(f"No metrics.json found in {experiment_dir}")
    
    with open(metrics_file, "r") as f:
        return json.load(f)


def format_metric(value: Any, format_type: str = "percent") -> str:
    """Format metric value for display."""
    if value is None:
        return "N/A"
    
    if format_type == "percent":
        return f"{value*100:6.2f}%"
    elif format_type == "float":
        return f"{value:6.3f}"
    elif format_type == "int":
        return f"{int(value):6d}"
    else:
        return f"{value:6}"


def format_delta(val1: float, val2: float, format_type: str = "percent") -> str:
    """Format the difference between two values."""
    if val1 is None or val2 is None:
        return ""
    
    delta = val2 - val1
    
    if format_type == "percent":
        delta_str = f"{delta*100:+.2f}%"
    else:
        delta_str = f"{delta:+.3f}"
    
    # Color coding (for terminal)
    if delta > 0:
        return f"  ↑ {delta_str}"
    elif delta < 0:
        return f"  ↓ {delta_str}"
    else:
        return f"  = {delta_str}"


def compare_experiments(exp1_dir: Path, exp2_dir: Path):
    """Compare two experiments side-by-side."""
    
    print(f"\n{'='*80}")
    print(f"Experiment Comparison")
    print(f"{'='*80}")
    print(f"Experiment 1: {exp1_dir.name}")
    print(f"Experiment 2: {exp2_dir.name}")
    print(f"{'='*80}\n")
    
    # Load metrics
    try:
        metrics1 = load_metrics(exp1_dir)
        metrics2 = load_metrics(exp2_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Compare detection metrics
    print("INCIDENT DETECTION:")
    print(f"{'Metric':<30} {'Exp1':>10} {'Exp2':>10} {'Delta':>15}")
    print("-" * 80)
    
    det1 = metrics1.get("detection", {})
    det2 = metrics2.get("detection", {})
    
    for metric in ["precision", "recall", "f1"]:
        v1 = det1.get(metric)
        v2 = det2.get(metric)
        delta = format_delta(v1, v2, "percent")
        print(f"  {metric.capitalize():<28} {format_metric(v1):>10} {format_metric(v2):>10} {delta:>15}")
    
    print("\n  Counts:")
    for metric in ["true_positives", "false_positives", "false_negatives"]:
        v1 = det1.get(metric)
        v2 = det2.get(metric)
        delta = format_delta(v1, v2, "int") if v1 is not None and v2 is not None else ""
        print(f"  {metric:<28} {format_metric(v1, 'int'):>10} {format_metric(v2, 'int'):>10} {delta:>15}")
    
    # Compare severity metrics
    print("\n\nSEVERITY CLASSIFICATION:")
    print(f"{'Metric':<30} {'Exp1':>10} {'Exp2':>10} {'Delta':>15}")
    print("-" * 80)
    
    sev1 = metrics1.get("severity", {})
    sev2 = metrics2.get("severity", {})
    
    for metric in ["accuracy", "macro_f1"]:
        v1 = sev1.get(metric)
        v2 = sev2.get(metric)
        delta = format_delta(v1, v2, "percent")
        print(f"  {metric.capitalize():<28} {format_metric(v1):>10} {format_metric(v2):>10} {delta:>15}")
    
    # Per-class severity
    if "per_class" in sev1 and "per_class" in sev2:
        print("\n  Per-class F1:")
        for sev_class in ["low", "medium", "high", "critical"]:
            v1 = sev1["per_class"].get(sev_class, {}).get("f1")
            v2 = sev2["per_class"].get(sev_class, {}).get("f1")
            delta = format_delta(v1, v2, "percent") if v1 is not None and v2 is not None else ""
            print(f"    {sev_class:<26} {format_metric(v1):>10} {format_metric(v2):>10} {delta:>15}")
    
    # Compare disposition metrics
    print("\n\nDISPOSITION CLASSIFICATION:")
    print(f"{'Metric':<30} {'Exp1':>10} {'Exp2':>10} {'Delta':>15}")
    print("-" * 80)
    
    disp1 = metrics1.get("disposition", {})
    disp2 = metrics2.get("disposition", {})
    
    for metric in ["accuracy", "macro_f1"]:
        v1 = disp1.get(metric)
        v2 = disp2.get(metric)
        delta = format_delta(v1, v2, "percent")
        print(f"  {metric.capitalize():<28} {format_metric(v1):>10} {format_metric(v2):>10} {delta:>15}")
    
    # Structured output metrics
    print("\n\nSTRUCTURED OUTPUT:")
    print(f"{'Metric':<30} {'Exp1':>10} {'Exp2':>10} {'Delta':>15}")
    print("-" * 80)
    
    struct1 = metrics1.get("structured_output", {})
    struct2 = metrics2.get("structured_output", {})
    
    v1 = struct1.get("valid_json_rate")
    v2 = struct2.get("valid_json_rate")
    delta = format_delta(v1, v2, "percent")
    print(f"  Valid JSON rate{'':<18} {format_metric(v1):>10} {format_metric(v2):>10} {delta:>15}")
    
    # Field completion
    if "field_completion" in struct1 and "field_completion" in struct2:
        print("\n  Field completion:")
        for field in ["summary", "ticket_title", "ticket_body", "next_steps"]:
            v1 = struct1["field_completion"].get(field)
            v2 = struct2["field_completion"].get(field)
            delta = format_delta(v1, v2, "percent") if v1 is not None and v2 is not None else ""
            print(f"    {field:<26} {format_metric(v1):>10} {format_metric(v2):>10} {delta:>15}")
    
    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    
    summary1 = metrics1.get("summary", {})
    summary2 = metrics2.get("summary", {})
    
    improvements = []
    regressions = []
    
    key_metrics = [
        ("detection_f1", "Detection F1"),
        ("severity_accuracy", "Severity Accuracy"),
        ("disposition_accuracy", "Disposition Accuracy"),
        ("valid_json_rate", "Valid JSON Rate"),
    ]
    
    for key, label in key_metrics:
        v1 = summary1.get(key)
        v2 = summary2.get(key)
        
        if v1 is not None and v2 is not None:
            delta = v2 - v1
            delta_str = f"{delta*100:+.2f}%"
            
            if delta > 0.01:  # Improvement threshold
                improvements.append(f"  ✓ {label}: {delta_str}")
            elif delta < -0.01:  # Regression threshold
                regressions.append(f"  ✗ {label}: {delta_str}")
    
    if improvements:
        print("\nImprovements (Exp2 vs Exp1):")
        for imp in improvements:
            print(imp)
    
    if regressions:
        print("\nRegressions (Exp2 vs Exp1):")
        for reg in regressions:
            print(reg)
    
    if not improvements and not regressions:
        print("\nNo significant changes detected")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python compare.py results/baseline results/improved
"""
    )
    
    parser.add_argument(
        "experiment1",
        type=str,
        help="Path to first experiment directory",
    )
    parser.add_argument(
        "experiment2",
        type=str,
        help="Path to second experiment directory",
    )
    
    args = parser.parse_args()
    
    exp1_dir = Path(args.experiment1)
    exp2_dir = Path(args.experiment2)
    
    if not exp1_dir.exists():
        print(f"❌ Experiment 1 not found: {exp1_dir}")
        sys.exit(1)
    
    if not exp2_dir.exists():
        print(f"❌ Experiment 2 not found: {exp2_dir}")
        sys.exit(1)
    
    compare_experiments(exp1_dir, exp2_dir)


if __name__ == "__main__":
    main()
