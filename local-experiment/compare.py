#!/usr/bin/env python3
"""
Compare metrics across multiple experiment runs.
"""

import argparse
import json
from pathlib import Path


def load_experiment_metrics(exp_dir: Path) -> dict:
    """Load metrics and metadata from experiment directory."""
    metrics_file = exp_dir / "metrics.json"
    metadata_file = exp_dir / "metadata.json"
    training_result_file = exp_dir / "training_result.json"
    
    data = {"experiment_id": exp_dir.name}
    
    if metrics_file.exists():
        with metrics_file.open() as f:
            data["metrics"] = json.load(f)
    
    if metadata_file.exists():
        with metadata_file.open() as f:
            metadata = json.load(f)
            data["config"] = metadata.get("config", {})
            data["git_sha"] = metadata.get("git_sha")
            data["dataset_sha256"] = metadata.get("dataset", {}).get("sha256")
    
    if training_result_file.exists():
        with training_result_file.open() as f:
            training_result = json.load(f)
            data["training_metrics"] = training_result.get("metrics", {})
    
    return data


def compare_experiments(experiments: list[dict]) -> str:
    """Generate comparison report."""
    lines = [
        "=" * 100,
        "EXPERIMENT COMPARISON",
        "=" * 100,
        "",
    ]
    
    # Table header
    lines.append(f"{'Experiment ID':<20} {'Det F1':>8} {'Sev Acc':>8} {'Sev F1':>8} {'Disp F1':>8} {'P50ms':>8} {'Valid':>6}")
    lines.append("-" * 100)
    
    # Table rows
    for exp in experiments:
        exp_id = exp.get("experiment_id", "unknown")
        metrics = exp.get("metrics", {})
        
        det_f1 = metrics.get("detection", {}).get("f1")
        sev_acc = metrics.get("severity", {}).get("accuracy")
        sev_f1 = metrics.get("severity", {}).get("macro_f1")
        disp_f1 = metrics.get("disposition", {}).get("macro_f1")
        p50 = metrics.get("latency", {}).get("p50_ms")
        valid = metrics.get("completion_rates", {}).get("valid_json")
        
        lines.append(
            f"{exp_id:<20} "
            f"{det_f1:>8.3f} " if det_f1 is not None else f"{exp_id:<20} {'N/A':>8} " +
            f"{sev_acc:>8.3f} " if sev_acc is not None else f"{'N/A':>8} " +
            f"{sev_f1:>8.3f} " if sev_f1 is not None else f"{'N/A':>8} " +
            f"{disp_f1:>8.3f} " if disp_f1 is not None else f"{'N/A':>8} " +
            f"{p50:>8.1f} " if p50 is not None else f"{'N/A':>8} " +
            f"{valid:>6.1%}" if valid is not None else f"{'N/A':>6}"
        )
    
    lines.append("")
    
    # Detailed differences
    if len(experiments) >= 2:
        lines.append("Configuration Differences:")
        lines.append("")
        
        # Compare configs
        for i in range(len(experiments) - 1):
            exp1 = experiments[i]
            exp2 = experiments[i + 1]
            
            config1 = exp1.get("config", {})
            config2 = exp2.get("config", {})
            
            lines.append(f"{exp1['experiment_id']} vs {exp2['experiment_id']}:")
            
            # Check LoRA params
            lora1 = config1.get("lora", {})
            lora2 = config2.get("lora", {})
            
            if lora1.get("rank") != lora2.get("rank"):
                lines.append(f"  LoRA rank: {lora1.get('rank')} → {lora2.get('rank')}")
            if lora1.get("alpha") != lora2.get("alpha"):
                lines.append(f"  LoRA alpha: {lora1.get('alpha')} → {lora2.get('alpha')}")
            
            # Check training params
            train1 = config1.get("training", {})
            train2 = config2.get("training", {})
            
            if train1.get("learning_rate") != train2.get("learning_rate"):
                lines.append(f"  Learning rate: {train1.get('learning_rate')} → {train2.get('learning_rate')}")
            if train1.get("num_epochs") != train2.get("num_epochs"):
                lines.append(f"  Epochs: {train1.get('num_epochs')} → {train2.get('num_epochs')}")
            
            lines.append("")
    
    lines.append("=" * 100)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare experiment runs")
    parser.add_argument(
        "experiments",
        nargs="+",
        type=Path,
        help="Experiment directories to compare"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save comparison to file"
    )
    
    args = parser.parse_args()
    
    # Load all experiments
    experiments = []
    for exp_dir in args.experiments:
        if not exp_dir.exists():
            print(f"WARNING: Experiment not found: {exp_dir}")
            continue
        
        try:
            exp_data = load_experiment_metrics(exp_dir)
            experiments.append(exp_data)
        except Exception as e:
            print(f"WARNING: Failed to load {exp_dir}: {e}")
    
    if len(experiments) < 2:
        print("ERROR: Need at least 2 valid experiments to compare")
        return
    
    # Generate comparison
    report = compare_experiments(experiments)
    
    # Print
    print(report)
    
    # Save if requested
    if args.output:
        with args.output.open("w") as f:
            f.write(report)
        print(f"\nComparison saved to: {args.output}")


if __name__ == "__main__":
    main()
