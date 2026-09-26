"""
End-to-end experiment orchestrator.

Runs the complete pipeline:
1. Validate dataset
2. Train model
3. Evaluate on test data
4. Generate metrics
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ {description} FAILED (exit code {result.returncode})")
        return False
    
    print(f"\n✓ {description} completed successfully")
    return True


def run_experiment(args):
    """Run complete experiment pipeline."""
    
    # Resolve paths
    config_path = Path(args.config).resolve()
    dataset_path = Path(args.dataset).resolve()
    eval_log_path = Path(args.evaluation_log).resolve()
    ground_truth_path = Path(args.ground_truth).resolve()
    output_dir = Path(args.output).resolve()
    
    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.name or f"experiment_{timestamp}"
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"IncidentLens Experiment: {experiment_name}")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Evaluation log: {eval_log_path}")
    print(f"Ground truth: {ground_truth_path}")
    print(f"Output: {experiment_dir}")
    print(f"{'='*60}\n")
    
    # Metadata
    metadata = {
        "experiment_name": experiment_name,
        "started_at": datetime.now().isoformat(),
        "config": str(config_path),
        "dataset": str(dataset_path),
        "evaluation_log": str(eval_log_path),
        "ground_truth": str(ground_truth_path),
        "steps_completed": [],
    }
    
    # Step 1: Validate dataset
    if not args.skip_validation:
        success = run_command(
            ["python", "validate_dataset.py", "--dataset", str(dataset_path)],
            "Step 1: Validate Training Dataset"
        )
        if not success:
            print("\n❌ Experiment failed at dataset validation")
            sys.exit(1)
        metadata["steps_completed"].append("validate_dataset")
    else:
        print("\nSkipping dataset validation (--skip-validation)")
    
    # Step 2: Validate ground truth
    if not args.skip_validation:
        success = run_command(
            ["python", "validate_ground_truth.py", "--ground-truth", str(ground_truth_path)],
            "Step 2: Validate Ground Truth"
        )
        if not success:
            print("\n❌ Experiment failed at ground truth validation")
            sys.exit(1)
        metadata["steps_completed"].append("validate_ground_truth")
    else:
        print("\nSkipping ground truth validation (--skip-validation)")
    
    # Step 3: Train model
    if not args.skip_training:
        train_output = experiment_dir / "training"
        success = run_command(
            [
                "python", "train.py",
                "--config", str(config_path),
                "--dataset", str(dataset_path),
                "--output", str(train_output),
            ],
            "Step 3: Train Model"
        )
        if not success:
            print("\n❌ Experiment failed at training")
            sys.exit(1)
        
        # Find the adapter directory
        train_dirs = sorted(train_output.glob("train_*"))
        if not train_dirs:
            print("\n❌ No training output found")
            sys.exit(1)
        
        latest_train_dir = train_dirs[-1]
        adapter_path = latest_train_dir / "adapter"
        
        if not adapter_path.exists():
            print(f"\n❌ Adapter not found at {adapter_path}")
            sys.exit(1)
        
        metadata["adapter_path"] = str(adapter_path)
        metadata["steps_completed"].append("train")
    else:
        if not args.adapter:
            print("\n❌ --skip-training requires --adapter")
            sys.exit(1)
        adapter_path = Path(args.adapter)
        metadata["adapter_path"] = str(adapter_path)
        print(f"\nSkipping training (--skip-training), using adapter: {adapter_path}")
    
    # Step 4: Evaluate model
    eval_output = experiment_dir / "evaluation"
    success = run_command(
        [
            "python", "evaluate.py",
            "--config", str(config_path),
            "--logs", str(eval_log_path),
            "--ground-truth", str(ground_truth_path),
            "--adapter", str(adapter_path),
            "--output", str(eval_output),
        ],
        "Step 4: Evaluate Model"
    )
    if not success:
        print("\n❌ Experiment failed at evaluation")
        sys.exit(1)
    
    metadata["steps_completed"].append("evaluate")
    
    # Find evaluation results
    eval_dirs = sorted(eval_output.glob("eval_*"))
    if not eval_dirs:
        print("\n❌ No evaluation output found")
        sys.exit(1)
    
    latest_eval_dir = eval_dirs[-1]
    metadata["evaluation_path"] = str(latest_eval_dir)
    
    # Copy metrics to experiment root for easy access
    metrics_file = latest_eval_dir / "metrics.json"
    if metrics_file.exists():
        import shutil
        shutil.copy(metrics_file, experiment_dir / "metrics.json")
        print(f"\n✓ Copied metrics to {experiment_dir / 'metrics.json'}")
    
    # Finalize metadata
    metadata["completed_at"] = datetime.now().isoformat()
    metadata["status"] = "completed"
    
    metadata_file = experiment_dir / "experiment_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Experiment Complete: {experiment_name}")
    print(f"{'='*60}")
    print(f"Results: {experiment_dir}")
    print(f"  - Training: {metadata.get('adapter_path', 'N/A')}")
    print(f"  - Evaluation: {metadata.get('evaluation_path', 'N/A')}")
    print(f"  - Metrics: {experiment_dir / 'metrics.json'}")
    
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        
        summary = metrics.get("summary", {})
        print(f"\nQuick Metrics:")
        print(f"  Detection F1: {summary.get('detection_f1', 0):.3f}")
        print(f"  Severity Accuracy: {summary.get('severity_accuracy', 0):.3f}")
        print(f"  Disposition Accuracy: {summary.get('disposition_accuracy', 0):.3f}")
        print(f"  Valid JSON Rate: {summary.get('valid_json_rate', 0):.3f}")
    
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete IncidentLens experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experiment
  python run_experiment.py \\
    --dataset ../data/dataset_v2.jsonl \\
    --evaluation-log data/evaluation.log \\
    --ground-truth ground_truth/evaluation.json

  # Run with custom name
  python run_experiment.py \\
    --name "qwen3b_baseline" \\
    --dataset ../data/dataset_v2.jsonl \\
    --evaluation-log data/evaluation.log \\
    --ground-truth ground_truth/evaluation.json

  # Evaluate existing adapter without retraining
  python run_experiment.py \\
    --skip-training \\
    --adapter results/train_20260925_120000/adapter \\
    --evaluation-log data/evaluation.log \\
    --ground-truth ground_truth/evaluation.json
"""
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to training dataset (JSONL)",
    )
    parser.add_argument(
        "--evaluation-log",
        type=str,
        required=True,
        help="Path to evaluation log file",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Path to ground truth JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for experiments",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Experiment name (default: experiment_TIMESTAMP)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip dataset and ground truth validation",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (requires --adapter)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        help="Path to existing adapter (for --skip-training)",
    )
    
    args = parser.parse_args()
    
    # Validate required files
    if not Path(args.config).exists():
        print(f"❌ Config not found: {args.config}")
        sys.exit(1)
    
    if not Path(args.dataset).exists():
        print(f"❌ Dataset not found: {args.dataset}")
        sys.exit(1)
    
    if not Path(args.evaluation_log).exists():
        print(f"❌ Evaluation log not found: {args.evaluation_log}")
        sys.exit(1)
    
    if not Path(args.ground_truth).exists():
        print(f"❌ Ground truth not found: {args.ground_truth}")
        sys.exit(1)
    
    if args.skip_training and not args.adapter:
        print(f"❌ --skip-training requires --adapter")
        sys.exit(1)
    
    run_experiment(args)


if __name__ == "__main__":
    main()
