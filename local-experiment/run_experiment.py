#!/usr/bin/env python3
"""
Run complete experiment: validate → train → evaluate → summary.

CRITICAL: Training and evaluation data MUST be separate!
- Training: data/train.jsonl (structured training examples)
- Evaluation: data/evaluation.log (raw unseen logs)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run command and return success status."""
    print(f"\n{'=' * 80}")
    print(f"{description}")
    print(f"{'=' * 80}")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ {description} FAILED")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run complete experiment with proper train/eval separation"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/train.jsonl"),
        help="Training dataset - JSONL format (default: data/train.jsonl)"
    )
    parser.add_argument(
        "--evaluation-log",
        type=Path,
        default=Path("data/evaluation.log"),
        help="Evaluation log file - raw logs, NOT training data (default: data/evaluation.log)"
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("ground_truth/evaluation.json"),
        help="Ground truth for evaluation (optional)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, only evaluate existing adapter"
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Pre-trained adapter path (required if --skip-training)"
    )
    
    args = parser.parse_args()
    
    if args.skip_training and not args.adapter:
        print("ERROR: --adapter required when --skip-training is set")
        sys.exit(1)
    
    # Safety check: ensure we're not using the same file for training and eval
    if args.dataset == args.evaluation_log:
        print("❌ CRITICAL ERROR: Training and evaluation files are the same!")
        print(f"   Training: {args.dataset}")
        print(f"   Evaluation: {args.evaluation_log}")
        print("\n   Evaluation MUST use unseen data separate from training.")
        sys.exit(1)
    
    script_dir = Path(__file__).parent
    
    # Step 1: Validate training dataset
    if not run_command(
        [sys.executable, str(script_dir / "validate_dataset.py"), "--dataset", str(args.dataset)],
        "STEP 1: VALIDATE TRAINING DATASET"
    ):
        sys.exit(1)
    
    # Step 2: Train (if not skipping)
    adapter_path = args.adapter
    if not args.skip_training:
        if not run_command(
            [
                sys.executable,
                str(script_dir / "train.py"),
                "--config", str(args.config),
                "--dataset", str(args.dataset),
            ],
            "STEP 2: TRAIN ADAPTER (on train.jsonl)"
        ):
            sys.exit(1)
        
        # Find the latest experiment directory
        results_dir = script_dir / "results"
        experiments = sorted([d for d in results_dir.iterdir() if d.is_dir()], reverse=True)
        if not experiments:
            print("ERROR: No experiment directory found after training")
            sys.exit(1)
        
        latest_exp = experiments[0]
        adapter_path = latest_exp / "adapter"
        print(f"\nLatest experiment: {latest_exp}")
    else:
        print(f"\nSkipping training, using adapter: {adapter_path}")
        latest_exp = adapter_path.parent
    
    # Step 3: Evaluate on UNSEEN evaluation log
    if args.evaluation_log.exists():
        if not run_command(
            [
                sys.executable,
                str(script_dir / "evaluate.py"),
                "--adapter", str(adapter_path),
                "--logs", str(args.evaluation_log),  # IMPORTANT: Using evaluation log, NOT training data
                "--ground-truth", str(args.ground_truth),
                "--config", str(args.config),
                "--output", str(latest_exp),
            ],
            "STEP 3: EVALUATE ADAPTER (on evaluation.log)"
        ):
            sys.exit(1)
    else:
        print(f"\n⚠️  WARNING: Evaluation log not found: {args.evaluation_log}")
        print("   Skipping evaluation. Place your evaluation log file at:")
        print(f"   {args.evaluation_log}")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nResults: {latest_exp}")
    print(f"  Adapter: {adapter_path}")
    print(f"  Predictions: {latest_exp / 'predictions.jsonl'}")
    print(f"  Summary: {latest_exp / 'summary.json'}")
    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
