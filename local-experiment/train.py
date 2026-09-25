#!/usr/bin/env python3
"""
Local QLoRA training using production IncidentLens components.

Isolated from AWS: no S3, no EC2, no Datadog, no production database.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add parent directory to path to import production modules
sys.path.insert(0, str(Path(__file__).parent.parent / "log-analyzer"))

from app.training.lora_trainer import TransformersPeftTrainingEngine
from app.training.training_engine import TrainingRequest
from app.training.training_profile import LoraTrainingProfile


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_file_sha256(path: Path) -> str:
    """Calculate SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_config(config_path: Path) -> dict:
    """Load experiment configuration."""
    with config_path.open() as f:
        return yaml.safe_load(f)


def create_experiment_dir(output_dir: Path) -> Path:
    """Create timestamped experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = output_dir / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_metadata(exp_dir: Path, config: dict, dataset_path: Path, git_sha: str):
    """Save experiment metadata for reproducibility."""
    import torch
    import transformers
    import peft
    import bitsandbytes
    
    metadata = {
        "experiment_id": exp_dir.name,
        "timestamp": datetime.now().isoformat(),
        "git_sha": git_sha,
        "dataset": {
            "path": str(dataset_path),
            "sha256": get_file_sha256(dataset_path),
        },
        "config": config,
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "peft_version": peft.__version__,
            "bitsandbytes_version": bitsandbytes.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    }
    
    with (exp_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def print_system_info():
    """Print GPU and system information."""
    import torch
    
    print("\n" + "=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  WARNING: No GPU detected - training will be extremely slow!")
    
    print("=" * 80 + "\n")


def load_and_validate_dataset(dataset_path: Path) -> tuple[bytes, int]:
    """Load dataset and count records."""
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)
    
    with dataset_path.open("rb") as f:
        content = f.read()
    
    # Count records
    record_count = 0
    for line in content.decode("utf-8").splitlines():
        if line.strip():
            record_count += 1
    
    return content, record_count


def main():
    parser = argparse.ArgumentParser(description="Local QLoRA training")
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
        help="Training dataset (default: data/train.jsonl)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory (default: results/)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate dataset, don't train"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if not args.config.exists():
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Print system info
    print_system_info()
    
    # Validate dataset
    print("Validating dataset...")
    validate_result = subprocess.run(
        [sys.executable, "validate_dataset.py", "--dataset", str(args.dataset)],
        cwd=Path(__file__).parent
    )
    
    if validate_result.returncode != 0:
        print("\n❌ Dataset validation failed!")
        sys.exit(1)
    
    if args.validate_only:
        print("\n✅ Dataset validation passed - exiting (--validate-only)")
        sys.exit(0)
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset_content, record_count = load_and_validate_dataset(args.dataset)
    print(f"  Records: {record_count}")
    print(f"  Size: {len(dataset_content) / 1024 / 1024:.2f} MB")
    
    # Create experiment directory
    exp_dir = create_experiment_dir(args.output_dir)
    print(f"\nExperiment directory: {exp_dir}")
    
    # Save metadata
    git_sha = get_git_sha()
    metadata = save_metadata(exp_dir, config, args.dataset, git_sha)
    print(f"Git SHA: {git_sha}")
    
    # Create LoRA profile from config
    lora_config = config["lora"]
    training_config = config["training"]
    
    profile = LoraTrainingProfile(
        rank=lora_config["rank"],
        alpha=lora_config["alpha"],
        dropout=lora_config["dropout"],
        target_modules=tuple(lora_config["target_modules"]),
        epochs=training_config["num_epochs"],  # Note: LoraTrainingProfile uses 'epochs'
        learning_rate=training_config["learning_rate"],
        batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        max_sequence_length=training_config["max_seq_length"],
        validation_fraction=config["data"]["eval_split"],
        seed=config["data"]["random_seed"],
    )
    
    # Create adapter output path (trainer requires this directory to exist and be empty)
    adapter_dir = exp_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = str(adapter_dir)
    
    # Create training request
    request = TrainingRequest(
        project_id="local-experiment",
        dataset_id=args.dataset.stem,
        dataset_version="local",
        base_model=config["base_model"],
        dataset_content=dataset_content,
        expected_record_count=record_count,
        adapter_output_path=adapter_path,
        profile=profile,
        progress_callback=None
    )
    
    # Train!
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Base model: {config['base_model']}")
    print(f"LoRA rank: {lora_config['rank']}, alpha: {lora_config['alpha']}")
    print(f"Learning rate: {training_config['learning_rate']}")
    print(f"Epochs: {training_config['num_epochs']}")
    print(f"Batch size: {training_config['per_device_train_batch_size']}")
    print(f"Gradient accumulation: {training_config['gradient_accumulation_steps']}")
    print(f"Effective batch size: {training_config['per_device_train_batch_size'] * training_config['gradient_accumulation_steps']}")
    print("=" * 80 + "\n")
    
    engine = TransformersPeftTrainingEngine()
    
    try:
        result = engine.train(request)
        
        if not result.succeeded:
            print(f"\n❌ Training failed: {result.error}")
            sys.exit(1)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Adapter saved: {result.adapter_path}")
        
        # Save training result
        result_data = {
            "succeeded": result.succeeded,
            "engine": result.engine,
            "adapter_path": result.adapter_path,
            "metrics": result.metrics,
            "framework_versions": result.framework_versions,
            "error": result.error,
        }
        
        with (exp_dir / "training_result.json").open("w") as f:
            json.dump(result_data, f, indent=2)
        
        # Print metrics
        if result.metrics:
            print("\nTraining Metrics:")
            for key, value in result.metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
        
        print("\n" + "=" * 80)
        print(f"Experiment saved to: {exp_dir}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
