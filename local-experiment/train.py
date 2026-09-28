#!/usr/bin/env python3
"""Run the application training engine locally; no separate training logic lives here."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "log-analyzer"))

from app.shared.model_config import configured_base_model
from app.training.lora_trainer import TransformersPeftTrainingEngine
from app.training.training_engine import TrainingRequest
from app.training.training_profile import LoraTrainingProfile


def main():
    parser = argparse.ArgumentParser(description="Train an IncidentLens LoRA adapter locally")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text()) or {}
    training = config.get("training", {})
    data = Path(args.data or training.get("data", "../data/dataset_v4.jsonl")).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = [line for line in data.read_bytes().splitlines() if line.strip()]
    profile = LoraTrainingProfile(
        rank=int(training.get("lora_r", 16)), alpha=int(training.get("lora_alpha", 32)),
        dropout=float(training.get("lora_dropout", 0.05)), epochs=float(training.get("epochs", 3)),
        learning_rate=float(training.get("learning_rate", 1e-4)), batch_size=int(training.get("batch_size", 2)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 8)),
        max_sequence_length=int(training.get("max_sequence_length", 1536)),
        validation_fraction=float(training.get("validation_split", 0.1)), seed=int(training.get("seed", 42)),
        target_modules=("all-linear",),
    )
    result = TransformersPeftTrainingEngine().train(TrainingRequest(
        project_id="local-experiment", dataset_id=data.stem, dataset_version=data.stem,
        base_model=args.model or config.get("model", {}).get("name") or configured_base_model(),
        dataset_content=data.read_bytes(), expected_record_count=len(rows), adapter_output_path=str(output), profile=profile,
        progress_callback=lambda current, total: print(f"progress {current}/{total}", flush=True),
    ))
    (output / "local-training-result.json").write_text(json.dumps({"metrics": result.metrics, "framework_versions": result.framework_versions}, indent=2) + "\n")
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
