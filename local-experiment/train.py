"""
Training script for incident analysis model.

Uses QLoRA fine-tuning with production-compatible system prompt.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

from data_loader import load_training_data, validate_training_data


# Production-compatible system prompt
SYSTEM_PROMPT = """You are an expert SRE analyzing production incidents.

You will receive incident metadata and sample logs. Analyze and respond with a JSON object:

{
  "severity": "low|medium|high|critical",
  "disposition": "NO_ACTION|OBSERVE|NEEDS_DEV|NEEDS_ONCALL|ESCALATE",
  "confidence": 0.0-1.0,
  "summary": "2-3 sentence summary",
  "suspected_root_cause": "short explanation or null",
  "next_steps": ["step1", "step2", "step3"],
  "ticket_title": "concise title under 100 chars",
  "ticket_body": "detailed description for developers"
}

Severity rules:
- CRITICAL: service down, data loss, OutOfMemoryError, heap exhaustion, segfaults
- HIGH: database connection errors, NPE, major features broken, cascades
- MEDIUM: partial degradation, intermittent errors
- LOW: single occurrence, cosmetic, known noise

Disposition rules:
- ESCALATE: page on-call NOW (critical/high + widespread impact)
- NEEDS_ONCALL: notify on-call during business hours
- NEEDS_DEV: create dev ticket
- OBSERVE: watch for recurrence
- NO_ACTION: known noise, benign activity"""


def format_training_example(example: Dict[str, str], tokenizer) -> Dict[str, str]:
    """Format example for chat-based fine-tuning."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    
    # Use tokenizer's chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    return {"text": text}


def prepare_training_dataset(
    data_path: Path,
    tokenizer,
    max_length: int = 2048
) -> Dataset:
    """Load and prepare training dataset."""
    print(f"Loading training data from {data_path}...")
    examples = load_training_data(data_path)
    
    print(f"\nValidating {len(examples)} examples...")
    report = validate_training_data(examples)
    
    print(f"\nValidation Results:")
    print(f"  Total: {report['total_examples']}")
    print(f"  Valid: {report['valid_examples']}")
    print(f"  Invalid: {report['invalid_examples']}")
    print(f"\nSeverity Distribution:")
    for sev, count in report['severity_distribution'].items():
        print(f"  {sev}: {count}")
    print(f"\nDisposition Distribution:")
    for disp, count in report['disposition_distribution'].items():
        print(f"  {disp}: {count}")
    print(f"\nBenign: {report['benign_count']}")
    print(f"Actionable: {report['actionable_count']}")
    
    if report['invalid_examples'] > 0:
        print(f"\n⚠️  Found {report['invalid_examples']} invalid examples!")
        for error in report['errors'][:5]:  # Show first 5
            print(f"  Example {error['example_index']}: {error['errors']}")
        raise ValueError(f"Dataset validation failed with {report['invalid_examples']} errors")
    
    # Convert to training format
    print("\nConverting to training format...")
    training_data = [ex.to_training_format() for ex in examples]
    
    # Format with tokenizer
    formatted = [format_training_example(ex, tokenizer) for ex in training_data]
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(formatted)
    
    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    return tokenized


def train(args):
    """Run training."""
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config["model"]
    lora_config = config["lora"]
    training_config = config["training"]
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"train_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"IncidentLens Training")
    print(f"{'='*60}")
    print(f"Model: {model_config['base_model']}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["base_model"],
        trust_remote_code=True,
    )
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset
    dataset = prepare_training_dataset(
        Path(args.dataset),
        tokenizer,
        max_length=training_config.get("max_seq_length", 2048),
    )
    
    print(f"\nPrepared {len(dataset)} training examples")
    
    # Quantization config for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    print("\nLoading base model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        target_modules=lora_config["target_modules"],
        lora_dropout=lora_config["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    print("\nApplying LoRA adapters...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config["epochs"],
        per_device_train_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        fp16=True,
        logging_steps=training_config.get("logging_steps", 10),
        save_strategy="epoch",
        save_total_limit=2,
        warmup_steps=training_config.get("warmup_steps", 100),
        weight_decay=training_config.get("weight_decay", 0.01),
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # Save metadata
    metadata = {
        "training_started": datetime.now().isoformat(),
        "base_model": model_config["base_model"],
        "dataset": str(args.dataset),
        "dataset_size": len(dataset),
        "lora_config": lora_config,
        "training_config": training_config,
        "git_sha": os.popen("git rev-parse HEAD").read().strip(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\nSaving adapter...")
    adapter_path = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    
    # Update metadata
    metadata["training_completed"] = datetime.now().isoformat()
    metadata["adapter_path"] = str(adapter_path)
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Adapter saved to: {adapter_path}")
    print(f"{'='*60}\n")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train incident analysis model")
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
        "--output",
        type=str,
        default="results",
        help="Output directory for trained model",
    )
    
    args = parser.parse_args()
    
    if not Path(args.dataset).exists():
        print(f"❌ Dataset not found: {args.dataset}")
        sys.exit(1)
    
    if not Path(args.config).exists():
        print(f"❌ Config not found: {args.config}")
        sys.exit(1)
    
    train(args)


if __name__ == "__main__":
    main()
