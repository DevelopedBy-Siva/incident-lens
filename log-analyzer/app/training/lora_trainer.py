import hashlib
import importlib.metadata
import json
import math
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

from app.training.training_engine import (
    TrainingEngine,
    TrainingRequest,
    TrainingResult,
)


class TrainingDatasetError(ValueError):
    pass


class _ListDataset:
    def __init__(self, records: list[dict[str, list[int]]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.records[index]


class TransformersPeftTrainingEngine(TrainingEngine):
    """Train and persist a real PEFT LoRA adapter with Transformers."""

    name = "transformers-peft-lora-v1"
    system_prompt = (
        "You are an expert SRE. Analyze the supplied incident evidence and return "
        "the confirmed structured incident outcome."
    )

    def train(self, request: TrainingRequest) -> TrainingResult:
        dependencies = self._load_dependencies()
        output_path = Path(request.adapter_output_path).expanduser().resolve()
        if not output_path.is_dir():
            raise FileNotFoundError(
                f"Prepared adapter directory does not exist: {output_path}"
            )
        if any(output_path.iterdir()):
            raise FileExistsError(f"Adapter directory is not empty: {output_path}")

        examples = self._parse_examples(
            request.dataset_content,
            request.expected_record_count,
        )
        started = perf_counter()
        tokenizer = dependencies["AutoTokenizer"].from_pretrained(
            request.base_model,
            token=self._huggingface_token(),
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = dependencies["AutoModelForCausalLM"].from_pretrained(
            request.base_model,
            token=self._huggingface_token(),
            dtype="auto",
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
        lora_config = dependencies["LoraConfig"](
            task_type=dependencies["TaskType"].CAUSAL_LM,
            inference_mode=False,
            r=request.profile.rank,
            lora_alpha=request.profile.alpha,
            lora_dropout=request.profile.dropout,
            target_modules=list(request.profile.target_modules),
            bias="none",
        )
        model = dependencies["get_peft_model"](model, lora_config)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        tokenized = [
            self._tokenize_example(
                tokenizer,
                example,
                request.profile.max_sequence_length,
            )
            for example in examples
        ]
        train_records, validation_records = self._split_records(
            tokenized,
            request.profile.validation_fraction,
        )
        training_arguments = dependencies["TrainingArguments"](
            output_dir=str(output_path / ".trainer"),
            num_train_epochs=request.profile.epochs,
            learning_rate=request.profile.learning_rate,
            per_device_train_batch_size=request.profile.batch_size,
            per_device_eval_batch_size=request.profile.batch_size,
            gradient_accumulation_steps=request.profile.gradient_accumulation_steps,
            eval_strategy="epoch" if validation_records else "no",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=1,
            report_to="none",
            remove_unused_columns=False,
            gradient_checkpointing=True,
            seed=request.profile.seed,
            data_seed=request.profile.seed,
            dataloader_pin_memory=False,
        )
        data_collator = dependencies["DataCollatorForSeq2Seq"](
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
        )
        trainer = dependencies["Trainer"](
            model=model,
            args=training_arguments,
            train_dataset=_ListDataset(train_records),
            eval_dataset=(
                _ListDataset(validation_records) if validation_records else None
            ),
            data_collator=data_collator,
            processing_class=tokenizer,
        )

        train_output = trainer.train()
        evaluation_metrics = trainer.evaluate() if validation_records else {}
        model.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)
        self._remove_trainer_directory(output_path / ".trainer")

        duration = perf_counter() - started
        training_loss = self._finite_metric(
            train_output.metrics.get("train_loss", train_output.training_loss)
        )
        validation_loss = self._finite_metric(evaluation_metrics.get("eval_loss"))
        return TrainingResult(
            succeeded=True,
            engine=self.name,
            adapter_path=str(output_path),
            metrics={
                "training_completed": True,
                "base_model": request.base_model,
                "dataset_version": request.dataset_version,
                "expected_records": request.expected_record_count,
                "records_seen": len(examples),
                "training_records": len(train_records),
                "validation_records": len(validation_records),
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "duration_seconds": round(duration, 6),
                "weights_created": True,
                "adapter_format": "peft-lora",
                "lora_configuration": request.profile.as_metadata(),
                "train_metrics": self._json_metrics(train_output.metrics),
                "validation_metrics": self._json_metrics(evaluation_metrics),
            },
            framework_versions=self._framework_versions(),
            artifact_files=self._artifact_manifest(output_path),
        )

    @staticmethod
    def _load_dependencies() -> dict[str, Any]:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                DataCollatorForSeq2Seq,
                Trainer,
                TrainingArguments,
            )
        except ImportError as exc:
            raise RuntimeError(
                "LoRA training dependencies are unavailable; install the backend "
                "requirements before running a training job"
            ) from exc
        return {
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "AutoTokenizer": AutoTokenizer,
            "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
            "LoraConfig": LoraConfig,
            "TaskType": TaskType,
            "Trainer": Trainer,
            "TrainingArguments": TrainingArguments,
            "get_peft_model": get_peft_model,
        }

    @staticmethod
    def _parse_examples(content: bytes, expected_count: int) -> list[dict[str, Any]]:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TrainingDatasetError("Dataset is not valid UTF-8") from exc

        examples = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingDatasetError(
                    f"Dataset line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(example, dict):
                raise TrainingDatasetError(
                    f"Dataset line {line_number} must contain a JSON object"
                )
            if not isinstance(example.get("input"), dict) or not isinstance(
                example.get("expected_output"), dict
            ):
                raise TrainingDatasetError(
                    f"Dataset line {line_number} does not match the training schema"
                )
            examples.append(example)

        if not examples:
            raise TrainingDatasetError("Dataset contains no training examples")
        if len(examples) != expected_count:
            raise TrainingDatasetError(
                "Dataset record count does not match registered metadata"
            )
        return examples

    def _tokenize_example(
        self,
        tokenizer,
        example: dict[str, Any],
        max_sequence_length: int,
    ) -> dict[str, list[int]]:
        user_messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    example["input"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ]
        messages = user_messages + [
            {
                "role": "assistant",
                "content": json.dumps(
                    example["expected_output"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        ]
        prompt_ids = self._template_token_ids(
            tokenizer,
            user_messages,
            add_generation_prompt=True,
        )
        full_ids = self._template_token_ids(
            tokenizer,
            messages,
            add_generation_prompt=False,
        )
        input_ids, labels = self._completion_labels(
            prompt_ids,
            full_ids,
            max_sequence_length,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    @staticmethod
    def _template_token_ids(
        tokenizer,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> list[int]:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        if isinstance(encoded, Mapping):
            encoded = encoded.get("input_ids")
        if encoded is None:
            raise TrainingDatasetError("Tokenizer did not produce input token ids")
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return list(encoded)

    @staticmethod
    def _completion_labels(
        prompt_ids: list[int],
        full_ids: list[int],
        max_sequence_length: int,
    ) -> tuple[list[int], list[int]]:
        if full_ids[: len(prompt_ids)] == prompt_ids:
            completion_ids = full_ids[len(prompt_ids) :]
            if not completion_ids:
                raise TrainingDatasetError("Training example has no completion tokens")
            completion_ids = completion_ids[:max_sequence_length]
            prompt_budget = max_sequence_length - len(completion_ids)
            prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids
            return input_ids, labels

        input_ids = full_ids[-max_sequence_length:]
        if not input_ids:
            raise TrainingDatasetError("Training example produced no tokens")
        return input_ids, list(input_ids)

    @staticmethod
    def _split_records(
        records: list[dict[str, list[int]]], validation_fraction: float
    ) -> tuple[list[dict[str, list[int]]], list[dict[str, list[int]]]]:
        if len(records) < 2 or validation_fraction <= 0:
            return records, []
        validation_count = max(1, math.floor(len(records) * validation_fraction))
        validation_count = min(validation_count, len(records) - 1)
        return records[:-validation_count], records[-validation_count:]

    @staticmethod
    def _artifact_manifest(output_path: Path) -> tuple[dict[str, Any], ...]:
        files = []
        for path in sorted(
            candidate for candidate in output_path.iterdir() if candidate.is_file()
        ):
            files.append(
                {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        return tuple(files)

    @staticmethod
    def _framework_versions() -> dict[str, str]:
        versions = {}
        for package in (
            "torch",
            "transformers",
            "peft",
            "accelerate",
            "safetensors",
        ):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "unavailable"
        return versions

    @staticmethod
    def _json_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }

    @staticmethod
    def _finite_metric(value) -> float | None:
        if value is None:
            return None
        metric = float(value)
        return metric if math.isfinite(metric) else None

    @staticmethod
    def _huggingface_token() -> str | None:
        return os.getenv("HF_TOKEN", "").strip() or None

    @staticmethod
    def _remove_trainer_directory(path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)
