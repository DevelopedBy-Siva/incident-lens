"""One QLoRA training implementation shared by local and EC2 execution."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

from app.shared.incident_policy import DISPOSITIONS, SEVERITIES, SYSTEM_PROMPT, build_training_messages, validate_output_schema
from app.training.training_engine import TrainingEngine, TrainingRequest, TrainingResult


class TrainingDatasetError(ValueError):
    pass


class _ListDataset:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def _progress_callback(base, callback):
    class Callback(base):
        def on_train_begin(self, args, state, control, **kwargs):
            callback(0, state.max_steps)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            callback(state.global_step, state.max_steps)
            return control

    return Callback


class TransformersPeftTrainingEngine(TrainingEngine):
    name = "transformers-peft-lora-v3"
    system_prompt = SYSTEM_PROMPT

    def train(self, request: TrainingRequest) -> TrainingResult:
        output = Path(request.adapter_output_path).resolve()
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError("Adapter output directory must exist and be empty")
        examples = self._parse_examples(request.dataset_content, request.expected_record_count)
        audit = self._audit(examples)
        self._enforce_gates(audit)
        dependencies = self._load_dependencies()
        tokenizer = dependencies["AutoTokenizer"].from_pretrained(request.base_model, token=os.getenv("HF_TOKEN") or None)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenized = [self._tokenize_example(tokenizer, example, request.profile.max_sequence_length) for example in examples]
        train_records, validation_records, split = self._split(examples, tokenized, request.profile.validation_fraction, request.profile.seed)
        model, quantized, dtype = self._load_model(dependencies, request.base_model)
        lora = dependencies["LoraConfig"](
            task_type=dependencies["TaskType"].CAUSAL_LM,
            inference_mode=False,
            r=request.profile.rank,
            lora_alpha=request.profile.alpha,
            lora_dropout=request.profile.dropout,
            target_modules="all-linear",
            bias="none",
        )
        model = dependencies["get_peft_model"](model, lora)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        args = dependencies["TrainingArguments"](
            output_dir=str(output / ".trainer"),
            num_train_epochs=request.profile.epochs,
            learning_rate=request.profile.learning_rate,
            per_device_train_batch_size=request.profile.batch_size,
            per_device_eval_batch_size=request.profile.batch_size,
            gradient_accumulation_steps=request.profile.gradient_accumulation_steps,
            eval_strategy="epoch" if validation_records else "no",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=10,
            report_to="none",
            remove_unused_columns=False,
            gradient_checkpointing=True,
            seed=request.profile.seed,
            data_seed=request.profile.seed,
            dataloader_pin_memory=False,
            bf16=dtype == "bfloat16",
            fp16=dtype == "float16",
        )
        callbacks = [_progress_callback(dependencies["TrainerCallback"], request.progress_callback)()] if request.progress_callback else None
        trainer = dependencies["Trainer"](
            model=model,
            args=args,
            train_dataset=_ListDataset(train_records),
            eval_dataset=_ListDataset(validation_records) if validation_records else None,
            data_collator=dependencies["DataCollatorForSeq2Seq"](tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100),
            processing_class=tokenizer,
            callbacks=callbacks,
        )
        started = perf_counter()
        trained = trainer.train()
        evaluated = trainer.evaluate() if validation_records else {}
        model.save_pretrained(output, safe_serialization=True)
        tokenizer.save_pretrained(output)
        shutil.rmtree(output / ".trainer", ignore_errors=True)
        train_loss = self._number(trained.metrics.get("train_loss", getattr(trained, "training_loss", None)))
        validation_loss = self._number(evaluated.get("eval_loss"))
        return TrainingResult(
            succeeded=True,
            engine=self.name,
            adapter_path=str(output),
            metrics={
                "training_completed": True,
                "base_model": request.base_model,
                "records_seen": len(examples),
                "training_records": len(train_records),
                "validation_records": len(validation_records),
                "training_loss": train_loss,
                "validation_loss": validation_loss,
                "duration_seconds": round(perf_counter() - started, 3),
                "quantized_4bit": quantized,
                "adapter_format": "peft-lora",
                "weights_created": True,
                "lora_configuration": request.profile.as_metadata(),
                "dataset_audit": audit,
                "split_stats": split,
            },
            framework_versions=self._versions(),
            artifact_files=self._artifact_manifest(output),
        )

    @staticmethod
    def _parse_examples(content: bytes, expected_count: int) -> list[dict[str, Any]]:
        try:
            rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TrainingDatasetError("Dataset must be valid UTF-8 JSONL") from exc
        if len(rows) != expected_count:
            raise TrainingDatasetError(f"Dataset record count {len(rows)} does not match expected {expected_count}")
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict) or set(row) != {"input", "expected_output"}:
                raise TrainingDatasetError(f"Dataset line {index}: expected keys 'input' and 'expected_output'")
            logs = row["input"].get("logs") if isinstance(row["input"], dict) else None
            if not isinstance(logs, list) or not logs or not all(isinstance(line, str) and line.strip() for line in logs):
                raise TrainingDatasetError(f"Dataset line {index}: input.logs must be a non-empty list")
            valid, reason = validate_output_schema(row["expected_output"])
            if not valid:
                raise TrainingDatasetError(f"Dataset line {index}: {reason}")
        return rows

    @staticmethod
    def _completion_labels(prompt_ids, full_ids, max_length):
        # A rendered chat prompt with ``add_generation_prompt=True`` is not
        # guaranteed to be an exact token prefix of the same conversation with
        # an assistant response.  Qwen (and several other templates) render a
        # different assistant marker in those two cases.  The user/system
        # context is still the common token prefix; train from the first token
        # after that shared context rather than rejecting an otherwise valid
        # example.
        boundary = 0
        for prompt_token, full_token in zip(prompt_ids, full_ids):
            if prompt_token != full_token:
                break
            boundary += 1
        if boundary == 0:
            raise TrainingDatasetError("Could not identify a shared chat context")
        context_ids = full_ids[:boundary]
        target = full_ids[boundary:]
        if not target:
            raise TrainingDatasetError("Assistant target is empty")
        if len(full_ids) > max_length:
            available_prompt = max_length - len(target)
            if available_prompt <= 0:
                raise TrainingDatasetError("Assistant target exceeds maximum sequence length")
            context_ids = context_ids[-available_prompt:]
            full_ids = context_ids + target
        return full_ids, [-100] * len(context_ids) + target

    def _tokenize_example(self, tokenizer, example, max_length):
        messages = build_training_messages(example["input"]["logs"], example["expected_output"])
        prompt = self._template_ids(tokenizer, messages[:-1], True)
        full = self._template_ids(tokenizer, messages, False)
        ids, labels = self._completion_labels(prompt, full, max_length)
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

    @staticmethod
    def _template_ids(tokenizer, messages, generation):
        try:
            rendered = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation, enable_thinking=False)
        except TypeError:
            rendered = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation)
        # Qwen3.5's tokenizer can return a BatchEncoding instead of a bare
        # token-id list.  Iterating a BatchEncoding yields field names
        # (``input_ids``, ``attention_mask``), which made the previous code
        # see an empty assistant response.  Normalize both forms here.
        if isinstance(rendered, dict) or hasattr(rendered, "input_ids"):
            rendered = rendered["input_ids"]
        if hasattr(rendered, "tolist"):
            rendered = rendered.tolist()
        if rendered and isinstance(rendered[0], (list, tuple)):
            if len(rendered) != 1:
                raise TrainingDatasetError("Chat template must render one training example")
            rendered = rendered[0]
        if not isinstance(rendered, (list, tuple)) or not all(isinstance(token, int) for token in rendered):
            raise TrainingDatasetError("Chat template did not return token IDs")
        return list(rendered)

    @staticmethod
    def _split(examples, records, fraction, seed):
        if fraction <= 0 or len(records) < 2:
            return records, [], {"strategy": "none", "train_examples": len(records), "validation_examples": 0}
        groups = defaultdict(list)
        for index, example in enumerate(examples):
            metadata = example["input"].get("metadata", {})
            key = str(metadata.get("scenario_group_id") or hashlib.sha256("\n".join(example["input"]["logs"]).encode()).hexdigest())
            groups[key].append(index)
        keys = list(groups)
        random.Random(seed).shuffle(keys)
        target = max(1, round(len(records) * fraction))
        selected, size = set(), 0
        for key in keys:
            if size >= target:
                break
            if len(groups[key]) >= len(records):
                continue
            selected.add(key)
            size += len(groups[key])
        validation_indices = {index for key in selected for index in groups[key]}
        if not validation_indices or len(validation_indices) == len(records):
            return records, [], {"strategy": "group", "train_examples": len(records), "validation_examples": 0}
        return ([record for index, record in enumerate(records) if index not in validation_indices], [record for index, record in enumerate(records) if index in validation_indices], {"strategy": "group", "train_examples": len(records) - len(validation_indices), "validation_examples": len(validation_indices), "scenario_groups": len(groups)})

    @staticmethod
    def _audit(examples):
        return {"examples": len(examples), "severity": dict(Counter(row["expected_output"]["severity"] for row in examples)), "disposition": dict(Counter(row["expected_output"]["disposition"] for row in examples))}

    @staticmethod
    def _enforce_gates(audit):
        if os.getenv("TRAINING_FAIL_ON_QUALITY_GATE", "1") in {"0", "false", "False"}:
            return
        missing = SEVERITIES - set(audit["severity"])
        missing_dispositions = DISPOSITIONS - set(audit["disposition"])
        if missing or missing_dispositions:
            raise TrainingDatasetError(f"Dataset must include every severity and disposition; missing severities={sorted(missing)}, dispositions={sorted(missing_dispositions)}")

    @staticmethod
    def _load_dependencies():
        try:
            import torch
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments
        except ImportError as exc:
            raise RuntimeError("Install the training dependencies before running LoRA training") from exc
        return locals()

    @staticmethod
    def _load_model(dependencies, base_model):
        torch = dependencies["torch"]
        cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else (torch.float16 if cuda else torch.float32)
        use_4bit = cuda and os.getenv("TRAINING_USE_4BIT", "auto").lower() not in {"0", "false", "no"}
        kwargs = {"token": os.getenv("HF_TOKEN") or None, "torch_dtype": dtype, "low_cpu_mem_usage": True}
        if use_4bit:
            kwargs["quantization_config"] = dependencies["BitsAndBytesConfig"](load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
            kwargs["device_map"] = "auto"
        model = dependencies["AutoModelForCausalLM"].from_pretrained(base_model, **kwargs)
        if use_4bit:
            model = dependencies["prepare_model_for_kbit_training"](model)
        return model, use_4bit, "bfloat16" if dtype == torch.bfloat16 else ("float16" if dtype == torch.float16 else "float32")

    @staticmethod
    def _number(value):
        try:
            value = float(value)
            return value if value == value and abs(value) != float("inf") else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _versions():
        return {name: importlib.metadata.version(name) for name in ("transformers", "peft", "torch") if _installed(name)}

    @staticmethod
    def _artifact_manifest(directory):
        return tuple({"name": path.name, "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(directory.iterdir()) if path.is_file())


def _installed(package):
    try:
        importlib.metadata.version(package)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False
