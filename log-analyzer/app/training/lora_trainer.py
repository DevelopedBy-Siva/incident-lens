"""
Production QLoRA/LoRA Training Engine for IncidentLens

This implementation matches the FINAL VERIFIED local training behavior:
- Uses canonical policy/prompt from app.shared.incident_policy
- Preserves complete assistant target tokens (NEVER truncates targets)
- Implements group-stratified validation split
- Uses enable_thinking=False for Qwen chat template
- Supports QLoRA with 4-bit quantization when available
- Implements dataset quality gates and validation
- Uses assistant-only target loss (prompt tokens masked)

Training configuration:
- Base model: Qwen/Qwen3.5-4B
- LoRA: r=8, alpha=16, dropout=0.05, target_modules=all-linear
- Learning rate: 1e-4, epochs=1, batch_size=1, gradient_accumulation=1
- Max sequence length: 1024
- Validation: 10% group-stratified
- Seed: 42
"""

import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

from app.shared.incident_policy import (
    DISPOSITIONS,
    POLICY_VERSION,
    REQUIRED_OUTPUT_FIELDS,
    SEVERITIES,
    SYSTEM_PROMPT,
    build_training_messages,
    validate_output_schema,
)
from app.training.training_engine import (
    TrainingEngine,
    TrainingRequest,
    TrainingResult,
)


class TrainingDatasetError(ValueError):
    pass


class _ListDataset:
    """Simple list-backed dataset for Transformers Trainer."""

    def __init__(self, records: list[dict[str, list[int]]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.records[index]


def _progress_callback_class(base_callback, report):
    """Create progress callback for Trainer."""

    class ProgressCallback(base_callback):
        def on_train_begin(self, args, state, control, **kwargs):
            report(0, state.max_steps)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            report(state.global_step, state.max_steps)
            return control

    return ProgressCallback


# Pattern matching for scenario normalization
ISO_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})")
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HEX_RE = re.compile(r"\b0x[0-9a-f]+\b", re.I)
VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?(?:-[a-z0-9]+(?:\.[a-z0-9]+)*)?\b", re.I)
API_PATH_RE = re.compile(r"/[a-z0-9_-]+(?:/[a-z0-9_-]+)+", re.I)
NUMBER_RE = re.compile(r"\b\d+\b")
BRACKET_SOURCE_RE = re.compile(r"\[([^\]]+)\]")
LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)")
RECOVERY_RE = re.compile(
    r"\b(recovered|recovery|restored|normalized|stabilized|healthy|draining|resolved|rollback complete|mitigation succeeded)\b",
    re.I,
)


class TransformersPeftTrainingEngine(TrainingEngine):
    """Production training engine matching final verified local implementation."""

    name = "transformers-peft-lora-v2"  # Incremented version for new implementation
    
    # Export system prompt for testing/validation
    system_prompt = SYSTEM_PROMPT

    def train(self, request: TrainingRequest) -> TrainingResult:
        """Execute complete training pipeline with final verified behavior."""
        dependencies = self._load_dependencies()
        output_path = Path(request.adapter_output_path).expanduser().resolve()
        if not output_path.is_dir():
            raise FileNotFoundError(
                f"Adapter output directory does not exist: {output_path}"
            )
        if any(output_path.iterdir()):
            raise FileExistsError(f"Adapter output directory is not empty: {output_path}")

        # Parse and validate dataset
        examples = self._parse_examples(
            request.dataset_content,
            request.expected_record_count,
        )

        # Run dataset audit and enforce quality gates
        audit = self._dataset_audit(examples)
        self._enforce_quality_gates(audit)

        started = perf_counter()

        # Load tokenizer
        tokenizer = dependencies["AutoTokenizer"].from_pretrained(
            request.base_model,
            token=self._huggingface_token(),
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize examples with complete target preservation
        tokenized = [
            self._tokenize_example(
                tokenizer,
                example,
                request.profile.max_sequence_length,
            )
            for example in examples
        ]

        # Group-stratified validation split
        train_records, validation_records, split_stats = self._group_stratified_split(
            examples,
            tokenized,
            request.profile.validation_fraction,
            request.profile.seed,
        )

        # Load model (with QLoRA if available and configured)
        model, quantized, dtype = self._load_model(
            dependencies,
            request.base_model,
            request.profile,
        )

        # Configure and apply LoRA
        lora_config = dependencies["LoraConfig"](
            task_type=dependencies["TaskType"].CAUSAL_LM,
            inference_mode=False,
            r=request.profile.rank,
            lora_alpha=request.profile.alpha,
            lora_dropout=request.profile.dropout,
            target_modules="all-linear",  # Match final local config
            bias="none",
        )
        model = dependencies["get_peft_model"](model, lora_config)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        # Configure training
        training_arguments = dependencies["TrainingArguments"](
            output_dir=str(output_path / ".trainer"),
            num_train_epochs=request.profile.epochs,
            learning_rate=request.profile.learning_rate,
            per_device_train_batch_size=request.profile.batch_size,
            per_device_eval_batch_size=1,
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
            bf16=(dtype == "bfloat16"),
            fp16=(dtype == "float16"),
        )

        # Data collator
        data_collator = dependencies["DataCollatorForSeq2Seq"](
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
        )

        # Progress callback
        callbacks = None
        if request.progress_callback is not None:
            progress_callback = _progress_callback_class(
                dependencies["TrainerCallback"], request.progress_callback
            )
            callbacks = [progress_callback()]

        # Create trainer
        trainer = dependencies["Trainer"](
            model=model,
            args=training_arguments,
            train_dataset=_ListDataset(train_records),
            eval_dataset=(_ListDataset(validation_records) if validation_records else None),
            data_collator=data_collator,
            processing_class=tokenizer,
            callbacks=callbacks,
        )

        # Train
        train_output = trainer.train()
        evaluation_metrics = trainer.evaluate() if validation_records else {}

        # Save adapter
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
                "policy_version": POLICY_VERSION,
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
                "quantized_4bit": quantized,
                "lora_configuration": request.profile.as_metadata(),
                "dataset_audit": audit,
                "split_stats": split_stats,
                "train_metrics": self._json_metrics(train_output.metrics),
                "validation_metrics": self._json_metrics(evaluation_metrics),
            },
            framework_versions=self._framework_versions(),
            artifact_files=self._artifact_manifest(output_path),
        )

    @staticmethod
    def _load_dependencies() -> dict[str, Any]:
        """Load all required ML dependencies."""
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                DataCollatorForSeq2Seq,
                Trainer,
                TrainerCallback,
                TrainingArguments,
            )

            import torch
        except ImportError as exc:
            raise RuntimeError(
                "LoRA training dependencies are unavailable; install backend requirements"
            ) from exc
        return {
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "AutoTokenizer": AutoTokenizer,
            "BitsAndBytesConfig": BitsAndBytesConfig,
            "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
            "LoraConfig": LoraConfig,
            "TaskType": TaskType,
            "Trainer": Trainer,
            "TrainingArguments": TrainingArguments,
            "TrainerCallback": TrainerCallback,
            "get_peft_model": get_peft_model,
            "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
            "torch": torch,
        }

    @staticmethod
    def _parse_examples(content: bytes, expected_count: int) -> list[dict[str, Any]]:
        """Parse and validate dataset examples."""
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
                    f"Dataset line {line_number} is not valid JSON: {exc}"
                ) from exc

            # Validate structure
            if not isinstance(example, dict):
                raise TrainingDatasetError(
                    f"Dataset line {line_number} must contain a JSON object"
                )
            if set(example.keys()) != {"input", "expected_output"}:
                raise TrainingDatasetError(
                    f"Dataset line {line_number}: expected keys 'input' and 'expected_output'"
                )

            inp = example.get("input")
            out = example.get("expected_output")

            if not isinstance(inp, dict) or not isinstance(out, dict):
                raise TrainingDatasetError(
                    f"Dataset line {line_number}: input and expected_output must be objects"
                )

            # Validate input has logs
            logs = inp.get("logs")
            if not isinstance(logs, list) or not logs:
                raise TrainingDatasetError(
                    f"Dataset line {line_number}: input.logs must be a non-empty list"
                )
            if not all(isinstance(x, str) and x.strip() for x in logs):
                raise TrainingDatasetError(
                    f"Dataset line {line_number}: input.logs must contain non-empty strings"
                )

            # Validate output schema using canonical validator
            is_valid, error = validate_output_schema(out)
            if not is_valid:
                raise TrainingDatasetError(
                    f"Dataset line {line_number}: {error}"
                )

            examples.append(example)

        if not examples:
            raise TrainingDatasetError("Dataset contains no training examples")
        if len(examples) != expected_count:
            raise TrainingDatasetError(
                f"Dataset record count {len(examples)} does not match expected {expected_count}"
            )
        return examples

    def _tokenize_example(
        self,
        tokenizer,
        example: dict[str, Any],
        max_sequence_length: int,
    ) -> dict[str, list[int]]:
        """
        Tokenize example with COMPLETE TARGET PRESERVATION.
        
        Critical: This implementation NEVER truncates the assistant target.
        Only log context is truncated if needed to fit max_sequence_length.
        """
        logs = example["input"]["logs"]
        expected_output = example["expected_output"]

        # Build complete messages using canonical policy
        messages = build_training_messages(logs, expected_output)

        # Get prompt IDs (system + user)
        prompt_messages = messages[:-1]  # Exclude assistant
        prompt_ids = self._template_token_ids(
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
        )

        # Get full IDs (system + user + assistant)
        full_ids = self._template_token_ids(
            tokenizer,
            messages,
            add_generation_prompt=False,
        )

        # Extract assistant target tokens
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise TrainingDatasetError(
                "Could not identify assistant token boundary from chat template"
            )

        assistant_target_ids = full_ids[len(prompt_ids) :]
        if not assistant_target_ids:
            raise TrainingDatasetError("Assistant target is empty after tokenization")

        # Check if truncation needed
        if len(full_ids) <= max_sequence_length:
            # Fits - use as-is
            labels = [-100] * len(prompt_ids) + assistant_target_ids
            return {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }

        # Need truncation - preserve complete assistant target, truncate log context
        # Calculate fixed overhead: system prompt + user prompt structure + assistant target
        empty_log_messages = build_training_messages([], expected_output)
        empty_prompt_messages = empty_log_messages[:-1]
        empty_prompt_ids = self._template_token_ids(
            tokenizer, empty_prompt_messages, add_generation_prompt=True
        )
        empty_full_ids = self._template_token_ids(
            tokenizer, empty_log_messages, add_generation_prompt=False
        )
        empty_target_ids = empty_full_ids[len(empty_prompt_ids) :]

        # Verify assistant target is independent of log content
        if assistant_target_ids != empty_target_ids:
            # This shouldn't happen but if it does, use the original target
            target_ids = assistant_target_ids
        else:
            target_ids = empty_target_ids

        # Calculate available budget for log text
        fixed_overhead = len(empty_prompt_ids)
        available_for_logs = max_sequence_length - fixed_overhead - len(target_ids)

        if available_for_logs <= 0:
            raise TrainingDatasetError(
                f"max_sequence_length={max_sequence_length} cannot fit "
                f"the complete assistant target ({len(target_ids)} tokens). "
                "Increase max_sequence_length or shorten the target."
            )

        # Truncate log text to fit budget
        log_text = "\n".join(logs)
        truncated_log_text = self._truncate_log_text(tokenizer, log_text, available_for_logs)

        # Rebuild with truncated logs
        truncated_messages = build_training_messages(
            truncated_log_text.split("\n") if truncated_log_text else [],
            expected_output,
        )
        truncated_prompt_messages = truncated_messages[:-1]
        truncated_prompt_ids = self._template_token_ids(
            tokenizer, truncated_prompt_messages, add_generation_prompt=True
        )
        truncated_full_ids = self._template_token_ids(
            tokenizer, truncated_messages, add_generation_prompt=False
        )

        # Verify we're within budget
        if len(truncated_full_ids) > max_sequence_length:
            # Emergency fallback: aggressively truncate
            budget = available_for_logs
            while len(truncated_full_ids) > max_sequence_length and budget > 0:
                budget -= 10
                truncated_log_text = self._truncate_log_text(tokenizer, log_text, budget)
                truncated_messages = build_training_messages(
                    truncated_log_text.split("\n") if truncated_log_text else [],
                    expected_output,
                )
                truncated_prompt_messages = truncated_messages[:-1]
                truncated_prompt_ids = self._template_token_ids(
                    tokenizer, truncated_prompt_messages, add_generation_prompt=True
                )
                truncated_full_ids = self._template_token_ids(
                    tokenizer, truncated_messages, add_generation_prompt=False
                )

        # Final validation: assistant target must be unchanged
        final_target_ids = truncated_full_ids[len(truncated_prompt_ids) :]
        if final_target_ids != assistant_target_ids:
            raise TrainingDatasetError(
                "Assistant target tokens changed during log truncation; refusing to train"
            )

        labels = [-100] * len(truncated_prompt_ids) + final_target_ids
        return {
            "input_ids": truncated_full_ids,
            "attention_mask": [1] * len(truncated_full_ids),
            "labels": labels,
        }

    @staticmethod
    def _template_token_ids(
        tokenizer,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> list[int]:
        """Apply chat template and return token IDs."""
        # Try with enable_thinking=False for Qwen models
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
                return_dict=False,
            )
        except TypeError:
            # Fallback for models/tokenizers without enable_thinking parameter
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_dict=False,
            )

        if isinstance(encoded, Mapping):
            encoded = encoded.get("input_ids")
        if encoded is None:
            raise TrainingDatasetError("Tokenizer did not produce input token IDs")
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return list(encoded)

    @staticmethod
    def _truncate_log_text(tokenizer, text: str, token_budget: int) -> str:
        """Truncate log text to fit within token budget, preserving head and tail."""
        if token_budget <= 0:
            return ""

        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= token_budget:
            return text

        # Add truncation marker
        marker = "\n[... log context truncated ...]\n"
        marker_ids = tokenizer(marker, add_special_tokens=False)["input_ids"]

        if token_budget <= len(marker_ids) + 4:
            # Just take tail
            return tokenizer.decode(ids[-token_budget:], skip_special_tokens=True)

        # Split budget between head and tail
        usable = token_budget - len(marker_ids)
        head_n = usable // 2
        tail_n = usable - head_n
        kept = ids[:head_n] + marker_ids + ids[-tail_n:]
        return tokenizer.decode(kept, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def _group_stratified_split(
        self,
        examples: list[dict[str, Any]],
        tokenized: list[dict[str, list[int]]],
        validation_fraction: float,
        seed: int,
    ) -> tuple[list[dict[str, list[int]]], list[dict[str, list[int]]], dict[str, Any]]:
        """
        Perform group-stratified validation split.
        
        Groups examples by normalized scenario, ensuring:
        - Whole groups stay together (no leakage)
        - Label distribution approximated
        - Each incident_type has training representation
        """
        if len(examples) < 2 or validation_fraction <= 0:
            return tokenized, [], {
                "strategy": "none",
                "train_examples": len(tokenized),
                "validation_examples": 0,
            }

        # Build scenario groups
        groups: dict[str, list[int]] = defaultdict(list)
        group_type: dict[str, str] = {}

        for idx, example in enumerate(examples):
            group_key = self._scenario_group_key(example)
            groups[group_key].append(idx)
            incident_type = str(
                example["input"].get("metadata", {}).get("incident_type", "__unknown__")
            )
            group_type[group_key] = incident_type

        type_groups: dict[str, set] = defaultdict(set)
        for g, t in group_type.items():
            type_groups[t].add(g)

        # Calculate target validation size
        target_n = max(1, round(len(examples) * validation_fraction))

        # Get label distributions
        full_sev = Counter(ex["expected_output"]["severity"] for ex in examples)
        full_disp = Counter(ex["expected_output"]["disposition"] for ex in examples)
        full_pair = Counter(
            (ex["expected_output"]["severity"], ex["expected_output"]["disposition"])
            for ex in examples
        )

        # Shuffle groups
        rng = random.Random(seed)
        remaining = list(groups.keys())
        rng.shuffle(remaining)

        selected = set()
        selected_by_type = Counter()
        val_n = 0
        cur_sev = Counter()
        cur_disp = Counter()
        cur_pair = Counter()

        # Pre-compute group characteristics
        group_chars = {}
        for g in remaining:
            sev = Counter(examples[i]["expected_output"]["severity"] for i in groups[g])
            disp = Counter(examples[i]["expected_output"]["disposition"] for i in groups[g])
            pair = Counter(
                (
                    examples[i]["expected_output"]["severity"],
                    examples[i]["expected_output"]["disposition"],
                )
                for i in groups[g]
            )
            group_chars[g] = (sev, disp, pair)

        # Selection function
        def can_select(g: str) -> bool:
            t = group_type[g]
            return selected_by_type[t] < len(type_groups[t]) - 1

        def score_after(g: str) -> float:
            size = len(groups[g])
            new_n = val_n + size
            sev_g, disp_g, pair_g = group_chars[g]

            # Penalize size deviation
            target_sev = {k: full_sev[k] * validation_fraction for k in SEVERITIES}
            target_disp = {k: full_disp[k] * validation_fraction for k in DISPOSITIONS}
            target_pair = {k: full_pair[k] * validation_fraction for k in full_pair}

            size_err = abs(new_n - target_n) / max(target_n, 1)
            sev_err = sum(
                abs((cur_sev[k] + sev_g[k]) - target_sev[k]) / max(target_sev[k], 1.0)
                for k in SEVERITIES
            )
            disp_err = sum(
                abs((cur_disp[k] + disp_g[k]) - target_disp[k]) / max(target_disp[k], 1.0)
                for k in DISPOSITIONS
            )
            pair_err = sum(
                abs((cur_pair[k] + pair_g[k]) - target_pair[k]) / max(target_pair[k], 1.0)
                for k in full_pair
            )

            # Bonus for filling missing labels
            missing_bonus = 0.0
            for k in SEVERITIES:
                if cur_sev[k] == 0 and sev_g[k] > 0:
                    missing_bonus -= 1.5
            for k in DISPOSITIONS:
                if cur_disp[k] == 0 and disp_g[k] > 0:
                    missing_bonus -= 1.5

            # Bonus for new incident types
            t = group_type[g]
            type_bonus = -0.75 if selected_by_type[t] == 0 and len(type_groups[t]) > 1 else 0.0

            return 2.0 * size_err + sev_err + disp_err + 0.75 * pair_err + missing_bonus + type_bonus

        # Main selection loop
        while val_n < target_n and remaining:
            candidates = [g for g in remaining if can_select(g)]
            if not candidates:
                break
            best = min(candidates, key=score_after)
            selected.add(best)
            remaining.remove(best)
            selected_by_type[group_type[best]] += 1
            sev_g, disp_g, pair_g = group_chars[best]
            cur_sev.update(sev_g)
            cur_disp.update(disp_g)
            cur_pair.update(pair_g)
            val_n += len(groups[best])

            # Early exit if good enough
            if (
                val_n >= target_n * 0.95
                and all(cur_sev[k] > 0 for k in SEVERITIES)
                and all(cur_disp[k] > 0 for k in DISPOSITIONS)
            ):
                break

        # Ensure all observed pairs represented
        for pair_key in full_pair:
            if cur_pair[pair_key] > 0:
                continue
            candidates = []
            for g in remaining:
                if not can_select(g):
                    continue
                if group_chars[g][2][pair_key] > 0:
                    candidates.append(g)
            if candidates:
                g = min(candidates, key=lambda x: len(groups[x]))
                selected.add(g)
                remaining.remove(g)
                selected_by_type[group_type[g]] += 1
                sev_g, disp_g, pair_g = group_chars[g]
                cur_sev.update(sev_g)
                cur_disp.update(disp_g)
                cur_pair.update(pair_g)
                val_n += len(groups[g])

        if not selected:
            raise TrainingDatasetError("Could not create group-stratified validation split")

        # Build train/val splits
        val_indices = {i for g in selected for i in groups[g]}
        train_tokenized = [tokenized[i] for i in range(len(examples)) if i not in val_indices]
        val_tokenized = [tokenized[i] for i in range(len(examples)) if i in val_indices]

        train_groups = {self._scenario_group_key(examples[i]) for i in range(len(examples)) if i not in val_indices}
        val_groups = {self._scenario_group_key(examples[i]) for i in range(len(examples)) if i in val_indices}
        overlap = train_groups & val_groups

        if overlap:
            raise TrainingDatasetError(f"Group-stratified split leaked {len(overlap)} groups")

        stats = {
            "strategy": "group-stratified",
            "requested_validation_fraction": validation_fraction,
            "actual_validation_fraction": len(val_tokenized) / len(examples),
            "train_examples": len(train_tokenized),
            "validation_examples": len(val_tokenized),
            "scenario_groups_total": len(groups),
            "train_scenario_groups": len(train_groups),
            "validation_scenario_groups": len(val_groups),
            "overlapping_scenario_groups": len(overlap),
        }

        return train_tokenized, val_tokenized, stats

    @staticmethod
    def _scenario_group_key(example: dict[str, Any]) -> str:
        """Create normalized scenario group key to prevent leakage."""
        metadata = example.get("input", {}).get("metadata", {}) or {}
        explicit = metadata.get("scenario_group_id")
        if explicit:
            return str(explicit)

        # Normalize text to collapse slot variations
        text = "\n".join(example["input"]["logs"]).lower()
        replacements = (
            (ISO_TS_RE, "<ts>"),
            (UUID_RE, "<uuid>"),
            (IP_RE, "<ip>"),
            (HEX_RE, "<hex>"),
            (VERSION_RE, "<version>"),
            (API_PATH_RE, "<endpoint>"),
            (NUMBER_RE, "<num>"),
        )
        for pattern, replacement in replacements:
            text = pattern.sub(replacement, text)
        text = BRACKET_SOURCE_RE.sub("[<source>]", text)
        text = re.sub(r"\s+", " ", text).strip()

        incident_type = str(metadata.get("incident_type", "__unknown__"))
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return f"{incident_type}:{digest}"

    def _dataset_audit(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Audit dataset quality metrics."""
        by_type_pairs: dict[str, set] = defaultdict(set)
        groups = set()
        recovery_pairs = Counter()
        chronological = 0
        parseable = 0
        target_counter = Counter()

        for ex in examples:
            inp, out = ex["input"], ex["expected_output"]
            md = inp.get("metadata", {}) or {}
            it = str(md.get("incident_type", "__unknown__"))
            pair = (out["severity"], out["disposition"])
            by_type_pairs[it].add(pair)
            groups.add(self._scenario_group_key(ex))

            if RECOVERY_RE.search("\n".join(inp.get("logs", []))):
                recovery_pairs[pair] += 1

            target_counter[json.dumps(out, sort_keys=True, ensure_ascii=False)] += 1

            # Check chronological ordering
            times = []
            for log in inp.get("logs", []):
                m = LOG_TS_RE.match(log)
                if m:
                    try:
                        from datetime import datetime

                        times.append(
                            datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))
                        )
                    except ValueError:
                        pass
            if times:
                parseable += 1
                if all(a <= b for a, b in zip(times, times[1:])):
                    chronological += 1

        sev = Counter(x["expected_output"]["severity"] for x in examples)
        disp = Counter(x["expected_output"]["disposition"] for x in examples)

        benign_like = {
            k
            for k in by_type_pairs
            if k.startswith("benign_")
            or k in {"transient_cache_miss", "single_4xx_error", "healthcheck_timeout_noise"}
        }
        non_benign = set(by_type_pairs) - benign_like
        non_benign_multi = sum(len(by_type_pairs[k]) > 1 for k in non_benign)

        high_urgency_recovered = sum(
            n
            for (sev, dispo), n in recovery_pairs.items()
            if sev in {"high", "critical"} or dispo in {"NEEDS_ONCALL", "ESCALATE"}
        )

        duplicate_examples = sum(n - 1 for n in target_counter.values() if n > 1)

        return {
            "examples": len(examples),
            "severity": dict(sev),
            "disposition": dict(disp),
            "scenario_groups": len(groups),
            "incident_types": len(by_type_pairs),
            "non_benign_incident_types": len(non_benign),
            "non_benign_incident_types_with_multiple_label_pairs": non_benign_multi,
            "non_benign_multilabel_type_fraction": (
                non_benign_multi / len(non_benign) if non_benign else 1.0
            ),
            "recovered_high_urgency_examples": high_urgency_recovered,
            "timestamp_parseable_examples": parseable,
            "chronological_examples": chronological,
            "chronological_pct": (
                100.0 * chronological / parseable if parseable else 100.0
            ),
            "unique_exact_targets": len(target_counter),
            "duplicate_exact_target_examples": duplicate_examples,
            "duplicate_exact_target_pct": (
                100.0 * duplicate_examples / len(examples) if examples else 0.0
            ),
        }

    @staticmethod
    def _enforce_quality_gates(audit: dict[str, Any]) -> None:
        """Enforce dataset quality gates (can be configured via env vars)."""
        # These are the FINAL quality requirements from local training
        failures = []

        min_chrono = float(os.getenv("TRAINING_MIN_CHRONOLOGICAL_PCT", "99.0"))
        if audit["chronological_pct"] < min_chrono:
            failures.append(
                f"chronological_pct={audit['chronological_pct']:.2f} < {min_chrono}"
            )

        if bool(int(os.getenv("TRAINING_REQUIRE_ALL_SEVERITIES", "1"))):
            missing = SEVERITIES - set(audit["severity"])
            if missing:
                failures.append(f"missing severities={sorted(missing)}")

        if bool(int(os.getenv("TRAINING_REQUIRE_ALL_DISPOSITIONS", "1"))):
            missing = DISPOSITIONS - set(audit["disposition"])
            if missing:
                failures.append(f"missing dispositions={sorted(missing)}")

        if bool(int(os.getenv("TRAINING_REQUIRE_RECOVERED_HIGH_URGENCY", "1"))):
            if audit["recovered_high_urgency_examples"] <= 0:
                failures.append("no recovered high-urgency examples")

        min_multi = float(
            os.getenv("TRAINING_MIN_NON_BENIGN_MULTILABEL_FRACTION", "0.75")
        )
        if audit["non_benign_multilabel_type_fraction"] < min_multi:
            failures.append(
                f"non_benign_multilabel_type_fraction="
                f"{audit['non_benign_multilabel_type_fraction']:.3f} < {min_multi}"
            )

        max_dup = float(os.getenv("TRAINING_MAX_DUPLICATE_TARGET_PCT", "15.0"))
        if audit["duplicate_exact_target_pct"] > max_dup:
            failures.append(
                f"duplicate_exact_target_pct={audit['duplicate_exact_target_pct']:.2f} > {max_dup}"
            )

        if failures and bool(int(os.getenv("TRAINING_FAIL_ON_QUALITY_GATE", "1"))):
            raise TrainingDatasetError(
                "Dataset quality gate failed: " + "; ".join(failures)
            )

    def _load_model(
        self,
        dependencies: dict[str, Any],
        base_model: str,
        profile,
    ) -> tuple[Any, bool, str]:
        """Load base model with optional QLoRA quantization."""
        torch_module = dependencies["torch"]

        # Determine dtype
        if torch_module.cuda.is_available():
            if torch_module.cuda.is_bf16_supported():
                dtype_name = "bfloat16"
                dtype = torch_module.bfloat16
            else:
                dtype_name = "float16"
                dtype = torch_module.float16
        else:
            dtype_name = "float32"
            dtype = torch_module.float32

        # Check if we can use QLoRA
        use_4bit_env = os.getenv("TRAINING_USE_4BIT", "auto").strip().lower()
        requested_4bit = use_4bit_env in {"true", "1", "yes", "auto"}
        can_4bit = False

        if requested_4bit and torch_module.cuda.is_available():
            try:
                import bitsandbytes  # noqa: F401

                can_4bit = True
            except ImportError:
                pass

        quantized = can_4bit if use_4bit_env == "auto" else (use_4bit_env in {"true", "1", "yes"})

        if quantized and not can_4bit:
            raise RuntimeError(
                "4-bit quantization requested but CUDA + bitsandbytes are not available"
            )

        load_kwargs: dict[str, Any] = {
            "token": self._huggingface_token(),
            "dtype": dtype,
        }

        if quantized:
            load_kwargs["quantization_config"] = dependencies["BitsAndBytesConfig"](
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = {"": torch_module.cuda.current_device()}
        else:
            load_kwargs["low_cpu_mem_usage"] = True

        model = dependencies["AutoModelForCausalLM"].from_pretrained(
            base_model, **load_kwargs
        )
        model.config.use_cache = False

        if quantized:
            model = dependencies["prepare_model_for_kbit_training"](
                model, use_gradient_checkpointing=True
            )
        else:
            model.gradient_checkpointing_enable()

        return model, quantized, dtype_name

    @staticmethod
    def _artifact_manifest(output_path: Path) -> tuple[dict[str, Any], ...]:
        """Generate artifact manifest with checksums."""
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
        """Get installed framework versions."""
        versions = {}
        for package in (
            "torch",
            "transformers",
            "peft",
            "accelerate",
            "safetensors",
            "bitsandbytes",
        ):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "unavailable"
        return versions

    @staticmethod
    def _json_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        """Filter metrics to JSON-serializable values."""
        return {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }

    @staticmethod
    def _finite_metric(value) -> float | None:
        """Convert metric to finite float or None."""
        if value is None:
            return None
        metric = float(value)
        return metric if math.isfinite(metric) else None

    @staticmethod
    def _completion_labels(
        prompt_ids: list[int],
        full_ids: list[int],
        max_length: int,
    ) -> tuple[list[int], list[int]]:
        """
        Create completion-only labels by masking prompt tokens.
        
        Args:
            prompt_ids: Token IDs for the prompt (system + user)
            full_ids: Complete token IDs (prompt + assistant)
            max_length: Maximum sequence length (unused, kept for compatibility)
        
        Returns:
            (input_ids, labels) tuple where labels mask prompt tokens with -100
        """
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        return full_ids, labels

    @staticmethod
    def _huggingface_token() -> str | None:
        """Get Hugging Face token from environment."""
        return os.getenv("HF_TOKEN", "").strip() or None

    @staticmethod
    def _remove_trainer_directory(path: Path) -> None:
        """Clean up trainer checkpoint directory."""
        if path.exists():
            shutil.rmtree(path)
