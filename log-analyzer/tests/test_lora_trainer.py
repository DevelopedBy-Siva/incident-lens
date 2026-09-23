import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.shared.model_config import DEFAULT_BASE_MODEL
from app.training.lora_trainer import (
    TrainingDatasetError,
    TransformersPeftTrainingEngine,
)
from app.training.training_engine import TrainingRequest
from app.training.training_profile import (
    LoraTrainingProfile,
    configured_training_profile,
)


class FakeTokenizer:
    pad_token_id = None
    pad_token = None
    eos_token = "</s>"

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return cls()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if len(messages) == 2:
            return [1, 2, 3]
        return [1, 2, 3, 4, 5]

    def save_pretrained(self, output_path):
        Path(output_path, "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "FakeTokenizer"}),
            encoding="utf-8",
        )


class FakeModel:
    def __init__(self, base_model):
        self.base_model = base_model
        self.config = SimpleNamespace(use_cache=True)

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return cls(model_name)

    def save_pretrained(self, output_path, safe_serialization):
        Path(output_path, "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": self.base_model,
                    "peft_type": "LORA",
                }
            ),
            encoding="utf-8",
        )
        Path(output_path, "adapter_model.safetensors").write_bytes(b"real-peft-adapter")


class FakeConfiguration:
    def __init__(self, **values):
        self.values = values


class FakeTrainer:
    def __init__(self, **values):
        self.values = values

    def train(self):
        callbacks = self.values.get("callbacks") or []
        state = SimpleNamespace(max_steps=4, global_step=0)
        control = SimpleNamespace()
        for callback in callbacks:
            callback.on_train_begin(None, state, control)
        state.global_step = state.max_steps
        for callback in callbacks:
            callback.on_step_end(None, state, control)
        return SimpleNamespace(
            metrics={"train_loss": 0.2, "train_runtime": 1.0},
            training_loss=0.2,
        )

    def evaluate(self):
        return {"eval_loss": 0.3}


class FakeTaskType:
    CAUSAL_LM = "CAUSAL_LM"


class FakeTrainerCallback:
    pass


class LoraTrainerTests(unittest.TestCase):
    def _dependencies(self):
        return {
            "AutoModelForCausalLM": FakeModel,
            "AutoTokenizer": FakeTokenizer,
            "DataCollatorForSeq2Seq": FakeConfiguration,
            "LoraConfig": FakeConfiguration,
            "TaskType": FakeTaskType,
            "Trainer": FakeTrainer,
            "TrainerCallback": FakeTrainerCallback,
            "TrainingArguments": FakeConfiguration,
            "get_peft_model": lambda model, config: model,
        }

    def _request(self, output_path: str, record_count: int = 2):
        example = {
            "input": {"incident": {"source": "checkout"}},
            "expected_output": {
                "severity": "medium",
                "disposition": "NEEDS_DEV",
            },
        }
        content = "\n".join(json.dumps(example) for _ in range(record_count)) + "\n"
        return TrainingRequest(
            project_id="project-1",
            dataset_id="dataset-1",
            dataset_version="dataset-v1",
            base_model=DEFAULT_BASE_MODEL,
            dataset_content=content.encode(),
            expected_record_count=record_count,
            adapter_output_path=output_path,
            profile=LoraTrainingProfile(validation_fraction=0.5),
        )

    def test_engine_trains_saves_adapter_and_reports_real_metrics(self):
        with tempfile.TemporaryDirectory() as artifact_directory:
            engine = TransformersPeftTrainingEngine()
            request = self._request(artifact_directory)

            with patch.object(
                engine,
                "_load_dependencies",
                return_value=self._dependencies(),
            ):
                result = engine.train(request)

            self.assertTrue(result.succeeded)
            self.assertEqual(result.engine, "transformers-peft-lora-v1")
            self.assertEqual(result.metrics["training_loss"], 0.2)
            self.assertEqual(result.metrics["validation_loss"], 0.3)
            self.assertTrue(result.metrics["training_completed"])
            self.assertTrue(result.metrics["weights_created"])
            self.assertEqual(result.metrics["training_records"], 1)
            self.assertEqual(result.metrics["validation_records"], 1)
            self.assertEqual(result.metrics["lora_configuration"]["rank"], 8)
            self.assertTrue(
                Path(artifact_directory, "adapter_model.safetensors").is_file()
            )
            self.assertTrue(Path(artifact_directory, "adapter_config.json").is_file())
            self.assertTrue(Path(artifact_directory, "tokenizer_config.json").is_file())
            self.assertIn(
                "adapter_model.safetensors",
                {entry["name"] for entry in result.artifact_files},
            )

    def test_engine_reports_step_progress(self):
        with tempfile.TemporaryDirectory() as artifact_directory:
            progress = []
            engine = TransformersPeftTrainingEngine()
            request = replace(
                self._request(artifact_directory),
                progress_callback=lambda current, total: progress.append(
                    (current, total)
                ),
            )

            with patch.object(
                engine,
                "_load_dependencies",
                return_value=self._dependencies(),
            ):
                engine.train(request)

            self.assertEqual(progress, [(0, 4), (4, 4)])

    def test_dataset_parser_rejects_schema_and_record_count_mismatches(self):
        with self.assertRaisesRegex(TrainingDatasetError, "training schema"):
            TransformersPeftTrainingEngine._parse_examples(b'{"input":{}}\n', 1)

        with self.assertRaisesRegex(TrainingDatasetError, "record count"):
            TransformersPeftTrainingEngine._parse_examples(
                b'{"input":{},"expected_output":{}}\n',
                2,
            )

    def test_completion_labels_mask_prompt_tokens(self):
        input_ids, labels = TransformersPeftTrainingEngine._completion_labels(
            [1, 2, 3],
            [1, 2, 3, 4, 5],
            10,
        )

        self.assertEqual(input_ids, [1, 2, 3, 4, 5])
        self.assertEqual(labels, [-100, -100, -100, 4, 5])

    def test_training_profile_is_configurable_and_validated(self):
        with patch.dict(
            os.environ,
            {
                "LORA_RANK": "16",
                "LORA_ALPHA": "32",
                "LORA_DROPOUT": "0.1",
                "LORA_EPOCHS": "2",
                "LORA_LEARNING_RATE": "0.0002",
                "LORA_BATCH_SIZE": "2",
            },
        ):
            profile = configured_training_profile()

        self.assertEqual(profile.rank, 16)
        self.assertEqual(profile.alpha, 32)
        self.assertEqual(profile.dropout, 0.1)
        self.assertEqual(profile.epochs, 2.0)
        self.assertEqual(profile.learning_rate, 0.0002)
        self.assertEqual(profile.batch_size, 2)
        with self.assertRaisesRegex(ValueError, "rank"):
            LoraTrainingProfile(rank=0)

    @unittest.skipUnless(
        os.getenv("RUN_REAL_LORA_TEST") == "1",
        "set RUN_REAL_LORA_TEST=1 to download and train the tiny Qwen model",
    )
    def test_real_transformers_peft_smoke_training(self):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        with tempfile.TemporaryDirectory() as artifact_directory:
            base_model = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
            request = self._request(artifact_directory, record_count=1)
            request = TrainingRequest(
                **{
                    **request.__dict__,
                    "base_model": base_model,
                    "profile": LoraTrainingProfile(
                        rank=2,
                        alpha=4,
                        epochs=1,
                        max_sequence_length=128,
                        validation_fraction=0,
                        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                    ),
                }
            )

            result = TransformersPeftTrainingEngine().train(request)

            self.assertTrue(result.succeeded)
            self.assertTrue(result.metrics["weights_created"])
            self.assertTrue(
                Path(artifact_directory, "adapter_model.safetensors").is_file()
            )
            self.assertGreater(
                Path(artifact_directory, "adapter_model.safetensors").stat().st_size,
                0,
            )
            reloaded = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(base_model),
                artifact_directory,
            )
            self.assertIn("default", reloaded.peft_config)


if __name__ == "__main__":
    unittest.main()
