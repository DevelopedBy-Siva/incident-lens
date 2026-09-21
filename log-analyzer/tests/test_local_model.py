import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.serving.local_model import (
    AdapterLoader,
    AdapterLoadError,
    LoadedBaseModel,
    LocalModelCache,
)
from app.serving.model_provider import LocalModelProvider
from app.shared.model_config import RuntimeModelSettings


class FakeInputs(dict):
    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __init__(self):
        self.requests = []

    def apply_chat_template(self, messages, **options):
        import torch

        self.requests.append((messages, options))
        return FakeInputs(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, output_ids, skip_special_tokens):
        return '{"severity":"low","disposition":"OBSERVE"}'


class FakeAdapterModel:
    def __init__(self):
        self.active_adapters = []
        self.generate_calls = []

    def set_adapter(self, adapter_name):
        self.active_adapters.append(adapter_name)

    def eval(self):
        return self

    def generate(self, **request):
        import torch

        self.generate_calls.append(request)
        return torch.tensor([[1, 2, 3, 4, 5]])


class FakeBaseLoader:
    def __init__(self):
        self.calls = 0
        self.tokenizer = FakeTokenizer()
        self.model = SimpleNamespace()

    def load(self, settings):
        self.calls += 1
        return LoadedBaseModel(settings.base_model, self.tokenizer, self.model)


class FakeAdapterLoader:
    def __init__(self):
        self.model = FakeAdapterModel()
        self.first = []
        self.additional = []

    def load_first(self, base_model, **request):
        self.first.append((base_model, request))
        return self.model

    def load_additional(self, model, **request):
        self.additional.append((model, request))


class LocalModelCacheTests(unittest.TestCase):
    def setUp(self):
        self.settings = RuntimeModelSettings(base_model="test-qwen")
        self.base_loader = FakeBaseLoader()
        self.adapter_loader = FakeAdapterLoader()
        self.cache = LocalModelCache(
            self.settings,
            base_loader=self.base_loader,
            adapter_loader=self.adapter_loader,
        )

    def test_base_model_and_adapters_are_loaded_once_and_reused(self):
        self.cache.initialize()
        self.cache.initialize()

        first = self.cache.generate(
            project_id="project-a",
            artifact_id="artifact-a1",
            adapter_path="/tmp/project-a/adapter-v1",
            messages=[{"role": "user", "content": "incident"}],
            tools=None,
            temperature=0.2,
            max_new_tokens=20,
        )
        self.cache.generate(
            project_id="project-a",
            artifact_id="artifact-a1",
            adapter_path="/tmp/project-a/adapter-v1",
            messages=[{"role": "user", "content": "incident again"}],
            tools=None,
            temperature=0.2,
            max_new_tokens=20,
        )
        self.cache.generate(
            project_id="project-b",
            artifact_id="artifact-b1",
            adapter_path="/tmp/project-b/adapter-v1",
            messages=[{"role": "user", "content": "other incident"}],
            tools=None,
            temperature=0.2,
            max_new_tokens=20,
        )

        self.assertEqual(self.base_loader.calls, 1)
        self.assertEqual(len(self.adapter_loader.first), 1)
        self.assertEqual(len(self.adapter_loader.additional), 1)
        self.assertEqual(
            self.cache.loaded_artifact_ids,
            ("artifact-a1", "artifact-b1"),
        )
        self.assertEqual(self.cache.active_artifact_for("project-a"), "artifact-a1")
        self.assertEqual(self.cache.active_artifact_for("project-b"), "artifact-b1")
        self.assertEqual(first.input_tokens, 3)
        self.assertEqual(first.output_tokens, 2)
        self.assertTrue(first.content.startswith("{"))

    def test_project_switches_to_new_active_artifact_without_reloading_base(self):
        for artifact_id, path in (
            ("artifact-a1", "/tmp/project-a/adapter-v1"),
            ("artifact-a2", "/tmp/project-a/adapter-v2"),
        ):
            self.cache.generate(
                project_id="project-a",
                artifact_id=artifact_id,
                adapter_path=path,
                messages=[{"role": "user", "content": "incident"}],
                tools=None,
                temperature=0,
                max_new_tokens=20,
            )

        self.assertEqual(self.base_loader.calls, 1)
        self.assertEqual(len(self.adapter_loader.first), 1)
        self.assertEqual(len(self.adapter_loader.additional), 1)
        self.assertEqual(self.cache.active_artifact_for("project-a"), "artifact-a2")
        self.assertIn("artifact_a2", self.adapter_loader.model.active_adapters[-1])

    def test_artifact_cache_prevents_cross_project_reuse(self):
        request = {
            "artifact_id": "shared-artifact-id",
            "adapter_path": "/tmp/project-a/adapter-v1",
            "messages": [{"role": "user", "content": "incident"}],
            "tools": None,
            "temperature": 0,
            "max_new_tokens": 20,
        }
        self.cache.generate(project_id="project-a", **request)

        with self.assertRaisesRegex(AdapterLoadError, "another project"):
            self.cache.generate(project_id="project-b", **request)


class FakePeftConfig:
    base_model_name_or_path = "test-qwen"

    @classmethod
    def from_pretrained(cls, path):
        return cls()


class AdapterLoaderTests(unittest.TestCase):
    def test_phase_six_artifact_shape_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "adapter_config.json").write_text("{}", encoding="utf-8")
            (path / "adapter_model.safetensors").write_bytes(b"adapter")
            loader = AdapterLoader(
                peft_model=SimpleNamespace(),
                peft_config=FakePeftConfig,
            )

            validated = loader._validate(str(path), "test-qwen")

            self.assertEqual(validated, path.resolve())

    def test_incompatible_base_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "adapter_config.json").write_text("{}", encoding="utf-8")
            (path / "adapter_model.safetensors").write_bytes(b"adapter")
            loader = AdapterLoader(
                peft_model=SimpleNamespace(),
                peft_config=FakePeftConfig,
            )

            with self.assertRaisesRegex(AdapterLoadError, "does not match"):
                loader._validate(str(path), "another-qwen")


@unittest.skipUnless(
    os.getenv("RUN_REAL_LOCAL_INFERENCE_TEST") == "1",
    "set RUN_REAL_LOCAL_INFERENCE_TEST=1 to train and serve the tiny Qwen model",
)
class RealLocalInferenceTests(unittest.TestCase):
    def test_trained_adapter_loads_and_generates_through_local_provider(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool

        from app.control.models import Project
        from app.serving.model_runtime import ModelRuntime
        from app.shared.database import Base
        from app.training.lora_trainer import TransformersPeftTrainingEngine
        from app.training.models import (
            Dataset,
            DatasetStatus,
            ModelArtifact,
            ModelArtifactStatus,
        )
        from app.training.training_engine import TrainingRequest
        from app.training.training_profile import LoraTrainingProfile

        base_model = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        example = {
            "input": {"incident": {"source": "checkout"}},
            "expected_output": {
                "severity": "low",
                "disposition": "OBSERVE",
                "summary": "Checkout warning",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            result = TransformersPeftTrainingEngine().train(
                TrainingRequest(
                    project_id="project-1",
                    dataset_id="dataset-1",
                    dataset_version="dataset-v1",
                    base_model=base_model,
                    dataset_content=(json.dumps(example) + "\n").encode(),
                    expected_record_count=1,
                    adapter_output_path=directory,
                    profile=LoraTrainingProfile(
                        rank=2,
                        alpha=4,
                        epochs=1,
                        max_sequence_length=128,
                        validation_fraction=0,
                        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                    ),
                )
            )
            settings = RuntimeModelSettings(
                base_model=base_model,
                max_new_tokens=8,
            )
            provider = LocalModelProvider(settings)
            engine = create_engine(
                "sqlite://",
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
            Base.metadata.create_all(engine)
            session_factory = sessionmaker(bind=engine)
            db = session_factory()
            project = Project(
                name="local-inference-project",
                password_hash="hash",
                base_model=base_model,
            )
            db.add(project)
            db.flush()
            dataset = Dataset(
                project_id=project.id,
                dataset_version="dataset-v1",
                storage_key="datasets/project-1/dataset-v1.jsonl",
                record_count=1,
                status=DatasetStatus.READY,
            )
            db.add(dataset)
            db.flush()
            artifact = ModelArtifact(
                project_id=project.id,
                artifact_version="adapter-v1",
                base_model=base_model,
                adapter_path=result.adapter_path,
                dataset_id=dataset.id,
                evaluation_score=1.0,
                status=ModelArtifactStatus.READY,
            )
            db.add(artifact)
            db.flush()
            project.active_artifact_id = artifact.id
            Path(directory, "metadata.json").write_text(
                json.dumps(
                    {
                        "artifact_id": artifact.id,
                        "project_id": project.id,
                        "base_model": base_model,
                        "contains_adapter_weights": True,
                        "contains_base_model_weights": False,
                    }
                ),
                encoding="utf-8",
            )
            db.commit()

            runtime = ModelRuntime(
                session_factory=session_factory,
                provider=provider,
                settings=settings,
            )
            runtime.initialize()
            runtime_session = runtime.resolve_project_model(project.id)

            response = runtime_session.complete(
                model=base_model,
                messages="Return a JSON incident decision.",
                temperature=0,
            )

            self.assertIsInstance(response.content, str)
            self.assertGreater(response.usage_metadata["output_tokens"], 0)
            self.assertEqual(provider.cache.loaded_artifact_ids, (artifact.id,))
            self.assertEqual(
                provider.cache.active_artifact_for(project.id),
                artifact.id,
            )
            db.close()
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
