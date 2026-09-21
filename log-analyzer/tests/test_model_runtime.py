import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.control.models import DEFAULT_BASE_MODEL, Project
from app.serving.local_model import AdapterLoadError, LocalGeneration
from app.serving.model_provider import (
    LocalModelProvider,
    ModelProvider,
    ProviderResponse,
)
from app.serving.model_runtime import ModelRuntime
from app.shared.database import Base
from app.shared.model_config import RuntimeModelSettings
from app.training.models import (
    Dataset,
    DatasetStatus,
    ModelArtifact,
    ModelArtifactStatus,
)


class AvailableProvider(ModelProvider):
    name = "test-provider"
    runtime_type = "test-runtime"
    default_model = "test-model"

    def model_candidates(self):
        return (self.default_model,)

    def capabilities(self):
        return {
            "text_generation": True,
            "structured_output": True,
            "tool_calling": True,
            "model_fallbacks": False,
        }

    def is_available(self, session):
        return True

    def complete(self, session, *, model, messages, temperature):
        return ProviderResponse(content="complete")

    def complete_with_tools(
        self,
        session,
        *,
        model,
        messages,
        tools,
        tool_choice,
        temperature,
        max_tokens,
    ):
        return ProviderResponse(content="complete-with-tools")


class ModelRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()
        self.project = Project(name="runtime-project", password_hash="hash")
        self.db.add(self.project)
        self.db.commit()
        self.db.refresh(self.project)
        self.runtime = ModelRuntime(
            session_factory=self.Session,
            provider=AvailableProvider(),
        )

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def _active_artifact(
        self,
        root: str,
        *,
        artifact_status=ModelArtifactStatus.READY,
        artifact_base_model=DEFAULT_BASE_MODEL,
        write_metadata=True,
    ):
        dataset = Dataset(
            project_id=self.project.id,
            dataset_version="dataset-v1",
            storage_key="datasets/runtime-project/dataset-v1.jsonl",
            record_count=1,
            status=DatasetStatus.READY,
        )
        self.db.add(dataset)
        self.db.flush()
        adapter_path = Path(root) / self.project.id / "adapter-v1"
        artifact = ModelArtifact(
            project_id=self.project.id,
            artifact_version="adapter-v1",
            base_model=artifact_base_model,
            adapter_path=str(adapter_path),
            dataset_id=dataset.id,
            evaluation_score=1.0,
            status=artifact_status,
        )
        self.db.add(artifact)
        self.db.flush()
        self.project.active_artifact_id = artifact.id
        self.db.commit()

        if write_metadata:
            adapter_path.mkdir(parents=True)
            (adapter_path / "metadata.json").write_text(
                json.dumps(
                    {
                        "artifact_id": artifact.id,
                        "project_id": self.project.id,
                        "base_model": artifact_base_model,
                        "contains_adapter_weights": True,
                        "contains_base_model_weights": False,
                    }
                ),
                encoding="utf-8",
            )
        return artifact

    def test_resolves_shared_base_model_without_an_artifact(self):
        session = self.runtime.resolve_project_model(self.project.id)

        self.assertEqual(session.project_id, self.project.id)
        self.assertEqual(session.base_model, DEFAULT_BASE_MODEL)
        self.assertIsNone(session.active_artifact)
        self.assertIsNone(session.adapter_path)
        self.assertEqual(session.provider, "test-provider")
        self.assertEqual(session.runtime_type, "test-runtime")
        self.assertFalse(session.capabilities["model_fallbacks"])
        self.assertTrue(session.capabilities["adapter_resolution"])
        self.assertTrue(session.capabilities["adapter_loading"])
        self.assertFalse(session.capabilities["weights_loaded"])
        self.assertEqual(session.validation_warnings, ())

    def test_resolves_ready_artifact_and_loads_its_metadata(self):
        with tempfile.TemporaryDirectory() as artifact_root:
            artifact = self._active_artifact(artifact_root)

            session = self.runtime.resolve_project_model(self.project.id)

            self.assertEqual(session.active_artifact.id, artifact.id)
            self.assertEqual(session.active_artifact.version, "adapter-v1")
            self.assertEqual(session.adapter_path, artifact.adapter_path)
            self.assertTrue(
                session.active_artifact.metadata["contains_adapter_weights"]
            )
            self.assertFalse(
                session.active_artifact.metadata["contains_base_model_weights"]
            )
            self.assertTrue(session.capabilities["adapter_metadata_available"])
            self.assertTrue(session.capabilities["adapter_loading"])
            self.assertEqual(session.validation_warnings, ())

    def test_missing_metadata_is_reported_to_the_runtime_session(self):
        with tempfile.TemporaryDirectory() as artifact_root:
            artifact = self._active_artifact(
                artifact_root,
                write_metadata=False,
            )

            session = self.runtime.resolve_project_model(self.project.id)

            self.assertEqual(session.active_artifact.id, artifact.id)
            self.assertIsNone(session.active_artifact.metadata)
            self.assertTrue(session.provider_available)
            self.assertFalse(session.capabilities["adapter_metadata_available"])
            self.assertIn("was not found", session.validation_warnings[0])

    def test_rejects_non_ready_or_incompatible_active_artifacts(self):
        scenarios = (
            (ModelArtifactStatus.FAILED, DEFAULT_BASE_MODEL, "not READY"),
            (ModelArtifactStatus.READY, "another-model", "does not match"),
        )
        for index, (status, base_model, expected_warning) in enumerate(scenarios):
            with self.subTest(status=status, base_model=base_model):
                if index:
                    self.db.query(ModelArtifact).delete()
                    self.db.query(Dataset).delete()
                    self.project.active_artifact_id = None
                    self.db.commit()
                with tempfile.TemporaryDirectory() as artifact_root:
                    self._active_artifact(
                        artifact_root,
                        artifact_status=status,
                        artifact_base_model=base_model,
                    )

                    session = self.runtime.resolve_project_model(self.project.id)

                    self.assertIsNone(session.active_artifact)
                    self.assertIn(expected_warning, session.validation_warnings[0])

    def test_unknown_project_is_not_resolved_as_the_system_project(self):
        with self.assertRaisesRegex(LookupError, "was not found"):
            self.runtime.resolve_project_model("missing-project")

    def test_supplied_project_must_match_project_id(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.runtime.resolve_project_model(
                "another-project",
                project=self.project,
            )

    def test_session_delegates_inference_to_the_provider_contract(self):
        session = self.runtime.resolve_project_model(self.project.id)

        self.assertEqual(
            session.complete(
                model="test-model",
                messages="prompt",
                temperature=0.3,
            ).content,
            "complete",
        )
        self.assertEqual(
            session.complete_with_tools(
                model="test-model",
                messages=[],
                tools=None,
                tool_choice=None,
                temperature=0.2,
                max_tokens=100,
            ).content,
            "complete-with-tools",
        )


class FakeLocalCache:
    initialized = True
    loaded_artifact_ids = ()

    def __init__(self, content="response text"):
        self.content = content
        self.initialized_with = 0
        self.requests = []

    def initialize(self):
        self.initialized_with += 1

    def generate(self, **request):
        self.requests.append(request)
        return LocalGeneration(
            content=self.content,
            input_tokens=5,
            output_tokens=3,
        )


class LocalModelProviderTests(unittest.TestCase):
    def setUp(self):
        self.settings = RuntimeModelSettings()
        self.cache = FakeLocalCache()
        self.provider = LocalModelProvider(self.settings, self.cache)
        self.session = SimpleNamespace(
            project_id="project-1",
            active_artifact=SimpleNamespace(
                id="artifact-1",
                adapter_path="/tmp/project-1/adapter-v1",
                metadata={"contains_adapter_weights": True},
            ),
            adapter_path="/tmp/project-1/adapter-v1",
            validation_warnings=(),
        )

    def test_initialize_and_model_candidates_use_one_local_model(self):
        self.provider.initialize(self.settings.base_model)

        self.assertEqual(self.cache.initialized_with, 1)
        self.assertEqual(self.provider.model_candidates(), (self.settings.base_model,))
        self.assertEqual(self.provider.name, "local")

    def test_text_completion_uses_project_adapter_and_reports_usage(self):
        response = self.provider.complete(
            self.session,
            model=self.settings.base_model,
            messages="prompt",
            temperature=0.3,
        )

        self.assertEqual(response.content, "response text")
        self.assertEqual(response.usage_metadata["input_tokens"], 5)
        self.assertEqual(self.cache.requests[0]["project_id"], "project-1")
        self.assertEqual(self.cache.requests[0]["artifact_id"], "artifact-1")
        self.assertEqual(
            self.cache.requests[0]["messages"],
            [{"role": "user", "content": "prompt"}],
        )

    def test_tool_completion_parses_qwen_tool_call_envelope(self):
        self.cache.content = (
            '<tool_call>{"name":"get_runbook","arguments":{"id":"rb"}}</tool_call>'
        )

        response = self.provider.complete_with_tools(
            self.session,
            model=self.settings.base_model,
            messages=[{"role": "user", "content": "prompt"}],
            tools=[{"type": "function"}],
            tool_choice="auto",
            temperature=0.2,
            max_tokens=1500,
        )

        self.assertEqual(response.content, "")
        self.assertEqual(response.tool_calls[0].name, "get_runbook")
        self.assertEqual(response.tool_calls[0].arguments, '{"id":"rb"}')
        self.assertEqual(response.usage_metadata["total_tokens"], 8)

    def test_missing_active_artifact_is_a_clear_runtime_error(self):
        self.session.active_artifact = None
        self.session.adapter_path = None

        with self.assertRaisesRegex(AdapterLoadError, "no READY active"):
            self.provider.complete(
                self.session,
                model=self.settings.base_model,
                messages="prompt",
                temperature=0,
            )


if __name__ == "__main__":
    unittest.main()
