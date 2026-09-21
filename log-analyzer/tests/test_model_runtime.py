import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.control.models import DEFAULT_BASE_MODEL, Project
from app.serving.model_provider import (
    DEFAULT_GROQ_MODEL,
    GroqProvider,
    ModelProvider,
    ProviderResponse,
)
from app.serving.model_runtime import ModelRuntime
from app.shared.database import Base
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
        self.assertFalse(session.capabilities["adapter_loading"])
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
            self.assertFalse(session.capabilities["adapter_loading"])
            self.assertEqual(session.validation_warnings, ())

    def test_missing_metadata_is_reported_without_disabling_remote_inference(self):
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


class GroqProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = GroqProvider()
        self.session = SimpleNamespace(_provider_api_key="project-key")

    def test_model_candidates_preserve_existing_order_and_remove_duplicates(self):
        with patch.dict(
            os.environ,
            {
                "GROQ_MODEL": "primary-model",
                "GROQ_MODEL_FALLBACKS": (
                    f"{DEFAULT_GROQ_MODEL},fallback-model,primary-model"
                ),
            },
            clear=False,
        ):
            self.assertEqual(
                self.provider.model_candidates(),
                ("primary-model", DEFAULT_GROQ_MODEL, "fallback-model"),
            )

    @patch("langchain_groq.ChatGroq")
    def test_text_completion_preserves_groq_request_and_normalizes_response(
        self, chat_groq
    ):
        client = chat_groq.return_value
        client.invoke.return_value = SimpleNamespace(
            content="response text",
            usage_metadata={"input_tokens": 5, "output_tokens": 3},
        )

        response = self.provider.complete(
            self.session,
            model="configured-model",
            messages="prompt",
            temperature=0.3,
        )

        chat_groq.assert_called_once_with(
            model="configured-model",
            temperature=0.3,
            api_key="project-key",
        )
        client.invoke.assert_called_once_with("prompt")
        self.assertEqual(response.content, "response text")
        self.assertEqual(response.usage_metadata["input_tokens"], 5)

    @patch("groq.Groq")
    def test_tool_completion_normalizes_groq_tool_calls(self, groq_client):
        raw_tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="get_runbook", arguments='{"id":"rb"}'),
        )
        raw_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[raw_tool_call],
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=4,
                total_tokens=14,
            ),
        )
        client = groq_client.return_value
        client.chat.completions.create.return_value = raw_response

        response = self.provider.complete_with_tools(
            self.session,
            model="configured-model",
            messages=[{"role": "user", "content": "prompt"}],
            tools=[{"type": "function"}],
            tool_choice="auto",
            temperature=0.2,
            max_tokens=1500,
        )

        groq_client.assert_called_once_with(api_key="project-key")
        self.assertEqual(response.content, "")
        self.assertEqual(response.tool_calls[0].id, "call-1")
        self.assertEqual(response.tool_calls[0].name, "get_runbook")
        self.assertEqual(response.tool_calls[0].arguments, '{"id":"rb"}')
        self.assertEqual(response.usage_metadata["total_tokens"], 14)


if __name__ == "__main__":
    unittest.main()
