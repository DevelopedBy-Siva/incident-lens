import os
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

from app.control.dataset_management import DatasetBuildCommand
from app.data.ingestion.log_source_watcher import fetch_logs_for_project
from app.serving.policy import PolicyDecision, evaluate
from app.shared import observability
from app.training.models import TrainingJobStatus
from app.training.worker import TrainingWorker


class RecordingSpan:
    def __init__(self):
        self.tags = {}
        self.metrics = {}

    def set_tag(self, key, value):
        self.tags[key] = value

    def set_metric(self, key, value):
        self.metrics[key] = value


class RecordingTracer:
    def __init__(self):
        self.calls = []
        self.span = RecordingSpan()

    @contextmanager
    def trace(self, name, **options):
        self.calls.append((name, options))
        yield self.span


class RecordingLLMObs:
    enabled = True
    calls: ClassVar[list] = []
    annotations: ClassVar[list] = []

    @classmethod
    @contextmanager
    def llm(cls, **options):
        cls.calls.append(options)
        yield object()

    @classmethod
    def annotate(cls, **annotation):
        cls.annotations.append(annotation)


class DatadogObservabilityTests(unittest.TestCase):
    def setUp(self):
        RecordingLLMObs.calls = []
        RecordingLLMObs.annotations = []

    def test_local_inference_uses_apm_and_llm_observability_without_secrets(self):
        tracer = RecordingTracer()
        with (
            patch.object(observability, "_datadog_tracer", tracer),
            patch.object(observability, "_DatadogLLMObs", RecordingLLMObs),
            patch.dict(os.environ, {"DD_LLMOBS_ENABLED": "1"}),
            observability.trace_llm_operation(
                "local_inference",
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                metadata={
                    "project_id": "project-1",
                    "incident_id": "incident-1",
                    "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "api_key": "must-not-appear",
                    "raw_prompt": "must-not-appear",
                },
            ) as span,
        ):
            span.metrics({"input_tokens": 10, "output_tokens": 4, "total_tokens": 14})

        self.assertEqual(tracer.calls[0][0], "incident_lens.serving.local_inference")
        self.assertEqual(RecordingLLMObs.calls[0]["model_provider"], "local")
        self.assertEqual(tracer.span.tags["project_id"], "project-1")
        self.assertNotIn("api_key", tracer.span.tags)
        self.assertNotIn("raw_prompt", tracer.span.tags)
        self.assertEqual(tracer.span.metrics["total_tokens"], 14)

    def test_log_ingestion_emits_data_plane_trace(self):
        operations = []

        @contextmanager
        def record(name, *, plane, metadata):
            operations.append((name, plane, metadata))
            yield SimpleNamespace(metrics=lambda _values: None, tag=lambda *_args: None)

        class Connector:
            def __init__(self, _config):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def fetch(self, _start, _end):
                return []

        project = SimpleNamespace(
            id="project-1",
            datadog_api_key="api-key",
            datadog_app_key="app-key",
            datadog_site="datadoghq.com",
            datadog_query="status:error",
            datadog_environment="prod",
            datadog_service=None,
        )
        now = datetime.now(timezone.utc)
        with patch("app.data.ingestion.log_source_watcher.trace_operation", record):
            fetch_logs_for_project(project, now, now, connector_factory=Connector)

        self.assertEqual(operations[0][0:2], ("log_ingestion", "data"))
        self.assertEqual(operations[0][2]["project_id"], "project-1")

    def test_dataset_training_and_policy_entrypoints_emit_traces(self):
        operations = []

        @contextmanager
        def record(name, *, plane, metadata):
            operations.append((name, plane))
            yield SimpleNamespace(
                tags=lambda _values: None,
                tag=lambda *_args: None,
                metrics=lambda _values: None,
            )

        dataset_command = DatasetBuildCommand.__new__(DatasetBuildCommand)
        dataset = SimpleNamespace(
            id="dataset-1",
            dataset_version="dataset-v1",
            record_count=2,
            status=SimpleNamespace(value="READY"),
        )
        worker = TrainingWorker.__new__(TrainingWorker)
        job = SimpleNamespace(
            id="job-1",
            dataset_id="dataset-1",
            artifact_id="artifact-1",
            status=TrainingJobStatus.PASSED,
        )
        incident = SimpleNamespace(id="incident-1", project_id="project-1")
        analysis = SimpleNamespace(analysis_source="local_llm")
        policy = PolicyDecision(
            allow=True,
            reason="allowed",
            effective_disposition="OBSERVE",
            allowed_actions=["auto_enrich"],
        )

        with (
            patch("app.control.dataset_management.trace_operation", record),
            patch.object(dataset_command, "_execute", return_value=dataset),
        ):
            dataset_command.execute("project-1")
        with (
            patch("app.training.worker.trace_operation", record),
            patch.object(worker, "_run", return_value=job),
        ):
            worker.run("job-1", "project-1")
        with (
            patch("app.serving.policy.trace_operation", record),
            patch("app.serving.policy._evaluate", return_value=policy),
        ):
            evaluate(incident, analysis)

        self.assertIn(("dataset_generation", "training"), operations)
        self.assertIn(("training_job_lifecycle", "training"), operations)
        self.assertIn(("policy_evaluation", "serving"), operations)


if __name__ == "__main__":
    unittest.main()
