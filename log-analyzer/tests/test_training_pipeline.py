import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import routes_model_management
from app.api.routes_auth import get_current_project
from app.control.model_management import ModelManagementService
from app.control.models import DEFAULT_BASE_MODEL, Project
from app.serving.model_runtime import ModelRuntime
from app.shared.database import Base
from app.training.artifact_metadata import LocalArtifactMetadataWriter
from app.training.dataset_storage import LocalDatasetStorage
from app.training.evaluation import BasicEvaluationService
from app.training.models import (
    DatasetStatus,
    ModelArtifact,
    ModelArtifactStatus,
    TrainingJob,
    TrainingJobStatus,
)
from app.training.training_engine import (
    TrainingEngine,
    TrainingRequest,
    TrainingResult,
)
from app.training.training_profile import LoraTrainingProfile
from app.training.worker import TrainingJobStateError, TrainingWorker


class FailingTrainingEngine(TrainingEngine):
    def train(self, request: TrainingRequest) -> TrainingResult:
        raise RuntimeError("test engine failure")


class SuccessfulTestTrainingEngine(TrainingEngine):
    def train(self, request: TrainingRequest) -> TrainingResult:
        output = Path(request.adapter_output_path)
        files = {
            "adapter_config.json": json.dumps(
                {
                    "base_model_name_or_path": request.base_model,
                    "peft_type": "LORA",
                }
            ).encode(),
            "adapter_model.safetensors": b"test-adapter-weights",
            "tokenizer_config.json": b"{}",
        }
        for name, content in files.items():
            (output / name).write_bytes(content)
        manifest = tuple(
            {
                "name": name,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, content in sorted(files.items())
        )
        return TrainingResult(
            succeeded=True,
            engine="test-peft-lora",
            adapter_path=str(output),
            metrics={
                "training_completed": True,
                "base_model": request.base_model,
                "training_loss": 0.25,
                "validation_loss": None,
                "duration_seconds": 1.25,
                "weights_created": True,
                "records_seen": request.expected_record_count,
                "lora_configuration": request.profile.as_metadata(),
            },
            framework_versions={"transformers": "test", "peft": "test"},
            artifact_files=manifest,
        )


class FailingArtifactWriter(LocalArtifactMetadataWriter):
    def write(self, project_id, artifact_version, metadata) -> None:
        raise OSError("artifact metadata unavailable")


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()
        self.project = Project(name="training-project", password_hash="hash")
        self.db.add(self.project)
        self.db.commit()
        self.db.refresh(self.project)
        self.management = ModelManagementService(self.db)

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def _ready_dataset(
        self,
        storage: LocalDatasetStorage,
        *,
        record_count: int = 1,
        content: bytes = b'{"input":{"incident":"x"},"expected_output":{"severity":"low"}}\n',
    ):
        key = storage.storage_key(self.project.id, "dataset-v1", "jsonl")
        storage.save(key, content)
        dataset = self.management.create_dataset_metadata(
            self.project.id, "dataset-v1", key
        )
        return self.management.mark_dataset_ready(
            self.project.id, dataset.id, record_count
        )

    def test_evaluation_checks_losses_dataset_and_adapter_integrity(self):
        with tempfile.TemporaryDirectory() as artifact_directory:
            request = TrainingRequest(
                project_id="project-1",
                dataset_id="dataset-1",
                dataset_version="dataset-v1",
                base_model=DEFAULT_BASE_MODEL,
                dataset_content=b'{"input":{},"expected_output":{}}\n',
                expected_record_count=1,
                adapter_output_path=artifact_directory,
                profile=LoraTrainingProfile(),
            )
            result = SuccessfulTestTrainingEngine().train(request)
            dataset = type("DatasetRecord", (), {"record_count": 1})()

            passed = BasicEvaluationService().evaluate(
                dataset, request.dataset_content, result
            )
            mismatched = BasicEvaluationService().evaluate(
                dataset, request.dataset_content * 2, result
            )

            self.assertTrue(passed.passed)
            self.assertEqual(passed.score, 1.0)
            self.assertTrue(passed.metrics["artifact_integrity"])
            self.assertFalse(mismatched.passed)
            self.assertFalse(mismatched.metrics["record_count_matches"])

    def test_worker_creates_real_adapter_metadata_and_activates_it(self):
        with tempfile.TemporaryDirectory() as dataset_directory:
            with tempfile.TemporaryDirectory() as artifact_directory:
                storage = LocalDatasetStorage(dataset_directory)
                dataset = self._ready_dataset(storage)
                job = self.management.create_training_job(self.project.id, dataset.id)
                worker = TrainingWorker(
                    self.db,
                    dataset_storage=storage,
                    engine=SuccessfulTestTrainingEngine(),
                    artifact_writer=LocalArtifactMetadataWriter(artifact_directory),
                )

                completed = worker.run(job.id, self.project.id)

                self.assertEqual(completed.status, TrainingJobStatus.PASSED)
                self.assertIsNotNone(completed.started_at)
                self.assertIsNotNone(completed.finished_at)
                artifact = self.db.query(ModelArtifact).one()
                self.assertEqual(artifact.status, ModelArtifactStatus.READY)
                self.assertEqual(artifact.artifact_version, "adapter-v1")
                self.assertEqual(artifact.dataset_id, dataset.id)
                self.assertEqual(artifact.base_model, DEFAULT_BASE_MODEL)
                self.assertEqual(artifact.evaluation_score, 1.0)
                self.assertEqual(completed.artifact_id, artifact.id)

                self.db.refresh(self.project)
                self.assertEqual(self.project.active_artifact_id, artifact.id)
                metadata_path = Path(artifact.adapter_path) / "metadata.json"
                self.assertTrue(metadata_path.is_file())
                self.assertTrue(
                    (metadata_path.parent / "adapter_model.safetensors").is_file()
                )
                self.assertTrue(
                    (metadata_path.parent / "adapter_config.json").is_file()
                )
                self.assertTrue(
                    (metadata_path.parent / "tokenizer_config.json").is_file()
                )
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.assertEqual(metadata["project_id"], self.project.id)
                self.assertEqual(metadata["dataset_id"], dataset.id)
                self.assertEqual(metadata["base_model"], DEFAULT_BASE_MODEL)
                self.assertEqual(metadata["artifact_version"], "adapter-v1")
                self.assertEqual(metadata["evaluation_score"], 1.0)
                self.assertEqual(
                    metadata["schema_version"], "incidentlens-lora-artifact-v1"
                )
                self.assertTrue(metadata["contains_adapter_weights"])
                self.assertFalse(metadata["contains_base_model_weights"])
                self.assertEqual(metadata["training_duration_seconds"], 1.25)
                self.assertEqual(metadata["lora_configuration"]["rank"], 8)
                self.assertEqual(metadata["framework_versions"]["peft"], "test")
                self.assertNotIn("training_simulated", metadata)

                runtime_session = ModelRuntime(
                    session_factory=self.Session
                ).resolve_project_model(self.project.id)
                self.assertEqual(runtime_session.active_artifact.id, artifact.id)
                self.assertEqual(runtime_session.adapter_path, artifact.adapter_path)
                self.assertTrue(
                    runtime_session.active_artifact.metadata["contains_adapter_weights"]
                )

                self.db.refresh(dataset)
                self.assertEqual(dataset.status, DatasetStatus.READY)

    def test_engine_failure_marks_job_failed_and_keeps_active_artifact(self):
        with tempfile.TemporaryDirectory() as dataset_directory:
            with tempfile.TemporaryDirectory() as artifact_directory:
                storage = LocalDatasetStorage(dataset_directory)
                dataset = self._ready_dataset(storage)
                existing = self.management.register_model_artifact(
                    self.project.id,
                    "adapter-v1",
                    dataset.id,
                    adapter_path="artifacts/existing/adapter-v1/",
                    evaluation_score=0.8,
                )
                self.management.update_active_artifact(self.project.id, existing.id)
                job = self.management.create_training_job(self.project.id, dataset.id)
                worker = TrainingWorker(
                    self.db,
                    dataset_storage=storage,
                    engine=FailingTrainingEngine(),
                    artifact_writer=LocalArtifactMetadataWriter(artifact_directory),
                )

                failed = worker.run(job.id, self.project.id)

                self.assertEqual(failed.status, TrainingJobStatus.FAILED)
                self.assertIsNotNone(failed.started_at)
                self.assertIsNotNone(failed.finished_at)
                self.assertIsNone(failed.artifact_id)
                self.assertEqual(self.db.query(ModelArtifact).count(), 1)
                self.db.refresh(self.project)
                self.assertEqual(self.project.active_artifact_id, existing.id)

    def test_evaluation_failure_creates_no_artifact(self):
        with tempfile.TemporaryDirectory() as dataset_directory:
            with tempfile.TemporaryDirectory() as artifact_directory:
                storage = LocalDatasetStorage(dataset_directory)
                dataset = self._ready_dataset(storage, record_count=2)
                job = self.management.create_training_job(self.project.id, dataset.id)
                worker = TrainingWorker(
                    self.db,
                    dataset_storage=storage,
                    engine=SuccessfulTestTrainingEngine(),
                    artifact_writer=LocalArtifactMetadataWriter(artifact_directory),
                )

                failed = worker.run(job.id, self.project.id)

                self.assertEqual(failed.status, TrainingJobStatus.FAILED)
                self.assertEqual(self.db.query(ModelArtifact).count(), 0)
                self.db.refresh(self.project)
                self.assertIsNone(self.project.active_artifact_id)

    def test_metadata_failure_marks_artifact_failed_without_activation(self):
        with tempfile.TemporaryDirectory() as dataset_directory:
            with tempfile.TemporaryDirectory() as artifact_directory:
                storage = LocalDatasetStorage(dataset_directory)
                dataset = self._ready_dataset(storage)
                job = self.management.create_training_job(self.project.id, dataset.id)
                worker = TrainingWorker(
                    self.db,
                    dataset_storage=storage,
                    engine=SuccessfulTestTrainingEngine(),
                    artifact_writer=FailingArtifactWriter(artifact_directory),
                )

                failed = worker.run(job.id, self.project.id)

                artifact = self.db.query(ModelArtifact).one()
                self.assertEqual(failed.status, TrainingJobStatus.FAILED)
                self.assertEqual(artifact.status, ModelArtifactStatus.FAILED)
                self.assertIsNone(failed.artifact_id)
                self.db.refresh(self.project)
                self.assertIsNone(self.project.active_artifact_id)

    def test_training_job_requires_ready_dataset(self):
        dataset = self.management.create_dataset_metadata(
            self.project.id,
            "dataset-v1",
            "projects/training-project/dataset-v1.jsonl",
        )

        with self.assertRaisesRegex(ValueError, "READY"):
            self.management.create_training_job(self.project.id, dataset.id)

    def test_run_next_reserves_oldest_job_and_completed_job_cannot_rerun(self):
        with tempfile.TemporaryDirectory() as dataset_directory:
            with tempfile.TemporaryDirectory() as artifact_directory:
                storage = LocalDatasetStorage(dataset_directory)
                dataset = self._ready_dataset(storage)
                job = self.management.create_training_job(self.project.id, dataset.id)
                worker = TrainingWorker(
                    self.db,
                    dataset_storage=storage,
                    engine=SuccessfulTestTrainingEngine(),
                    artifact_writer=LocalArtifactMetadataWriter(artifact_directory),
                )

                completed = worker.run_next(self.project.id)

                self.assertEqual(completed.id, job.id)
                self.assertEqual(completed.status, TrainingJobStatus.PASSED)
                self.assertIsNone(worker.run_next(self.project.id))
                with self.assertRaises(TrainingJobStateError):
                    worker.run(job.id, self.project.id)

    def test_successive_jobs_version_artifacts_and_activate_latest(self):
        with tempfile.TemporaryDirectory() as dataset_directory:
            with tempfile.TemporaryDirectory() as artifact_directory:
                storage = LocalDatasetStorage(dataset_directory)
                dataset = self._ready_dataset(storage)
                worker = TrainingWorker(
                    self.db,
                    dataset_storage=storage,
                    engine=SuccessfulTestTrainingEngine(),
                    artifact_writer=LocalArtifactMetadataWriter(artifact_directory),
                )

                first_job = self.management.create_training_job(
                    self.project.id, dataset.id
                )
                first = worker.run(first_job.id, self.project.id)
                second_job = self.management.create_training_job(
                    self.project.id, dataset.id
                )
                second = worker.run(second_job.id, self.project.id)

                artifacts = (
                    self.db.query(ModelArtifact)
                    .order_by(ModelArtifact.artifact_version.asc())
                    .all()
                )
                self.assertEqual(
                    [artifact.artifact_version for artifact in artifacts],
                    ["adapter-v1", "adapter-v2"],
                )
                self.assertNotEqual(first.artifact_id, second.artifact_id)
                self.db.refresh(self.project)
                self.assertEqual(self.project.active_artifact_id, second.artifact_id)

    def test_training_api_queues_and_runs_job(self):
        with tempfile.TemporaryDirectory() as dataset_directory:
            with tempfile.TemporaryDirectory() as artifact_directory:
                storage = LocalDatasetStorage(dataset_directory)
                dataset = self._ready_dataset(storage)

                api = FastAPI()
                api.include_router(routes_model_management.router, prefix="/api")
                api.dependency_overrides[get_current_project] = lambda: self.project

                def override_db():
                    yield self.db

                api.dependency_overrides[routes_model_management.get_db] = override_db

                with patch.dict(
                    os.environ,
                    {
                        "DATASET_STORAGE_PATH": dataset_directory,
                        "ARTIFACT_STORAGE_PATH": artifact_directory,
                    },
                ):
                    with patch(
                        "app.training.worker.configured_training_engine",
                        return_value=SuccessfulTestTrainingEngine(),
                    ):
                        client = TestClient(api)
                        queued_response = client.post(
                            "/api/training-jobs", json={"dataset_id": dataset.id}
                        )
                        self.assertEqual(queued_response.status_code, 201)
                        queued = queued_response.json()
                        self.assertEqual(queued["status"], "QUEUED")

                        run_response = client.post(
                            f"/api/training-jobs/{queued['id']}/run"
                        )

                self.assertEqual(run_response.status_code, 200)
                completed = run_response.json()
                self.assertEqual(completed["status"], "PASSED")
                self.assertIsNotNone(completed["artifact_id"])
                self.db.refresh(self.project)
                self.assertEqual(
                    self.project.active_artifact_id, completed["artifact_id"]
                )
                self.assertEqual(self.db.query(TrainingJob).count(), 1)


if __name__ == "__main__":
    unittest.main()
