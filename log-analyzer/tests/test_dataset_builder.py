import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import routes_model_management
from app.api.routes_auth import get_current_project
from app.control.dataset_management import DatasetBuildCommand, NoEligibleIncidentsError
from app.control.models import Project
from app.data.models import Incident
from app.serving.models import Analysis, ActionLog, InvestigationRun
from app.shared.database import Base
from app.training.dataset_builder import (
    DATASET_EXAMPLE_SCHEMA_VERSION,
    DatasetBuilder,
)
from app.training.dataset_serializer import JsonLinesDatasetSerializer
from app.training.dataset_storage import (
    DatasetAlreadyExistsError,
    LocalDatasetStorage,
)
from app.training.models import Dataset, DatasetStatus, ModelArtifact, TrainingJob
from app.training.repositories import DatasetSourceRepository


class DatasetBuilderTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()
        self.project = Project(name="dataset-project", password_hash="hash")
        self.other_project = Project(name="other-project", password_hash="hash")
        self.db.add_all([self.project, self.other_project])
        self.db.commit()
        self.db.refresh(self.project)
        self.db.refresh(self.other_project)

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def _incident(self, **overrides):
        values = {
            "project_id": self.project.id,
            "source": "checkout-api",
            "environment": "prod",
            "signature": "database timeout",
            "sample_lines": ["ERROR database timeout"],
            "status": "closed",
        }
        values.update(overrides)
        incident = Incident(**values)
        self.db.add(incident)
        self.db.flush()
        return incident

    def _analysis(self, incident, **overrides):
        values = {
            "incident_id": incident.id,
            "severity": "high",
            "disposition": "NEEDS_ONCALL",
            "confidence": 0.9,
            "summary": "Database connections are exhausted",
            "next_steps": ["Inspect the connection pool"],
            "analysis_source": "runbook",
            "created_at": datetime.utcnow(),
        }
        values.update(overrides)
        analysis = Analysis(**values)
        self.db.add(analysis)
        self.db.flush()
        return analysis

    def test_builder_uses_latest_persisted_facts_and_skips_ineligible_rows(self):
        incident = self._incident(
            root_cause_incident_id="root-1",
            cause_explanation="Pool exhaustion caused request timeouts",
        )
        self._analysis(
            incident,
            severity="low",
            summary="Old analysis",
            created_at=datetime.utcnow() - timedelta(minutes=1),
        )
        latest_analysis = self._analysis(incident)
        investigation = InvestigationRun(
            incident_id=incident.id,
            project_id=self.project.id,
            evidence_snapshot="Persisted investigation evidence",
            evidence_samples=1,
            evidence_related_count=2,
            evidence_runbook="Database pool exhausted",
            started_at=datetime.utcnow(),
            finished_at=datetime.utcnow(),
            iterations=2,
        )
        action = ActionLog(
            incident_id=incident.id,
            project_id=self.project.id,
            requested_actions=["notify_oncall"],
            allowed_actions=["notify_oncall"],
            actions_taken=["notify_oncall"],
            outcome="resolved",
        )
        self.db.add_all([investigation, action])

        no_analysis = self._incident(signature="missing analysis")
        no_evidence = self._incident(signature="missing evidence", sample_lines=[])
        self._analysis(no_evidence)
        malformed = self._incident(signature="malformed", sample_lines={"bad": True})
        self._analysis(malformed)
        other = self._incident(
            project_id=self.other_project.id, signature="other project"
        )
        self._analysis(other)
        self.db.commit()

        examples = DatasetBuilder(DatasetSourceRepository(self.db)).build(
            self.project.id
        )

        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(set(example), {"input", "expected_output"})
        self.assertEqual(example["input"]["incident"]["id"], incident.id)
        self.assertEqual(
            example["input"]["metadata"]["analysis"]["id"], latest_analysis.id
        )
        self.assertEqual(
            example["input"]["evidence"]["snapshot"],
            "Persisted investigation evidence",
        )
        self.assertEqual(example["input"]["metadata"]["action"]["outcome"], "resolved")
        self.assertEqual(
            example["input"]["metadata"]["schema_version"],
            DATASET_EXAMPLE_SCHEMA_VERSION,
        )
        self.assertEqual(
            example["expected_output"]["root_cause"],
            {
                "incident_id": "root-1",
                "explanation": "Pool exhaustion caused request timeouts",
            },
        )
        self.assertEqual(
            example["expected_output"]["recommended_actions"],
            ["notify_oncall"],
        )
        self.assertNotIn(no_analysis.id, json.dumps(examples))

    def test_serializer_is_deterministic(self):
        serializer = JsonLinesDatasetSerializer()
        examples = [{"input": {"logs": ["é"]}, "expected_output": {"severity": "low"}}]

        first = serializer.serialize(examples)
        second = serializer.serialize(examples)

        self.assertEqual(first, second)
        self.assertTrue(first.endswith(b"\n"))
        self.assertEqual(json.loads(first), examples[0])

    def test_local_storage_never_overwrites_a_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = LocalDatasetStorage(directory)
            key = storage.storage_key("project-1", "dataset-v1", "jsonl")
            storage.save(key, b"first\n")

            with self.assertRaises(DatasetAlreadyExistsError):
                storage.save(key, b"second\n")

            self.assertEqual((Path(directory) / key).read_bytes(), b"first\n")

    def test_command_creates_immutable_incrementing_ready_datasets(self):
        incident = self._incident()
        self._analysis(incident)
        self.db.commit()

        with tempfile.TemporaryDirectory() as directory:
            command = DatasetBuildCommand(
                self.db, storage=LocalDatasetStorage(directory)
            )
            first = command.execute(self.project.id)
            second = command.execute(self.project.id)

            self.assertEqual(first.dataset_version, "dataset-v1")
            self.assertEqual(second.dataset_version, "dataset-v2")
            self.assertEqual(first.status, DatasetStatus.READY)
            self.assertEqual(first.record_count, 1)
            self.assertNotEqual(first.storage_key, second.storage_key)
            self.assertTrue((Path(directory) / first.storage_key).is_file())
            self.assertTrue((Path(directory) / second.storage_key).is_file())

        self.assertEqual(self.db.query(Dataset).count(), 2)
        self.assertEqual(self.db.query(TrainingJob).count(), 0)
        self.assertEqual(self.db.query(ModelArtifact).count(), 0)

    def test_command_rejects_project_without_eligible_incidents(self):
        with tempfile.TemporaryDirectory() as directory:
            command = DatasetBuildCommand(
                self.db, storage=LocalDatasetStorage(directory)
            )
            with self.assertRaises(NoEligibleIncidentsError):
                command.execute(self.project.id)

        self.assertEqual(self.db.query(Dataset).count(), 0)

    def test_storage_failure_registers_failed_dataset_without_starting_training(self):
        class FailingStorage(LocalDatasetStorage):
            def save(self, storage_key: str, content: bytes) -> None:
                raise OSError("storage unavailable")

        incident = self._incident()
        self._analysis(incident)
        self.db.commit()

        with tempfile.TemporaryDirectory() as directory:
            command = DatasetBuildCommand(self.db, storage=FailingStorage(directory))
            with self.assertRaisesRegex(OSError, "storage unavailable"):
                command.execute(self.project.id)

        dataset = self.db.query(Dataset).one()
        self.assertEqual(dataset.status, DatasetStatus.FAILED)
        self.assertEqual(dataset.record_count, 0)
        self.assertEqual(self.db.query(TrainingJob).count(), 0)

    def test_build_endpoint_returns_registered_dataset(self):
        incident = self._incident()
        self._analysis(incident)
        self.db.commit()

        api = FastAPI()
        api.include_router(routes_model_management.router, prefix="/api")
        api.dependency_overrides[get_current_project] = lambda: self.project

        def override_db():
            yield self.db

        api.dependency_overrides[routes_model_management.get_db] = override_db

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"DATASET_STORAGE_PATH": directory}):
                response = TestClient(api).post("/api/datasets/build")

            self.assertEqual(response.status_code, 201)
            payload = response.json()
            self.assertEqual(payload["dataset_version"], "dataset-v1")
            self.assertEqual(payload["status"], "READY")
            self.assertEqual(payload["record_count"], 1)
            self.assertTrue((Path(directory) / payload["storage_key"]).is_file())


if __name__ == "__main__":
    unittest.main()
