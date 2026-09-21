import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import routes_model_management
from app.api.routes_auth import get_current_project
from app.control.model_management import ModelManagementService
from app.control.models import DEFAULT_BASE_MODEL, Project
from app.shared.database import Base
from app.shared.migrations.versions.v0002_model_lifecycle import upgrade as upgrade_v2
from app.shared.migrations.versions.v0003_dataset_record_count import (
    upgrade as upgrade_v3,
)
from app.training.models import (
    DatasetStatus,
    ModelArtifactStatus,
    TrainingJobStatus,
)
from app.training.repositories import (
    DatasetRepository,
    ModelArtifactRepository,
    TrainingJobRepository,
)


class ModelLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()

        self.project = Project(name="alpha", password_hash="hash")
        self.other_project = Project(name="beta", password_hash="hash")
        self.db.add_all([self.project, self.other_project])
        self.db.commit()
        self.db.refresh(self.project)
        self.db.refresh(self.other_project)

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def test_project_defaults_to_shared_base_model(self):
        self.assertEqual(self.project.base_model, DEFAULT_BASE_MODEL)
        self.assertIsNone(self.project.active_artifact_id)

    def test_service_creates_metadata_lifecycle_and_activates_ready_artifact(self):
        service = ModelManagementService(self.db)

        dataset = service.create_dataset_metadata(
            self.project.id, "v1", "datasets/alpha/v1.jsonl"
        )
        dataset = service.mark_dataset_ready(self.project.id, dataset.id, 1)
        job = service.create_training_job(self.project.id, dataset.id)
        artifact = service.register_model_artifact(
            self.project.id,
            "v1",
            dataset.id,
            adapter_path="artifacts/alpha/v1",
            evaluation_score=0.91,
        )
        activated_project = service.update_active_artifact(self.project.id, artifact.id)

        self.assertEqual(dataset.status, DatasetStatus.READY)
        self.assertEqual(job.status, TrainingJobStatus.QUEUED)
        self.assertEqual(artifact.status, ModelArtifactStatus.READY)
        self.assertEqual(artifact.base_model, DEFAULT_BASE_MODEL)
        self.assertEqual(activated_project.active_artifact_id, artifact.id)

    def test_repositories_scope_reads_to_project(self):
        service = ModelManagementService(self.db)
        dataset = service.create_dataset_metadata(
            self.project.id, "v1", "datasets/alpha/v1.jsonl"
        )
        dataset = service.mark_dataset_ready(self.project.id, dataset.id, 1)
        job = service.create_training_job(self.project.id, dataset.id)
        artifact = service.register_model_artifact(
            self.project.id, "v1", dataset.id, adapter_path="artifacts/alpha/v1"
        )

        self.assertEqual(
            DatasetRepository(self.db).list_for_project(self.project.id), [dataset]
        )
        self.assertEqual(
            TrainingJobRepository(self.db).list_for_project(self.project.id), [job]
        )
        self.assertEqual(
            ModelArtifactRepository(self.db).list_for_project(self.project.id),
            [artifact],
        )
        self.assertIsNone(
            DatasetRepository(self.db).get_for_project(
                dataset.id, self.other_project.id
            )
        )

    def test_non_ready_artifact_cannot_be_activated(self):
        service = ModelManagementService(self.db)
        dataset = service.create_dataset_metadata(
            self.project.id, "v1", "datasets/alpha/v1.jsonl"
        )
        dataset = service.mark_dataset_ready(self.project.id, dataset.id, 1)
        artifact = service.register_model_artifact(
            self.project.id,
            "v1",
            dataset.id,
            status=ModelArtifactStatus.FAILED,
        )

        with self.assertRaisesRegex(ValueError, "READY"):
            service.update_active_artifact(self.project.id, artifact.id)

    def test_read_only_api_lists_and_gets_project_scoped_entities(self):
        service = ModelManagementService(self.db)
        dataset = service.create_dataset_metadata(
            self.project.id, "v1", "datasets/alpha/v1.jsonl"
        )
        dataset = service.mark_dataset_ready(self.project.id, dataset.id, 1)
        job = service.create_training_job(self.project.id, dataset.id)
        artifact = service.register_model_artifact(
            self.project.id, "v1", dataset.id, adapter_path="artifacts/alpha/v1"
        )

        api = FastAPI()
        api.include_router(routes_model_management.router, prefix="/api")
        api.dependency_overrides[get_current_project] = lambda: self.project

        def override_db():
            yield self.db

        api.dependency_overrides[routes_model_management.get_db] = override_db
        client = TestClient(api)

        expected = {
            "/api/datasets": dataset.id,
            "/api/training-jobs": job.id,
            "/api/model-artifacts": artifact.id,
        }
        for path, entity_id in expected.items():
            with self.subTest(path=path):
                response = client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()[0]["id"], entity_id)

                detail = client.get(f"{path}/{entity_id}")
                self.assertEqual(detail.status_code, 200)
                self.assertEqual(detail.json()["project_id"], self.project.id)

                missing = client.get(f"{path}/missing")
                self.assertEqual(missing.status_code, 404)


class ModelLifecycleMigrationTests(unittest.TestCase):
    def test_upgrade_preserves_legacy_projects_and_adds_lifecycle_schema(self):
        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE projects ("
                    "id VARCHAR PRIMARY KEY, "
                    "name VARCHAR NOT NULL, "
                    "password_hash VARCHAR NOT NULL"
                    ")"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO projects (id, name, password_hash) "
                    "VALUES ('project-1', 'legacy', 'hash')"
                )
            )
            upgrade_v2(connection)

        inspector = inspect(engine)
        project_columns = {
            column["name"] for column in inspector.get_columns("projects")
        }
        self.assertIn("base_model", project_columns)
        self.assertIn("active_artifact_id", project_columns)
        self.assertTrue(inspector.has_table("datasets"))
        self.assertTrue(inspector.has_table("training_jobs"))
        self.assertTrue(inspector.has_table("model_artifacts"))

        with engine.connect() as connection:
            base_model = connection.execute(
                text("SELECT base_model FROM projects WHERE id = 'project-1'")
            ).scalar_one()
        self.assertEqual(base_model, DEFAULT_BASE_MODEL)
        engine.dispose()

    def test_record_count_migration_upgrades_existing_dataset_table(self):
        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE datasets ("
                    "id VARCHAR PRIMARY KEY, "
                    "project_id VARCHAR NOT NULL, "
                    "dataset_version VARCHAR NOT NULL, "
                    "storage_key VARCHAR NOT NULL, "
                    "status VARCHAR NOT NULL, "
                    "created_at DATETIME NOT NULL"
                    ")"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO datasets "
                    "(id, project_id, dataset_version, storage_key, status, created_at) "
                    "VALUES ('dataset-1', 'project-1', 'dataset-v1', "
                    "'projects/project-1/dataset-v1.jsonl', 'READY', CURRENT_TIMESTAMP)"
                )
            )
            upgrade_v3(connection)

        self.assertIn(
            "record_count",
            {column["name"] for column in inspect(engine).get_columns("datasets")},
        )
        with engine.connect() as connection:
            count = connection.execute(
                text("SELECT record_count FROM datasets WHERE id = 'dataset-1'")
            ).scalar_one()
        self.assertEqual(count, 0)
        engine.dispose()


if __name__ == "__main__":
    unittest.main()
