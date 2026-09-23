import unittest

from sqlalchemy import create_engine, text

from app.shared.migrations.versions import v0012_qwen35_base_model
from app.shared.model_config import DEFAULT_BASE_MODEL


class Qwen35BaseModelMigrationTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE projects ("
                    "id VARCHAR PRIMARY KEY, "
                    "base_model VARCHAR NOT NULL, "
                    "active_artifact_id VARCHAR"
                    ")"
                )
            )
            connection.execute(
                text(
                    "CREATE TABLE model_artifacts ("
                    "id VARCHAR PRIMARY KEY, "
                    "project_id VARCHAR NOT NULL, "
                    "base_model VARCHAR NOT NULL"
                    ")"
                )
            )

    def tearDown(self):
        self.engine.dispose()

    def test_updates_project_and_detaches_incompatible_adapter(self):
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO projects "
                    "(id, base_model, active_artifact_id) "
                    "VALUES ('project-1', 'legacy-base-model', 'artifact-1')"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO model_artifacts (id, project_id, base_model) "
                    "VALUES ('artifact-1', 'project-1', "
                    "'legacy-base-model')"
                )
            )

            v0012_qwen35_base_model.upgrade(connection)

            project = connection.execute(
                text(
                    "SELECT base_model, active_artifact_id FROM projects "
                    "WHERE id = 'project-1'"
                )
            ).one()
            artifact_model = connection.execute(
                text(
                    "SELECT base_model FROM model_artifacts "
                    "WHERE id = 'artifact-1'"
                )
            ).scalar_one()

        self.assertEqual(project.base_model, DEFAULT_BASE_MODEL)
        self.assertIsNone(project.active_artifact_id)
        self.assertEqual(artifact_model, "legacy-base-model")

    def test_keeps_compatible_qwen35_adapter_active(self):
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO projects "
                    "(id, base_model, active_artifact_id) "
                    "VALUES ('project-1', 'legacy-model', 'artifact-1')"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO model_artifacts (id, project_id, base_model) "
                    "VALUES ('artifact-1', 'project-1', :base_model)"
                ),
                {"base_model": DEFAULT_BASE_MODEL},
            )

            v0012_qwen35_base_model.upgrade(connection)

            project = connection.execute(
                text(
                    "SELECT base_model, active_artifact_id FROM projects "
                    "WHERE id = 'project-1'"
                )
            ).one()

        self.assertEqual(project.base_model, DEFAULT_BASE_MODEL)
        self.assertEqual(project.active_artifact_id, "artifact-1")


if __name__ == "__main__":
    unittest.main()
