import unittest
from unittest.mock import patch

from app.api import routes_auth, routes_incidents
from app.api.routes_auth import get_current_project
from app.control.models import Project
from app.data.models import Incident
from app.serving.models import ActionLog, Analysis, InvestigationRun
from app.shared.database import Base, get_db
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


class ProjectConfigurationApiTests(unittest.TestCase):
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

        app = FastAPI()
        app.include_router(routes_auth.router, prefix="/api/auth")
        app.include_router(routes_incidents.router, prefix="/api")
        app.dependency_overrides[get_current_project] = lambda: self.project

        def override_db():
            yield self.db

        app.dependency_overrides[get_db] = override_db
        app.dependency_overrides[routes_auth.get_db] = override_db
        app.dependency_overrides[routes_incidents.get_db] = override_db
        self.client = TestClient(app)

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def test_datadog_verification_uses_proposed_values_without_saving(self):
        captured = {}

        class VerifyingConnector:
            def __init__(self, config, page_size):
                captured["config"] = config
                captured["page_size"] = page_size

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def verify_access(self, start, end):
                captured["verified"] = start < end

        payload = {
            "datadog_api_key": "api-key",
            "datadog_app_key": "app-key",
            "datadog_site": "datadoghq.com",
            "datadog_query": "status:error",
            "datadog_service": "checkout",
        }
        with patch.object(routes_auth, "DatadogLogConnector", VerifyingConnector):
            response = self.client.post(
                "/api/auth/settings/datadog/verify", json=payload
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["valid"])
        self.assertTrue(captured["verified"])
        self.assertEqual(captured["config"].service_filter, "checkout")
        self.db.refresh(self.project)
        self.assertIsNone(self.project.datadog_api_key)

    def test_global_environment_does_not_make_project_configured(self):
        with patch.dict(
            "os.environ",
            {
                "DATADOG_API_KEY": "global-api",
                "DATADOG_APP_KEY": "global-app",
                "DATADOG_SITE": "datadoghq.com",
                "DATADOG_QUERY": "status:error",
                "DATADOG_SERVICE": "checkout",
            },
        ):
            response = self.client.get("/api/auth/settings/status")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["datadog"]["configured"])

    def test_clear_deletes_only_current_project_incident_data(self):
        incident = Incident(
            project_id=self.project.id,
            source="checkout",
            signature="current",
            sample_lines=["ERROR current"],
        )
        other_incident = Incident(
            project_id=self.other_project.id,
            source="billing",
            signature="other",
            sample_lines=["ERROR other"],
        )
        self.db.add_all([incident, other_incident])
        self.db.flush()
        self.db.add_all(
            [
                Analysis(incident_id=incident.id, summary="current"),
                Analysis(incident_id=other_incident.id, summary="other"),
                ActionLog(project_id=self.project.id, incident_id=incident.id),
                ActionLog(
                    project_id=self.other_project.id, incident_id=other_incident.id
                ),
                InvestigationRun(project_id=self.project.id, incident_id=incident.id),
                InvestigationRun(
                    project_id=self.other_project.id, incident_id=other_incident.id
                ),
            ]
        )
        self.db.commit()

        response = self.client.delete("/api/incidents")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["deleted"]["incidents"], 1)
        self.assertEqual(
            self.db.query(Incident)
            .filter(Incident.project_id == self.project.id)
            .count(),
            0,
        )
        self.assertEqual(
            self.db.query(Incident)
            .filter(Incident.project_id == self.other_project.id)
            .count(),
            1,
        )
        self.assertEqual(self.db.query(Analysis).count(), 1)
        self.assertEqual(self.db.query(ActionLog).count(), 1)
        self.assertEqual(self.db.query(InvestigationRun).count(), 1)

    def test_delete_project_removes_only_authenticated_project(self):
        incident = Incident(
            project_id=self.project.id,
            source="checkout",
            signature="delete-me",
            sample_lines=["ERROR delete me"],
        )
        self.db.add(incident)
        self.db.commit()

        response = self.client.delete("/api/auth/project")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "deleted")
        self.assertIsNone(
            self.db.query(Project).filter(Project.id == self.project.id).first()
        )
        self.assertIsNotNone(
            self.db.query(Project).filter(Project.id == self.other_project.id).first()
        )
        self.assertEqual(
            self.db.query(Incident)
            .filter(Incident.project_id == self.project.id)
            .count(),
            0,
        )


if __name__ == "__main__":
    unittest.main()
