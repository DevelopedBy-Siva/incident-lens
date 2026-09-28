from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.data.clustering import _observed_tags
from app.data.parser import ParsedLog, resolve_source
from app.data.signatures import generate_signature
from app.serving.models import Analysis
from app.shared.database import Base
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


def test_kubernetes_oom_and_service_sigkill_share_an_incident_window():
    oom = ParsedLog(
        "2026-09-22T14:09:54.410Z ERROR k8s event namespace=prod "
        "Pod reporting-service-8b5d7c9f6-a9q75 container reporting-service "
        "OOMKilled, exit code 137"
    )
    killed = ParsedLog(
        "2026-09-22T14:09:55.410Z ERROR [reporting-service] worker process "
        "killed signal=SIGKILL host=reporting-service-8b5d7c9f6-a9q75"
    )

    oom_source = resolve_source(oom, "project-2")
    killed_source = resolve_source(killed, "project-2")
    assert oom_source == killed_source == "reporting-service"
    assert oom.entity == killed.entity == "reporting-service-8b5d7c9f6-a9q75"
    assert generate_signature(oom_source, oom) == generate_signature(killed_source, killed)


def test_provider_metadata_is_preserved_as_incident_evidence():
    parsed = ParsedLog(
        "ERROR [checkout-api] dependency request failed",
        attributes={"namespace": "payments", "region": "us-east-1", "ignored": "secret"},
    )

    assert _observed_tags(parsed) == {"namespace": "payments", "region": "us-east-1"}


def test_analysis_persistence_retries_after_connection_closes_during_commit():
    from app.serving import orchestrator

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)

    class FailingCommitSession:
        def __init__(self, session):
            self.session = session

        def commit(self):
            raise OperationalError("INSERT INTO analyses", {}, RuntimeError("connection closed"))

        def __getattr__(self, name):
            return getattr(self.session, name)

    result = SimpleNamespace(
        severity="high",
        disposition="NEEDS_ONCALL",
        confidence=0.9,
        summary="notification-worker has sustained consumer lag.",
        suspected_root_cause="The logs establish consumer lag but not its underlying cause.",
        next_steps=["Inspect consumer health."],
        ticket_title="notification-worker: consumer lag",
        ticket_body="Investigate consumer lag.",
    )
    first = FailingCommitSession(sessions())
    second = sessions()
    dispose = Mock()
    with patch.object(orchestrator, "SessionLocal", side_effect=[first, second]), patch.object(orchestrator.engine, "dispose", dispose):
        analysis = orchestrator._persist_analysis(Analysis, "incident-1", result, "model")

    check = sessions()
    try:
        stored = check.query(Analysis).filter(Analysis.incident_id == "incident-1").one()
        assert analysis.summary == stored.summary == result.summary
        assert stored.suspected_root_cause == result.suspected_root_cause
        dispose.assert_called_once()
    finally:
        check.close()
        engine.dispose()
