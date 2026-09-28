import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.training.dataset_builder import DatasetBuilder
from app.training.dataset_serializer import DatasetValidationError, JsonLinesDatasetSerializer


def canonical_record():
    return {
        "input": {"logs": ["2026-01-01T00:00:00Z ERROR [checkout] connection pool exhausted"], "service": "checkout", "environment": "prod", "metadata": {"scenario_group_id": "checkout-pool"}},
        "expected_output": {"severity": "high", "disposition": "NEEDS_ONCALL", "confidence": 0.9, "summary": "Checkout cannot obtain database connections.", "suspected_root_cause": "The pool is exhausted.", "next_steps": ["Inspect pool usage."], "ticket_title": "Checkout database pool exhaustion", "ticket_body": "Investigate pool exhaustion."},
    }


def test_committed_dataset_is_canonical_and_loadable():
    serializer = JsonLinesDatasetSerializer()
    records = serializer.deserialize((Path(__file__).parents[2] / "data" / "dataset_v4.jsonl").read_bytes())
    assert len(records) == 2200
    assert {record["expected_output"]["severity"] for record in records} == {"low", "medium", "high", "critical"}


def test_serializer_rejects_partial_legacy_output():
    record = canonical_record()
    del record["expected_output"]["confidence"]
    with pytest.raises(DatasetValidationError, match="Missing required fields"):
        JsonLinesDatasetSerializer().serialize([record])


def test_builder_emits_the_exact_inference_contract():
    incident = SimpleNamespace(id="incident-1", source="checkout", environment="prod", signature="sig", count=3, sample_lines=canonical_record()["input"]["logs"], cause_explanation=None)
    analysis = SimpleNamespace(severity="high", disposition="NEEDS_ONCALL", confidence=0.9, summary="Checkout cannot obtain database connections.", next_steps=["Inspect pool usage."], ticket_title="Checkout database pool exhaustion", ticket_body="Investigate pool exhaustion.")
    source = SimpleNamespace(incident=incident, analysis=analysis)
    records = DatasetBuilder(SimpleNamespace(list_for_project=lambda _project: [source])).build("project-1")
    JsonLinesDatasetSerializer().validate(records)
    assert records[0]["expected_output"]["severity"] == "high"
