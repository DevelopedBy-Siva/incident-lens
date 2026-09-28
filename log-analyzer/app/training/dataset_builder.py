"""Create canonical training records from reviewed production incidents."""

from __future__ import annotations

from typing import Any

from app.training.repositories import DatasetSourceRepository


DATASET_EXAMPLE_SCHEMA_VERSION = "incident-training-example-v2"


class DatasetEligibilityRules:
    """Compatibility placeholder; eligibility is intentionally defined in DatasetBuilder."""


class DatasetBuilder:
    """Only completed, evidence-backed analyses are eligible for model training."""

    def __init__(self, sources: DatasetSourceRepository, rules=None):
        self.sources = sources

    def build(self, project_id: str) -> list[dict[str, Any]]:
        records = []
        for source in self.sources.list_for_project(project_id):
            incident, analysis = source.incident, source.analysis
            if not isinstance(incident.sample_lines, list):
                continue
            logs = list(incident.sample_lines)
            if not analysis or not logs or not all(isinstance(line, str) and line.strip() for line in logs):
                continue
            # Rule-only fallbacks are operational safety output, not ground
            # truth. Training on them teaches the adapter repeated boilerplate.
            if getattr(analysis, "analysis_source", None) == "rules":
                continue
            if not all(isinstance(getattr(analysis, field, None), str) and getattr(analysis, field).strip() for field in ("severity", "disposition", "summary")):
                continue
            records.append({
                "input": {
                    "logs": logs,
                    "service": incident.source,
                    "environment": incident.environment,
                    "metadata": {
                        "schema_version": DATASET_EXAMPLE_SCHEMA_VERSION,
                        "incident_id": incident.id,
                        "signature": incident.signature,
                        "count": incident.count,
                        "scenario_group_id": incident.signature,
                        "label_source": getattr(analysis, "analysis_source", "reviewed"),
                    },
                },
                "expected_output": {
                    "severity": analysis.severity.lower(),
                    "disposition": analysis.disposition.upper(),
                    "confidence": float(analysis.confidence or 0.7),
                    "summary": analysis.summary.strip(),
                    "suspected_root_cause": incident.cause_explanation or None,
                    "next_steps": [step for step in (analysis.next_steps or []) if isinstance(step, str)],
                    "ticket_title": (analysis.ticket_title or f"{incident.source}: investigate incident").strip(),
                    "ticket_body": (analysis.ticket_body or analysis.summary).strip(),
                },
            })
        return records
