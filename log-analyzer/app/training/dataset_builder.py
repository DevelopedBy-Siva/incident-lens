from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.training.repositories import DatasetSourceRecord, DatasetSourceRepository

DATASET_EXAMPLE_SCHEMA_VERSION = "incident-training-example-v1"


@dataclass(frozen=True)
class DatasetEligibilityRules:
    require_completed_analysis: bool = True
    require_evidence: bool = True

    def allows(self, source: DatasetSourceRecord) -> bool:
        incident = source.incident
        if not all(
            self._present_string(value)
            for value in (
                incident.id,
                incident.project_id,
                incident.source,
                incident.signature,
            )
        ):
            return False

        if incident.sample_lines is not None and (
            not isinstance(incident.sample_lines, list)
            or not all(isinstance(line, str) for line in incident.sample_lines)
        ):
            return False

        if self.require_completed_analysis:
            analysis = source.analysis
            if not analysis or not all(
                self._present_string(value)
                for value in (
                    analysis.severity,
                    analysis.disposition,
                    analysis.summary,
                )
            ):
                return False

        if self.require_evidence:
            has_logs = any(line.strip() for line in (incident.sample_lines or []))
            has_investigation_evidence = bool(
                source.investigation
                and source.investigation.evidence_snapshot
                and source.investigation.evidence_snapshot.strip()
            )
            if not has_logs and not has_investigation_evidence:
                return False

        return True

    @staticmethod
    def _present_string(value) -> bool:
        return isinstance(value, str) and bool(value.strip())


class DatasetBuilder:
    """Transform eligible persisted incidents into deterministic examples."""

    def __init__(
        self,
        sources: DatasetSourceRepository,
        rules: DatasetEligibilityRules | None = None,
    ):
        self.sources = sources
        self.rules = rules or DatasetEligibilityRules()

    def build(self, project_id: str) -> list[dict[str, Any]]:
        return [
            self._to_example(source)
            for source in self.sources.list_for_project(project_id)
            if self.rules.allows(source)
        ]

    def _to_example(self, source: DatasetSourceRecord) -> dict[str, Any]:
        incident = source.incident
        analysis = source.analysis
        investigation = source.investigation
        action_log = source.action_log

        root_cause = None
        if incident.root_cause_incident_id or incident.cause_explanation:
            root_cause = {
                "incident_id": incident.root_cause_incident_id,
                "explanation": incident.cause_explanation,
            }

        recommended_actions = []
        if action_log and isinstance(action_log.requested_actions, list):
            recommended_actions = list(action_log.requested_actions)
        elif analysis and isinstance(analysis.next_steps, list):
            recommended_actions = list(analysis.next_steps)

        return {
            "input": {
                "evidence": {
                    "snapshot": (
                        investigation.evidence_snapshot if investigation else None
                    ),
                    "sample_count": (
                        investigation.evidence_samples if investigation else None
                    ),
                    "related_incident_count": (
                        investigation.evidence_related_count if investigation else None
                    ),
                    "runbook": (
                        investigation.evidence_runbook if investigation else None
                    ),
                },
                "incident": {
                    "id": incident.id,
                    "source": incident.source,
                    "environment": incident.environment,
                    "signature": incident.signature,
                    "first_seen": self._timestamp(incident.first_seen),
                    "last_seen": self._timestamp(incident.last_seen),
                    "count": incident.count,
                    "status": incident.status,
                    "auto_tags": incident.auto_tags,
                },
                "logs": list(incident.sample_lines or []),
                "metadata": {
                    "schema_version": DATASET_EXAMPLE_SCHEMA_VERSION,
                    "analysis": {
                        "id": analysis.id if analysis else None,
                        "source": analysis.analysis_source if analysis else None,
                        "confidence": analysis.confidence if analysis else None,
                        "matched_runbook_id": (
                            analysis.matched_runbook_id if analysis else None
                        ),
                        "runbook_match_score": (
                            analysis.runbook_match_score if analysis else None
                        ),
                    },
                    "investigation": self._investigation_metadata(investigation),
                    "action": self._action_metadata(action_log),
                },
            },
            "expected_output": {
                "severity": analysis.severity if analysis else None,
                "disposition": analysis.disposition if analysis else None,
                "root_cause": root_cause,
                "summary": analysis.summary if analysis else None,
                "recommended_actions": recommended_actions,
            },
        }

    def _investigation_metadata(self, investigation) -> dict[str, Any] | None:
        if not investigation:
            return None
        return {
            "id": investigation.id,
            "started_at": self._timestamp(investigation.started_at),
            "finished_at": self._timestamp(investigation.finished_at),
            "iterations": investigation.iterations,
            "fallback_used": investigation.fallback_used,
            "analysis_source": investigation.analysis_source,
            "effective_disposition": investigation.effective_disposition,
            "verifier_outcome": investigation.verifier_outcome,
        }

    def _action_metadata(self, action_log) -> dict[str, Any] | None:
        if not action_log:
            return None
        return {
            "id": action_log.id,
            "requested_actions": action_log.requested_actions,
            "allowed_actions": action_log.allowed_actions,
            "blocked_actions": action_log.blocked_actions,
            "actions_taken": action_log.actions_taken,
            "outcome": action_log.outcome,
            "policy_reason": action_log.policy_reason,
            "actioned_at": self._timestamp(action_log.actioned_at),
            "resolved_at": self._timestamp(action_log.resolved_at),
        }

    @staticmethod
    def _timestamp(value: datetime | None) -> str | None:
        return value.isoformat() if value else None
