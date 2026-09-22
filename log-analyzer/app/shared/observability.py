"""Datadog-only tracing helpers for IncidentLens lifecycle operations."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

try:
    from ddtrace import tracer as _datadog_tracer
except ImportError:  # Keep local unit tests importable before dependencies install.
    _datadog_tracer = None

try:
    from ddtrace.llmobs import LLMObs as _DatadogLLMObs
except ImportError:  # APM remains available if an older tracer is present.
    _DatadogLLMObs = None


SERVICE_NAME = os.getenv("DD_SERVICE", "incident-lens")
ML_APP = os.getenv("DD_LLMOBS_ML_APP", "incident-lens")

_SAFE_METADATA_FIELDS = {
    "action_count",
    "adapter_version",
    "artifact_id",
    "artifact_version",
    "base_model",
    "candidate_count",
    "dataset_id",
    "dataset_version",
    "decision_source",
    "device",
    "environment",
    "error_type",
    "incident_id",
    "input_tokens",
    "log_count",
    "model_provider",
    "output_tokens",
    "policy_result",
    "project_id",
    "record_count",
    "result",
    "runtime_type",
    "source",
    "status",
    "training_engine",
    "training_job_id",
}


class _NoOpSpan:
    def set_tag(self, *_args, **_kwargs):
        return self

    def set_metric(self, *_args, **_kwargs):
        return self


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def safe_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return low-cardinality metadata that cannot contain credentials."""
    cleaned: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if key not in _SAFE_METADATA_FIELDS or value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            cleaned[key] = value
    return cleaned


@dataclass
class DatadogSpan:
    """Small adapter shared by APM and LLM Observability spans."""

    apm_span: Any = None
    llm_span: Any = None

    def tag(self, key: str, value: Any) -> None:
        values = safe_metadata({key: value})
        if key not in values:
            return
        if self.apm_span is not None:
            self.apm_span.set_tag(key, values[key])
        if self.llm_span is not None and _DatadogLLMObs is not None:
            _DatadogLLMObs.annotate(span=self.llm_span, tags={key: values[key]})

    def tags(self, metadata: Mapping[str, Any] | None) -> None:
        values = safe_metadata(metadata)
        for key, value in values.items():
            if self.apm_span is not None:
                self.apm_span.set_tag(key, value)
        if values and self.llm_span is not None and _DatadogLLMObs is not None:
            _DatadogLLMObs.annotate(span=self.llm_span, tags=values)

    def metrics(self, metrics: Mapping[str, int | float | None]) -> None:
        values = {
            key: value
            for key, value in metrics.items()
            if value is not None and isinstance(value, (int, float))
        }
        for key, value in values.items():
            if self.apm_span is not None:
                self.apm_span.set_metric(key, value)
        if values and self.llm_span is not None and _DatadogLLMObs is not None:
            _DatadogLLMObs.annotate(span=self.llm_span, metrics=values)


@contextmanager
def trace_operation(
    name: str,
    *,
    plane: str,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[DatadogSpan]:
    """Create a Datadog APM span for a lifecycle operation."""
    operation_name = f"incident_lens.{plane}.{name}"
    if _datadog_tracer is None:
        yield DatadogSpan(apm_span=_NoOpSpan())
        return

    with _datadog_tracer.trace(
        operation_name,
        service=SERVICE_NAME,
        resource=name,
    ) as span:
        wrapped = DatadogSpan(apm_span=span)
        wrapped.tags(metadata)
        yield wrapped


@contextmanager
def trace_llm_operation(
    name: str,
    *,
    model_name: str,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[DatadogSpan]:
    """Trace a local invocation in APM and Datadog LLM Observability."""
    with trace_operation(name, plane="serving", metadata=metadata) as apm:
        llmobs_enabled = (
            _DatadogLLMObs is not None
            and _enabled(os.getenv("DD_LLMOBS_ENABLED"))
            and bool(getattr(_DatadogLLMObs, "enabled", False))
        )
        if not llmobs_enabled:
            yield apm
            return

        with _DatadogLLMObs.llm(
            model_name=model_name,
            name=name,
            model_provider="local",
            ml_app=ML_APP,
        ) as llm_span:
            wrapped = DatadogSpan(apm_span=apm.apm_span, llm_span=llm_span)
            wrapped.tags(metadata)
            yield wrapped
