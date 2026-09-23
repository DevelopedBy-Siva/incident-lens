# Log Analyzer

The IncidentLens backend is a modular monolith organized into Control, Data,
Serving, and Training planes. Datadog is the single observability platform,
providing Logs, APM, LLM Observability, and metrics.

## Log ingestion

The Data Plane exposes a provider-neutral `LogSourceConnector` contract. The
Datadog connector calls the v2 Logs Search API, follows response cursors,
normalizes results into internal log envelopes, and passes the original message
text to the existing parser and incident pipeline. Parser, signature,
clustering, evidence, serving, policy, notification, and training behavior do
not depend on Datadog response objects.

Configure each project's Datadog connection through the Settings page. The API
key, application key, site, query, and service are stored on that project; the
worker has no deployment-level credential fallback.

```env
DATADOG_LOOKBACK_SECONDS=30
POLL_INTERVAL=30

DD_API_KEY=your_rotated_api_key
DD_SITE=datadoghq.com
DD_LLMOBS_ENABLED=1
DD_LLMOBS_AGENTLESS_ENABLED=1
DD_LLMOBS_ML_APP=incident-lens
DD_TRACE_ENABLED=1
```

The application key must have `logs_read_data` permission. Query and service
are required, and the analyzer always adds `env:prod`. The Verify Datadog
Configuration button tests credentials and log-search access without saving;
the normal Save Settings button persists them. A polling cursor advances after
a successful query and remains unchanged after a request or processing
failure.

On startup, migration `0006_datadog_log_source` adds the Datadog project
settings and removes the retired provider columns. No incident or model
lifecycle schema is changed.

Authenticated projects can clear their incident-processing records with
`DELETE /api/incidents`. The project account, Datadog configuration, datasets,
and model artifacts are preserved.

The Settings danger zone exposes this incident reset and permanent project
deletion. Deleting a project also removes its datasets, training jobs, model
artifacts, and project-scoped objects from the configured storage backend.

## Local PostgreSQL

From the repository root, copy `.env.example` to `.env` and run:

```bash
docker compose up -d postgres
```

The local backend then uses
`postgresql://incidentlens:incidentlens@localhost:5432/incidentlens`. Database
state is persisted in the `incidentlens-postgres-data` Docker volume.

## Datadog LLM Observability

The backend installs `ddtrace` and its container entrypoint runs Uvicorn with
`ddtrace-run`. Local processes should also use `ddtrace-run`. In agentless mode,
set `DD_LLMOBS_AGENTLESS_ENABLED=1` in addition to `DD_SITE`, `DD_API_KEY`,
`DD_LLMOBS_ENABLED`, and `DD_LLMOBS_ML_APP` before the process starts.

Manual instrumentation covers log ingestion, normalization, parsing,
clustering, evidence generation, runtime and adapter resolution, local
inference, validation, policy and action handling, dataset creation, training,
evaluation, artifact lifecycle, and Control Plane mutations. Metadata is
allowlisted so credentials, prompts, and raw model output are not emitted.
