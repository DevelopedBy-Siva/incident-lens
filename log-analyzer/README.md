# Log Analyzer

The IncidentLens backend is a modular monolith organized into Control, Data,
Serving, and Training planes. Datadog Logs is the only external log provider.

## Log ingestion

The Data Plane exposes a provider-neutral `LogSourceConnector` contract. The
Datadog connector calls the v2 Logs Search API, follows response cursors,
normalizes results into internal log envelopes, and passes the original message
text to the existing parser and incident pipeline. Parser, signature,
clustering, evidence, serving, policy, notification, and training behavior do
not depend on Datadog response objects.

Configure ingestion globally with deployment environment variables or per
project through the Settings page:

```env
DATADOG_API_KEY=your_api_key
DATADOG_APP_KEY=your_application_key
DATADOG_SITE=datadoghq.com
DATADOG_QUERY=status:(error OR warn OR critical)
DATADOG_LOOKBACK_SECONDS=30
DATADOG_ENVIRONMENT=prod
DATADOG_SERVICE=
POLL_INTERVAL=30
```

The application key must have `logs_read_data` permission. A project-level API
key, application key, query, environment, and optional service filter are used
only for that project's polling cycle. A polling cursor advances after a
successful query and remains unchanged after a request or processing failure.

On startup, migration `0006_datadog_log_source` adds the Datadog project
settings and removes the retired provider columns. No incident or model
lifecycle schema is changed.
