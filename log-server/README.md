# Log Server

A small FastAPI service that generates realistic application logs and sends
them to Datadog Logs.

## What it does

- Generates info, warning, and error logs in memory
- Simulates production issues such as database timeouts, OOM failures, and
  authentication cascades
- Sends logs directly to the Datadog HTTP intake API
- Preserves the structured text format consumed by the IncidentLens parser

## Data flow

```text
Log generator
      |
      v
Datadog HTTP intake
      |
      v
Datadog Logs
      |
      v
IncidentLens Log Source Connector
```

The simulator has no environment configuration. The start request supplies its
write-only Datadog API key, site, and service through request headers. The
analyzer proxy fills these headers from the authenticated project's database
configuration. Logs are always tagged as `env:prod`.

## Run locally

```bash
uvicorn server:app --host 0.0.0.0 --port 5001 --reload
```

- API: [http://localhost:5001](http://localhost:5001)
- Docs: [http://localhost:5001/docs](http://localhost:5001/docs)

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/api/start` | Start log generation |
| POST | `/api/stop` | Stop generation |

Start the generator:

```bash
curl -X POST "http://localhost:5001/api/start?duration=60" \
  -H "X-Datadog-API-Key: your-write-api-key" \
  -H "X-Datadog-Site: datadoghq.com" \
  -H "X-Datadog-Service: project-1-api"
```

## View logs

In Datadog Log Explorer, query:

```text
service:project-1-api
```

Useful refinements:

```text
service:project-1-api status:error
```
