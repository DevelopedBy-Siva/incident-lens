# Log Server

A small FastAPI service that streams realistic application logs from
`data/data.log` to Datadog Logs.

## What it does

- Reads one log line at a time from `data/data.log`
- Preserves the mix of info, warning, and error logs from the file
- Sends logs directly to the Datadog HTTP intake API
- Preserves the structured text format consumed by the IncidentLens parser

## Data flow

```text
data/data.log
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

The streamer has no environment configuration. The start request supplies its
write-only Datadog API key and service through request headers. The optional
site header defaults to `datadoghq.com`. The
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
| POST | `/api/start` | Start streaming `data/data.log` |
| POST | `/api/stop` | Stop generation |

Start the file streamer:

```bash
curl -X POST "http://localhost:5001/api/start?duration=60&interval_seconds=0.25" \
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
