# Log Server

A small FastAPI service that generates realistic application logs and sends
them to Datadog Logs.

## What it does

- Generates info, warning, and error logs in memory
- Simulates production issues such as database timeouts, OOM failures, and
  authentication cascades
- Sends logs directly to the Datadog HTTP intake API
- Exposes controllable scenarios for demos and pipeline testing
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

## Environment variables

```env
DATADOG_API_KEY=your_api_key
DATADOG_SITE=datadoghq.com
DATADOG_ENVIRONMENT=prod
LOG_SERVICE_NAME=log-server
CORS_ORIGINS=http://localhost:3000
```

The simulator only writes logs, so it requires a Datadog API key but not an
application key. The analyzer separately requires an application key with
`logs_read_data` permission to search those events.

## Run locally

```bash
uvicorn server:app --host 0.0.0.0 --port 5001 --reload
```

- API: [http://localhost:5001](http://localhost:5001)
- Docs: [http://localhost:5001/docs](http://localhost:5001/docs)

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Send a Datadog connectivity event |
| GET | `/ready` | Readiness/status probe |
| GET | `/db-health` | Synthetic DB health endpoint |
| POST | `/api/start` | Start log generation |
| POST | `/api/stop` | Stop generation |
| GET | `/api/status` | Current generator and transport stats |
| POST | `/api/scenario/{name}` | Run a correlated scenario |
| GET | `/api/scenario` | List scenarios |
| POST | `/api/recover` | Stop generation and emit a recovery event |

Start the generator:

```bash
curl -X POST "http://localhost:5001/api/start?duration=60"
```

Run a scenario:

```bash
curl -X POST http://localhost:5001/api/scenario/db_pool_exhaustion
```

## View logs

In Datadog Log Explorer, query:

```text
service:log-server
```

Useful refinements:

```text
service:log-server status:error
scenario:db_pool_exhaustion
```

## Scenarios

- `healthcheck_timeout_noise`
- `db_pool_exhaustion`
- `payment_gateway_degraded`
- `api_gateway_5xx_spike`
- `memory_pressure_or_oom`
- `auth_failure_cascade`
- `deployment_regression`
- `queue_backlog`
- `vendor_api_timeout`
- `false_suppression_trap`
- `low_frequency_high_impact`
- `ambiguous_cascade`

Legacy scenario aliases remain available for compatibility. Each scenario
emits parser-compatible plain text with structured key-value fields:

```text
[timestamp] LEVEL: timestamp=... level=... service=... source=... environment=prod message="..." request_id=req_... trace_id=trace_... endpoint=... operation=... status_code=... latency_ms=... error_type=... host=... pod=...
```

Scenario metadata includes the expected runbook, severity, disposition,
allowed actions, and blocked actions so the evaluation tests can reuse the
registry.
