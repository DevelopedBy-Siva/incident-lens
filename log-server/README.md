# Log Server

A simple FastAPI service that generates realistic application logs and sends them directly to Grafana Loki.

---

## What it does

- Generates logs (info, warnings, errors) in memory
- Simulates real production issues (DB timeouts, OOM, auth failures, etc.)
- Ships logs directly to **Grafana Loki** (no Redis)
- Lets you trigger **scenarios** for demo/debugging
- Exposes a small API to control everything

---

## How it works

1. Logs are generated in memory
2. Buffered briefly
3. Sent to Loki using HTTP (`/loki/api/v1/push`)
4. Viewed in Grafana (Explore or dashboards)

---

## Environment variables

```env
LOGSHIPPER_API_KEY=your_api_key

LOKI_URL=https://logs-prod-XXX.grafana.net
LOKI_USERNAME=your_numeric_id
LOKI_API_KEY=your_loki_token

LOG_SERVICE_NAME=log-server
CORS_ORIGINS=http://localhost:3000
```

---

## Run locally

```bash
uvicorn server:app --host 0.0.0.0 --port 5001 --reload
```

- API: [http://localhost:5001](http://localhost:5001)
- Docs: [http://localhost:5001/docs](http://localhost:5001/docs)

---

## API

| Method | Endpoint               | Description           |
| ------ | ---------------------- | --------------------- |
| GET    | `/health`              | Check Loki connection |
| GET    | `/ready`               | Readiness/status probe |
| GET    | `/db-health`           | Synthetic DB health endpoint |
| POST   | `/api/start`           | Start log generation  |
| POST   | `/api/stop`            | Stop generation       |
| GET    | `/api/status`          | Current stats         |
| POST   | `/api/scenario/{name}` | Run a scenario        |
| GET    | `/api/scenario`        | List scenarios        |
| POST   | `/api/recover`         | Stop generation and emit recovery log |

All `/api/*` endpoints require:

```
X-Api-Key: dev
```

---

## Quick test

Start generator:

```bash
curl -X POST "http://localhost:5001/api/start?duration=60" \
  -H "X-Api-Key: dev"
```

Or run a scenario:

```bash
curl -X POST http://localhost:5001/api/scenario/db_pool_exhaustion \
  -H "X-Api-Key: dev"
```

---

## View logs (Grafana)

Go to **Explore → Loki** and run:

```logql
{service="log-server"}
```

Useful filters:

```logql
{service="log-server"} |= "ERROR"
{scenario="db_pool_exhaustion"}
```

---

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

Legacy aliases still work:

- `db_cascade` -> `ambiguous_cascade`
- `auth_cascade` -> `auth_failure_cascade`
- `deployment_gone_wrong` -> `deployment_regression`
- `memory_leak` -> `memory_pressure_or_oom`

Each scenario emits a sequence of related production-like logs over time. Scenario
logs keep the parser-compatible prefix:

```text
[timestamp] LEVEL: timestamp=... level=... service=... source=... environment=prod message="..." request_id=req_... trace_id=trace_... endpoint=... operation=... status_code=... latency_ms=... error_type=... host=... pod=...
```

Scenario metadata includes expected runbook, severity, disposition, allowed
actions, and blocked actions so evaluation tests can reuse the registry.

---

## Notes

- Logs are plain text with structured key-value fields for parser compatibility
- Loki is the only transport (Redis removed)
- Most logs are INFO by default (~70%)

---

## Use case

- Demoing observability setups
- Testing log pipelines
- Simulating production incidents

---

That’s it — start the server, generate logs, and watch them in Grafana.
