# IncidentLens

**Turn noisy application logs into explainable incidents, policy-approved actions, and project-specific model improvements.**

IncidentLens is a policy-bound AIOps platform for incident triage. It reads application logs from Datadog, groups repeated failures into incidents, investigates them with [Qwen3.5 4B](https://huggingface.co/Qwen/Qwen3.5-4B), and applies deterministic safety rules before any action is executed. Reviewed incident data can be versioned and used to train a project-specific LoRA adapter.

![IncidentLens dashboard](imgs/dashboard.png)

![IncidentLens incident investigation](imgs/incident.png)

## The problem

Application log streams are repetitive, high-volume, and full of values that change from one event to the next. Raw alerts reveal that something failed, but they rarely explain whether several errors belong to the same incident, whether one failure caused another, or what response is safe.

An unconstrained model is not an execution boundary. It can produce a useful diagnosis while still understating severity, overreacting to a single event, or suggesting an unsafe remediation. IncidentLens keeps model reasoning and action authority separate.

## The solution

IncidentLens combines:

- Datadog log ingestion with cursor-based polling and pagination
- Message normalization and signature-based incident clustering
- Evidence-grounded, tool-assisted investigation with Qwen3.5 4B
- Structured severity, disposition, root-cause, summary, and ticket output
- A deterministic policy engine that derives, allows, or blocks actions
- A complete investigation and action audit trail in PostgreSQL
- Human-reviewed JSONL datasets and project-specific LoRA adapters
- Discord and email notification routing

The model explains the incident. The policy engine controls the response.

## Architecture

```text
React dashboard
      │
      │ JWT-authenticated API calls
      ▼
FastAPI modular monolith
      │
      ├── Control Plane
      │     Projects, settings, authentication, model lifecycle
      │
      ├── Data Plane
      │     Datadog polling, parsing, normalization, clustering, evidence
      │
      ├── Serving Plane
      │     Qwen3.5 4B, investigation tools, policy, actions, notifications
      │
      └── Training Plane
            JSONL datasets, record review, LoRA training, artifacts
      │
      ├── PostgreSQL ── incidents, analyses, audit records, lifecycle state
      └── Local storage ── versioned datasets and LoRA adapters
```

| Plane | Responsibility |
| --- | --- |
| **Control** | Project registration, authentication, settings, lifecycle operations, and maintenance |
| **Data** | Datadog ingestion, log parsing, signature normalization, clustering, and evidence construction |
| **Serving** | Local inference, tool-assisted investigation, root-cause reasoning, policy checks, actions, and notifications |
| **Training** | Dataset import/build, record approval, LoRA fine-tuning, artifact validation, and activation |

## User setup flow

IncidentLens is project-scoped. Credentials, incidents, datasets, adapters, and notifications belong to the authenticated project.

```text
Open IncidentLens
      │
      ▼
Register project
  • project name
  • password
      │
      ▼
Configure Datadog in Settings
  • API key
  • application key with logs_read_data
  • Datadog site
  • log query
  • service filter
      │
      ▼
Verify connection
  read-only credential and log-search check
      │
      ▼
Save project settings
      │
      ├──► Optional: add Discord webhooks and an email recipient
      │
      ▼
Open Training
      │
      ├──► Upload data/dataset_v1.jsonl to bootstrap the project
      │          or
      └──► Build a dataset later from analyzed incident history
                    │
                    ▼
            Review records and select examples
                    │
                    ▼
               Approve dataset
                    │
                    ▼
             Run LoRA training job
                    │
                    ▼
          Validate and register adapter
                    │
                    ▼
       READY adapter is activated for the project
                    │
                    ▼
Dashboard is ready for model-backed incident monitoring
```

The Datadog verification action does not save the submitted values. Saving settings is a separate step. The UI considers setup complete when the project has both a valid Datadog configuration and an active model artifact.

## IncidentLens application flow

### 1. Ingestion and incident formation

```text
Configured project
      │
      ▼ every polling interval
Query one contiguous Datadog time window
      │
      ├── paginate until every matching log is fetched
      └── advance the in-memory cursor only after success
      │
      ▼
Normalize each result into a provider-neutral envelope
      │
      ▼
Group by source and environment
      │
      ▼
Parse level, timestamp, message, and exception type
      │
      ├── INFO / DEBUG ──► ignore
      └── WARN / ERROR / CRITICAL
                        │
                        ▼
              Normalize volatile values
        UUIDs, IDs, hosts, durations, memory sizes
                        │
                        ▼
           Hash source + level + normalized message
                        │
                        ▼
               Look for an open matching incident
                    within the 2-minute cluster window
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
        Match found            No match
        increment count        create incident
        retain up to           analyze immediately
        10 samples
              │
              └── re-analyze when count reaches 5, 10, or 20
```

This reduces repeated log lines to one evolving incident while preserving representative evidence.

### 2. Evidence and model investigation

```text
New or threshold-triggered investigation
      │
      ▼
Build bounded evidence
  • up to 8 stored log samples
  • up to 5 related open incidents from the last 15 minutes
  • previously known root-cause link, when available
      │
      ▼
Resolve project model
  shared Qwen3.5 4B base + active project LoRA adapter
      │
      ▼
Tool-assisted investigation loop (up to 4 rounds)
      │
      ├── get_recent_logs
      ├── get_related_incidents
      └── get_incident_timeline
      │
      ▼
Parse and validate structured result
  • severity
  • disposition
  • confidence
  • summary and suspected root cause
  • next steps
  • ticket title and body
      │
      ├── invalid tool-loop output ──► single-shot model fallback
      └── valid output ──────────────► persist analysis
      │
      ▼
Compare a new incident with earlier incidents
and store a causal link when the relationship is plausible
```

Critical patterns and severity/disposition consistency are checked after generation. Model or adapter failures do not fall through to a remote provider or an unadapted base model.

### 3. Policy, actions, and audit trail

The model does not directly choose an executable operation. The policy layer maps the validated disposition to default actions, applies safety constraints, and records both allowed and blocked results.

```text
Validated analysis
      │
      ▼
Derive requested actions from disposition
      │
      ▼
Policy checks
  • incident is still open
  • model confidence is at least 0.55
  • no action was taken during the 20-minute cooldown
  • low-count non-critical escalations are downgraded
  • disposition meets the minimum floor for its severity
      │
      ▼
Evaluate every action
      │
      ├── safe ─────────► allow
      ├── conditional ──► check severity, confidence, and disposition
      └── dangerous or unknown ──► block
      │
      ▼
Execute only allowed actions
      │
      ├── enrich the incident
      ├── suppress qualifying low-severity noise
      └── route Discord or email notifications
      │
      ▼
Persist InvestigationRun + ActionLog
  evidence, tool calls, result, policy reason,
  allowed actions, blocked actions, and actions taken
```

Safe actions include enrichment, incident summaries, evidence attachment, notifications, and verification checks. Infrastructure changes, restarts, database changes, secret rotation, deletion, and scaling are blocked. Unknown action names are also blocked.

Automatic suppression is allowed only for low-severity `NO_ACTION` decisions with confidence of at least `0.80`.

<table>
  <tr>
    <td><img src="imgs/discord.png" alt="IncidentLens Discord notification"></td>
    <td><img src="imgs/email.png" alt="IncidentLens email notification"></td>
  </tr>
</table>

## Training dataset

[`data/dataset_v1.jsonl`](data/dataset_v1.jsonl) is the current seed dataset for project adapter training. It contains 1,984 supervised incident examples used to teach the project adapter.

| Dataset property | Value |
| --- | ---: |
| Records | 1,984 |
| Incident types | 30 |
| Services | 35 |
| Regions | 26 |
| Log lines per record | 3–6, median 5 |
| Records with related-incident context | 311 |

### Label distribution

| Field | Value | Records |
| --- | --- | ---: |
| Severity | High | 1,195 |
| Severity | Medium | 462 |
| Severity | Critical | 327 |
| Disposition | `NEEDS_ONCALL` | 890 |
| Disposition | `ESCALATE` | 552 |
| Disposition | `NEEDS_DEV` | 542 |

The dataset focuses on actionable incidents. Its 30 failure categories include database and cache exhaustion, message-queue backlog, payment timeouts, memory leaks, certificate failures, pod failures, search degradation, configuration errors, and model or adapter loading failures.

Each line follows this shape:

```text
input
├── logs[]
├── service
├── environment
├── count
├── related_incidents[]
└── metadata
    ├── region
    ├── host
    ├── trace_id
    └── request_id

output
├── incident_type
├── severity
├── disposition
├── summary
└── recommended_actions[]
```

On upload, the backend validates every record and normalizes the dataset's `output` object to the internal `expected_output` schema. The dataset remains pending review until a user selects at least one record and approves it.

## Model lifecycle

```text
Uploaded seed data or analyzed incident history
      │
      ▼
Immutable versioned JSONL dataset
      │
      ▼
Record review and selection
      │
      ▼
QUEUED ──► RUNNING ──► EVALUATING ──► PASSED
                 │             │
                 └─────────────┴────► FAILED
      │
      ▼
Versioned LoRA adapter + tokenizer metadata + file manifest
      │
      ▼
READY artifact
      │
      ▼ automatic activation after a successful run
Active project adapter
```

Training applies the Qwen chat template and masks prompt tokens so loss is computed on the expected incident response. The default LoRA profile covers Qwen3.5's full-attention, linear-attention, and MLP projections with rank `8`, alpha `16`, dropout `0.05`, and a maximum sequence length of `1024`.

Datasets and adapters are versioned rather than overwritten. A successful run activates its new artifact; the Models page can later switch to any READY artifact owned by the project. A failed training or validation step leaves the previously active artifact unchanged. The shared base model is loaded once, adapters are loaded lazily, and adapter selection and generation share a lock to prevent cross-project model use.

## Tech stack

| Layer | Technology |
| --- | --- |
| Frontend | React 19, React Router, Tailwind CSS, Recharts, Axios |
| API | FastAPI, Pydantic, Uvicorn |
| Persistence | PostgreSQL 16, SQLAlchemy |
| Log source | Datadog Logs API |
| Model | [Qwen3.5 4B](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Model runtime | PyTorch, Transformers, PEFT |
| Training | LoRA, Hugging Face Datasets, Safetensors |
| Observability | Datadog APM, Logs, LLM Observability, and metrics |
| Notifications | Discord webhooks and SMTP |

## Local development

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker with Docker Compose
- Datadog API and application keys; the application key needs `logs_read_data`
- Network access to download the configured base model

CPU execution is supported, but loading and training a 4B model is resource-intensive. A compatible accelerator is recommended.

### 1. Configure the project

```bash
git clone https://github.com/DevelopedBy-Siva/incident-lens.git
cd incident-lens
cp .env.example .env
```

The defaults in `.env.example` connect the backend to the bundled PostgreSQL service. Replace `SECRET_KEY` and add Datadog observability settings if you want to trace IncidentLens itself. Credentials used to read application logs are entered per project in the UI and stored in PostgreSQL.

### 2. Start PostgreSQL

```bash
docker compose up -d postgres
docker compose ps
```

### 3. Start the backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r log-analyzer/requirements.txt

ddtrace-run python -m uvicorn app.main:app \
  --app-dir log-analyzer \
  --host 0.0.0.0 \
  --port 8000
```

The backend loads the configured shared base model during startup and fails fast if the model or database cannot be initialized.

### 4. Start the frontend

In a second terminal:

```bash
cd log-analyzer-frontend
npm ci
npm start
```

Open [http://localhost:3000](http://localhost:3000), register a project, configure Datadog under **Settings**, then upload and activate a model from **Training**. The backend API is available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 5. Stream sample logs (optional)

The local simulator sends the included sample stream to Datadog so the complete ingestion path can be exercised.

```bash
python3 -m venv .venv-log-server
source .venv-log-server/bin/activate
pip install -r log-server/requirements.txt
python -m uvicorn server:app --app-dir log-server --port 5001
```

With `LOG_SERVER_URL=http://localhost:5001`, start and stop the stream from the dashboard.

## Configuration

The complete template is in [`.env.example`](.env.example). The primary settings are:

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | PostgreSQL connection used for application and audit state |
| `SECRET_KEY` | JWT signing key |
| `CORS_ORIGINS` | Allowed dashboard origins |
| `BASE_MODEL` | Shared model identifier; defaults to `Qwen/Qwen3.5-4B` |
| `DEVICE` | PyTorch runtime device |
| `DTYPE` | Model weight data type |
| `DATASET_STORAGE_PATH` | Dataset storage root |
| `ARTIFACT_STORAGE_PATH` | LoRA adapter storage root |
| `POLL_INTERVAL` | Datadog polling interval in seconds |
| `DATADOG_LOOKBACK_SECONDS` | Initial log-search window in seconds |
| `LOG_SERVER_URL` | Optional local simulator URL |

Datadog application-observability variables (`DD_*`) are separate from the per-project credentials used to search logs.

## Testing

```bash
# Backend
cd log-analyzer
python -m pytest -q

# Frontend build
cd ../log-analyzer-frontend
npm run build
```

## Repository layout

```text
incident-lens/
├── data/
│   └── dataset_v1.jsonl    Seed incident-training dataset
├── log-analyzer/           FastAPI backend and Python test suite
│   └── app/
│       ├── control/        Project setup and lifecycle management
│       ├── data/           Ingestion and incident formation
│       ├── serving/        Investigation, policy, and actions
│       └── training/       Datasets, LoRA training, and artifacts
├── log-analyzer-frontend/  React dashboard
├── log-server/             Optional Datadog log simulator
├── imgs/                   Product screenshots
└── compose.yaml            Local PostgreSQL
```

## Known limitations

- Datadog is currently the only implemented log-source connector.
- The polling cursor is kept in process memory and resets when the backend restarts.
- Model-backed investigation requires a READY active adapter for the project.
- There is no remote-model or base-model-only inference fallback.
- Training runs synchronously and is resource-intensive on CPU.
- Polling and verification workers run inside the API process, so the current architecture assumes one backend instance.

## What I learned

- Safe automation needs an enforcement boundary outside the model.
- Incident quality depends as much on normalization and evidence selection as it does on model capability.
- Human-reviewed operational history can form a reproducible training loop when datasets and artifacts are immutable.
- A model lifecycle is trustworthy only when failures cannot replace the last known-good artifact.
