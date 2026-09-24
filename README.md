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

## How it fits together

```text
Datadog Logs ──► IncidentLens API ──► PostgreSQL
                       │
                       ├──► Qwen3.5 4B + project LoRA adapter
                       ├──► S3 datasets and adapters
                       └──► Discord and email

React dashboard ◄─────► IncidentLens API
```

FastAPI owns project setup, ingestion, investigation, policy enforcement, training, and audit history. PostgreSQL stores application state. S3 is the durable store for versioned JSONL datasets and complete LoRA adapter directories.

## User setup flow

Each project has its own credentials, incidents, datasets, adapter, and notification settings.

```text
Register project
      ▼
Configure and verify Datadog
      ▼
Upload dataset → review records → train adapter
      ▼
Adapter becomes active
      ▼
Monitor incidents in the dashboard
```

Datadog setup requires an API key, an application key with `logs_read_data`, a site, a query, and a service filter. Verification performs a read-only check; the user must still save the settings. Discord webhooks and an email recipient are optional.

To bootstrap model training, upload [`data/dataset_v1.jsonl`](data/dataset_v1.jsonl), review the records, approve the selection, and run training. The resulting READY adapter is activated automatically. Setup is complete when both Datadog and an active adapter are configured.

## IncidentLens application flow

```text
Datadog logs
      ▼
Normalize and cluster repeated errors
      ▼
Build incident evidence
      ▼
Qwen3.5 4B + project adapter investigates
      ▼
Validate severity, disposition, root cause, and next steps
      ▼
Policy allows safe actions and blocks unsafe actions
      ▼
Save the audit trail and send configured notifications
```

The polling worker queries contiguous Datadog windows and processes warning, error, and critical events. Volatile values are removed before signatures are generated, so repeated messages join the same open incident within a two-minute window. New incidents are investigated immediately; growing incidents are reconsidered at counts `5`, `10`, and `20`.

The evidence bundle contains representative logs, recent related incidents, and any known causal link. The model can request recent logs, related incidents, or an incident timeline before returning structured analysis.

The model cannot execute operations directly. Policy checks confidence, severity, incident state, cooldowns, and action type. Enrichment, qualifying suppression, and configured notifications can run; infrastructure changes, restarts, database changes, secret rotation, deletion, scaling, and unknown actions are blocked. Evidence, tool calls, analysis, policy reasons, and action outcomes are saved for inspection.

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

1. A user uploads seed data or builds a dataset from analyzed incidents.
2. The user reviews the records and approves the examples to use.
3. Training creates a project-specific LoRA adapter for Qwen3.5 4B.
4. Integrity checks verify the adapter files and metadata.
5. A successful adapter is uploaded to S3, registered as READY, and activated.

### Training execution modes

IncidentLens supports two training execution modes:

#### Local/Synchronous (Development Mode)

**Default behavior:** Training runs on the application server, blocking until completion.

- **When:** `TRAINING_EC2_ENABLED=false` (default)
- **Resources:** GPU/CPU on the same EC2 instance as the application
- **Duration:** 5–15 minutes depending on dataset size and hardware
- **Use case:** Local development, small datasets, testing

Training applies the Qwen chat template and masks prompt tokens so loss is computed on the expected incident response. The default LoRA profile covers Qwen3.5's full-attention, linear-attention, and MLP projections with rank `8`, alpha `16`, dropout `0.05`, and a maximum sequence length of `1024`.

#### EC2 Remote/Asynchronous (Production Mode)

**Scalable alternative:** Training runs on a temporary GPU-enabled EC2 instance (g6.xlarge), returning immediately with status `RUNNING`.

- **When:** `TRAINING_EC2_ENABLED=true`
- **Resources:** Temporary g6.xlarge GPU instance launched on-demand
- **Duration:** Instance boots (~2 min), trains (~5–15 min), terminates (~1 min)
- **Cost:** Only GPU usage during training, no idle instance costs
- **Use case:** Production deployments, large datasets, frequent retraining
- **Response:** API returns `202 Accepted` with running job; training completes asynchronously
- **Monitoring:** Poll `GET /training-jobs/{job_id}` to track progress and completion
- **Status updates:** Bootstrap script updates job status via direct database connection

**Architecture:**
1. Application receives training request, validates dataset, creates configuration
2. Determines current Git commit SHA (or uses configured SHA from environment)
3. Launches temporary g6.xlarge EC2 instance with training configuration in User Data
4. Returns immediately with job status `RUNNING` and instance ID (202 Accepted)
5. Temporary instance boots and runs `/opt/incident-lens/bootstrap-training.py` from AMI
6. Bootstrap script clones repository from Git and checks out the specified commit SHA
7. Bootstrap activates pre-built training environment and configures Python path
8. Bootstrap downloads dataset from S3, runs QLoRA fine-tuning, uploads artifacts
9. Training job status updated to `PASSED` or `FAILED` in database
10. Instance terminates automatically after completion

**Key Design:**
- Training AMI contains stable GPU drivers, Python environment (`/home/ubuntu/training-env`), and bootstrap script
- AMI does **not** contain application source code
- Application code is cloned fresh at runtime and pinned to a Git commit SHA
- This ensures training uses code matching the application version that launched the job
- AMI can be updated for GPU/Python dependencies without rebuilding for code changes

Datasets and adapters are versioned rather than overwritten. A successful run activates its new artifact; the Models page can later switch to any READY artifact owned by the project. A failed training or validation step leaves the previously active artifact unchanged.

Training needs a local working directory, and inference needs local model files. Those directories are caches, not the durable store: completed adapter files are uploaded to S3 and downloaded again when a cache is empty. Datasets are written directly to S3. If `S3_BUCKET` is not configured, the filesystem implementations remain available for local development and tests.

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
- A Hugging Face token with access to `Qwen/Qwen3.5-4B`
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

Open [http://localhost:3000](http://localhost:3000), register a project, configure Datadog under **Settings**, then upload a dataset and train an adapter under **Training**. The backend API is available at [http://localhost:8000/docs](http://localhost:8000/docs).

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
| `HF_TOKEN` | Hugging Face token used to download the base model |
| `DEVICE` | PyTorch runtime device |
| `DTYPE` | Model weight data type |
| `AWS_REGION` | Region containing the S3 bucket |
| `S3_BUCKET` | Private bucket for datasets and adapters; setting it enables S3 storage |
| `S3_DATASET_PREFIX` | Dataset object prefix; defaults to `datasets` |
| `S3_ARTIFACT_PREFIX` | Adapter object prefix; defaults to `artifacts` |
| `DATASET_STORAGE_PATH` | Filesystem fallback used only when `S3_BUCKET` is empty |
| `ARTIFACT_STORAGE_PATH` | Local training/inference cache and filesystem fallback |
| `POLL_INTERVAL` | Datadog polling interval in seconds |
| `DATADOG_LOOKBACK_SECONDS` | Initial log-search window in seconds |
| `LOG_SERVER_URL` | Optional local simulator URL |

### EC2 GPU Training Configuration (Production)

To enable asynchronous training on temporary GPU instances, configure these settings:

**GitHub Secrets (deployment-specific values):**

| Variable | Purpose |
| --- | --- |
| `TRAINING_EC2_ENABLED` | Enable EC2 remote training; set to `true` (defaults to `false`) |
| `TRAINING_EC2_AMI_ID` | **Required.** Pre-built training AMI with PyTorch, Transformers, PEFT installed (e.g., `ami-0c54a4fe99a96cfc3`) |
| `TRAINING_EC2_IAM_INSTANCE_PROFILE` | **Required.** IAM instance profile name with S3 and PostgreSQL access |

**Application defaults (non-sensitive configuration):**

These values use sensible defaults and do not need to be stored as GitHub Secrets:

| Variable | Purpose | Default |
| --- | --- | --- |
| `TRAINING_EC2_INSTANCE_TYPE` | GPU instance type | `g6.xlarge` (85 GB GPU memory) |
| `TRAINING_EC2_MAX_WAIT_SECONDS` | Max time to wait for training completion | `3600` (1 hour) |
| `TRAINING_EC2_DETAILED_MONITORING` | Enable CloudWatch detailed monitoring | `false` |
| `TRAINING_EC2_TERMINATE_ON_COMPLETION` | Auto-terminate instance after completion | `true` |
| `TRAINING_EC2_ASSOCIATE_PUBLIC_IP` | Assign public IP to training instances | `false` |
| `TRAINING_EC2_SUBNET_ID` | VPC subnet for instance (optional) | Uses default VPC |
| `TRAINING_EC2_SECURITY_GROUP_IDS` | Comma-separated security group IDs (optional) | Uses default security group |

The application must have AWS credentials (via IAM role) to launch EC2 instances. No AWS access keys are required as GitHub Secrets.

Datadog application-observability variables (`DD_*`) are separate from the per-project credentials used to search logs.

With S3 enabled, datasets use keys such as `datasets/projects/<project-id>/dataset-v1.jsonl`. Adapter directories are uploaded under `artifacts/<project-id>/adapter-vN/`, including weights, adapter configuration, tokenizer files, metadata, and the integrity manifest.

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
- Local training runs synchronously and is resource-intensive on CPU; use EC2 mode for production.
- EC2 training requires a pre-built AMI and IAM instance profile configuration.
- Polling and verification workers run inside the API process, so the current architecture assumes one backend instance.

## What I learned

- Safe automation needs an enforcement boundary outside the model.
- Incident quality depends as much on normalization and evidence selection as it does on model capability.
- Human-reviewed operational history can form a reproducible training loop when datasets and artifacts are immutable.
- A model lifecycle is trustworthy only when failures cannot replace the last known-good artifact.
