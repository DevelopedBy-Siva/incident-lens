# IncidentLens

A policy-bound AIOps agent that turns noisy application logs into auditable incidents.

IncidentLens queries application logs from Datadog, clusters them into incidents,
routes known failures through YAML runbooks, uses LLM-assisted investigation for
ambiguous incidents, and gates every requested action through a backend policy
engine.

---

## What It Does

- Normalizes noisy log lines into stable signatures
- Clusters repeated logs into incidents using time windows
- Builds bounded evidence bundles for each incident
- Matches known failures against deterministic runbooks
- Uses an LLM for ambiguous incidents, cascade reasoning, and weak runbook tie-breaking
- Allows low-risk actions such as enrichment, notifications, verification, and high-confidence suppression
- Blocks high-impact actions such as restarts, deployments, infrastructure changes, database mutations, secret rotation, deletion, and cluster scaling
- Stores incident state, policy decisions, and action outcomes in PostgreSQL
- Sends Discord/email notifications
- Tracks LLM calls and tool-loop behavior with Langfuse

---

## Architecture

```text
Log Server / App Logs
        |
        v
Datadog Logs
        |
        v
Log Source Connector
        |
        v
Parser + Signature Normalization
        |
        v
Incident Clustering
        |
        v
Evidence Bundle
        |
        v
Runbook Match
   |             |
   | strong      | weak / ambiguous
   v             v
Runbook Path   LLM Investigation
   |             |
   +------> Policy Engine
                  |
                  v
        Allowed + Blocked Actions
                  |
                  v
        ActionLog + Notifications
```

Deployment:

```text
React Dashboard        -> Vercel
FastAPI Backend        -> Render
Log Simulator          -> Render
Incident State         -> Neon PostgreSQL
Raw Logs               -> Datadog Logs
Model Inference        -> Local Qwen + project LoRA adapter
LLM Tracing            -> Langfuse
Notifications          -> Discord / SMTP
```

---

## Policy-Gated Automation

IncidentLens separates reasoning from execution.

Runbooks and LLM analysis may propose actions, but every action is checked by the policy engine before it can run.

Allowed actions include:

- `auto_enrich`
- `create_incident_summary`
- `attach_evidence_bundle`
- `notify_oncall`
- `send_discord_notification`
- `send_email_notification`
- `run_verification_check`
- `auto_suppress`, only for low-severity, high-confidence `NO_ACTION` cases

Blocked actions include:

- `restart_service`
- `deploy_code`
- `rollback_release`
- `change_infrastructure`
- `modify_database_config`
- `rotate_secrets`
- `delete_data`
- `scale_cluster`

The LLM cannot directly execute remediation.

---

## Versioned Training Datasets

IncidentLens can export confirmed incident history as immutable, project-scoped
JSON Lines datasets. An incident is initially eligible when it has a complete
persisted analysis, is structurally valid, and has either stored log samples or
a persisted investigation evidence snapshot.

Each example contains:

- Incident metadata and stored log samples
- The latest persisted analysis
- The latest investigation and evidence snapshot, when available
- The latest policy/action outcome, when available
- Expected severity, disposition, root cause, summary, and recorded recommended
  actions

The builder never asks an LLM for labels. Missing optional facts remain `null`,
and malformed or incomplete incidents are excluded. Records are sorted and
serialized deterministically using the `incident-training-example-v1` schema.

Build a dataset synchronously with:

```text
POST /api/datasets/build
```

Each successful build reserves a new version such as `dataset-v1` or
`dataset-v2`, stores it without overwriting earlier files, and registers its
storage key and record count in PostgreSQL. Local files default to `datasets/`;
set `DATASET_STORAGE_PATH` to change the root. Storage is behind a dedicated
interface so a future S3 backend does not require changes to the builder.

Datasets are immutable because future training jobs must be reproducible. A
later training job will reference one dataset row and read its registered
storage key; building a dataset does not create a training job or model artifact.

### Training lifecycle

A READY dataset can be queued and run synchronously through the Training Plane:

```text
POST /api/training-jobs
POST /api/training-jobs/{id}/run
```

The authenticated project is the job owner; the create request supplies the
`dataset_id`. Jobs follow explicit transitions:

```text
success:            QUEUED -> RUNNING -> EVALUATING -> PASSED
training failure:   QUEUED -> RUNNING -> FAILED
evaluation failure: QUEUED -> RUNNING -> EVALUATING -> FAILED
```

The training engine performs real supervised fine-tuning with Hugging Face
Transformers and PEFT. Every project uses the same configurable shared base
model, `Qwen/Qwen2.5-0.5B-Instruct` by default, and produces only a
project-specific LoRA adapter. The base-model weights are never copied into a
project artifact.

The immutable Dataset Builder JSONL is consumed directly. Each `input` object
becomes the user-side incident context and each `expected_output` object becomes
the assistant completion. The trainer applies the Qwen chat template and masks
prompt tokens so loss is calculated on the confirmed output rather than on the
incident evidence.

The default LoRA profile is:

```text
rank:                        8
alpha:                       16
dropout:                     0.05
epochs:                      1
learning rate:               0.0001
batch size:                  1
gradient accumulation:       1
maximum sequence length:     1024
validation fraction:         0.10
```

The target modules are Qwen attention and MLP projections: `q_proj`, `k_proj`,
`v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. All profile values
are configurable through the environment variables listed below.

Before training, the worker reserves an immutable
`artifacts/<project>/<adapter-vN>/` directory. PEFT writes the real adapter and
tokenizer metadata into it:

```text
adapter_model.safetensors
adapter_config.json
tokenizer_config.json
metadata.json
```

Tokenizer implementations may write additional tokenizer files. The final
metadata includes training duration, training and validation loss, dataset
version, the complete LoRA profile, evaluation metrics, framework versions, and
a SHA-256 manifest of generated adapter files. It explicitly distinguishes
adapter weights from base-model weights.

Evaluation checks dataset availability and record count, successful completion,
finite training loss, finite validation loss when a validation split exists,
and adapter integrity. An artifact is registered as READY and activated only
after every check passes. Partial output is removed when training or evaluation
fails; an adapter retained after a metadata-write failure is registered FAILED
and is never activated.

Artifact registration, project activation, and the final PASSED job transition
are coordinated by the Training Worker. If training, evaluation, or metadata
writing fails, the job becomes FAILED and the project's existing active
artifact remains unchanged. Successful artifacts are activated and become the
project's local inference adapter.

The production backend uses portable Transformers + PEFT rather than Unsloth.
Unsloth is not enabled because the current deployment contract does not
guarantee a supported NVIDIA/CUDA environment. `TrainingEngine` remains the
replacement boundary for adding an Unsloth or hosted training backend without
changing jobs, evaluation, artifact registration, or activation.

### Model runtime

All Serving Plane inference now enters through the Model Runtime:

```text
Incident evidence
      |
      v
Project -> active_artifact_id -> READY ModelArtifact
      |
      v
Model Runtime
      |-- Shared base model (loaded once)
      |-- READY active artifact
      |-- Adapter path and metadata
      |-- Project adapter cache
      v
Local Qwen + active LoRA adapter
      |
      v
Existing decision output
```

`resolve_project_model(project_id)` returns a runtime session that describes the
project, shared base model, validated active artifact, adapter path, local
runtime type, and runtime capabilities.
An active artifact is accepted only when it belongs to the project, is READY,
and was trained for the project's base model. Its `metadata.json` is loaded and
checked against the database record; validation problems are exposed as session
warnings and become explicit adapter-loading errors during inference.

At application startup, the runtime loads the configured shared base model. A
base-model loading failure prevents startup. On the first inference request for
a project artifact, PEFT loads that adapter into the shared model under a unique
name. Later requests reuse it. Adapter selection and generation share one lock,
so concurrent project requests cannot generate with another project's adapter.
When `Project.active_artifact_id` changes, the next request resolves and selects
the new artifact automatically.

The local provider uses the model's chat template for existing prompts and tool
schemas. Generated text continues through the existing Pydantic parsing,
validation, policy, action, notification, and audit paths. A missing, failed,
incompatible, or unreadable adapter produces a local runtime error; there is no
remote inference or base-model-only fallback.

---

## Example Decisions

### Allowed: Health-Check Noise

```text
Severity:          low
Disposition:       NO_ACTION
Requested actions: auto_enrich, auto_suppress
Policy decision:   allowed
Executed actions:  auto_enrich, auto_suppress
```

### Blocked: DB Pool Exhaustion

```text
Severity:        high
Disposition:     NEEDS_ONCALL
Allowed actions: notify_oncall, create_incident_summary, attach_evidence_bundle
Blocked actions: restart_service, modify_database_config
Reason:          high-impact remediation requires human approval
```

---

## Evaluation

IncidentLens includes a labeled evaluation suite for runbook cases, LLM reasoning cases, misleading log-volume signals, low-frequency high-impact incidents, and policy edge cases.

A case is counted as correctly triaged only when severity, disposition, and expected root cause or runbook match.

```text
Correct triage:              194/210 (92.4%)
Unsafe automation:             2/210 (1.0%)
False suppression:             2/210 (1.0%)
Unsafe disposition:           16/210 (7.6%)
Dangerous actions allowed:     0/210 (0.0%)
Dangerous actions executed:    0/210 (0.0%)
Runbook match accuracy:      178/190 (93.7%)
Runbook sample coverage:       68/68
```

Action-policy fixture:

```text
Policy block accuracy:        2/2 (100.0%)
Dangerous action block rate:  3/3 (100.0%)
Policy allow accuracy:        2/2 (100.0%)
```

Run evaluations:

```bash
python log-analyzer/scripts/metrics_report.py triage-eval
python log-analyzer/scripts/metrics_report.py triage-eval --dataset log-analyzer/evals/action_policy_cases.json
python log-analyzer/scripts/check_runbook_coverage.py
```

## Screenshots

### Dashboard

![Incident dashboard](./imgs/dashboard.png)

### Incidents

![Incidents view](./imgs/incident.png)

### AI platform dashboard

The existing incident dashboard now includes the project's local model status.
The **Models** page shows the shared base model, active adapter, evaluation
scores, and immutable artifact history; any READY artifact can be activated.
The **Training** page exposes the full dataset → training job → model artifact
workflow and only polls while a job is active. Settings displays the read-only
local runtime and dataset/artifact storage configuration. See the
[frontend guide](./log-analyzer-frontend/README.md) for the UI workflow and API
mapping.

### Discord

![Discord](./imgs/discord.png)

### Email

![Email](./imgs/email.png)

---

## Tech Stack

**Backend:** FastAPI, SQLAlchemy, PostgreSQL / Neon

**Frontend:** React, Tailwind CSS, Vercel

**Logs:** Datadog Logs API

**LLM:** Local Qwen 2.5 Instruct, Transformers, PEFT, LoRA, LangChain

**Notifications:** Discord, SMTP

**Deployment:** Vercel, Render, Neon PostgreSQL

---

## Quick Start

```bash
# Backend
pip install -r log-analyzer/requirements.txt
python -m uvicorn app.main:app --port 8000 --app-dir log-analyzer

# Log simulator
pip install -r log-server/requirements.txt
python -m uvicorn server:app --port 5001 --app-dir log-server

# Frontend
cd log-analyzer-frontend
npm ci
npm start
```

The first backend startup downloads and loads the configured shared base model;
startup fails if that model cannot be loaded. Local LLM inference also requires
the project to have a READY active artifact produced by the training lifecycle.

---

## Environment Variables

```env
DATABASE_URL=postgresql://user:pass@localhost:5432/log_analyzer

DATADOG_API_KEY=your_api_key
DATADOG_APP_KEY=your_application_key
DATADOG_SITE=datadoghq.com
DATADOG_QUERY=status:(error OR warn OR critical)
DATADOG_LOOKBACK_SECONDS=30
DATADOG_ENVIRONMENT=prod
DATADOG_SERVICE=
POLL_INTERVAL=30

MODEL_PROVIDER=local
BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
DEVICE=cpu
DTYPE=auto

LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_EPOCHS=1
LORA_LEARNING_RATE=0.0001
LORA_BATCH_SIZE=1
LORA_GRADIENT_ACCUMULATION_STEPS=1
LORA_MAX_SEQUENCE_LENGTH=1024
LORA_VALIDATION_FRACTION=0.1
LORA_SEED=42
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
ARTIFACT_STORAGE_PATH=artifacts
DATASET_STORAGE_PATH=datasets
HF_TOKEN=optional_hugging_face_token

LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

SMTP_HOST=smtp.example.com
SMTP_USER=alerts@example.com
SMTP_PASSWORD=your_password

LOG_SERVER_URL=http://localhost:5001
CORS_ORIGINS=http://localhost:3000
```

The application key used by the analyzer must have Datadog's
`logs_read_data` permission. `DATADOG_SITE` is the Datadog site domain for the
organization, such as `datadoghq.com`, `datadoghq.eu`, or
`us5.datadoghq.com`. Credentials may be configured per project on the Settings
page; environment values are deployment-level fallbacks.

The Data Plane depends on the provider-neutral `LogSourceConnector` contract.
Its Datadog implementation performs cursor-paginated searches over contiguous
polling windows, converts each result to an internal log envelope, and only
then hands plain log batches to the existing parser and incident pipeline.
Project credentials, queries, environment tags, and optional service filters
remain isolated during polling.

The log simulator submits logs to Datadog's HTTP intake with
`DATADOG_API_KEY`, `DATADOG_SITE`, `DATADOG_ENVIRONMENT`, and
`LOG_SERVICE_NAME`. It does not require the application key, which is used only
by the analyzer to search logs.
