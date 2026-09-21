# IncidentLens

A policy-bound AIOps agent that turns noisy application logs into auditable incidents.

IncidentLens clusters logs from Grafana Loki, routes known failures through YAML runbooks, uses LLM-assisted investigation for ambiguous incidents, and gates every requested action through a backend policy engine.

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
Grafana Loki
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
Raw Logs               -> Grafana Loki
LLM Provider           -> Groq / GPT-OSS
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

The current training engine is deliberately a placeholder. It exercises the
complete lifecycle and reports realistic execution metadata, but it does not
run LoRA, create an adapter, use a GPU, or produce model weights. Evaluation
checks dataset availability, record count, and successful engine execution.

After evaluation passes, IncidentLens registers an `adapter-vN` ModelArtifact
and writes only an `artifacts/<project>/<version>/metadata.json` file. The file
explicitly records that training was simulated and that it contains no model
weights. Set `ARTIFACT_STORAGE_PATH` to change the local artifact root.

Artifact registration, project activation, and the final PASSED job transition
are coordinated by the Training Worker. If training, evaluation, or metadata
writing fails, the job becomes FAILED and the project's existing active
artifact remains unchanged. Successful artifacts are activated for future use,
but the current Groq inference path does not load adapter weights.

### Model runtime

All Serving Plane inference now enters through the Model Runtime:

```text
Incident evidence
      |
      v
Model Runtime
      |-- Project base model
      |-- READY active artifact
      |-- Adapter path and metadata
      |-- Provider and capabilities
      v
Groq Provider
      |
      v
Existing decision output
```

`resolve_project_model(project_id)` returns a runtime session that
describes the project, base model, validated active artifact, adapter path,
provider, runtime type, configured model candidates, and runtime capabilities.
An active artifact is accepted only when it belongs to the project, is READY,
and was trained for the project's base model. Its `metadata.json` is loaded and
checked against the database record; validation problems are exposed as session
warnings rather than silently loading incompatible state.

The runtime is metadata-only in this phase. It explicitly reports
`adapter_loading=false` and `weights_loaded=false`, and the Groq provider keeps
the same API keys, model ordering, fallback behavior, prompts, temperatures,
tool calls, parsing, and outputs used by the previous inference path. Dataset
building and training remain separate and are never invoked by the runtime.

Future local LoRA support plugs in at the provider boundary. A local provider
can use the already-resolved base model and adapter metadata to load weights and
advertise adapter capabilities; decision analysis, investigation, runbook
tie-breaking, evidence creation, policy evaluation, and notification callers
continue using the same runtime-session interface.

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

The local Groq-backed eval was run with `openai/gpt-oss-20b`.

---

## Screenshots

### Dashboard

![Incident dashboard](./imgs/dashboard.png)

### Incidents

![Incidents view](./imgs/incident.png)

### Settings

![Settings](./imgs/settings.png)

### Discord

![Discord](./imgs/discord.png)

### Email

![Email](./imgs/email.png)

---

## Tech Stack

**Backend:** FastAPI, SQLAlchemy, PostgreSQL / Neon

**Frontend:** React, Tailwind CSS, Vercel

**Logs:** Grafana Loki

**LLM:** Groq, GPT-OSS, LangChain

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
npm install
npm start
```

---

## Environment Variables

```env
DATABASE_URL=postgresql://user:pass@localhost:5432/log_analyzer

LOKI_URL=https://logs-prod-xxx.grafana.net
LOKI_USERNAME=your_username
LOKI_API_KEY=your_token

GROQ_API_KEY=your_key
GROQ_API_KEY_2=your_second_key
GROQ_API_KEY_3=your_third_key
GROQ_MODEL=openai/gpt-oss-20b
GROQ_MODEL_FALLBACKS=openai/gpt-oss-20b,openai/gpt-oss-120b

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
