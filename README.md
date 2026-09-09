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
