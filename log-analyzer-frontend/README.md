# IncidentLens Frontend

The React dashboard exposes IncidentLens incident operations and the complete
project model lifecycle without duplicating backend workflow logic.

## Pages

- **Dashboard** — incident statistics, filters, simulator controls, and incident
  investigation details, including the serving model and decision source.
- **Models** — shared base model, active project adapter, artifact history,
  evaluation scores, and activation controls for READY artifacts.
- **Training** — immutable dataset history, synchronous dataset builds, training
  job creation/execution, status progress, and produced artifacts.
- **Settings** — Datadog API/application credentials and site, notification and
  security settings, plus a read-only view of the local AI runtime/storage
  configuration. There is no remote inference provider configuration.

## Model lifecycle

The UI follows the backend-owned lifecycle:

```text
Confirmed incidents
      ↓
Build immutable dataset
      ↓
Create and run training job
      ↓
Evaluate and register LoRA artifact
      ↓
Activate READY artifact
      ↓
Future incidents use the active adapter
```

Training is started synchronously through the existing API. While a job is
`QUEUED`, `RUNNING`, or `EVALUATING`, the Training page refreshes lifecycle data
every four seconds. Polling stops after the job reaches `PASSED` or `FAILED`.
Other lifecycle reads share a ten-second in-memory cache, which is invalidated
after every mutation.

Datasets and artifacts are never deleted or overwritten by the frontend.

## API usage

The frontend uses the authenticated backend routes:

- `GET /api/model-runtime`
- `GET /api/datasets` and `POST /api/datasets/build`
- `GET|POST /api/training-jobs`
- `POST /api/training-jobs/{id}/run`
- `GET /api/model-artifacts`
- `POST /api/model-artifacts/{id}/activate`

All lifecycle ownership, eligibility, training, evaluation, activation, and
inference decisions remain in the backend.

## Development

```bash
npm ci
npm start
```

Set `REACT_APP_API_URL` when the backend is not available at
`http://localhost:8000`.

Run a production verification build with:

```bash
npm run build
```
