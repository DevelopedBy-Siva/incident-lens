# Log analyzer deployment (EC2 + Neon + S3)

Only `log-analyzer` is deployed by this setup. The frontend remains on Vercel,
and the log simulator/server is not deployed.

## 1. Prepare AWS once

- Create an EC2 instance with enough RAM and disk for the local Qwen model and
  LoRA training. Install Docker Engine, the Docker Compose plugin, and `curl`.
- Allow inbound TCP `8000` only from the clients/proxy that need it. For HTTPS,
  place your preferred reverse proxy or load balancer in front later.
- Create one private S3 bucket. Block public access and keep default encryption
  enabled.
- Attach the IAM role with S3 access to the EC2 instance. The current demo role
  with `AmazonS3FullAccess` works. It can be narrowed to `s3:GetObject`,
  `s3:PutObject`, `s3:DeleteObject`, and `s3:ListBucket` for this bucket later.
- Copy the Neon pooled PostgreSQL connection string. Keep `sslmode=require` in
  the URL.

The EC2 login user must be able to run `docker` without an interactive sudo
prompt. No repository checkout or GitHub token is needed on the instance.

## 2. Add GitHub production secrets

Create a GitHub Environment named `production`, then add these required secrets:

| Secret | Value |
| --- | --- |
| `EC2_HOST` | EC2 public DNS name or IP |
| `EC2_USER` | SSH user, such as `ubuntu` |
| `EC2_SSH_KEY` | Complete private SSH key |
| `DATABASE_URL` | Neon pooled PostgreSQL URL with `sslmode=require` |
| `SECRET_KEY` | JWT signing key; generate once with `openssl rand -hex 32` |
| `CORS_ORIGINS` | Exact Vercel URL; comma-separate multiple origins |
| `AWS_REGION` | Bucket region, such as `us-east-1` |
| `S3_BUCKET` | Private bucket name |
| `HF_TOKEN` | Hugging Face token used to download `Qwen/Qwen3.5-4B` |

`EC2_SSH_KEY` is the content of the `.pem` file downloaded when the EC2 key
pair was created. It only permits GitHub Actions to SSH into the instance; it
does not grant S3 access.

S3 authentication comes from the attached EC2 IAM role. Do not create or add
`AWS_ACCESS_KEY_ID` or `AWS_SECRET_ACCESS_KEY` GitHub secrets. The only optional
secret is:

| Secret | Value |
| --- | --- |
| `EC2_DEPLOY_PATH` | Custom path; defaults to `/home/<user>/incident-lens` |

`DASHBOARD_URL` is not needed unless SMTP email notifications are enabled. No
deployment-level Datadog key is used: each IncidentLens user configures their
own Datadog connection in project settings. Datadog application telemetry is
disabled in this demo deployment.

Do not add the log server URL: production ingestion reads each project's saved
Datadog configuration, so the undeployed simulator is not required.

## 3. Deploy

Push a backend/deployment change to `main`, or run **Deploy log analyzer to
EC2** manually from GitHub Actions. The workflow:

1. runs the Python test suite;
2. copies a source bundle and generated `.env.production` over SSH;
3. builds and starts the container with Docker Compose;
4. waits for `/ready` to confirm Neon connectivity.

The service is available at `http://EC2_HOST:8000`. Point the Vercel
`REACT_APP_API_URL` at the public HTTPS backend URL once DNS/TLS is configured.

Setting `S3_BUCKET` selects S3 automatically. Datasets are stored under
`datasets/`, and trained adapters under `artifacts/`. Local adapter and Hugging
Face directories are working caches, so ordinary deployments do not need to
download unchanged files again.

## Operations

On EC2, inspect the service with:

```bash
cd ~/incident-lens/current
docker compose -p incident-lens -f compose.prod.yaml ps
docker compose -p incident-lens -f compose.prod.yaml logs -f --tail=200
curl --fail http://localhost:8000/ready
```
