# Log analyzer deployment (EC2 + Neon + S3 + GPU Training)

Only `log-analyzer` is deployed by this setup. The frontend remains on Vercel,
and the log simulator/server is not deployed.

## Training architecture

IncidentLens supports two training modes:

- **Local (default):** Training runs on the application EC2 instance, blocking the API during training
- **EC2 GPU (production):** Training runs on temporary g6.xlarge GPU instances, enabling scalable on-demand training

For production deployments with frequent retraining, use EC2 GPU training to avoid blocking the application server.

## 1. Prepare AWS once

### Application EC2 Instance

- Create an EC2 instance with enough RAM and disk for the local Qwen model and
  LoRA training cache (if using local training mode). Install Docker Engine, the Docker Compose plugin, and `curl`.
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

### GPU Training EC2 (Optional, for production training)

For on-demand GPU training via temporary instances, complete these additional steps:

#### 1. Create Training AMI

Create a pre-built AMI with PyTorch, Transformers, and PEFT already installed to speed up instance launch:

```bash
# On a temporary GPU-enabled EC2 instance (e.g., g6.xlarge):
sudo apt update && sudo apt install -y python3 python3-pip python3-venv
sudo pip3 install torch transformers peft safetensors datasets accelerate boto3 sqlalchemy psycopg2-binary

# Copy bootstrap script to /opt/incident-lens
sudo mkdir -p /opt/incident-lens
sudo cp scripts/bootstrap-training.py /opt/incident-lens/
sudo chmod +x /opt/incident-lens/bootstrap-training.py

# Configure systemd or similar to auto-run bootstrap script at startup:
# (or include it in a startup service that runs post-boot)

# Create and save AMI from this instance
# Example in AWS console or CLI:
# aws ec2 create-image --instance-id i-xxxxx --name incident-lens-training-gpu-v1
```

Save the resulting AMI ID (e.g., `ami-0123456789abcdef0`) and configure it in `TRAINING_EC2_AMI_ID`.

#### 2. Create IAM Instance Profile

Training instances need permissions to access S3 and PostgreSQL (for status updates):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::your-bucket",
        "arn:aws:s3:::your-bucket/*"
      ]
    }
  ]
}
```

Create an EC2 instance profile named `incident-lens-training-role` and attach this policy.
Use this profile name in `TRAINING_EC2_IAM_INSTANCE_PROFILE`.

#### 3. Configure Networking

Training instances need outbound access to:
- S3 (to download datasets and upload adapters)
- PostgreSQL (to update job status)
- Hugging Face Hub (to download base model)

Security group should allow:
- Outbound HTTPS (port 443) to S3, Hugging Face, and PostgreSQL
- Inbound: None required (no external access to training instances)

#### 4. Application IAM Role

The application EC2 instance needs EC2 permissions to launch instances:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:RunInstances",
        "ec2:DescribeInstances",
        "ec2:TerminateInstances",
        "ec2:CreateTags",
        "iam:PassRole"
      ],
      "Resource": "*"
    }
  ]
}
```

Attach this policy to the application instance's IAM role.

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

Optional secrets for EC2 GPU training:

| Secret | Value |
| --- | --- |
| `TRAINING_EC2_ENABLED` | Set to `true` to enable GPU training (defaults to `false`) |
| `TRAINING_EC2_AMI_ID` | Pre-built training AMI with PyTorch/PEFT installed |
| `TRAINING_EC2_INSTANCE_TYPE` | GPU instance type, e.g., `g6.xlarge` (defaults to `g6.xlarge`) |
| `TRAINING_EC2_IAM_INSTANCE_PROFILE` | IAM instance profile name for training instances |
| `TRAINING_EC2_SUBNET_ID` | VPC subnet ID for training instances (optional, uses default) |
| `TRAINING_EC2_SECURITY_GROUP_IDS` | Comma-separated security group IDs (optional, uses default) |
| `TRAINING_EC2_MAX_WAIT_SECONDS` | Max time to wait for training; defaults to 3600 |
| `EC2_DEPLOY_PATH` | Custom path; defaults to `/home/<user>/incident-lens` |

`EC2_SSH_KEY` is the content of the `.pem` file downloaded when the EC2 key
pair was created. It only permits GitHub Actions to SSH into the instance; it
does not grant S3 or EC2 access.

S3 authentication comes from the attached EC2 IAM role. EC2 instance management
uses the application EC2 instance's IAM role (no separate credentials needed).
Do not create or add `AWS_ACCESS_KEY_ID` or `AWS_SECRET_ACCESS_KEY` GitHub secrets.

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

### GPU Training Operations

If EC2 GPU training is enabled, monitor training jobs via the dashboard or API:

**Track training job status:**
```bash
curl http://EC2_HOST:8000/api/training-jobs/{job_id}
```

**Response codes indicate execution mode:**
- `202 Accepted`: Training delegated to EC2 GPU instance (EC2 mode)
- `200 OK`: Training completed locally (local mode)
- `ec2_instance_id` field in response: ID of the temporary training instance

**Monitor EC2 training instances:**
```bash
# From the application EC2 instance
aws ec2 describe-instances \
  --filters "Name=tag:managed_by,Values=incident-lens-training-orchestrator" \
  --query 'Reservations[].Instances[].[InstanceId, State.Name, LaunchTime, Tags[?Key==`job_id`].Value[]]'
```

**Troubleshoot training failure:**
1. Check job status: `GET /api/training-jobs/{job_id}` → look for `FAILED` status
2. Check EC2 instance: Verify instance launched, check if it terminated (training succeeded)
3. Check bootstrap logs: On a running instance, SSH and check `/var/log/incident-lens-training.log`
4. Manual cleanup: If an instance is stuck, terminate it manually via AWS console or CLI

**Configuration tuning:**
- `TRAINING_EC2_MAX_WAIT_SECONDS`: Increase if training frequently times out (GPU-intensive datasets)
- `TRAINING_EC2_INSTANCE_TYPE`: Use larger instance (e.g., `g6.2xlarge` for 200 GB GPU) for very large datasets
- `TRAINING_EC2_TERMINATE_ON_COMPLETION`: Set to `false` if you want to inspect completed instances (remember to terminate manually)
