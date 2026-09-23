"""
EC2 Training Orchestrator
Manages the lifecycle of temporary GPU EC2 instances for fine-tuning jobs.
"""

import json
import logging
import os
from base64 import b64encode
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EC2TrainingConfig:
    """Configuration for EC2-based training execution."""

    # Enable/disable EC2 remote execution (local execution is default)
    enabled: bool

    # EC2 instance configuration
    instance_type: str  # e.g., "g6.xlarge"
    ami_id: str  # Training AMI with dependencies pre-installed
    subnet_id: str | None  # VPC subnet for instance (optional, uses default)
    security_group_ids: list[str] | None  # Security groups (optional, uses default)
    iam_instance_profile: str  # IAM instance profile with S3, DB, CloudWatch access

    # Monitoring and cleanup
    max_wait_seconds: int  # Max time to wait for training completion (default: 3600)
    enable_detailed_monitoring: bool  # Enable CloudWatch detailed monitoring
    terminate_on_completion: bool  # Auto-terminate after training (default: True)

    # Networking
    associate_public_ip: bool  # Whether to assign public IP (default: False for security)


def configured_ec2_training_config() -> EC2TrainingConfig | None:
    """Load EC2 training configuration from environment variables.
    
    Returns None if EC2 training is disabled (development mode).
    """
    enabled = os.getenv("TRAINING_EC2_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    if not enabled:
        return None

    # Validate required configuration
    required_vars = {
        "TRAINING_EC2_AMI_ID": "Training AMI ID",
        "TRAINING_EC2_INSTANCE_TYPE": "EC2 instance type (e.g., g6.xlarge)",
        "TRAINING_EC2_IAM_INSTANCE_PROFILE": "IAM instance profile name",
    }

    missing = []
    for var, desc in required_vars.items():
        if not os.getenv(var, "").strip():
            missing.append(f"{var} ({desc})")

    if missing:
        raise RuntimeError(
            "EC2 training enabled but missing required configuration: "
            + ", ".join(missing)
        )

    # Parse optional security group list
    sg_str = os.getenv("TRAINING_EC2_SECURITY_GROUP_IDS", "").strip()
    security_group_ids = [sg.strip() for sg in sg_str.split(",") if sg.strip()] or None

    return EC2TrainingConfig(
        enabled=True,
        instance_type=os.getenv("TRAINING_EC2_INSTANCE_TYPE", "g6.xlarge").strip(),
        ami_id=os.getenv("TRAINING_EC2_AMI_ID", "").strip(),
        subnet_id=os.getenv("TRAINING_EC2_SUBNET_ID", "").strip() or None,
        security_group_ids=security_group_ids,
        iam_instance_profile=os.getenv(
            "TRAINING_EC2_IAM_INSTANCE_PROFILE", ""
        ).strip(),
        max_wait_seconds=int(
            os.getenv("TRAINING_EC2_MAX_WAIT_SECONDS", "3600").strip()
        ),
        enable_detailed_monitoring=os.getenv(
            "TRAINING_EC2_DETAILED_MONITORING", ""
        ).strip().lower()
        in {"1", "true", "yes"},
        terminate_on_completion=os.getenv(
            "TRAINING_EC2_TERMINATE_ON_COMPLETION", "true"
        ).strip().lower()
        not in {"0", "false", "no"},
        associate_public_ip=os.getenv("TRAINING_EC2_ASSOCIATE_PUBLIC_IP", "")
        .strip()
        .lower()
        in {"1", "true", "yes"},
    )


class EC2TrainingOrchestrator:
    """Orchestrates temporary GPU EC2 instances for training job execution."""

    def __init__(
        self, config: EC2TrainingConfig, ec2_client=None, region_name: str | None = None
    ):
        """Initialize EC2 orchestrator.

        Args:
            config: EC2 training configuration
            ec2_client: Optional boto3 EC2 client (for testing)
            region_name: AWS region (auto-detected if not provided)
        """
        self.config = config
        if ec2_client is None:
            region = region_name or os.getenv("AWS_REGION", "").strip() or None
            ec2_client = boto3.client("ec2", region_name=region)
        self.ec2_client = ec2_client

    def launch_training_instance(
        self,
        job_id: str,
        project_id: str,
        dataset_id: str,
        dataset_storage_key: str,
        database_url: str,
        training_config_json: str,
    ) -> str:
        """Launch a temporary GPU EC2 instance for training job execution.

        Args:
            job_id: Training job ID
            project_id: Project ID (for tagging and isolation)
            dataset_id: Dataset ID
            dataset_storage_key: S3 storage key for dataset
            database_url: Database connection URL (for job status updates)
            training_config_json: Serialized training job configuration

        Returns:
            instance_id: The launched EC2 instance ID

        Raises:
            RuntimeError: If instance launch fails
        """
        # Prepare User Data script with job configuration
        user_data = self._prepare_user_data(
            job_id=job_id,
            project_id=project_id,
            dataset_id=dataset_id,
            dataset_storage_key=dataset_storage_key,
            database_url=database_url,
            training_config_json=training_config_json,
        )

        # Build launch parameters
        launch_params = {
            "ImageId": self.config.ami_id,
            "InstanceType": self.config.instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "IamInstanceProfile": {"Name": self.config.iam_instance_profile},
            "UserData": user_data,
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": f"incident-lens-training-{job_id}"},
                        {"Key": "job_id", "Value": job_id},
                        {"Key": "project_id", "Value": project_id},
                        {
                            "Key": "managed_by",
                            "Value": "incident-lens-training-orchestrator",
                        },
                    ],
                },
                {
                    "ResourceType": "volume",
                    "Tags": [
                        {"Key": "Name", "Value": f"incident-lens-training-{job_id}"},
                        {"Key": "job_id", "Value": job_id},
                    ],
                },
            ],
        }

        # Add optional VPC/networking configuration
        if self.config.subnet_id:
            launch_params["SubnetId"] = self.config.subnet_id
        if self.config.security_group_ids:
            launch_params["SecurityGroupIds"] = self.config.security_group_ids

        if self.config.associate_public_ip:
            launch_params["AssociatePublicIpAddress"] = True

        # Add monitoring
        if self.config.enable_detailed_monitoring:
            launch_params["Monitoring"] = {"Enabled": True}

        try:
            response = self.ec2_client.run_instances(**launch_params)
            instance_id = response["Instances"][0]["InstanceId"]
            logger.info(
                f"Launched EC2 training instance {instance_id} for job {job_id} "
                f"(project={project_id})"
            )
            return instance_id
        except ClientError as exc:
            logger.error(
                f"Failed to launch EC2 instance for job {job_id}: {exc.response['Error']['Message']}"
            )
            raise RuntimeError(
                f"Failed to launch training instance: {exc.response['Error']['Message']}"
            ) from exc
        except KeyError as exc:
            logger.error(f"Unexpected EC2 response format: {exc}")
            raise RuntimeError("Unexpected EC2 response format") from exc

    def get_instance_status(self, instance_id: str) -> dict:
        """Get current status of a training instance.

        Args:
            instance_id: EC2 instance ID

        Returns:
            dict: Instance status information
                - state: Instance state (pending, running, shutting-down, terminated, etc.)
                - state_transition_reason: Reason for current state
                - launch_time: Instance launch time
                - public_ip: Public IP (if assigned)
                - private_ip: Private IP
        """
        try:
            response = self.ec2_client.describe_instances(InstanceIds=[instance_id])
            if not response["Reservations"]:
                raise RuntimeError(f"Instance {instance_id} not found")

            instance = response["Reservations"][0]["Instances"][0]
            return {
                "state": instance["State"]["Name"],
                "state_transition_reason": instance.get(
                    "StateTransitionReason", "unknown"
                ),
                "launch_time": instance.get("LaunchTime"),
                "public_ip": instance.get("PublicIpAddress"),
                "private_ip": instance.get("PrivateIpAddress"),
            }
        except ClientError as exc:
            logger.error(f"Failed to get instance status for {instance_id}: {exc}")
            raise RuntimeError(f"Failed to get instance status: {exc}") from exc

    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate a training instance.

        Args:
            instance_id: EC2 instance ID

        Returns:
            bool: True if termination was initiated successfully
        """
        try:
            self.ec2_client.terminate_instances(InstanceIds=[instance_id])
            logger.info(f"Initiated termination of instance {instance_id}")
            return True
        except ClientError as exc:
            if "InvalidInstanceID.NotFound" in str(exc):
                logger.warning(f"Instance {instance_id} not found (already terminated)")
                return True
            logger.error(f"Failed to terminate instance {instance_id}: {exc}")
            return False

    def _prepare_user_data(
        self,
        job_id: str,
        project_id: str,
        dataset_id: str,
        dataset_storage_key: str,
        database_url: str,
        training_config_json: str,
    ) -> str:
        """Prepare User Data script that configures the training instance.

        The User Data script is base64-encoded and executed as root when the instance starts.
        It sets up environment variables and invokes the training bootstrap script.

        Args:
            job_id: Training job ID
            project_id: Project ID
            dataset_id: Dataset ID
            dataset_storage_key: S3 key for dataset
            database_url: Database URL
            training_config_json: Serialized training configuration

        Returns:
            base64-encoded User Data script
        """
        # Create configuration JSON for the instance
        config = {
            "job_id": job_id,
            "project_id": project_id,
            "dataset_id": dataset_id,
            "dataset_storage_key": dataset_storage_key,
            "database_url": database_url,
            "training_config": json.loads(training_config_json),
        }

        # Create shell script that runs training bootstrap
        script = f"""#!/bin/bash
set -e

# Log all output
exec 1> >(logger -s -t incident-lens-training)
exec 2>&1

echo "[TRAINING] Temporary GPU instance launched for job {job_id}"

# Export environment for training script
export INCIDENT_LENS_JOB_ID="{job_id}"
export INCIDENT_LENS_PROJECT_ID="{project_id}"
export INCIDENT_LENS_DATASET_ID="{dataset_id}"
export INCIDENT_LENS_DATASET_STORAGE_KEY="{dataset_storage_key}"
export INCIDENT_LENS_DATABASE_URL="{database_url}"

# Write training configuration to file for script access
cat > /tmp/incident-lens-training-config.json << 'CONFIGEOF'
{json.dumps(config, indent=2)}
CONFIGEOF

echo "[TRAINING] Configuration prepared at /tmp/incident-lens-training-config.json"

# Execute training bootstrap script
# The script is expected to be at /opt/incident-lens/bootstrap-training.py
if [ -f /opt/incident-lens/bootstrap-training.py ]; then
    echo "[TRAINING] Starting training bootstrap"
    cd /opt/incident-lens
    python bootstrap-training.py
    BOOTSTRAP_EXIT=$?
    echo "[TRAINING] Bootstrap exited with code $BOOTSTRAP_EXIT"
else
    echo "[TRAINING] ERROR: Bootstrap script not found at /opt/incident-lens/bootstrap-training.py"
    exit 1
fi

# Auto-terminate if configured (controlled via launch parameters)
# The orchestrator will handle termination via EC2 API
echo "[TRAINING] Awaiting orchestrator termination signal"
"""

        # Base64 encode the script for EC2 User Data
        encoded = b64encode(script.encode("utf-8")).decode("utf-8")
        return encoded

    @staticmethod
    def is_training_instance_idle(instance_status: dict) -> bool:
        """Check if a training instance has completed or failed.

        Args:
            instance_status: Instance status dict from get_instance_status()

        Returns:
            bool: True if instance is in terminal state
        """
        terminal_states = {"stopped", "stopping", "terminated", "terminating"}
        return instance_status["state"] in terminal_states

    @staticmethod
    def should_terminate_instance(instance_status: dict) -> bool:
        """Check if an instance should be terminated for cleanup.

        Args:
            instance_status: Instance status dict from get_instance_status()

        Returns:
            bool: True if instance is ready for termination
        """
        # Terminate if running or stopped
        return instance_status["state"] in {"running", "stopped"}
