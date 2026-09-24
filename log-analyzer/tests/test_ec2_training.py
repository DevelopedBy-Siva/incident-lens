"""
Tests for EC2-based training orchestration and local fallback mode.

Tests verify:
1. EC2 orchestrator instance lifecycle management
2. Training worker mode selection (local vs EC2)
3. EC2 configuration loading from environment
4. Training job configuration serialization
5. Job status transitions for EC2 remote execution
6. Backward compatibility with local training mode
"""

import json
import os
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.shared.database import Base
from app.training.ec2_orchestrator import (
    EC2TrainingConfig,
    EC2TrainingOrchestrator,
    configured_ec2_training_config,
)
from app.training.job_config import (
    TrainingJobConfig,
    create_training_job_config,
)
from app.training.models import TrainingJob, TrainingJobStatus
from app.training.repositories import TrainingJobRepository
from app.training.training_profile import LoraTrainingProfile
from app.training.worker import TrainingWorker


class EC2ConfigurationTests(unittest.TestCase):
    """Test EC2 configuration loading from environment variables."""

    def tearDown(self):
        """Clean up environment variables after each test."""
        for key in [
            "TRAINING_EC2_ENABLED",
            "TRAINING_EC2_AMI_ID",
            "TRAINING_EC2_INSTANCE_TYPE",
            "TRAINING_EC2_IAM_INSTANCE_PROFILE",
            "TRAINING_EC2_SUBNET_ID",
            "TRAINING_EC2_SECURITY_GROUP_IDS",
            "TRAINING_EC2_MAX_WAIT_SECONDS",
            "TRAINING_EC2_DETAILED_MONITORING",
            "TRAINING_EC2_TERMINATE_ON_COMPLETION",
            "TRAINING_EC2_ASSOCIATE_PUBLIC_IP",
        ]:
            if key in os.environ:
                del os.environ[key]

    def test_ec2_disabled_by_default(self):
        """Test that EC2 training is disabled by default."""
        config = configured_ec2_training_config()
        self.assertIsNone(config)

    def test_ec2_requires_ami_id(self):
        """Test that EC2 training requires AMI ID."""
        os.environ["TRAINING_EC2_ENABLED"] = "true"

        with self.assertRaisesRegex(RuntimeError, "TRAINING_EC2_AMI_ID"):
            configured_ec2_training_config()

    def test_ec2_requires_iam_instance_profile(self):
        """Test that EC2 training requires IAM instance profile."""
        os.environ["TRAINING_EC2_ENABLED"] = "true"
        os.environ["TRAINING_EC2_AMI_ID"] = "ami-12345"

        with self.assertRaisesRegex(RuntimeError, "TRAINING_EC2_IAM_INSTANCE_PROFILE"):
            configured_ec2_training_config()

    def test_ec2_full_configuration(self):
        """Test loading complete EC2 configuration."""
        os.environ["TRAINING_EC2_ENABLED"] = "true"
        os.environ["TRAINING_EC2_AMI_ID"] = "ami-12345"
        os.environ["TRAINING_EC2_INSTANCE_TYPE"] = "g6.2xlarge"
        os.environ["TRAINING_EC2_IAM_INSTANCE_PROFILE"] = "incident-lens-training"
        os.environ["TRAINING_EC2_SUBNET_ID"] = "subnet-xyz"
        os.environ["TRAINING_EC2_SECURITY_GROUP_IDS"] = "sg-1,sg-2"
        os.environ["TRAINING_EC2_MAX_WAIT_SECONDS"] = "7200"
        os.environ["TRAINING_EC2_DETAILED_MONITORING"] = "true"
        os.environ["TRAINING_EC2_ASSOCIATE_PUBLIC_IP"] = "false"

        config = configured_ec2_training_config()

        self.assertIsNotNone(config)
        self.assertTrue(config.enabled)
        self.assertEqual(config.ami_id, "ami-12345")
        self.assertEqual(config.instance_type, "g6.2xlarge")
        self.assertEqual(config.iam_instance_profile, "incident-lens-training")
        self.assertEqual(config.subnet_id, "subnet-xyz")
        self.assertEqual(config.security_group_ids, ["sg-1", "sg-2"])
        self.assertEqual(config.max_wait_seconds, 7200)
        self.assertTrue(config.enable_detailed_monitoring)
        self.assertFalse(config.associate_public_ip)

    def test_ec2_defaults(self):
        """Test default values for optional EC2 configuration."""
        os.environ["TRAINING_EC2_ENABLED"] = "true"
        os.environ["TRAINING_EC2_AMI_ID"] = "ami-12345"
        os.environ["TRAINING_EC2_IAM_INSTANCE_PROFILE"] = "incident-lens-training"

        config = configured_ec2_training_config()

        self.assertEqual(config.instance_type, "g6.xlarge")
        self.assertIsNone(config.subnet_id)
        self.assertIsNone(config.security_group_ids)
        self.assertEqual(config.max_wait_seconds, 3600)
        self.assertFalse(config.enable_detailed_monitoring)
        self.assertTrue(config.terminate_on_completion)
        self.assertFalse(config.associate_public_ip)


class EC2OrchestratorTests(unittest.TestCase):
    """Test EC2 orchestrator instance lifecycle management."""

    def setUp(self):
        """Set up test orchestrator with mocked EC2 client."""
        self.config = EC2TrainingConfig(
            enabled=True,
            instance_type="g6.xlarge",
            ami_id="ami-12345",
            subnet_id="subnet-xyz",
            security_group_ids=["sg-1"],
            iam_instance_profile="incident-lens-training",
            max_wait_seconds=3600,
            enable_detailed_monitoring=False,
            terminate_on_completion=True,
            associate_public_ip=False,
        )
        self.ec2_client = Mock()
        self.orchestrator = EC2TrainingOrchestrator(self.config, self.ec2_client)

    def test_launch_instance(self):
        """Test launching a training instance."""
        self.ec2_client.run_instances.return_value = {
            "Instances": [{"InstanceId": "i-training-123"}]
        }

        instance_id = self.orchestrator.launch_training_instance(
            job_id="job-1",
            project_id="project-1",
            dataset_id="dataset-1",
            dataset_storage_key="datasets/project-1/dataset-v1.jsonl",
            database_url="postgresql://localhost/db",
            training_config_json='{"job_id": "job-1"}',
        )

        self.assertEqual(instance_id, "i-training-123")
        self.ec2_client.run_instances.assert_called_once()

        # Verify launch parameters
        call_args = self.ec2_client.run_instances.call_args
        kwargs = call_args.kwargs
        self.assertEqual(kwargs["ImageId"], "ami-12345")
        self.assertEqual(kwargs["InstanceType"], "g6.xlarge")
        self.assertEqual(kwargs["SubnetId"], "subnet-xyz")
        self.assertEqual(kwargs["SecurityGroupIds"], ["sg-1"])
        self.assertIn("UserData", kwargs)
        self.assertIn("TagSpecifications", kwargs)

    def test_launch_instance_without_subnet(self):
        """Test launching instance without specifying subnet."""
        config = EC2TrainingConfig(
            enabled=True,
            instance_type="g6.xlarge",
            ami_id="ami-12345",
            subnet_id=None,
            security_group_ids=None,
            iam_instance_profile="incident-lens-training",
            max_wait_seconds=3600,
            enable_detailed_monitoring=False,
            terminate_on_completion=True,
            associate_public_ip=False,
        )
        orchestrator = EC2TrainingOrchestrator(config, self.ec2_client)
        self.ec2_client.run_instances.return_value = {
            "Instances": [{"InstanceId": "i-training-456"}]
        }

        orchestrator.launch_training_instance(
            job_id="job-2",
            project_id="project-1",
            dataset_id="dataset-1",
            dataset_storage_key="datasets/project-1/dataset-v1.jsonl",
            database_url="postgresql://localhost/db",
            training_config_json='{"job_id": "job-2"}',
        )

        call_args = self.ec2_client.run_instances.call_args.kwargs
        self.assertNotIn("SubnetId", call_args)

    def test_get_instance_status(self):
        """Test retrieving instance status."""
        self.ec2_client.describe_instances.return_value = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "State": {"Name": "running"},
                            "StateTransitionReason": "User initiated",
                            "LaunchTime": datetime.now(),
                            "PublicIpAddress": "1.2.3.4",
                            "PrivateIpAddress": "10.0.0.1",
                        }
                    ]
                }
            ]
        }

        status = self.orchestrator.get_instance_status("i-training-123")

        self.assertEqual(status["state"], "running")
        self.assertEqual(status["state_transition_reason"], "User initiated")
        self.assertEqual(status["public_ip"], "1.2.3.4")
        self.assertEqual(status["private_ip"], "10.0.0.1")

    def test_terminate_instance(self):
        """Test terminating an instance."""
        self.ec2_client.terminate_instances.return_value = {}

        result = self.orchestrator.terminate_instance("i-training-123")

        self.assertTrue(result)
        self.ec2_client.terminate_instances.assert_called_once_with(
            InstanceIds=["i-training-123"]
        )

    def test_terminate_nonexistent_instance(self):
        """Test terminating an already-terminated instance."""
        from botocore.exceptions import ClientError

        error = ClientError(
            {"Error": {"Code": "InvalidInstanceID.NotFound", "Message": "Not found"}},
            "TerminateInstances",
        )
        self.ec2_client.terminate_instances.side_effect = error

        result = self.orchestrator.terminate_instance("i-nonexistent")

        self.assertTrue(result)  # Should return True (already gone)

    def test_is_training_instance_idle(self):
        """Test checking if instance is in terminal state."""
        running_status = {"state": "running"}
        stopped_status = {"state": "stopped"}
        terminated_status = {"state": "terminated"}

        self.assertFalse(
            EC2TrainingOrchestrator.is_training_instance_idle(running_status)
        )
        self.assertTrue(
            EC2TrainingOrchestrator.is_training_instance_idle(stopped_status)
        )
        self.assertTrue(
            EC2TrainingOrchestrator.is_training_instance_idle(terminated_status)
        )


class TrainingJobConfigTests(unittest.TestCase):
    """Test training job configuration serialization."""

    def test_config_serialization(self):
        """Test serializing training job configuration to JSON."""
        config = TrainingJobConfig(
            job_id="job-1",
            project_id="project-1",
            dataset_id="dataset-1",
            dataset_storage_key="datasets/project-1/dataset-v1.jsonl",
            s3_bucket="incident-lens-data",
            database_url="postgresql://localhost/db",
            base_model="Qwen/Qwen3.5-4B",
            git_repository_url="https://github.com/DevelopedBy-Siva/incident-lens.git",
            git_commit_sha="abc123def456789012345678901234567890abcd",
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.05,
            lora_epochs=1,
            lora_learning_rate=0.0001,
            lora_batch_size=1,
            lora_gradient_accumulation_steps=1,
            lora_max_sequence_length=1024,
            lora_validation_fraction=0.1,
            lora_seed=42,
            lora_target_modules=["q_proj", "k_proj"],
        )

        json_str = config.to_json()
        data = json.loads(json_str)

        self.assertEqual(data["job_id"], "job-1")
        self.assertEqual(data["project_id"], "project-1")
        self.assertEqual(data["lora_rank"], 8)
        self.assertEqual(data["lora_target_modules"], ["q_proj", "k_proj"])
        self.assertEqual(data["git_repository_url"], "https://github.com/DevelopedBy-Siva/incident-lens.git")
        self.assertEqual(data["git_commit_sha"], "abc123def456789012345678901234567890abcd")

    def test_config_deserialization(self):
        """Test deserializing configuration from JSON."""
        json_str = json.dumps(
            {
                "job_id": "job-1",
                "project_id": "project-1",
                "dataset_id": "dataset-1",
                "dataset_storage_key": "datasets/project-1/dataset-v1.jsonl",
                "s3_bucket": "incident-lens-data",
                "database_url": "postgresql://localhost/db",
                "base_model": "Qwen/Qwen3.5-4B",
                "git_repository_url": "https://github.com/DevelopedBy-Siva/incident-lens.git",
                "git_commit_sha": "abc123def456789012345678901234567890abcd",
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_epochs": 1,
                "lora_learning_rate": 0.0001,
                "lora_batch_size": 1,
                "lora_gradient_accumulation_steps": 1,
                "lora_max_sequence_length": 1024,
                "lora_validation_fraction": 0.1,
                "lora_seed": 42,
                "lora_target_modules": ["q_proj", "k_proj"],
                "selected_record_indices": None,
                "huggingface_token": None,
            }
        )

        config = TrainingJobConfig.from_json(json_str)

        self.assertEqual(config.job_id, "job-1")
        self.assertEqual(config.project_id, "project-1")
        self.assertEqual(config.lora_rank, 8)
        self.assertEqual(config.git_repository_url, "https://github.com/DevelopedBy-Siva/incident-lens.git")
        self.assertEqual(config.git_commit_sha, "abc123def456789012345678901234567890abcd")

    def test_config_from_profile(self):
        """Test creating configuration from training profile."""
        profile = LoraTrainingProfile(
            rank=8,
            alpha=16,
            dropout=0.05,
            epochs=1,
            learning_rate=0.0001,
            batch_size=1,
            gradient_accumulation_steps=1,
            max_sequence_length=1024,
            validation_fraction=0.1,
            seed=42,
            target_modules={"q_proj", "k_proj"},
        )

        config = create_training_job_config(
            job_id="job-1",
            project_id="project-1",
            dataset_id="dataset-1",
            dataset_storage_key="datasets/project-1/dataset-v1.jsonl",
            s3_bucket="incident-lens-data",
            database_url="postgresql://localhost/db",
            base_model="Qwen/Qwen3.5-4B",
            git_repository_url="https://github.com/DevelopedBy-Siva/incident-lens.git",
            git_commit_sha="abc123def456789012345678901234567890abcd",
            training_profile=profile,
        )

        self.assertEqual(config.lora_rank, 8)
        self.assertEqual(config.lora_alpha, 16)
        self.assertEqual(config.lora_epochs, 1)
        self.assertEqual(config.git_repository_url, "https://github.com/DevelopedBy-Siva/incident-lens.git")
        self.assertEqual(config.git_commit_sha, "abc123def456789012345678901234567890abcd")


class TrainingWorkerModeTests(unittest.TestCase):
    """Test training worker mode selection and execution."""

    def setUp(self):
        """Set up test database and worker."""
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()

    def tearDown(self):
        """Clean up database."""
        self.db.close()
        self.engine.dispose()
        for key in [
            "TRAINING_EC2_ENABLED",
            "TRAINING_EC2_AMI_ID",
            "TRAINING_EC2_IAM_INSTANCE_PROFILE",
        ]:
            if key in os.environ:
                del os.environ[key]

    def test_worker_local_mode_by_default(self):
        """Test that worker defaults to local mode."""
        worker = TrainingWorker(self.db)

        self.assertFalse(worker.use_ec2_remote)
        self.assertIsNone(worker.ec2_orchestrator)

    def test_worker_ec2_mode_when_enabled(self):
        """Test that worker uses EC2 mode when configured."""
        os.environ["TRAINING_EC2_ENABLED"] = "true"
        os.environ["TRAINING_EC2_AMI_ID"] = "ami-12345"
        os.environ["TRAINING_EC2_IAM_INSTANCE_PROFILE"] = "incident-lens-training"

        mock_orchestrator = Mock()
        worker = TrainingWorker(self.db, ec2_orchestrator=mock_orchestrator)

        self.assertTrue(worker.use_ec2_remote)
        self.assertIsNotNone(worker.ec2_orchestrator)

    def test_worker_executes_local_by_default(self):
        """Test that job execution uses local path by default."""
        # Create a mock job
        job = TrainingJob(
            id="job-1",
            project_id="project-1",
            dataset_id="dataset-1",
            status=TrainingJobStatus.QUEUED,
        )
        self.db.add(job)
        self.db.commit()

        worker = TrainingWorker(self.db)

        # Verify worker would use local execution
        self.assertFalse(worker.use_ec2_remote)
        self.assertTrue(hasattr(worker, "_run_local"))

    def test_ec2_orchestrator_initialization(self):
        """Test EC2 orchestrator is properly initialized in worker."""
        os.environ["TRAINING_EC2_ENABLED"] = "true"
        os.environ["TRAINING_EC2_AMI_ID"] = "ami-12345"
        os.environ["TRAINING_EC2_IAM_INSTANCE_PROFILE"] = "incident-lens-training"

        # Mock boto3.client to avoid real AWS calls
        mock_ec2_client = MagicMock()
        with patch("app.training.ec2_orchestrator.boto3.client", return_value=mock_ec2_client) as mock_boto3:
            worker = TrainingWorker(self.db)

            # Verify boto3.client was called with correct parameters
            mock_boto3.assert_called_once_with("ec2", region_name=None)

        self.assertTrue(worker.use_ec2_remote)
        self.assertIsInstance(worker.ec2_orchestrator, EC2TrainingOrchestrator)
        self.assertEqual(worker.ec2_orchestrator.ec2_client, mock_ec2_client)


class TrainingJobStatusTransitionTests(unittest.TestCase):
    """Test training job status transitions for EC2 execution."""

    def setUp(self):
        """Set up test database."""
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()

    def tearDown(self):
        """Clean up database."""
        self.db.close()
        self.engine.dispose()

    def test_job_transitions_to_running_for_ec2(self):
        """Test job transitions from QUEUED to RUNNING for EC2 execution."""
        job = TrainingJob(
            id="job-1",
            project_id="project-1",
            dataset_id="dataset-1",
            status=TrainingJobStatus.QUEUED,
        )
        self.db.add(job)
        self.db.commit()

        # Simulate EC2 execution: job should be reserved (transitioned to RUNNING)
        job_repo = TrainingJobRepository(self.db)
        reserved_job = job_repo.reserve("job-1", "project-1", datetime.utcnow())

        self.assertIsNotNone(reserved_job)
        self.assertEqual(reserved_job.status, TrainingJobStatus.RUNNING)
        self.assertIsNotNone(reserved_job.started_at)

    def test_job_includes_instance_id(self):
        """Test job model includes ec2_instance_id field."""
        job = TrainingJob(
            id="job-1",
            project_id="project-1",
            dataset_id="dataset-1",
            status=TrainingJobStatus.RUNNING,
        )
        self.db.add(job)
        self.db.commit()

        retrieved_job = self.db.query(TrainingJob).filter_by(id="job-1").first()

        self.assertTrue(hasattr(retrieved_job, "ec2_instance_id"))
        self.assertIsNone(retrieved_job.ec2_instance_id)  # Not yet set

        # Update with instance ID
        retrieved_job.ec2_instance_id = "i-training-123"
        self.db.commit()

        updated_job = self.db.query(TrainingJob).filter_by(id="job-1").first()
        self.assertEqual(updated_job.ec2_instance_id, "i-training-123")


if __name__ == "__main__":
    unittest.main()
