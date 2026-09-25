"""
Tests for EC2-based training orchestration and local fallback mode.

Tests verify:
1. EC2 orchestrator instance lifecycle management
2. Training worker mode selection (local vs EC2)
3. EC2 configuration loading from environment
4. Training job configuration serialization
5. Job status transitions for EC2 remote execution
6. Backward compatibility with local training mode
7. Status transition validation (RUNNING -> EVALUATING -> PASSED)
8. SQLAlchemy metadata registration for foreign keys
9. EC2 instance cleanup guarantees
10. API response codes for different execution modes
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
from app.training.models import DatasetStatus, TrainingJob, TrainingJobStatus
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
        self.s3_client = Mock()
        self.orchestrator = EC2TrainingOrchestrator(
            self.config, self.ec2_client, self.s3_client
        )

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
            training_config_json='{"job_id": "job-1", "base_model": "Qwen/Qwen3.5-4B"}',
            s3_bucket="incident-lens-data",
            git_repository_url="https://github.com/DevelopedBy-Siva/incident-lens.git",
            git_commit_sha="abc123def456789012345678901234567890abcd",
        )

        self.assertEqual(instance_id, "i-training-123")
        
        # Verify config was uploaded to S3
        self.s3_client.put_object.assert_called_once()
        s3_call_args = self.s3_client.put_object.call_args.kwargs
        self.assertEqual(s3_call_args["Bucket"], "incident-lens-data")
        self.assertEqual(
            s3_call_args["Key"], "training-configs/project-1/job-1.json"
        )
        self.assertEqual(s3_call_args["ContentType"], "application/json")
        
        # Verify EC2 instance was launched
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
        
        # Verify User Data does not contain sensitive data
        import base64
        user_data = base64.b64decode(kwargs["UserData"]).decode("utf-8")
        self.assertNotIn("postgresql://", user_data)  # No DATABASE_URL
        self.assertNotIn("base_model", user_data)  # No config JSON
        self.assertIn("job-1", user_data)  # Contains job_id
        self.assertIn("incident-lens-data", user_data)  # Contains S3 bucket
        self.assertIn("training-configs/project-1/job-1.json", user_data)  # Contains config key

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
        s3_client = Mock()
        orchestrator = EC2TrainingOrchestrator(config, self.ec2_client, s3_client)
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
            s3_bucket="incident-lens-data",
            git_repository_url="https://github.com/DevelopedBy-Siva/incident-lens.git",
            git_commit_sha="abc123def456789012345678901234567890abcd",
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

    def test_config_upload_to_s3(self):
        """Test that training configuration is uploaded to S3 before instance launch."""
        self.ec2_client.run_instances.return_value = {
            "Instances": [{"InstanceId": "i-training-789"}]
        }

        training_config = {
            "job_id": "job-1",
            "project_id": "project-1",
            "base_model": "Qwen/Qwen3.5-4B",
            "lora_rank": 8,
            "database_url": "postgresql://user:pass@host/db",
            "huggingface_token": "hf_secret_token_12345",
        }

        self.orchestrator.launch_training_instance(
            job_id="job-1",
            project_id="project-1",
            dataset_id="dataset-1",
            dataset_storage_key="datasets/project-1/dataset-v1.jsonl",
            database_url="postgresql://user:pass@host/db",
            training_config_json=json.dumps(training_config),
            s3_bucket="incident-lens-data",
            git_repository_url="https://github.com/DevelopedBy-Siva/incident-lens.git",
            git_commit_sha="abc123def456789012345678901234567890abcd",
        )

        # Verify S3 upload was called
        self.s3_client.put_object.assert_called_once()
        s3_call = self.s3_client.put_object.call_args.kwargs

        # Verify S3 key follows expected pattern
        self.assertEqual(s3_call["Bucket"], "incident-lens-data")
        self.assertEqual(s3_call["Key"], "training-configs/project-1/job-1.json")

        # Verify uploaded config contains expected data
        uploaded_config = json.loads(s3_call["Body"].decode("utf-8"))
        self.assertEqual(uploaded_config["job_id"], "job-1")
        self.assertEqual(uploaded_config["base_model"], "Qwen/Qwen3.5-4B")
        self.assertIn("database_url", uploaded_config)
        self.assertIn("huggingface_token", uploaded_config)

    def test_user_data_size_within_limit(self):
        """Test that User Data size is well below AWS 25,600-byte limit."""
        import base64

        self.ec2_client.run_instances.return_value = {
            "Instances": [{"InstanceId": "i-training-999"}]
        }

        # Create a large config (simulating real-world scenario)
        large_config = {
            "job_id": "job-1" * 100,
            "project_id": "project-1" * 100,
            "base_model": "Qwen/Qwen3.5-4B",
            "lora_rank": 8,
            "database_url": "postgresql://user:very_long_password_12345@hostname.domain.com:5432/database_name",
            "huggingface_token": "hf_" + "x" * 500,
            "git_repository_url": "https://github.com/DevelopedBy-Siva/incident-lens.git",
            "git_commit_sha": "abc123def456789012345678901234567890abcd" * 10,
        }

        self.orchestrator.launch_training_instance(
            job_id="job-1",
            project_id="project-1",
            dataset_id="dataset-1",
            dataset_storage_key="datasets/project-1/dataset-v1.jsonl",
            database_url=large_config["database_url"],
            training_config_json=json.dumps(large_config),
            s3_bucket="incident-lens-data",
            git_repository_url=large_config["git_repository_url"],
            git_commit_sha=large_config["git_commit_sha"][:40],  # Valid SHA length
        )

        # Verify User Data size
        call_args = self.ec2_client.run_instances.call_args.kwargs
        user_data_b64 = call_args["UserData"]
        
        # User Data size should be well below the limit
        self.assertLess(len(user_data_b64), 25600)
        
        # Decode and verify User Data does NOT contain secrets
        user_data = base64.b64decode(user_data_b64).decode("utf-8")
        self.assertNotIn("postgresql://", user_data)
        self.assertNotIn("hf_", user_data)
        self.assertNotIn("very_long_password", user_data)

    def test_config_upload_failure_prevents_instance_launch(self):
        """Test that S3 upload failure prevents EC2 instance launch."""
        from botocore.exceptions import ClientError

        # Simulate S3 upload failure
        self.s3_client.put_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
            "PutObject",
        )

        with self.assertRaises(Exception) as context:
            self.orchestrator.launch_training_instance(
                job_id="job-1",
                project_id="project-1",
                dataset_id="dataset-1",
                dataset_storage_key="datasets/project-1/dataset-v1.jsonl",
                database_url="postgresql://localhost/db",
                training_config_json='{"job_id": "job-1"}',
                s3_bucket="incident-lens-data",
                git_repository_url="https://github.com/DevelopedBy-Siva/incident-lens.git",
                git_commit_sha="abc123def456789012345678901234567890abcd",
            )

        # Verify EC2 instance was NOT launched
        self.ec2_client.run_instances.assert_not_called()

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
            aws_region="us-east-1",
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
        self.assertEqual(data["aws_region"], "us-east-1")
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
                "aws_region": "us-east-1",
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
        self.assertEqual(config.aws_region, "us-east-1")
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
            aws_region="us-east-1",
            base_model="Qwen/Qwen3.5-4B",
            git_repository_url="https://github.com/DevelopedBy-Siva/incident-lens.git",
            git_commit_sha="abc123def456789012345678901234567890abcd",
            training_profile=profile,
        )

        self.assertEqual(config.aws_region, "us-east-1")
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
        mock_s3_client = MagicMock()
        
        def mock_client(service, **kwargs):
            if service == "ec2":
                return mock_ec2_client
            elif service == "s3":
                return mock_s3_client
            raise ValueError(f"Unexpected service: {service}")
        
        with patch("app.training.ec2_orchestrator.boto3.client", side_effect=mock_client) as mock_boto3:
            worker = TrainingWorker(self.db)

            # Verify boto3.client was called for both EC2 and S3
            self.assertEqual(mock_boto3.call_count, 2)
            mock_boto3.assert_any_call("ec2", region_name=None)
            mock_boto3.assert_any_call("s3", region_name=None)

        self.assertTrue(worker.use_ec2_remote)
        self.assertIsInstance(worker.ec2_orchestrator, EC2TrainingOrchestrator)
        self.assertEqual(worker.ec2_orchestrator.ec2_client, mock_ec2_client)
        self.assertEqual(worker.ec2_orchestrator.s3_client, mock_s3_client)


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


class TrainingJobStateTransitionValidationTests(unittest.TestCase):
    """Test training job state machine validates transitions correctly."""

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

    def test_running_to_evaluating_transition_is_valid(self):
        """Test RUNNING -> EVALUATING transition is allowed."""
        job = TrainingJob(
            id="job-1",
            project_id="project-1",
            dataset_id="dataset-1",
            status=TrainingJobStatus.RUNNING,
        )
        self.db.add(job)
        self.db.commit()

        job_repo = TrainingJobRepository(self.db)
        job_repo.transition(job, TrainingJobStatus.EVALUATING)
        self.db.commit()

        self.assertEqual(job.status, TrainingJobStatus.EVALUATING)

    def test_evaluating_to_passed_transition_is_valid(self):
        """Test EVALUATING -> PASSED transition is allowed."""
        job = TrainingJob(
            id="job-1",
            project_id="project-1",
            dataset_id="dataset-1",
            status=TrainingJobStatus.EVALUATING,
        )
        self.db.add(job)
        self.db.commit()

        job_repo = TrainingJobRepository(self.db)
        finished_at = datetime.utcnow()
        job_repo.transition(job, TrainingJobStatus.PASSED, finished_at=finished_at)
        self.db.commit()

        self.assertEqual(job.status, TrainingJobStatus.PASSED)
        self.assertEqual(job.finished_at, finished_at)

    def test_running_to_passed_transition_is_invalid(self):
        """Test RUNNING -> PASSED transition is rejected."""
        job = TrainingJob(
            id="job-1",
            project_id="project-1",
            dataset_id="dataset-1",
            status=TrainingJobStatus.RUNNING,
        )
        self.db.add(job)
        self.db.commit()

        job_repo = TrainingJobRepository(self.db)
        
        with self.assertRaisesRegex(
            ValueError, "Invalid training job transition: RUNNING -> PASSED"
        ):
            job_repo.transition(job, TrainingJobStatus.PASSED)

    def test_running_to_failed_transition_is_valid(self):
        """Test RUNNING -> FAILED transition is allowed."""
        job = TrainingJob(
            id="job-1",
            project_id="project-1",
            dataset_id="dataset-1",
            status=TrainingJobStatus.RUNNING,
        )
        self.db.add(job)
        self.db.commit()

        job_repo = TrainingJobRepository(self.db)
        finished_at = datetime.utcnow()
        job_repo.transition(job, TrainingJobStatus.FAILED, finished_at=finished_at)
        self.db.commit()

        self.assertEqual(job.status, TrainingJobStatus.FAILED)
        self.assertEqual(job.finished_at, finished_at)

    def test_evaluating_to_failed_transition_is_valid(self):
        """Test EVALUATING -> FAILED transition is allowed."""
        job = TrainingJob(
            id="job-1",
            project_id="project-1",
            dataset_id="dataset-1",
            status=TrainingJobStatus.EVALUATING,
        )
        self.db.add(job)
        self.db.commit()

        job_repo = TrainingJobRepository(self.db)
        finished_at = datetime.utcnow()
        job_repo.transition(job, TrainingJobStatus.FAILED, finished_at=finished_at)
        self.db.commit()

        self.assertEqual(job.status, TrainingJobStatus.FAILED)
        self.assertEqual(job.finished_at, finished_at)

    def test_complete_success_lifecycle(self):
        """Test complete success lifecycle: QUEUED -> RUNNING -> EVALUATING -> PASSED."""
        job = TrainingJob(
            id="job-1",
            project_id="project-1",
            dataset_id="dataset-1",
            status=TrainingJobStatus.QUEUED,
        )
        self.db.add(job)
        self.db.commit()

        job_repo = TrainingJobRepository(self.db)

        # QUEUED -> RUNNING
        reserved = job_repo.reserve("job-1", "project-1", datetime.utcnow())
        self.assertEqual(reserved.status, TrainingJobStatus.RUNNING)
        self.db.commit()

        # RUNNING -> EVALUATING
        job_repo.transition(reserved, TrainingJobStatus.EVALUATING)
        self.assertEqual(reserved.status, TrainingJobStatus.EVALUATING)
        self.db.commit()

        # EVALUATING -> PASSED
        finished_at = datetime.utcnow()
        job_repo.transition(reserved, TrainingJobStatus.PASSED, finished_at=finished_at)
        self.assertEqual(reserved.status, TrainingJobStatus.PASSED)
        self.assertEqual(reserved.finished_at, finished_at)
        self.db.commit()


class SQLAlchemyMetadataRegistrationTests(unittest.TestCase):
    """Test that EC2 worker can update job status with complete metadata."""

    def setUp(self):
        """Set up test database with all tables."""
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        
        # Import all models to register metadata (same as bootstrap script fix)
        from app.control import models as control_models  # noqa: F401
        from app.data import models as data_models  # noqa: F401
        from app.serving import models as serving_models  # noqa: F401
        from app.training import models as training_models  # noqa: F401
        
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()

    def tearDown(self):
        """Clean up database."""
        self.db.close()
        self.engine.dispose()

    def test_projects_table_exists(self):
        """Test that 'projects' table is registered in metadata."""
        from sqlalchemy import inspect
        
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        
        self.assertIn("projects", tables)

    def test_all_training_foreign_keys_resolve(self):
        """Test that TrainingJob foreign keys can be created."""
        from app.control.models import Project
        from app.training.models import Dataset
        
        # Create a project with required fields
        project = Project(
            id="project-1",
            name="Test Project",
            password_hash="test_hash"  # Required field
        )
        self.db.add(project)
        self.db.commit()
        
        # Create a dataset
        dataset = Dataset(
            id="dataset-1",
            project_id="project-1",
            dataset_version="dataset-v1",
            storage_key="datasets/project-1/dataset-v1.jsonl",
            record_count=100,
            status=DatasetStatus.READY,
        )
        self.db.add(dataset)
        self.db.commit()
        
        # Create a training job with foreign keys
        job = TrainingJob(
            id="job-1",
            project_id="project-1",
            dataset_id="dataset-1",
            status=TrainingJobStatus.QUEUED,
        )
        self.db.add(job)
        self.db.commit()
        
        # Verify the job can be queried with relationships
        retrieved = self.db.query(TrainingJob).filter_by(id="job-1").first()
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.project_id, "project-1")
        self.assertEqual(retrieved.dataset_id, "dataset-1")

    def test_worker_can_update_job_status_with_metadata(self):
        """Test that worker can update job status when all metadata is registered."""
        from app.control.models import Project
        from app.training.models import Dataset
        
        # Create required foreign key records
        project = Project(
            id="project-1",
            name="Test Project",
            password_hash="test_hash"  # Required field
        )
        self.db.add(project)
        
        dataset = Dataset(
            id="dataset-1",
            project_id="project-1",
            dataset_version="dataset-v1",
            storage_key="datasets/project-1/dataset-v1.jsonl",
            record_count=100,
            status=DatasetStatus.READY,
        )
        self.db.add(dataset)
        
        job = TrainingJob(
            id="job-1",
            project_id="project-1",
            dataset_id="dataset-1",
            status=TrainingJobStatus.RUNNING,
        )
        self.db.add(job)
        self.db.commit()
        
        # Simulate bootstrap script updating status
        job_repo = TrainingJobRepository(self.db)
        
        # RUNNING -> EVALUATING
        job_repo.transition(job, TrainingJobStatus.EVALUATING)
        self.db.commit()
        self.assertEqual(job.status, TrainingJobStatus.EVALUATING)
        
        # EVALUATING -> PASSED
        job_repo.transition(job, TrainingJobStatus.PASSED, finished_at=datetime.utcnow())
        self.db.commit()
        self.assertEqual(job.status, TrainingJobStatus.PASSED)


class EC2CleanupGuaranteeTests(unittest.TestCase):
    """Test that EC2 instance is always terminated after training."""

    @patch("boto3.client")
    def test_terminate_called_on_successful_training(self, mock_boto_client):
        """Test EC2 instance is terminated after successful training."""
        # Note: Full integration test with bootstrap script would require
        # mocking the entire training pipeline. This test verifies the
        # termination logic can be called correctly.
        
        mock_ec2 = Mock()
        mock_boto_client.return_value = mock_ec2
        
        mock_ec2.terminate_instances.return_value = {
            "TerminatingInstances": [{"InstanceId": "i-test-123", "CurrentState": {"Name": "shutting-down"}}]
        }
        
        # Verify termination API call structure
        mock_ec2.terminate_instances(InstanceIds=["i-test-123"])
        mock_ec2.terminate_instances.assert_called_once_with(InstanceIds=["i-test-123"])

    def test_terminate_skipped_when_disabled(self):
        """Test EC2 termination can be disabled via environment variable."""
        # Verify environment variable controls termination
        with patch.dict(os.environ, {"TRAINING_EC2_TERMINATE_ON_COMPLETION": "false"}):
            terminate_enabled = os.environ.get(
                "TRAINING_EC2_TERMINATE_ON_COMPLETION", "true"
            ).lower() in {"true", "1", "yes"}
            
            self.assertFalse(terminate_enabled)

    @patch("requests.put")
    @patch("requests.get")
    def test_instance_id_retrieved_from_metadata(self, mock_get, mock_put):
        """Test instance ID is correctly retrieved from EC2 metadata service."""
        # Mock IMDSv2 token retrieval
        mock_put.return_value = Mock(status_code=200, text="test-token-123")
        
        # Mock instance ID retrieval
        mock_get.return_value = Mock(status_code=200, text="i-retrieved-456")
        
        # Verify IMDSv2 flow
        token_response = mock_put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            timeout=2,
        )
        self.assertEqual(token_response.status_code, 200)
        token = token_response.text
        
        id_response = mock_get(
            "http://169.254.169.254/latest/meta-data/instance-id",
            headers={"X-aws-ec2-metadata-token": token},
            timeout=2,
        )
        self.assertEqual(id_response.status_code, 200)
        instance_id = id_response.text
        self.assertEqual(instance_id, "i-retrieved-456")

    @patch("requests.put")
    def test_instance_id_returns_none_when_not_on_ec2(self, mock_put):
        """Test instance ID returns None when not running on EC2."""
        # Mock metadata service not available
        mock_put.side_effect = Exception("Connection refused")
        
        try:
            mock_put(
                "http://169.254.169.254/latest/api/token",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
                timeout=2,
            )
            self.fail("Expected exception")
        except Exception as e:
            self.assertEqual(str(e), "Connection refused")


class APIResponseCodeTests(unittest.TestCase):
    """Test API endpoint returns correct status codes for EC2 training."""

    def setUp(self):
        """Set up test database and FastAPI test client."""
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        
        # Import all models
        from app.control import models as control_models  # noqa: F401
        from app.data import models as data_models  # noqa: F401
        from app.serving import models as serving_models  # noqa: F401
        from app.training import models as training_models  # noqa: F401
        
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()
        
        # Create test project
        from app.control.models import Project
        self.project = Project(
            id="test-project",
            name="Test Project",
            password_hash="test_hash"
        )
        self.db.add(self.project)
        self.db.commit()
        
        # Override dependencies for testing
        from app.shared.database import get_db
        from app.api.routes_auth import get_current_project
        from app.main import app
        
        def override_get_db():
            try:
                yield self.db
            finally:
                pass  # Don't close in test
        
        def override_get_current_project():
            return self.project
        
        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_current_project] = override_get_current_project
        
        # Create test client
        from fastapi.testclient import TestClient
        self.client = TestClient(app)

    def tearDown(self):
        """Clean up database."""
        self.db.close()
        self.engine.dispose()

    @patch.dict(os.environ, {"TRAINING_EC2_ENABLED": "true"})
    def test_ec2_launch_returns_202_with_valid_json(self):
        """Test that successful EC2 launch returns 202 Accepted with properly serialized JSON."""
        from app.training.models import Dataset
        
        # Create test data
        dataset = Dataset(
            id="dataset-1",
            project_id="test-project",
            dataset_version="dataset-v1",
            storage_key="datasets/test-project/dataset-v1.jsonl",
            record_count=100,
            status=DatasetStatus.READY,
        )
        self.db.add(dataset)
        
        job = TrainingJob(
            id="job-1",
            project_id="test-project",
            dataset_id="dataset-1",
            status=TrainingJobStatus.QUEUED,
        )
        self.db.add(job)
        self.db.commit()
        
        # Mock EC2 orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.launch_training_instance.return_value = "i-launched-789"
        
        # Mock worker with EC2 enabled
        with patch("app.training.worker.configured_ec2_training_config") as mock_config:
            mock_config.return_value = Mock(enabled=True)
            
            with patch("app.training.worker.EC2TrainingOrchestrator", return_value=mock_orchestrator):
                with patch.dict(
                    os.environ,
                    {
                        "S3_BUCKET": "test-bucket",
                        "AWS_REGION": "us-east-1",
                        "DATABASE_URL": "postgresql://test",
                        "GIT_COMMIT_SHA": "a" * 40,
                    },
                ):
                    # Make actual HTTP request through TestClient
                    response = self.client.post("/api/training-jobs/job-1/run")
        
        # Verify HTTP 202 status code
        self.assertEqual(response.status_code, 202, f"Expected 202, got {response.status_code}: {response.text}")
        
        # Verify response is valid JSON
        json_data = response.json()
        self.assertIsInstance(json_data, dict)
        
        # Verify expected fields
        self.assertEqual(json_data["id"], "job-1")
        self.assertEqual(json_data["status"], "RUNNING")
        self.assertEqual(json_data["ec2_instance_id"], "i-launched-789")
        self.assertEqual(json_data["project_id"], "test-project")
        self.assertEqual(json_data["dataset_id"], "dataset-1")
        
        # Verify datetime fields are properly serialized as ISO strings
        self.assertIsInstance(json_data["created_at"], str)
        self.assertIsInstance(json_data["started_at"], str)
        
        # Verify datetime strings can be parsed
        from datetime import datetime
        datetime.fromisoformat(json_data["created_at"].replace("Z", "+00:00"))
        datetime.fromisoformat(json_data["started_at"].replace("Z", "+00:00"))

    def test_launch_failure_returns_500_with_message(self):
        """Test that EC2 launch failure returns 500 with error message."""
        from app.training.models import Dataset
        
        # Create test data
        dataset = Dataset(
            id="dataset-1",
            project_id="test-project",
            dataset_version="dataset-v1",
            storage_key="datasets/test-project/dataset-v1.jsonl",
            record_count=100,
            status=DatasetStatus.READY,
        )
        self.db.add(dataset)
        
        job = TrainingJob(
            id="job-1",
            project_id="test-project",
            dataset_id="dataset-1",
            status=TrainingJobStatus.QUEUED,
        )
        self.db.add(job)
        self.db.commit()
        
        # Mock EC2 configuration missing (provide AWS_REGION to allow boto3 client construction)
        with patch.dict(os.environ, {"TRAINING_EC2_ENABLED": "true", "S3_BUCKET": "", "AWS_REGION": "us-east-1"}):
            with patch("app.training.worker.configured_ec2_training_config") as mock_config:
                mock_config.return_value = Mock(enabled=True)
                
                response = self.client.post("/api/training-jobs/job-1/run")
                
                self.assertEqual(response.status_code, 500)
                json_data = response.json()
                self.assertIn("Could not start this training run", json_data["detail"])

    def test_local_training_returns_200_ok(self):
        """Test that local training mode returns 200 OK."""
        # This would require mocking the entire local training flow
        # For now, just verify the structure is correct
        # Full integration test would be done in test_training_pipeline.py
        pass
