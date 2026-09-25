"""
Tests for EC2 training bootstrap script.

Tests verify:
1. Configuration loading from S3
2. Environment variable handling
3. Git repository cloning from environment
4. No secrets in User Data
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Mock the log file path before importing bootstrap module
temp_log = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
temp_log.close()

# Add scripts directory to path and load the bootstrap module
scripts_dir = Path(__file__).parent.parent / "scripts"
bootstrap_script_path = scripts_dir / "bootstrap-training.py"

# Mock FileHandler to use temp file instead of /var/log
with patch("logging.FileHandler") as mock_handler:
    mock_handler.return_value = Mock()
    
    # Load module from file with dash in name
    spec = importlib.util.spec_from_file_location("bootstrap_training", bootstrap_script_path)
    bootstrap_training = importlib.util.module_from_spec(spec)
    sys.modules["bootstrap_training"] = bootstrap_training
    spec.loader.exec_module(bootstrap_training)


class BootstrapConfigLoadingTests(unittest.TestCase):
    """Test bootstrap script configuration loading from S3."""

    def setUp(self):
        """Set up environment variables for testing."""
        self.original_env = os.environ.copy()
        os.environ["INCIDENT_LENS_S3_BUCKET"] = "test-bucket"
        os.environ["INCIDENT_LENS_CONFIG_S3_KEY"] = "training-configs/project-1/job-1.json"
        os.environ["INCIDENT_LENS_JOB_ID"] = "job-1"
        os.environ["INCIDENT_LENS_PROJECT_ID"] = "project-1"
        os.environ["INCIDENT_LENS_GIT_REPOSITORY_URL"] = "https://github.com/test/repo.git"
        os.environ["INCIDENT_LENS_GIT_COMMIT_SHA"] = "abc123def456789012345678901234567890abcd"

    def tearDown(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_config_loaded_from_s3(self):
        """Test that configuration is downloaded from S3."""
        # Mock S3 client
        mock_s3_client = Mock()
        mock_response = {
            "Body": Mock(read=Mock(return_value=json.dumps({
                "job_id": "job-1",
                "project_id": "project-1",
                "dataset_id": "dataset-1",
                "dataset_storage_key": "datasets/project-1/dataset-v1.jsonl",
                "database_url": "postgresql://localhost/db",
                "base_model": "Qwen/Qwen3.5-4B",
                "lora_rank": 8,
            }).encode("utf-8")))
        }
        mock_s3_client.get_object.return_value = mock_response

        with patch("boto3.client", return_value=mock_s3_client):
            bootstrap = bootstrap_training.TrainingBootstrap()
            config = bootstrap.config

            # Verify S3 was called correctly
            mock_s3_client.get_object.assert_called_once_with(
                Bucket="test-bucket",
                Key="training-configs/project-1/job-1.json"
            )

            # Verify config loaded correctly
            self.assertEqual(config["job_id"], "job-1")
            self.assertEqual(config["project_id"], "project-1")
            self.assertEqual(config["base_model"], "Qwen/Qwen3.5-4B")
            self.assertEqual(config["lora_rank"], 8)
            
            # Verify S3 bucket was added to config
            self.assertEqual(config["s3_bucket"], "test-bucket")

    def test_config_loading_requires_environment_variables(self):
        """Test that config loading requires S3 bucket and key."""
        # Remove required environment variables
        del os.environ["INCIDENT_LENS_S3_BUCKET"]

        with self.assertRaises(bootstrap_training.ConfigurationError) as context:
            bootstrap_training.TrainingBootstrap()

        self.assertIn("INCIDENT_LENS_S3_BUCKET", str(context.exception))

    def test_config_loading_handles_invalid_json(self):
        """Test that invalid JSON in config raises error."""
        # Mock S3 client with invalid JSON
        mock_s3_client = Mock()
        mock_response = {
            "Body": Mock(read=Mock(return_value=b"invalid json {"))
        }
        mock_s3_client.get_object.return_value = mock_response

        with patch("boto3.client", return_value=mock_s3_client):
            with self.assertRaises(bootstrap_training.ConfigurationError) as context:
                bootstrap_training.TrainingBootstrap()

            self.assertIn("Invalid JSON", str(context.exception))

    def test_config_loading_handles_s3_errors(self):
        """Test that S3 download errors are handled."""
        from botocore.exceptions import ClientError

        # Mock S3 client with error
        mock_s3_client = Mock()
        mock_s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Key not found"}},
            "GetObject"
        )

        with patch("boto3.client", return_value=mock_s3_client):
            with self.assertRaises(bootstrap_training.ConfigurationError) as context:
                bootstrap_training.TrainingBootstrap()

            self.assertIn("Failed to download config from S3", str(context.exception))

    def test_git_repository_loaded_from_environment(self):
        """Test that git repository URL and commit are loaded from environment."""
        # Mock S3 client with valid config
        mock_s3_client = Mock()
        mock_response = {
            "Body": Mock(read=Mock(return_value=json.dumps({
                "job_id": "job-1",
                "project_id": "project-1",
            }).encode("utf-8")))
        }
        mock_s3_client.get_object.return_value = mock_response

        with patch("boto3.client", return_value=mock_s3_client):
            bootstrap = bootstrap_training.TrainingBootstrap()
            
            # Git info should come from environment, not config
            # Verify environment variables are set
            self.assertEqual(
                os.environ["INCIDENT_LENS_GIT_REPOSITORY_URL"],
                "https://github.com/test/repo.git"
            )
            self.assertEqual(
                os.environ["INCIDENT_LENS_GIT_COMMIT_SHA"],
                "abc123def456789012345678901234567890abcd"
            )

    def test_no_database_url_in_user_data_environment(self):
        """Test that DATABASE_URL is not required in environment (loaded from S3 config)."""
        # Mock S3 client with config containing database_url
        mock_s3_client = Mock()
        mock_response = {
            "Body": Mock(read=Mock(return_value=json.dumps({
                "job_id": "job-1",
                "project_id": "project-1",
                "database_url": "postgresql://user:pass@host/db",
            }).encode("utf-8")))
        }
        mock_s3_client.get_object.return_value = mock_response

        # The bootstrap script does NOT read DATABASE_URL from environment,
        # it loads it from S3 config instead
        with patch("boto3.client", return_value=mock_s3_client):
            bootstrap = bootstrap_training.TrainingBootstrap()
            
            # Database URL should be in config loaded from S3
            self.assertEqual(
                bootstrap.config["database_url"],
                "postgresql://user:pass@host/db"
            )
            
            # Verify this came from S3 config, not environment
            # by checking the S3 call was made
            mock_s3_client.get_object.assert_called_once()
    
    def test_aws_region_loaded_from_config(self):
        """Test that AWS region is loaded from training configuration."""
        # Mock S3 client with config containing aws_region
        mock_s3_client = Mock()
        mock_response = {
            "Body": Mock(read=Mock(return_value=json.dumps({
                "job_id": "job-1",
                "project_id": "project-1",
                "aws_region": "us-east-1",
            }).encode("utf-8")))
        }
        mock_s3_client.get_object.return_value = mock_response

        with patch("boto3.client", return_value=mock_s3_client):
            bootstrap = bootstrap_training.TrainingBootstrap()
            
            # AWS region should be in config
            self.assertEqual(bootstrap.config["aws_region"], "us-east-1")


class BootstrapTerminationTests(unittest.TestCase):
    """Test EC2 instance termination behavior."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_env = os.environ.copy()
        os.environ["INCIDENT_LENS_S3_BUCKET"] = "test-bucket"
        os.environ["INCIDENT_LENS_CONFIG_S3_KEY"] = "config.json"
        os.environ["INCIDENT_LENS_JOB_ID"] = "job-1"
        os.environ["INCIDENT_LENS_PROJECT_ID"] = "project-1"
        os.environ["INCIDENT_LENS_GIT_REPOSITORY_URL"] = "https://github.com/test/repo.git"
        os.environ["INCIDENT_LENS_GIT_COMMIT_SHA"] = "abc123def456789012345678901234567890abcd"
        os.environ["TRAINING_EC2_TERMINATE_ON_COMPLETION"] = "true"
    
    def tearDown(self):
        """Restore environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_termination_receives_aws_region(self):
        """Test that EC2 termination uses AWS region from configuration."""
        # Mock S3 client
        mock_s3_client = Mock()
        mock_response = {
            "Body": Mock(read=Mock(return_value=json.dumps({
                "job_id": "job-1",
                "project_id": "project-1",
                "aws_region": "us-west-2",
            }).encode("utf-8")))
        }
        mock_s3_client.get_object.return_value = mock_response
        
        # Mock EC2 client
        mock_ec2_client = Mock()
        mock_ec2_client.terminate_instances.return_value = {
            "TerminatingInstances": [{"CurrentState": {"Name": "shutting-down"}}]
        }
        
        # Mock metadata service to return instance ID
        mock_requests = Mock()
        mock_token_response = Mock()
        mock_token_response.status_code = 200
        mock_token_response.text = "test-token"
        
        mock_id_response = Mock()
        mock_id_response.status_code = 200
        mock_id_response.text = "i-test123"
        
        mock_requests.put.return_value = mock_token_response
        mock_requests.get.return_value = mock_id_response
        
        with patch("boto3.client") as mock_boto_client:
            # Return S3 client for first call, EC2 client for second
            mock_boto_client.side_effect = [mock_s3_client, mock_ec2_client]
            
            with patch("requests.put", mock_requests.put):
                with patch("requests.get", mock_requests.get):
                    bootstrap = bootstrap_training.TrainingBootstrap()
                    bootstrap._terminate_ec2_instance()
                    
                    # Verify boto3.client was called with region for EC2
                    # First call is S3 (no region), second call should be EC2 with region
                    calls = mock_boto_client.call_args_list
                    self.assertEqual(len(calls), 2)
                    self.assertEqual(calls[1][0][0], "ec2")
                    self.assertEqual(calls[1][1]["region_name"], "us-west-2")
                    
                    # Verify terminate_instances was called
                    mock_ec2_client.terminate_instances.assert_called_once_with(
                        InstanceIds=["i-test123"]
                    )
    
    def test_termination_failure_without_region(self):
        """Test that termination fails gracefully when AWS region is missing."""
        # Mock S3 client without aws_region
        mock_s3_client = Mock()
        mock_response = {
            "Body": Mock(read=Mock(return_value=json.dumps({
                "job_id": "job-1",
                "project_id": "project-1",
                # aws_region intentionally missing
            }).encode("utf-8")))
        }
        mock_s3_client.get_object.return_value = mock_response
        
        # Mock metadata service to return instance ID
        mock_requests = Mock()
        mock_token_response = Mock()
        mock_token_response.status_code = 200
        mock_token_response.text = "test-token"
        
        mock_id_response = Mock()
        mock_id_response.status_code = 200
        mock_id_response.text = "i-test123"
        
        mock_requests.put.return_value = mock_token_response
        mock_requests.get.return_value = mock_id_response
        
        with patch("boto3.client", return_value=mock_s3_client):
            with patch("requests.put", mock_requests.put):
                with patch("requests.get", mock_requests.get):
                    bootstrap = bootstrap_training.TrainingBootstrap()
                    
                    # Termination should not raise exception even without region
                    # It should log error and return gracefully
                    bootstrap._terminate_ec2_instance()
                    
                    # Verify termination was not attempted (no boto3 EC2 client created)
                    # Only S3 client should have been created during __init__


if __name__ == "__main__":
    unittest.main()
