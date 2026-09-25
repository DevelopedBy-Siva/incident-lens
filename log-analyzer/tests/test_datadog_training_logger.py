"""
Tests for Datadog training telemetry logging.

Verifies that:
- Training logs contain proper correlation attributes
- Progress events are emitted with correct structure
- Datadog delivery failures do not fail training
- Credentials are never included in log messages
- Training works when Datadog configuration is absent
- Local logging remains functional
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time

from app.training.datadog_logger import (
    DatadogLogger,
    TrainingLogContext,
    create_training_logger,
    DATADOG_LOG_INTAKE_URLS,
    SERVICE_NAME,
    SOURCE_NAME,
)


class TrainingLogContextTests(unittest.TestCase):
    """Test TrainingLogContext correlation attributes."""
    
    def test_context_includes_required_attributes(self):
        """Test that context includes all required correlation attributes."""
        context = TrainingLogContext(
            project_id="project-123",
            training_job_id="job-456",
            dataset_id="dataset-789",
            ec2_instance_id="i-abc123",
            environment="production",
        )
        
        tags = context.to_tags()
        
        self.assertEqual(tags["project_id"], "project-123")
        self.assertEqual(tags["training_job_id"], "job-456")
        self.assertEqual(tags["dataset_id"], "dataset-789")
        self.assertEqual(tags["ec2_instance_id"], "i-abc123")
        self.assertEqual(tags["environment"], "production")
    
    def test_context_without_instance_id(self):
        """Test context when EC2 instance ID is not available."""
        context = TrainingLogContext(
            project_id="project-123",
            training_job_id="job-456",
            dataset_id="dataset-789",
            ec2_instance_id=None,
        )
        
        tags = context.to_tags()
        
        self.assertNotIn("ec2_instance_id", tags)
        self.assertEqual(tags["environment"], "production")


class DatadogLoggerTests(unittest.TestCase):
    """Test DatadogLogger functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.context = TrainingLogContext(
            project_id="project-test",
            training_job_id="job-test",
            dataset_id="dataset-test",
            ec2_instance_id="i-test123",
        )
    
    def test_logger_disabled_without_credentials(self):
        """Test that logger is disabled when credentials are missing."""
        logger = DatadogLogger(None, None, self.context)
        self.assertFalse(logger.enabled)
        
        logger = DatadogLogger("api-key", None, self.context)
        self.assertFalse(logger.enabled)
        
        logger = DatadogLogger(None, "datadoghq.com", self.context)
        self.assertFalse(logger.enabled)
    
    def test_logger_disabled_for_unsupported_site(self):
        """Test that logger is disabled for unsupported Datadog site."""
        logger = DatadogLogger(
            "api-key",
            "unsupported.example.com",
            self.context,
        )
        
        self.assertFalse(logger.enabled)
    
    def test_logger_enabled_with_valid_credentials(self):
        """Test that logger is enabled with valid credentials and site."""
        logger = DatadogLogger(
            "api-key",
            "datadoghq.com",
            self.context,
        )
        
        self.assertTrue(logger.enabled)
        self.assertIsNotNone(logger._session)
        self.assertEqual(
            logger._intake_url,
            DATADOG_LOG_INTAKE_URLS["datadoghq.com"]
        )
    
    def test_logger_normalizes_site_url(self):
        """Test that logger normalizes site URLs correctly."""
        logger = DatadogLogger(
            "api-key",
            "HTTPS://DATADOGHQ.EU/",
            self.context,
        )
        
        self.assertTrue(logger.enabled)
        self.assertEqual(
            logger._intake_url,
            DATADOG_LOG_INTAKE_URLS["datadoghq.eu"]
        )
    
    @patch("app.training.datadog_logger.requests.Session")
    def test_info_event_structure(self, mock_session_class):
        """Test that info events have correct structure and attributes."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        logger = DatadogLogger("api-key", "datadoghq.com", self.context)
        logger.info("training_started", {"base_model": "Qwen3.5-4B"})
        
        # Verify POST was called
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        
        # Verify URL
        self.assertEqual(
            call_args[0][0],
            DATADOG_LOG_INTAKE_URLS["datadoghq.com"]
        )
        
        # Verify payload structure
        payload = call_args[1]["json"]
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 1)
        
        log_entry = payload[0]
        self.assertEqual(log_entry["ddsource"], SOURCE_NAME)
        self.assertEqual(log_entry["service"], SERVICE_NAME)
        self.assertEqual(log_entry["level"], "info")
        self.assertEqual(log_entry["event"], "training_started")
        self.assertEqual(log_entry["project_id"], "project-test")
        self.assertEqual(log_entry["training_job_id"], "job-test")
        self.assertEqual(log_entry["dataset_id"], "dataset-test")
        self.assertEqual(log_entry["base_model"], "Qwen3.5-4B")
        self.assertIn("timestamp", log_entry)
        self.assertIn("message", log_entry)
    
    @patch("app.training.datadog_logger.requests.Session")
    def test_progress_event_structure(self, mock_session_class):
        """Test that progress events include proper training metrics."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        logger = DatadogLogger("api-key", "datadoghq.com", self.context)
        logger.progress(
            current_step=50,
            total_steps=100,
            training_loss=0.234567,
            elapsed_seconds=123.456,
        )
        
        mock_session.post.assert_called_once()
        payload = mock_session.post.call_args[1]["json"]
        log_entry = payload[0]
        
        self.assertEqual(log_entry["event"], "training_progress")
        self.assertEqual(log_entry["current_step"], 50)
        self.assertEqual(log_entry["total_steps"], 100)
        self.assertEqual(log_entry["progress_percent"], 50.0)
        self.assertEqual(log_entry["training_loss"], 0.234567)
        self.assertEqual(log_entry["elapsed_seconds"], 123.5)
        
        # Verify visible message contains progress details
        self.assertIn("50.0%", log_entry["message"])
        self.assertIn("50/100", log_entry["message"])
        self.assertIn("Training progress:", log_entry["message"])
    
    @patch("app.training.datadog_logger.requests.Session")
    def test_progress_message_at_100_percent(self, mock_session_class):
        """Test progress message format at 100% completion."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        logger = DatadogLogger("api-key", "datadoghq.com", self.context)
        logger.progress(
            current_step=1786,
            total_steps=1786,
        )
        
        mock_session.post.assert_called_once()
        payload = mock_session.post.call_args[1]["json"]
        log_entry = payload[0]
        
        self.assertEqual(log_entry["progress_percent"], 100.0)
        self.assertIn("100.0%", log_entry["message"])
        self.assertIn("1786/1786", log_entry["message"])
    
    @patch("app.training.datadog_logger.requests.Session")
    def test_error_event_structure(self, mock_session_class):
        """Test that error events are logged with correct level."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        logger = DatadogLogger("api-key", "datadoghq.com", self.context)
        logger.error("training_failed", {
            "error_type": "OOMError",
            "error_message": "Out of memory",
        })
        
        mock_session.post.assert_called_once()
        payload = mock_session.post.call_args[1]["json"]
        log_entry = payload[0]
        
        self.assertEqual(log_entry["level"], "error")
        self.assertEqual(log_entry["event"], "training_failed")
        self.assertEqual(log_entry["error_type"], "OOMError")
    
    @patch("app.training.datadog_logger.requests.Session")
    def test_none_values_filtered_from_attributes(self, mock_session_class):
        """Test that None values are filtered from log attributes."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        logger = DatadogLogger("api-key", "datadoghq.com", self.context)
        logger.info("test_event", {
            "value_present": "exists",
            "value_none": None,
            "value_zero": 0,
        })
        
        payload = mock_session.post.call_args[1]["json"]
        log_entry = payload[0]
        
        self.assertIn("value_present", log_entry)
        self.assertIn("value_zero", log_entry)
        self.assertNotIn("value_none", log_entry)
    
    @patch("app.training.datadog_logger.requests.Session")
    def test_non_serializable_values_filtered(self, mock_session_class):
        """Test that non-serializable values are filtered from attributes."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        logger = DatadogLogger("api-key", "datadoghq.com", self.context)
        logger.info("test_event", {
            "string_value": "ok",
            "int_value": 42,
            "float_value": 3.14,
            "bool_value": True,
            "list_value": [1, 2, 3],  # Should be filtered
            "dict_value": {"key": "val"},  # Should be filtered
        })
        
        payload = mock_session.post.call_args[1]["json"]
        log_entry = payload[0]
        
        self.assertIn("string_value", log_entry)
        self.assertIn("int_value", log_entry)
        self.assertIn("float_value", log_entry)
        self.assertIn("bool_value", log_entry)
        self.assertNotIn("list_value", log_entry)
        self.assertNotIn("dict_value", log_entry)
    
    @patch("app.training.datadog_logger.requests.Session")
    def test_timeout_does_not_raise_exception(self, mock_session_class):
        """Test that Datadog timeout does not raise exception."""
        import requests
        
        mock_session = MagicMock()
        mock_session.post.side_effect = requests.Timeout("Connection timeout")
        mock_session_class.return_value = mock_session
        
        logger = DatadogLogger("api-key", "datadoghq.com", self.context)
        
        # Should not raise exception
        logger.info("test_event")
        
        # Verify the log was attempted
        mock_session.post.assert_called_once()
    
    @patch("app.training.datadog_logger.requests.Session")
    def test_network_error_does_not_raise_exception(self, mock_session_class):
        """Test that network errors do not raise exception."""
        import requests
        
        mock_session = MagicMock()
        mock_session.post.side_effect = requests.ConnectionError("Network unreachable")
        mock_session_class.return_value = mock_session
        
        logger = DatadogLogger("api-key", "datadoghq.com", self.context)
        
        # Should not raise exception
        logger.error("training_failed", {"error": "test"})
        
        mock_session.post.assert_called_once()
    
    @patch("app.training.datadog_logger.requests.Session")
    def test_unexpected_error_does_not_raise_exception(self, mock_session_class):
        """Test that unexpected errors do not raise exception."""
        mock_session = MagicMock()
        mock_session.post.side_effect = RuntimeError("Unexpected error")
        mock_session_class.return_value = mock_session
        
        logger = DatadogLogger("api-key", "datadoghq.com", self.context)
        
        # Should not raise exception
        logger.warning("test_warning")
        
        mock_session.post.assert_called_once()
    
    @patch("app.training.datadog_logger.requests.Session")
    def test_disabled_logger_does_not_send_logs(self, mock_session_class):
        """Test that disabled logger does not attempt to send logs."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        logger = DatadogLogger(None, None, self.context, enabled=False)
        
        logger.info("test_event")
        logger.error("test_error")
        logger.progress(10, 100)
        
        # Verify no HTTP calls were made
        mock_session.post.assert_not_called()
    
    @patch("app.training.datadog_logger.requests.Session")
    def test_session_closed_properly(self, mock_session_class):
        """Test that HTTP session is closed when logger is closed."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        logger = DatadogLogger("api-key", "datadoghq.com", self.context)
        logger.close()
        
        mock_session.close.assert_called_once()
    
    @patch("app.training.datadog_logger.requests.Session")
    def test_context_manager_closes_session(self, mock_session_class):
        """Test that context manager properly closes session."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        with DatadogLogger("api-key", "datadoghq.com", self.context) as logger:
            logger.info("test_event")
        
        mock_session.close.assert_called_once()
    
    @patch("app.training.datadog_logger.requests.Session")
    def test_api_key_not_in_log_payload(self, mock_session_class):
        """Test that API key is never included in log payload."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        logger = DatadogLogger("secret-api-key", "datadoghq.com", self.context)
        logger.info("test_event", {"some_field": "some_value"})
        
        payload = mock_session.post.call_args[1]["json"]
        log_entry = payload[0]
        
        # Verify API key is not in payload
        payload_str = str(log_entry)
        self.assertNotIn("secret-api-key", payload_str)
        self.assertNotIn("api_key", log_entry)
        self.assertNotIn("datadog_api_key", log_entry)


class CreateTrainingLoggerTests(unittest.TestCase):
    """Test create_training_logger factory function."""
    
    @patch("app.training.datadog_logger.DatadogLogger")
    def test_factory_creates_logger_with_context(self, mock_logger_class):
        """Test that factory creates logger with proper context."""
        create_training_logger(
            api_key="test-key",
            site="datadoghq.com",
            project_id="proj-1",
            training_job_id="job-1",
            dataset_id="dataset-1",
            ec2_instance_id="i-instance123",
        )
        
        mock_logger_class.assert_called_once()
        call_kwargs = mock_logger_class.call_args.kwargs
        
        self.assertEqual(call_kwargs["api_key"], "test-key")
        self.assertEqual(call_kwargs["site"], "datadoghq.com")
        
        context = call_kwargs["context"]
        self.assertEqual(context.project_id, "proj-1")
        self.assertEqual(context.training_job_id, "job-1")
        self.assertEqual(context.dataset_id, "dataset-1")
        self.assertEqual(context.ec2_instance_id, "i-instance123")
    
    @patch("app.training.datadog_logger.DatadogLogger")
    def test_factory_creates_disabled_logger_without_credentials(self, mock_logger_class):
        """Test that factory creates disabled logger when credentials absent."""
        create_training_logger(
            api_key=None,
            site=None,
            project_id="proj-1",
            training_job_id="job-1",
            dataset_id="dataset-1",
        )
        
        call_args = mock_logger_class.call_args
        self.assertEqual(call_args[1]["enabled"], False)


class IntegrationTests(unittest.TestCase):
    """Integration tests for training telemetry."""
    
    @patch("app.training.datadog_logger.requests.Session")
    def test_complete_training_flow_with_datadog(self, mock_session_class):
        """Test complete training flow emits expected events."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        context = TrainingLogContext(
            project_id="project-integration",
            training_job_id="job-integration",
            dataset_id="dataset-integration",
        )
        
        with DatadogLogger("api-key", "datadoghq.com", context) as logger:
            # Simulate training lifecycle
            logger.info("training_worker_started")
            logger.info("repository_clone_completed")
            logger.info("dataset_download_started")
            logger.info("dataset_download_completed", {"dataset_size_mb": 10.5})
            logger.info("model_loading_started")
            logger.info("model_loading_completed")
            logger.info("training_started", {"epochs": 3})
            
            # Simulate progress updates
            for step in [25, 50, 75, 100]:
                logger.progress(step, 100, elapsed_seconds=step * 1.5)
            
            logger.info("training_completed", {"final_loss": 0.123})
            logger.info("artifact_upload_started")
            logger.info("artifact_upload_completed")
            logger.info("job_status_updated", {"status": "PASSED"})
        
        # Verify expected number of events (11 info + 4 progress = 15)
        self.assertEqual(mock_session.post.call_count, 15)
    
    @patch("app.training.datadog_logger.requests.Session")
    def test_training_continues_on_datadog_failure(self, mock_session_class):
        """Test that training continues even when Datadog fails."""
        import requests
        
        mock_session = MagicMock()
        # Simulate intermittent failures
        mock_session.post.side_effect = [
            None,  # Success
            requests.Timeout(),  # Failure
            None,  # Success
            requests.ConnectionError(),  # Failure
            None,  # Success
        ]
        mock_session_class.return_value = mock_session
        
        context = TrainingLogContext(
            project_id="project-resilient",
            training_job_id="job-resilient",
            dataset_id="dataset-resilient",
        )
        
        # Should not raise any exceptions
        with DatadogLogger("api-key", "datadoghq.com", context) as logger:
            logger.info("event_1")  # Success
            logger.info("event_2")  # Timeout
            logger.info("event_3")  # Success
            logger.info("event_4")  # ConnectionError
            logger.info("event_5")  # Success
        
        # All attempts were made despite failures
        self.assertEqual(mock_session.post.call_count, 5)


if __name__ == "__main__":
    unittest.main()
