"""
Tests for incident clustering logic.

Verifies that local evaluation clustering matches production behavior.
"""

import pytest
from datetime import datetime, timedelta
from incident_signatures import normalize_message, generate_signature, extract_exception_type
from incident_clustering import ParsedLog, IncidentClusterer, cluster_logs


class TestMessageNormalization:
    """Test that message normalization matches production."""
    
    def test_uuid_normalization(self):
        msg = "Request 550e8400-e29b-41d4-a716-446655440000 failed"
        assert normalize_message(msg) == "request uuid failed"
    
    def test_number_normalization(self):
        msg = "Connection timeout after 5000ms"
        assert normalize_message(msg) == "connection timeout after nms"
    
    def test_preserve_http_status(self):
        msg = "HTTP 503 service unavailable"
        assert normalize_message(msg) == "http 503 service unavailable"
        
        msg2 = "HTTP 404 not found"
        assert normalize_message(msg2) == "http 404 not found"
    
    def test_memory_size_normalization(self):
        msg = "Heap usage 1907MB exceeds limit"
        assert normalize_message(msg) == "heap usage nmb exceeds limit"
    
    def test_order_id_normalization(self):
        msg = "Order ORD-12345 failed"
        assert normalize_message(msg) == "order ord-n failed"
    
    def test_identical_patterns_same_signature(self):
        """Verify that identical error patterns generate same signature."""
        msg1 = "Connection timeout after 5000ms"
        msg2 = "Connection timeout after 8000ms"
        
        assert normalize_message(msg1) == normalize_message(msg2)


class TestSignatureGeneration:
    """Test signature generation."""
    
    def test_signature_includes_service(self):
        sig1 = generate_signature("payment-api", "ERROR", "connection failed", None)
        sig2 = generate_signature("order-api", "ERROR", "connection failed", None)
        
        # Different services = different signatures
        assert sig1 != sig2
    
    def test_signature_includes_level(self):
        sig1 = generate_signature("payment-api", "ERROR", "connection failed", None)
        sig2 = generate_signature("payment-api", "WARN", "connection failed", None)
        
        # Different levels = different signatures
        assert sig1 != sig2
    
    def test_signature_includes_exception(self):
        sig1 = generate_signature("payment-api", "ERROR", "failed", "TimeoutException")
        sig2 = generate_signature("payment-api", "ERROR", "failed", "NullPointerException")
        
        # Different exceptions = different signatures
        assert sig1 != sig2
    
    def test_signature_deterministic(self):
        """Signature must be deterministic."""
        sig1 = generate_signature("payment-api", "ERROR", "connection failed", None)
        sig2 = generate_signature("payment-api", "ERROR", "connection failed", None)
        
        assert sig1 == sig2


class TestExceptionExtraction:
    """Test exception type extraction from messages."""
    
    def test_extract_timeout_exception(self):
        msg = "PaymentGatewayTimeout: Stripe did not respond"
        assert extract_exception_type(msg) == "PaymentGatewayTimeout"
    
    def test_extract_null_pointer(self):
        msg = "NullPointerException at line 42"
        assert extract_exception_type(msg) == "NullPointerException"
    
    def test_no_exception(self):
        msg = "Request completed successfully"
        assert extract_exception_type(msg) is None


class TestIncidentClustering:
    """Test incident clustering logic."""
    
    def create_log(self, timestamp: datetime, service: str, level: str, message: str) -> ParsedLog:
        """Helper to create a ParsedLog."""
        exception_type = extract_exception_type(message)
        raw = f"{timestamp.isoformat()}Z {level} [{service}] {message}"
        
        return ParsedLog(
            timestamp=timestamp,
            level=level,
            service=service,
            message=message,
            exception_type=exception_type,
            raw=raw
        )
    
    def test_same_signature_same_incident(self):
        """Logs with same signature within time window -> same incident."""
        base_time = datetime(2026, 9, 25, 10, 0, 0)
        
        logs = [
            self.create_log(base_time, "payment-api", "ERROR", "connection timeout after 5000ms"),
            self.create_log(base_time + timedelta(seconds=30), "payment-api", "ERROR", "connection timeout after 8000ms"),
        ]
        
        incidents = cluster_logs(logs)
        
        assert len(incidents) == 1
        assert incidents[0].count == 2
        assert incidents[0].service == "payment-api"
    
    def test_different_signature_different_incident(self):
        """Different error signatures -> different incidents."""
        base_time = datetime(2026, 9, 25, 10, 0, 0)
        
        logs = [
            self.create_log(base_time, "payment-api", "ERROR", "connection timeout"),
            self.create_log(base_time + timedelta(seconds=10), "payment-api", "ERROR", "null pointer exception"),
        ]
        
        incidents = cluster_logs(logs)
        
        assert len(incidents) == 2
    
    def test_time_window_creates_separate_incident(self):
        """Logs with same signature but outside time window -> separate incidents."""
        base_time = datetime(2026, 9, 25, 10, 0, 0)
        
        logs = [
            self.create_log(base_time, "payment-api", "ERROR", "connection timeout"),
            self.create_log(base_time + timedelta(minutes=5), "payment-api", "ERROR", "connection timeout"),
        ]
        
        incidents = cluster_logs(logs, time_window_minutes=2)
        
        assert len(incidents) == 2
        assert incidents[0].count == 1
        assert incidents[1].count == 1
    
    def test_multiple_incidents_per_service(self):
        """ONE SERVICE CAN PRODUCE MULTIPLE INCIDENTS (critical requirement)."""
        base_time = datetime(2026, 9, 25, 10, 0, 0)
        
        logs = [
            # Kafka lag incident
            self.create_log(base_time, "notification-worker", "WARN", "kafka consumer lag topic=emails lag=5000"),
            self.create_log(base_time + timedelta(seconds=30), "notification-worker", "WARN", "kafka consumer lag topic=emails lag=8000"),
            
            # RabbitMQ incident (different signature)
            self.create_log(base_time + timedelta(seconds=45), "notification-worker", "ERROR", "RabbitMQ queue backlog"),
            
            # Container OOM (different signature)
            self.create_log(base_time + timedelta(minutes=1), "notification-worker", "CRITICAL", "OOMKilled container restarting"),
        ]
        
        incidents = cluster_logs(logs)
        
        # Should create 3 distinct incidents from one service
        assert len(incidents) == 3
        
        # All from same service
        assert all(inc.service == "notification-worker" for inc in incidents)
        
        # Different signatures
        signatures = [inc.signature for inc in incidents]
        assert len(set(signatures)) == 3
    
    def test_normal_logs_not_incidents(self):
        """INFO logs should create incidents too (for completeness), but can be filtered later."""
        base_time = datetime(2026, 9, 25, 10, 0, 0)
        
        logs = [
            self.create_log(base_time, "user-service", "INFO", "heartbeat ok"),
            self.create_log(base_time + timedelta(seconds=30), "user-service", "INFO", "heartbeat ok"),
        ]
        
        incidents = cluster_logs(logs)
        
        # Clustering creates incident, but disposition should be NO_ACTION
        assert len(incidents) == 1
        assert incidents[0].count == 2
    
    def test_cross_service_separate_incidents(self):
        """Same error signature from different services -> different incidents."""
        base_time = datetime(2026, 9, 25, 10, 0, 0)
        
        logs = [
            self.create_log(base_time, "payment-api", "ERROR", "database connection failed"),
            self.create_log(base_time + timedelta(seconds=10), "order-api", "ERROR", "database connection failed"),
        ]
        
        incidents = cluster_logs(logs)
        
        # Different services -> different signatures -> different incidents
        assert len(incidents) == 2
    
    def test_sample_line_limit(self):
        """Verify sample lines are limited to MAX_SAMPLES."""
        base_time = datetime(2026, 9, 25, 10, 0, 0)
        
        # Create 20 logs with same signature
        logs = [
            self.create_log(base_time + timedelta(seconds=i*5), "payment-api", "ERROR", "connection timeout")
            for i in range(20)
        ]
        
        incidents = cluster_logs(logs)
        
        assert len(incidents) == 1
        assert incidents[0].count == 20
        assert len(incidents[0].sample_lines) <= 10  # MAX_SAMPLES
    
    def test_incident_boundaries(self):
        """Verify first_seen and last_seen are tracked correctly."""
        base_time = datetime(2026, 9, 25, 10, 0, 0)
        
        logs = [
            self.create_log(base_time, "payment-api", "ERROR", "connection timeout"),
            self.create_log(base_time + timedelta(seconds=60), "payment-api", "ERROR", "connection timeout"),
        ]
        
        incidents = cluster_logs(logs)
        
        assert len(incidents) == 1
        assert incidents[0].first_seen == base_time
        assert incidents[0].last_seen == base_time + timedelta(seconds=60)


class TestClustererStatistics:
    """Test clustering statistics."""
    
    def test_statistics_multi_incident_detection(self):
        """Verify statistics correctly identify services with multiple incidents."""
        base_time = datetime(2026, 9, 25, 10, 0, 0)
        
        logs = [
            ParsedLog(base_time, "ERROR", "svc-a", "error1", None, "raw1"),
            ParsedLog(base_time + timedelta(seconds=10), "ERROR", "svc-a", "error2", None, "raw2"),
            ParsedLog(base_time + timedelta(seconds=20), "ERROR", "svc-b", "error3", None, "raw3"),
        ]
        
        clusterer = IncidentClusterer()
        for idx, log in enumerate(logs):
            clusterer.process_log(log, idx)
        
        clusterer.close_all()
        stats = clusterer.get_statistics()
        
        assert stats["total_incidents"] == 3
        assert stats["unique_services"] == 2
        assert "svc-a" in stats["multi_incident_services"]
        assert "svc-b" not in stats["multi_incident_services"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
