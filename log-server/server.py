import asyncio
import base64
import json
import logging
import os
import random
import time
from collections import deque
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

LOKI_URL = os.getenv("LOKI_URL", "")
LOKI_USERNAME = os.getenv("LOKI_USERNAME", "")
LOKI_API_KEY = os.getenv("LOKI_API_KEY", "")
SERVICE_NAME = os.getenv("LOG_SERVICE_NAME", "log-server")

cors_origins = os.getenv("CORS_ORIGINS", "")
origins = [o.strip() for o in cors_origins.split(",") if o.strip()]

app = FastAPI(title="Log Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _loki_auth_header() -> str:
    """Basic auth header for Grafana Cloud Loki."""
    token = base64.b64encode(f"{LOKI_USERNAME}:{LOKI_API_KEY}".encode()).decode()
    return f"Basic {token}"


async def push_to_loki(lines: list[str], extra_labels: dict | None = None) -> bool:
    """
    Push a list of log lines to Loki in a single HTTP request
    """
    if not LOKI_URL or not LOKI_USERNAME or not LOKI_API_KEY:
        print("[LOKI] Credentials not set — dropping logs")
        return False

    labels = {"service": SERVICE_NAME, "env": "prod"}
    if extra_labels:
        labels.update(extra_labels)

    now_ns = str(int(time.time() * 1_000_000_000))
    values = [[now_ns, line] for line in lines]

    payload = {
        "streams": [
            {
                "stream": labels,
                "values": values,
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{LOKI_URL}/loki/api/v1/push",
                headers={
                    "Authorization": _loki_auth_header(),
                    "Content-Type": "application/json",
                },
                content=json.dumps(payload),
            )
            if response.status_code == 204:
                return True
            print(f"[LOKI] Push failed: {response.status_code} — {response.text}")
            return False
    except Exception as e:
        print(f"[LOKI] Push exception: {e}")
        return False


class ErrorPatterns:

    @staticmethod
    def db_connection_timeout():
        host = random.choice(["db-primary-1", "db-replica-2", "db-analytics-3"])
        return f"DatabaseConnectionError: Connection timeout after 30s to {host}:5432"

    @staticmethod
    def db_deadlock():
        table = random.choice(["orders", "payments", "inventory", "sessions"])
        return f"DeadlockException: Transaction deadlock detected on table '{table}' — rolled back"

    @staticmethod
    def db_pool_exhausted():
        return f"ConnectionPoolExhaustedError: All {random.choice([10, 20, 50])} connections in use — request queued"

    @staticmethod
    def db_replication_lag():
        lag = random.randint(5, 60)
        return f"ReplicationLagWarning: Replica is {lag}s behind primary — read consistency degraded"

    @staticmethod
    def auth_token_expired():
        return f"TokenExpiredError: JWT expired at {datetime.now(timezone.utc).strftime('%H:%M:%S')} — user forced to re-login"

    @staticmethod
    def auth_invalid_signature():
        return "InvalidSignatureError: JWT signature verification failed — possible token tampering detected"

    @staticmethod
    def auth_rate_limited():
        ip = f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"
        return f"RateLimitExceeded: Too many login attempts from {ip} — blocked for 15 minutes"

    @staticmethod
    def session_store_unavailable():
        return "SessionStoreError: Redis session store unreachable — falling back to stateless mode"

    @staticmethod
    def payment_gateway_timeout():
        gateway = random.choice(["Stripe", "PayPal", "Braintree", "Adyen"])
        return f"PaymentGatewayTimeout: {gateway} did not respond within 10s — transaction aborted"

    @staticmethod
    def payment_card_declined():
        code = random.choice(
            ["insufficient_funds", "card_expired", "do_not_honor", "lost_card"]
        )
        return f"CardDeclinedError: Payment declined — reason: {code}"

    @staticmethod
    def payment_fraud_detected():
        return "FraudDetectionAlert: Transaction flagged by risk engine — score exceeded threshold"

    @staticmethod
    def payment_double_charge():
        return "IdempotencyViolation: Duplicate payment request detected — second charge blocked"

    @staticmethod
    def service_unavailable():
        svc = random.choice(
            [
                "email-service",
                "notification-service",
                "recommendation-engine",
                "search-service",
                "analytics-pipeline",
                "image-processor",
            ]
        )
        return f"ServiceUnavailableError: {svc} returned 503 after 3 retries — circuit breaker opened"

    @staticmethod
    def message_queue_full():
        queue = random.choice(
            ["email-queue", "sms-queue", "webhook-queue", "export-queue"]
        )
        depth = random.randint(10000, 99999)
        return f"QueueDepthCritical: {queue} has {depth} pending messages — consumer lag growing"

    @staticmethod
    def cache_stampede():
        key = random.choice(
            ["product-catalog", "user-permissions", "config-flags", "pricing-table"]
        )
        return f"CacheStampedeDetected: Cache miss storm on key '{key}' — {random.randint(50, 500)} simultaneous DB queries"

    @staticmethod
    def disk_space_critical():
        mount = random.choice(["/var/log", "/data", "/tmp", "/var/lib/postgresql"])
        pct = random.randint(90, 99)
        return f"DiskSpaceCritical: {mount} is {pct}% full — write operations may fail"

    @staticmethod
    def null_pointer():
        cls = random.choice(
            [
                "UserService.getProfile",
                "OrderRepository.findById",
                "PaymentController.process",
                "CartService.checkout",
                "NotificationService.send",
                "ReportGenerator.build",
            ]
        )
        return f"NullPointerException: Unexpected null reference in {cls}() at line {random.randint(40, 300)}"

    @staticmethod
    def stack_overflow():
        cls = random.choice(
            ["TreeParser", "RecursiveResolver", "XmlDeserializer", "GraphTraversal"]
        )
        return f"StackOverflowError: Maximum call depth exceeded in {cls} — possible circular reference"

    @staticmethod
    def out_of_memory():
        heap_mb = random.randint(1900, 2048)
        return f"OutOfMemoryError: Java heap space exhausted ({heap_mb}MB/2048MB) — GC overhead limit exceeded"

    @staticmethod
    def unhandled_exception():
        exc = random.choice(
            [
                "IndexOutOfBoundsException",
                "ClassCastException",
                "IllegalStateException",
                "ConcurrentModificationException",
                "NumberFormatException",
            ]
        )
        return f"UnhandledException: {exc} propagated to global handler — request returned 500"

    @staticmethod
    def data_validation_failed():
        field = random.choice(
            ["email", "phone_number", "postal_code", "tax_id", "iban"]
        )
        return (
            f"ValidationError: Field '{field}' failed schema validation — data rejected"
        )

    @staticmethod
    def api_schema_mismatch():
        api = random.choice(["Salesforce", "HubSpot", "Shopify", "Twilio", "SendGrid"])
        return f"SchemaMismatchError: {api} API response shape changed — expected field missing in response"

    @staticmethod
    def file_upload_failed():
        ext = random.choice([".pdf", ".csv", ".xlsx", ".zip", ".jpg"])
        return f"FileUploadError: Failed to write {ext} to object storage — S3 returned 500"

    @staticmethod
    def csv_parse_error():
        return f"CSVParseError: Malformed row at line {random.randint(100, 9999)} — unexpected column count"

    @staticmethod
    def sql_injection_attempt():
        return "SecurityAlert: SQL injection pattern detected in request — query blocked and IP flagged"

    @staticmethod
    def xss_attempt():
        return "SecurityAlert: XSS payload detected in user input — sanitization applied, incident logged"

    @staticmethod
    def brute_force_detected():
        endpoint = random.choice(
            ["/api/login", "/api/reset-password", "/api/verify-otp"]
        )
        return f"BruteForceDetected: {random.randint(50, 500)} failed attempts on {endpoint} in 60s"

    @staticmethod
    def kubernetes_oom_kill():
        pod = random.choice(
            ["api-server-7d9f", "worker-node-3b2c", "scheduler-pod-1a4e"]
        )
        return f"OOMKilled: Container {pod} exceeded memory limit and was killed by the kernel"

    @staticmethod
    def grpc_deadline_exceeded():
        svc = random.choice(["inventory-grpc", "pricing-grpc", "shipping-grpc"])
        return f"DeadlineExceeded: gRPC call to {svc} timed out after 5000ms — context cancelled by client"

    @staticmethod
    def elasticsearch_shard_failure():
        index = random.choice(["logs-2024", "products-v3", "users-search"])
        return f"ShardFailureException: Primary shard for index '{index}' unavailable — search results degraded"

    @staticmethod
    def websocket_connection_dropped():
        user_id = random.randint(10000, 99999)
        reason = random.choice(
            ["ping timeout", "transport close", "server namespace disconnect"]
        )
        return (
            f"WebSocketError: Connection dropped for user {user_id} — reason: {reason}"
        )

    @staticmethod
    def feature_flag_service_timeout():
        flag = random.choice(
            [
                "checkout-v2",
                "new-pricing-engine",
                "dark-mode-rollout",
                "ab-test-homepage",
            ]
        )
        return f"FeatureFlagTimeout: Failed to evaluate flag '{flag}' — defaulting to off, service unreachable"

    @staticmethod
    def cdn_origin_pull_failed():
        asset = random.choice(
            [
                "/static/js/main.chunk.js",
                "/static/css/app.css",
                "/images/hero-banner.webp",
            ]
        )
        return f"CDNOriginError: Origin pull failed for {asset} — 502 from origin, serving stale cache"


ERROR_GENERATORS = [
    ErrorPatterns.db_connection_timeout,
    ErrorPatterns.db_deadlock,
    ErrorPatterns.db_pool_exhausted,
    ErrorPatterns.db_replication_lag,
    ErrorPatterns.auth_token_expired,
    ErrorPatterns.auth_invalid_signature,
    ErrorPatterns.auth_rate_limited,
    ErrorPatterns.session_store_unavailable,
    ErrorPatterns.payment_gateway_timeout,
    ErrorPatterns.payment_card_declined,
    ErrorPatterns.payment_fraud_detected,
    ErrorPatterns.payment_double_charge,
    ErrorPatterns.service_unavailable,
    ErrorPatterns.message_queue_full,
    ErrorPatterns.cache_stampede,
    ErrorPatterns.disk_space_critical,
    ErrorPatterns.null_pointer,
    ErrorPatterns.stack_overflow,
    ErrorPatterns.out_of_memory,
    ErrorPatterns.unhandled_exception,
    ErrorPatterns.data_validation_failed,
    ErrorPatterns.api_schema_mismatch,
    ErrorPatterns.file_upload_failed,
    ErrorPatterns.csv_parse_error,
    ErrorPatterns.sql_injection_attempt,
    ErrorPatterns.xss_attempt,
    ErrorPatterns.brute_force_detected,
    ErrorPatterns.kubernetes_oom_kill,
    ErrorPatterns.grpc_deadline_exceeded,
    ErrorPatterns.elasticsearch_shard_failure,
    ErrorPatterns.websocket_connection_dropped,
    ErrorPatterns.feature_flag_service_timeout,
    ErrorPatterns.cdn_origin_pull_failed,
]

ERROR_RATE = 0.20
SLOW_REQUEST_RATE = 0.10


REQUIRED_SCENARIO_FIELDS = {
    "timestamp",
    "level",
    "service",
    "source",
    "environment",
    "message",
    "request_id",
    "trace_id",
    "endpoint",
    "operation",
    "status_code",
    "latency_ms",
    "error_type",
    "host",
    "pod",
}


def _quote_value(value) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"' if any(ch.isspace() for ch in text) else text


def _scenario_step(
    message: str,
    *,
    level: str = "ERROR",
    service: str,
    source: str | None = None,
    environment: str = "prod",
    endpoint: str = "/api/checkout",
    operation: str = "request",
    status_code: int = 500,
    latency_ms: int = 1200,
    error_type: str = "ApplicationError",
    host: str = "prod-app-01",
    pod: str | None = None,
    delay_seconds: float = 1,
) -> dict:
    pod = pod or f"{service}-7d9f"
    return {
        "delay_seconds": delay_seconds,
        "level": level,
        "service": service,
        "source": source or service,
        "environment": environment,
        "message": message,
        "endpoint": endpoint,
        "operation": operation,
        "status_code": status_code,
        "latency_ms": latency_ms,
        "error_type": error_type,
        "host": host,
        "pod": pod,
    }


def format_scenario_log(step: dict, scenario_name: str, run_idx: int, step_idx: int) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    fields = {
        "timestamp": ts,
        "level": step.get("level", "ERROR"),
        "service": step.get("service", SERVICE_NAME),
        "source": step.get("source", step.get("service", SERVICE_NAME)),
        "environment": step.get("environment", "prod"),
        "scenario": scenario_name,
        "run": run_idx,
        "step": step_idx,
        "message": step["message"],
        "request_id": step.get("request_id", f"req_{random.randint(100000, 999999)}"),
        "trace_id": step.get("trace_id", f"trace_{random.randint(10000000, 99999999)}"),
        "endpoint": step.get("endpoint", "unknown"),
        "operation": step.get("operation", "unknown"),
        "status_code": step.get("status_code", 0),
        "latency_ms": step.get("latency_ms", 0),
        "error_type": step.get("error_type", "None"),
        "host": step.get("host", "unknown"),
        "pod": step.get("pod", "unknown"),
    }
    kv = " ".join(f"{key}={_quote_value(value)}" for key, value in fields.items())
    return f"[{ts}] {fields['level']}: {kv}"


SCENARIOS = {
    "healthcheck_timeout_noise": {
        "description": "Synthetic health checks timing out without customer impact.",
        "services": ["synthetic-monitor", "checkout-api"],
        "expected_runbook": "healthcheck_timeout_noise",
        "expected_severity": "low",
        "expected_disposition": "NO_ACTION",
        "expected_allowed_actions": ["auto_suppress"],
        "expected_blocked_actions": [],
        "steps": [
            _scenario_step("synthetic health check timeout from us-east probe no customer impact", level="WARN", service="synthetic-monitor", endpoint="/health", operation="healthcheck", status_code=504, latency_ms=2100, error_type="HealthCheckTimeout", host="monitor-01", pod="synthetic-monitor-6c4d", delay_seconds=0),
            _scenario_step("health probe failed readiness probe timeout no customer impact", level="WARN", service="checkout-api", endpoint="/ready", operation="readiness_probe", status_code=503, latency_ms=1800, error_type="ReadinessProbeTimeout", host="worker-02", pod="checkout-api-54fd", delay_seconds=1),
            _scenario_step("readiness probe timeout recovered traffic healthy no customer impact", level="INFO", service="checkout-api", endpoint="/ready", operation="readiness_probe", status_code=200, latency_ms=42, error_type="None", host="worker-02", pod="checkout-api-54fd", delay_seconds=1),
        ],
    },
    "db_pool_exhaustion": {
        "description": "Checkout and payment failures caused by exhausted database connections.",
        "services": ["checkout-api", "payment-worker", "api-gateway", "postgres"],
        "expected_runbook": "db_connection_pool_exhausted",
        "expected_severity": "critical",
        "expected_disposition": "ESCALATE",
        "expected_allowed_actions": ["notify_oncall", "create_incident_summary", "attach_evidence_bundle"],
        "expected_blocked_actions": ["restart_service", "modify_database_config"],
        "steps": [
            _scenario_step("database connection pool exhausted all connections in use request queued", service="postgres", endpoint="/db/orders", operation="checkout_query", status_code=500, latency_ms=30000, error_type="ConnectionPoolExhaustedError", host="db-primary-1", pod="postgres-primary-0", delay_seconds=0),
            _scenario_step("too many connections from checkout-api checkout database timeout", service="checkout-api", endpoint="/api/checkout", operation="create_order", status_code=500, latency_ms=12000, error_type="DatabaseConnectionTimeout", host="app-01", pod="checkout-api-6f8d", delay_seconds=1),
            _scenario_step("payment-worker transaction retries exceeded after checkout database timeout", service="payment-worker", endpoint="/jobs/payment-authorize", operation="authorize_payment", status_code=500, latency_ms=15000, error_type="RetryLimitExceeded", host="worker-03", pod="payment-worker-7b2a", delay_seconds=1),
            _scenario_step("api-gateway 5xx rising checkout returned 500 from upstream service error", service="api-gateway", endpoint="/api/checkout", operation="route_request", status_code=502, latency_ms=4100, error_type="UpstreamServiceError", host="edge-01", pod="api-gateway-9a11", delay_seconds=1),
        ],
    },
    "payment_gateway_degraded": {
        "description": "External payment provider degradation causing authorization failures.",
        "services": ["checkout-api", "payment-worker"],
        "expected_runbook": "payment_gateway",
        "expected_severity": "high",
        "expected_disposition": "NEEDS_ONCALL",
        "expected_allowed_actions": ["notify_oncall", "create_incident_summary"],
        "expected_blocked_actions": ["restart_service"],
        "steps": [
            _scenario_step("PaymentGatewayTimeout: Stripe did not respond payment gateway timeout transaction aborted", service="payment-worker", endpoint="/jobs/payment-authorize", operation="authorize_payment", status_code=504, latency_ms=10000, error_type="PaymentGatewayTimeout", host="worker-01", pod="payment-worker-22ca", delay_seconds=0),
            _scenario_step("PayPal did not respond transaction authorization failed payment retry limit exceeded", service="payment-worker", endpoint="/jobs/payment-authorize", operation="retry_authorization", status_code=504, latency_ms=9800, error_type="PaymentGatewayTimeout", host="worker-02", pod="payment-worker-54de", delay_seconds=1),
            _scenario_step("Braintree unavailable transaction authorization failed transaction aborted", service="checkout-api", endpoint="/api/checkout", operation="submit_payment", status_code=502, latency_ms=6200, error_type="ProviderUnavailable", host="app-03", pod="checkout-api-3bd4", delay_seconds=1),
        ],
    },
    "api_gateway_5xx_spike": {
        "description": "Customer-facing 5xx spike at the gateway across checkout routes.",
        "services": ["api-gateway", "checkout-api"],
        "expected_runbook": "api_gateway_5xx_spike",
        "expected_severity": "high",
        "expected_disposition": "NEEDS_ONCALL",
        "expected_allowed_actions": ["notify_oncall", "create_incident_summary"],
        "expected_blocked_actions": ["restart_service"],
        "steps": [
            _scenario_step("api gateway 5xx spike gateway returned 502 upstream service error", service="api-gateway", endpoint="/api/checkout", operation="route_request", status_code=502, latency_ms=2500, error_type="UpstreamServiceError", host="edge-02", pod="api-gateway-11ab", delay_seconds=0),
            _scenario_step("gateway returned 503 service unavailable edge 5xx rate elevated", service="api-gateway", endpoint="/api/cart", operation="route_request", status_code=503, latency_ms=3100, error_type="ServiceUnavailableError", host="edge-02", pod="api-gateway-11ab", delay_seconds=1),
            _scenario_step("gateway returned 504 gateway timeout upstream service error 5xx rate above threshold", service="api-gateway", endpoint="/api/orders", operation="route_request", status_code=504, latency_ms=8000, error_type="GatewayTimeout", host="edge-03", pod="api-gateway-27fe", delay_seconds=1),
        ],
    },
    "memory_pressure_or_oom": {
        "description": "Memory pressure grows into a customer-impacting OOM kill.",
        "services": ["checkout-api"],
        "expected_runbook": "kubernetes_oom_kill",
        "expected_severity": "critical",
        "expected_disposition": "ESCALATE",
        "expected_allowed_actions": ["notify_oncall", "create_incident_summary", "attach_evidence_bundle"],
        "expected_blocked_actions": ["restart_service", "change_infrastructure"],
        "steps": [
            _scenario_step("MemoryPressureWarning: Heap at 91% gc overhead critical Java heap space", level="WARN", service="checkout-api", endpoint="/api/checkout", operation="create_order", status_code=200, latency_ms=1900, error_type="MemoryPressureWarning", host="app-04", pod="checkout-api-7d9f", delay_seconds=0),
            _scenario_step("OutOfMemoryError Java heap space container memory limit exceeded", service="checkout-api", endpoint="/api/checkout", operation="create_order", status_code=500, latency_ms=4200, error_type="OutOfMemoryError", host="app-04", pod="checkout-api-7d9f", delay_seconds=1),
            _scenario_step("OOMKilled pod checkout-api-7d9f exceeded memory limit killed by the kernel", level="CRITICAL", service="checkout-api", endpoint="/api/checkout", operation="pod_restart", status_code=500, latency_ms=0, error_type="OOMKilled", host="app-04", pod="checkout-api-7d9f", delay_seconds=1),
        ],
    },
    "auth_failure_cascade": {
        "description": "Auth token validation failures cascade across login and session checks.",
        "services": ["auth-service", "session-api", "api-gateway"],
        "expected_runbook": "auth_failure_cascade",
        "expected_severity": "high",
        "expected_disposition": "NEEDS_ONCALL",
        "expected_allowed_actions": ["notify_oncall", "create_incident_summary"],
        "expected_blocked_actions": ["rotate_secrets", "restart_service"],
        "steps": [
            _scenario_step("authentication failure cascade auth token validation failed JWKS fetch failed", service="auth-service", endpoint="/oauth/token", operation="validate_token", status_code=401, latency_ms=3400, error_type="TokenValidationFailed", host="auth-01", pod="auth-service-33df", delay_seconds=0),
            _scenario_step("session verification failed jwt validation failed across services 401 increase", service="session-api", endpoint="/api/session", operation="verify_session", status_code=401, latency_ms=2200, error_type="SessionVerificationFailed", host="auth-02", pod="session-api-21ac", delay_seconds=1),
            _scenario_step("login failure spike token introspection timeout 403 increase", service="api-gateway", endpoint="/api/login", operation="route_request", status_code=403, latency_ms=5100, error_type="TokenIntrospectionTimeout", host="edge-01", pod="api-gateway-9a11", delay_seconds=1),
        ],
    },
    "deployment_regression": {
        "description": "Error rate rises shortly after a new checkout deployment and feature flag enablement.",
        "services": ["checkout-api", "api-gateway"],
        "expected_runbook": "deployment_regression",
        "expected_severity": "high",
        "expected_disposition": "NEEDS_ONCALL",
        "expected_allowed_actions": ["notify_oncall", "create_incident_summary", "attach_evidence_bundle"],
        "expected_blocked_actions": ["rollback_release", "deploy_code"],
        "steps": [
            _scenario_step("new deployment version checkout-api v2.3.1 feature flag enabled checkout-v2", level="INFO", service="checkout-api", endpoint="/deployments/checkout-api", operation="deploy", status_code=200, latency_ms=300, error_type="None", host="deploy-01", pod="checkout-api-5ac1", delay_seconds=0),
            _scenario_step("deployment regression error rate increased after deploy post deploy 5xx spike", service="checkout-api", endpoint="/api/checkout", operation="create_order", status_code=500, latency_ms=3800, error_type="DeploymentRegression", host="app-02", pod="checkout-api-5ac1", delay_seconds=1),
            _scenario_step("canary health degraded new release causing failures rollback candidate requires human approval", service="api-gateway", endpoint="/api/checkout", operation="route_request", status_code=502, latency_ms=2700, error_type="CanaryHealthDegraded", host="edge-03", pod="api-gateway-27fe", delay_seconds=1),
        ],
    },
    "queue_backlog": {
        "description": "Message queue backlog and dead-letter growth from slow consumers.",
        "services": ["order-worker", "message-queue"],
        "expected_runbook": "message_queue",
        "expected_severity": "high",
        "expected_disposition": "NEEDS_ONCALL",
        "expected_allowed_actions": ["notify_oncall", "create_incident_summary"],
        "expected_blocked_actions": ["scale_cluster"],
        "steps": [
            _scenario_step("QueueDepthCritical: email-queue has 42991 pending messages consumer lag growing queue backlog increasing", service="message-queue", endpoint="/queues/email-queue", operation="poll_depth", status_code=200, latency_ms=600, error_type="QueueDepthCritical", host="mq-01", pod="message-queue-0", delay_seconds=0),
            _scenario_step("message queue full consumer lag growing message retry exhausted", service="order-worker", endpoint="/jobs/order-created", operation="consume_message", status_code=500, latency_ms=7500, error_type="MessageRetryExhausted", host="worker-04", pod="order-worker-41cd", delay_seconds=1),
            _scenario_step("dead letter queue receiving checkout events pending messages continue rising", service="message-queue", endpoint="/queues/dead-letter", operation="dead_letter", status_code=500, latency_ms=500, error_type="DeadLetterQueueGrowth", host="mq-01", pod="message-queue-0", delay_seconds=1),
        ],
    },
    "vendor_api_timeout": {
        "description": "Third-party vendor API timeouts and rate limiting in integrations.",
        "services": ["integration-worker"],
        "expected_runbook": "vendor_api_timeout",
        "expected_severity": "medium",
        "expected_disposition": "NEEDS_DEV",
        "expected_allowed_actions": ["create_incident_summary", "send_discord_notification"],
        "expected_blocked_actions": ["deploy_code"],
        "steps": [
            _scenario_step("vendor API timeout Salesforce external vendor timed out partner api did not respond", service="integration-worker", endpoint="/jobs/sync-crm", operation="sync_contact", status_code=504, latency_ms=10000, error_type="VendorApiTimeout", host="worker-05", pod="integration-worker-63ae", delay_seconds=0),
            _scenario_step("HubSpot unavailable third-party rate limited vendor gateway timeout", service="integration-worker", endpoint="/jobs/sync-crm", operation="sync_company", status_code=429, latency_ms=8500, error_type="ThirdPartyRateLimited", host="worker-05", pod="integration-worker-63ae", delay_seconds=1),
            _scenario_step("Shopify unavailable third party api timeout fallback queue preserved", service="integration-worker", endpoint="/jobs/sync-orders", operation="sync_order", status_code=504, latency_ms=9300, error_type="VendorApiTimeout", host="worker-06", pod="integration-worker-18bf", delay_seconds=1),
        ],
    },
    "false_suppression_trap": {
        "description": "Noisy repeated failures that include clear customer-impact signals and must not be suppressed.",
        "services": ["checkout-api", "payment-worker", "api-gateway"],
        "expected_runbook": "api_gateway_5xx_spike",
        "expected_severity": "high",
        "expected_disposition": "NEEDS_ONCALL",
        "expected_allowed_actions": ["notify_oncall", "create_incident_summary", "attach_evidence_bundle"],
        "expected_blocked_actions": ["auto_suppress"],
        "steps": [
            _scenario_step("checkout returned 500 customer request failed api gateway 5xx spike", service="checkout-api", endpoint="/api/checkout", operation="create_order", status_code=500, latency_ms=3600, error_type="CustomerRequestFailed", host="app-01", pod="checkout-api-6f8d", delay_seconds=0),
            _scenario_step("payment authorization failed customer request failed repeated timeout", service="payment-worker", endpoint="/jobs/payment-authorize", operation="authorize_payment", status_code=500, latency_ms=7800, error_type="PaymentAuthorizationFailed", host="worker-01", pod="payment-worker-22ca", delay_seconds=1),
            _scenario_step("db-health degraded checkout returned 500 5xx rate above threshold", service="api-gateway", endpoint="/db-health", operation="db_health", status_code=503, latency_ms=2900, error_type="DbHealthDegraded", host="edge-01", pod="api-gateway-9a11", delay_seconds=1),
        ],
    },
    "low_frequency_high_impact": {
        "description": "One or two low-frequency events with severe impact signals.",
        "services": ["checkout-api", "ledger-api"],
        "expected_runbook": "kubernetes_oom_kill",
        "expected_severity": "critical",
        "expected_disposition": "ESCALATE",
        "expected_allowed_actions": ["notify_oncall", "create_incident_summary", "attach_evidence_bundle"],
        "expected_blocked_actions": ["auto_suppress", "delete_data"],
        "steps": [
            _scenario_step("OutOfMemoryError Java heap space fatal error data loss detected", level="CRITICAL", service="ledger-api", endpoint="/api/ledger/post", operation="post_transaction", status_code=500, latency_ms=0, error_type="OutOfMemoryError", host="app-09", pod="ledger-api-81ba", delay_seconds=0),
            _scenario_step("segmentation fault fatal error pod OOMKilled exceeded memory limit killed by the kernel", level="CRITICAL", service="ledger-api", endpoint="/api/ledger/post", operation="recover_transaction", status_code=500, latency_ms=0, error_type="SegmentationFault", host="app-09", pod="ledger-api-81ba", delay_seconds=1),
        ],
    },
    "ambiguous_cascade": {
        "description": "Multi-service cascade requiring correlation across DB, checkout, payment, and gateway.",
        "services": ["postgres", "checkout-api", "payment-worker", "api-gateway"],
        "expected_runbook": "db_connection_pool_exhausted",
        "expected_severity": "critical",
        "expected_disposition": "ESCALATE",
        "expected_allowed_actions": ["notify_oncall", "create_incident_summary", "attach_evidence_bundle"],
        "expected_blocked_actions": ["restart_service", "modify_database_config"],
        "steps": [
            _scenario_step("db-primary connection pool exhausted all connections in use request queued", service="postgres", endpoint="/db/orders", operation="checkout_query", status_code=500, latency_ms=30000, error_type="ConnectionPoolExhaustedError", host="db-primary-1", pod="postgres-primary-0", delay_seconds=0),
            _scenario_step("checkout-api timeout checkout database timeout customer request failed", service="checkout-api", endpoint="/api/checkout", operation="create_order", status_code=500, latency_ms=14000, error_type="CheckoutDatabaseTimeout", host="app-01", pod="checkout-api-6f8d", delay_seconds=1),
            _scenario_step("payment-worker retry limit exceeded payment-worker transaction retries exceeded", service="payment-worker", endpoint="/jobs/payment-authorize", operation="authorize_payment", status_code=500, latency_ms=15000, error_type="RetryLimitExceeded", host="worker-02", pod="payment-worker-54de", delay_seconds=1),
            _scenario_step("api-gateway 502 spike gateway returned 502 upstream service error", service="api-gateway", endpoint="/api/checkout", operation="route_request", status_code=502, latency_ms=4300, error_type="UpstreamServiceError", host="edge-01", pod="api-gateway-9a11", delay_seconds=1),
        ],
    },
}

SCENARIO_ALIASES = {
    "db_cascade": "ambiguous_cascade",
    "auth_cascade": "auth_failure_cascade",
    "deployment_gone_wrong": "deployment_regression",
    "memory_leak": "memory_pressure_or_oom",
}


async def _run_scenario(scenario_name: str, repeat: int = 1, speed: float = 1.0):
    """Execute a scenario — push each step to Loki with controlled timing."""
    scenario_name = SCENARIO_ALIASES.get(scenario_name, scenario_name)
    scenario = SCENARIOS[scenario_name]
    steps = scenario["steps"]
    speed = max(speed, 0.01)
    print(
        f"[SCENARIO] Starting '{scenario_name}' — {len(steps)} steps x{repeat} at {speed:.2f}x"
    )

    for run_idx in range(repeat):
        for i, step in enumerate(steps):
            delay = step["delay_seconds"] / speed
            if delay > 0:
                print(
                    f"[SCENARIO] Run {run_idx+1}/{repeat} step {i+1}/{len(steps)} — waiting {delay:.2f}s"
                )
                await asyncio.sleep(delay)

            log_line = format_scenario_log(step, scenario_name, run_idx + 1, i + 1)

            success = await push_to_loki(
                [log_line],
                extra_labels={
                    "scenario": scenario_name,
                    "service": step.get("service", SERVICE_NAME),
                    "env": step.get("environment", "prod"),
                    "step": str(i + 1),
                    "run": str(run_idx + 1),
                },
            )
            status = "pushed" if success else "FAILED"
            print(
                f"[SCENARIO] Run {run_idx+1}/{repeat} step {i+1}/{len(steps)} {status}: {log_line[:80]}…"
            )

    print(f"[SCENARIO] '{scenario_name}' complete")


class CustomFormatter(logging.Formatter):
    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created).isoformat()
        return f"[{timestamp}] {record.levelname}: {record.getMessage()}"


class InMemoryHandler(logging.Handler):
    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer

    def emit(self, record):
        self.buffer.append(self.format(record))


class LogGenerator:

    def __init__(self):
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self.log_buffer = deque(maxlen=50)
        self.stats = {
            "logs_generated": 0,
            "logs_shipped": 0,
            "batches_pushed": 0,
            "push_errors": 0,
        }
        self.last_push_at: str | None = None
        self.last_error: str | None = None
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("log-generator")
        logger.setLevel(logging.INFO)
        logger.handlers = []
        handler = InMemoryHandler(self.log_buffer)
        handler.setFormatter(CustomFormatter())
        logger.addHandler(handler)
        return logger

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(
        self,
        duration: int = 300,
        interval_seconds: float = 3.0,
        batch_size: int = 1,
        error_rate: float = ERROR_RATE,
        slow_rate: float = SLOW_REQUEST_RATE,
    ):
        async with self._lock:
            if self.running:
                return False, "Already running"
            self._stop_event.clear()
            self.stats = {k: 0 for k in self.stats}
            self.last_error = None
            self._task = asyncio.create_task(
                self._run(
                    duration=duration,
                    interval_seconds=max(interval_seconds, 0.01),
                    batch_size=max(1, batch_size),
                    error_rate=min(max(error_rate, 0.0), 1.0),
                    slow_rate=min(max(slow_rate, 0.0), 1.0),
                )
            )
            print(
                f"[LOG-SERVER] Started — {duration}s, interval={interval_seconds}s, "
                f"batch_size={batch_size}, error_rate={error_rate:.2f}, slow_rate={slow_rate:.2f}"
            )
            return True, "started"

    async def stop(self):
        async with self._lock:
            if not self.running:
                return False, "Not running"
            self._stop_event.set()
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._task.cancel()
        print("[LOG-SERVER] Stopped")
        return True, "stopped"

    async def _run(
        self,
        duration: int = 300,
        interval_seconds: float = 3.0,
        batch_size: int = 1,
        error_rate: float = ERROR_RATE,
        slow_rate: float = SLOW_REQUEST_RATE,
    ):
        print(
            f"[LOG-SERVER] Running for {duration}s at interval={interval_seconds}s "
            f"batch_size={batch_size}"
        )
        start_time = asyncio.get_event_loop().time()
        try:
            while not self._stop_event.is_set():
                if asyncio.get_event_loop().time() - start_time >= duration:
                    break
                self._generate_logs(batch_size, error_rate, slow_rate)
                if self.log_buffer:
                    await self._flush_to_loki()
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=interval_seconds
                    )
                except asyncio.TimeoutError:
                    pass

            if self.log_buffer:
                await self._flush_to_loki()

            print(f"[LOG-SERVER] Finished. Stats: {self.stats}")
        except asyncio.CancelledError:
            print("[LOG-SERVER] Task cancelled")

    def _generate_logs(
        self,
        count: int,
        error_rate: float = ERROR_RATE,
        slow_rate: float = SLOW_REQUEST_RATE,
    ):
        for _ in range(count):
            self.stats["logs_generated"] += 1
            rand = random.random()
            if rand < error_rate:
                self.logger.error(random.choice(ERROR_GENERATORS)())
            elif rand < error_rate + slow_rate:
                svc = random.choice(["checkout", "search", "auth", "upload", "report"])
                delay = random.uniform(2, 8)
                self.logger.warning(
                    f"SlowRequestWarning: {svc} endpoint took {delay:.2f}s — SLA breach"
                )
            else:
                endpoints = [
                    "GET /api/users/{} 200 12ms",
                    "POST /api/orders/{} 201 45ms",
                    "GET /api/products/{} 200 8ms",
                    "PUT /api/cart/{} 200 23ms",
                    "GET /health 200 1ms",
                ]
                self.logger.info(
                    random.choice(endpoints).format(random.randint(1000, 9999))
                )

    async def _flush_to_loki(self):
        if not self.log_buffer:
            return

        logs = list(self.log_buffer)
        self.log_buffer.clear()

        success = await push_to_loki(logs)
        if success:
            self.stats["logs_shipped"] += len(logs)
            self.stats["batches_pushed"] += 1
            self.last_push_at = datetime.now(timezone.utc).isoformat()
            print(f"[LOG-SERVER] Pushed {len(logs)} logs to Loki")
        else:
            self.stats["push_errors"] += 1
            self.last_error = "Loki push failed"
            self.log_buffer.extendleft(reversed(logs))


log_generator: LogGenerator | None = None


@app.on_event("startup")
async def startup_event():
    global log_generator
    log_generator = LogGenerator()

    if not all([LOKI_URL, LOKI_USERNAME, LOKI_API_KEY]):
        print(
            "[LOG-SERVER] WARNING: LOKI_URL / LOKI_USERNAME / LOKI_API_KEY not fully set"
        )
    else:
        ok = await push_to_loki(["[startup] Log server connected to Loki"])
        if ok:
            print(f"[LOG-SERVER] Loki connected at {LOKI_URL}")
        else:
            print(
                "[LOG-SERVER] WARNING: Loki connection test failed — check credentials"
            )


@app.post("/api/start")
async def start_generation(
    duration: int = 300,
    interval_seconds: float = 3.0,
    batch_size: int = 1,
    error_rate: float = ERROR_RATE,
    slow_rate: float = SLOW_REQUEST_RATE,
):
    ok, msg = await log_generator.start(
        duration=duration,
        interval_seconds=interval_seconds,
        batch_size=batch_size,
        error_rate=error_rate,
        slow_rate=slow_rate,
    )
    return {
        "message": msg,
        "status": "running" if log_generator.running else "idle",
        "transport": "loki",
        "duration_seconds": duration,
        "interval_seconds": interval_seconds,
        "batch_size": batch_size,
        "error_rate": error_rate,
        "slow_rate": slow_rate,
    }


@app.post("/api/stop")
async def stop_generation():
    ok, msg = await log_generator.stop()
    return {
        "message": msg,
        "stats": log_generator.stats,
        "status": "idle",
    }


@app.get("/api/status")
async def get_status():
    return {
        "status": "running" if log_generator.running else "idle",
        "stats": log_generator.stats,
        "last_push_at": log_generator.last_push_at,
        "last_error": log_generator.last_error,
        "transport": "loki",
        "loki_url": LOKI_URL,
    }


@app.post("/api/scenario/{scenario_name}")
async def run_scenario(scenario_name: str, repeat: int = 1, speed: float = 1.0):
    """
    Fire a correlated error scenario against Loki.
    """
    requested_name = scenario_name
    scenario_name = SCENARIO_ALIASES.get(scenario_name, scenario_name)
    if scenario_name not in SCENARIOS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario '{requested_name}'. Available: {list(SCENARIOS.keys())}",
        )
    if repeat < 1:
        raise HTTPException(status_code=400, detail="repeat must be >= 1")
    if speed <= 0:
        raise HTTPException(status_code=400, detail="speed must be > 0")

    asyncio.create_task(_run_scenario(scenario_name, repeat=repeat, speed=speed))

    scenario = SCENARIOS[scenario_name]
    steps = scenario["steps"]
    total_delay = sum(s["delay_seconds"] for s in steps) * repeat / max(speed, 0.01)
    return {
        "scenario": scenario_name,
        "requested_scenario": requested_name,
        "description": scenario["description"],
        "status": "started",
        "steps": len(steps),
        "repeat": repeat,
        "speed": speed,
        "estimated_duration_seconds": total_delay,
        "message": (
            f"Scenario '{scenario_name}' is running in the background. "
            f"{len(steps) * repeat} log entries will be pushed to Loki over ~{total_delay:.1f}s."
        ),
    }


@app.post("/api/burst")
async def generate_burst(
    count: int = 100,
    error_rate: float = 1.0,
    slow_rate: float = 0.0,
):
    """
    Generate and push a burst of logs immediately.
    Useful for throughput and clustering tests without waiting for the background loop.
    """
    if count < 1:
        raise HTTPException(status_code=400, detail="count must be >= 1")
    if error_rate < 0 or slow_rate < 0 or error_rate + slow_rate > 1:
        raise HTTPException(
            status_code=400,
            detail="error_rate and slow_rate must be >= 0 and sum to <= 1",
        )

    log_generator._generate_logs(count, error_rate=error_rate, slow_rate=slow_rate)
    await log_generator._flush_to_loki()
    return {
        "message": "burst_generated",
        "count": count,
        "error_rate": error_rate,
        "slow_rate": slow_rate,
        "stats": log_generator.stats,
    }


@app.get("/api/scenario")
async def list_scenarios():
    """List all available test scenarios."""
    return {
        "scenarios": {
            name: {
                "description": scenario["description"],
                "services": scenario.get("services", []),
                "expected_runbook": scenario.get("expected_runbook"),
                "expected_severity": scenario.get("expected_severity"),
                "expected_disposition": scenario.get("expected_disposition"),
                "expected_allowed_actions": scenario.get("expected_allowed_actions", []),
                "expected_blocked_actions": scenario.get("expected_blocked_actions", []),
                "steps": len(scenario["steps"]),
                "estimated_duration_seconds": sum(
                    s["delay_seconds"] for s in scenario["steps"]
                ),
            }
            for name, scenario in SCENARIOS.items()
        },
        "aliases": SCENARIO_ALIASES,
    }


@app.get("/health")
async def health():
    loki_ok = await push_to_loki(["[healthcheck] ping"])
    return {
        "status": "healthy",
        "loki": "connected" if loki_ok else "unreachable",
    }


@app.get("/ready")
async def ready():
    return {
        "status": "ready",
        "generator": "running" if log_generator and log_generator.running else "idle",
        "transport": "loki",
    }


@app.get("/db-health")
async def db_health():
    return {
        "status": "degraded" if not LOKI_URL else "healthy",
        "database": "simulated",
        "message": "db-health endpoint is synthetic; no real database dependency is checked",
    }


@app.post("/api/recover")
async def recover():
    """
    Stop active background generation and emit a short recovery signal.
    """
    if log_generator and log_generator.running:
        await log_generator.stop()

    recovery_line = format_scenario_log(
        _scenario_step(
            "RecoveryComplete: service metrics returned to baseline manual remediation verified",
            level="INFO",
            service=SERVICE_NAME,
            endpoint="/api/recover",
            operation="recover",
            status_code=200,
            latency_ms=120,
            error_type="None",
            host="log-server-01",
            pod="log-server",
            delay_seconds=0,
        ),
        "recovery",
        1,
        1,
    )
    pushed = await push_to_loki([recovery_line], extra_labels={"scenario": "recovery"})
    return {
        "status": "recovered",
        "generator": "idle",
        "recovery_log_pushed": pushed,
    }


@app.get("/")
async def root():
    return {
        "service": "log-server",
        "status": "running",
        "transport": "loki",
        "loki_url": LOKI_URL or "not configured",
        "scenarios": list(SCENARIOS.keys()),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)
