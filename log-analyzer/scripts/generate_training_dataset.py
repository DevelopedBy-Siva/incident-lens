#!/usr/bin/env python3
"""
Generate a balanced, realistic training dataset with correct output schema.

This script creates training examples that match the inference contract exactly:
- severity, disposition, confidence, summary, suspected_root_cause, next_steps, ticket_title, ticket_body
- Includes benign/no-incident examples
- Balanced severity distribution (20% low, 30% medium, 35% high, 15% critical)
- Realistic production-style logs
- Diverse incident patterns matching evaluation scenarios
"""

import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

random.seed(42)  # Deterministic generation


@dataclass
class GenerationStats:
    total: int = 0
    no_action_low: int = 0  # Benign/no-incident examples (low + NO_ACTION)
    low: int = 0
    medium: int = 0
    high: int = 0
    critical: int = 0
    by_incident_type: dict[str, int] = None
    by_disposition: dict[str, int] = None
    min_logs: int = 999
    max_logs: int = 0
    total_logs: int = 0

    def __post_init__(self):
        if self.by_incident_type is None:
            self.by_incident_type = {}
        if self.by_disposition is None:
            self.by_disposition = {}

    def record(self, example: dict):
        self.total += 1
        severity = example["expected_output"]["severity"]
        disposition = example["expected_output"]["disposition"]
        log_count = len(example["input"]["logs"])

        if severity == "low":
            self.low += 1
            if disposition == "NO_ACTION":
                self.no_action_low += 1
        elif severity == "medium":
            self.medium += 1
        elif severity == "high":
            self.high += 1
        elif severity == "critical":
            self.critical += 1

        # Track incident type (internal, not part of model output)
        incident_type = example["input"]["metadata"].get("incident_type", "unknown")
        self.by_incident_type[incident_type] = (
            self.by_incident_type.get(incident_type, 0) + 1
        )

        self.by_disposition[disposition] = self.by_disposition.get(disposition, 0) + 1

        self.min_logs = min(self.min_logs, log_count)
        self.max_logs = max(self.max_logs, log_count)
        self.total_logs += log_count

    def report(self) -> str:
        lines = [
            "="* 80,
            "DATASET GENERATION REPORT",
            "="* 80,
            f"\nTotal examples: {self.total}",
            f"\nSeverity Distribution:",
            f"  Low (NO_ACTION): {self.no_action_low:4d} ({100*self.no_action_low/self.total if self.total else 0:5.1f}%) [benign/no-incident]",
            f"  Low (total):     {self.low:4d} ({100*self.low/self.total if self.total else 0:5.1f}%)",
            f"  Medium:          {self.medium:4d} ({100*self.medium/self.total if self.total else 0:5.1f}%)",
            f"  High:            {self.high:4d} ({100*self.high/self.total if self.total else 0:5.1f}%)",
            f"  Critical:        {self.critical:4d} ({100*self.critical/self.total if self.total else 0:5.1f}%)",
            f"\nDisposition Distribution:",
        ]
        for disp in sorted(self.by_disposition.keys()):
            count = self.by_disposition[disp]
            pct = 100 * count / self.total if self.total else 0
            lines.append(f"  {disp:20s} {count:4d} ({pct:5.1f}%)")

        lines.extend([
            f"\nIncident Type Distribution:",
        ])
        for inc_type in sorted(self.by_incident_type.keys(), key=lambda k: self.by_incident_type[k], reverse=True)[:15]:
            count = self.by_incident_type[inc_type]
            pct = 100 * count / self.total if self.total else 0
            lines.append(f"  {inc_type:40s} {count:4d} ({pct:5.1f}%)")

        if len(self.by_incident_type) > 15:
            lines.append(f"  ... and {len(self.by_incident_type) - 15} more types")

        avg_logs = self.total_logs / self.total if self.total else 0
        lines.extend([
            f"\nLogs per Example:",
            f"  Minimum: {self.min_logs}",
            f"  Maximum: {self.max_logs}",
            f"  Average: {avg_logs:.1f}",
            "="* 80,
        ])
        return "\n".join(lines)


# Global counters for realistic IDs
_request_counter = random.randint(100000, 999999)
_trace_counter = random.randint(10000000, 99999999)
_host_counter = random.randint(1000, 9999)


def _random_timestamp(base=None) -> str:
    if base is None:
        base = datetime.now(timezone.utc) - timedelta(days=random.randint(1, 30))
    offset = timedelta(seconds=random.randint(-3600, 3600))
    return (base + offset).isoformat().replace("+00:00", "Z")


def _random_service() -> str:
    return random.choice([
        "auth-service", "checkout-api", "payment-worker", "notification-worker",
        "order-orchestrator", "inventory-service", "user-profile-api", "cart-service",
        "billing-service", "analytics-ingest", "search-api", "ml-inference",
        "webhook-dispatcher", "reporting-service", "media-processor", "gateway-edge",
        "session-service", "recommendation-service", "shipping-service",
    ])


def _random_host(service: str) -> str:
    global _host_counter
    _host_counter += 1
    return f"{service}-{random.choice(['aks', 'eks', 'gce'])}-{_host_counter % 10000:04x}"


def _random_region() -> str:
    return random.choice([
        "us-east-1", "us-east-2", "us-west-1", "us-west-2",
        "eu-west-1", "eu-central-1", "ap-southeast-1", "ap-northeast-1"
    ])


def _random_request_id() -> str:
    global _request_counter
    _request_counter += 1
    return f"req_{_request_counter:016x}"


def _random_trace_id() -> str:
    global _trace_counter
    _trace_counter += 1
    return f"{_trace_counter:016x}"


def _vary_duration(base: float, variation: float = 0.2) -> float:
    """Add variation to duration values."""
    return base * random.uniform(1 - variation, 1 + variation)


def _vary_count(base: int, variation: float = 0.3) -> int:
    """Add variation to count values."""
    return max(1, int(base * random.uniform(1 - variation, 1 + variation)))


def _random_failure_wording(base: str) -> str:
    """Add variation to failure descriptions."""
    synonyms = {
        "failed": ["failed", "unsuccessful", "error", "did not complete"],
        "timeout": ["timeout", "timed out", "did not respond", "exceeded timeout"],
        "slow": ["slow", "degraded", "elevated latency", "high latency"],
        "unavailable": ["unavailable", "unreachable", "down", "not responding"],
    }
    result = base
    for key, variants in synonyms.items():
        if key in base.lower():
            result = result.replace(key, random.choice(variants))
    return result


# ============================================================
# BENIGN / NO-INCIDENT GENERATORS
# ============================================================

def generate_benign_examples(count: int) -> list[dict]:
    """Generate benign/normal production logs that should NOT trigger incidents."""
    examples = []

    benign_patterns = [
        ("successful_request", generate_successful_requests),
        ("healthy_heartbeat", generate_healthy_heartbeats),
        ("successful_deployment", generate_successful_deployments),
        ("normal_cache_hits", generate_normal_cache_operations),
        ("successful_queue_processing", generate_successful_queue_processing),
        ("normal_k8s_events", generate_normal_k8s_events),
        ("successful_db_queries", generate_successful_db_queries),
    ]

    per_pattern = count // len(benign_patterns)
    remainder = count % len(benign_patterns)

    for pattern_name, generator in benign_patterns:
        pattern_count = per_pattern + (1 if remainder > 0 else 0)
        remainder -= 1
        examples.extend(generator(pattern_count))

    return examples


def generate_successful_requests(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = _random_service()
        endpoint = random.choice(["/api/checkout", "/api/cart", "/api/orders", "/health", "/metrics"])
        status = random.choice([200, 201, 204, 304])
        duration = random.uniform(10, 300)
        
        logs = [
            f"{_random_timestamp()} INFO [{service}] GET {endpoint} {status} duration={duration:.1f}ms trace={_random_trace_id()} region={_random_region()}",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 1,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "benign_successful_request",
                    "region": _random_region(),
                    "host": _random_host(service),
                    "trace_id": _random_trace_id(),
                    "request_id": _random_request_id(),
                }
            },
            "expected_output": {
                "severity": "low",
                "disposition": "NO_ACTION",
                "confidence": 0.95,
                "summary": "Normal successful HTTP request with healthy response time and 2xx status code.",
                "suspected_root_cause": None,
                "next_steps": [],
                "ticket_title": "",
                "ticket_body": "",
            }
        })
    return examples


def generate_healthy_heartbeats(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = _random_service()
        uptime = random.randint(10000, 900000)
        
        logs = [
            f"{_random_timestamp()} INFO [{service}] heartbeat ok host={_random_host(service)} uptime={uptime}s",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 1,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "benign_heartbeat",
                    "region": _random_region(),
                    "host": _random_host(service),
                }
            },
            "expected_output": {
                "severity": "low",
                "disposition": "NO_ACTION",
                "confidence": 0.98,
                "summary": "Service heartbeat indicates healthy operation with stable uptime.",
                "suspected_root_cause": None,
                "next_steps": [],
                "ticket_title": "",
                "ticket_body": "",
            }
        })
    return examples


def generate_successful_deployments(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = _random_service()
        version = f"v{random.randint(1, 20)}.{random.randint(0, 50)}.{random.randint(0, 100)}"
        replicas = random.randint(2, 10)
        
        logs = [
            f"{_random_timestamp()} INFO deployment {service} rollout status: {replicas}/{replicas} replicas updated to {version}",
            f"{_random_timestamp()} INFO deployment {service} rollout complete all health checks passed",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 2,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "benign_deployment",
                    "region": _random_region(),
                    "deployment_version": version,
                }
            },
            "expected_output": {
                "severity": "low",
                "disposition": "NO_ACTION",
                "confidence": 0.97,
                "summary": f"Successful deployment of {service} {version} completed with all replicas healthy.",
                "suspected_root_cause": None,
                "next_steps": [],
                "ticket_title": "",
                "ticket_body": "",
            }
        })
    return examples


def generate_normal_cache_operations(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = _random_service()
        key = f"{random.choice(['session', 'user_profile', 'cart', 'pricing'])}:{random.randint(1000, 9999)}"
        latency = random.uniform(0.1, 5.0)
        
        logs = [
            f"{_random_timestamp()} INFO [{service}] redis GET key={key} HIT latency={latency:.1f}ms",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 1,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "benign_cache_hit",
                    "region": _random_region(),
                    "host": _random_host(service),
                }
            },
            "expected_output": {
                "severity": "low",
                "disposition": "NO_ACTION",
                "confidence": 0.99,
                "summary": "Normal cache operation with successful hit and healthy latency.",
                "suspected_root_cause": None,
                "next_steps": [],
                "ticket_title": "",
                "ticket_body": "",
            }
        })
    return examples


def generate_successful_queue_processing(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        worker = f"{random.choice(['email', 'invoice', 'report', 'image-resize', 'payout'])}-worker"
        batch = random.randint(100, 1000)
        items = random.randint(50, 500)
        duration = random.uniform(1, 20)
        
        logs = [
            f"{_random_timestamp()} INFO worker {worker} batch={batch} completed items={items} duration={duration:.1f}s",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": worker,
                "count": 1,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "benign_queue_processing",
                    "region": _random_region(),
                    "host": _random_host(worker),
                }
            },
            "expected_output": {
                "severity": "low",
                "disposition": "NO_ACTION",
                "confidence": 0.98,
                "summary": f"Batch processing completed successfully with {items} items processed.",
                "suspected_root_cause": None,
                "next_steps": [],
                "ticket_title": "",
                "ticket_body": "",
            }
        })
    return examples


def generate_normal_k8s_events(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = _random_service()
        pod = f"{service}-{random.randint(1000, 9999):04x}"
        
        logs = [
            f"{_random_timestamp()} INFO k8s event namespace=prod Pod {pod} started container {service}",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 1,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "benign_k8s_event",
                    "region": _random_region(),
                    "pod": pod,
                }
            },
            "expected_output": {
                "severity": "low",
                "disposition": "NO_ACTION",
                "confidence": 0.97,
                "summary": "Normal Kubernetes pod lifecycle event indicating container started successfully.",
                "suspected_root_cause": None,
                "next_steps": [],
                "ticket_title": "",
                "ticket_body": "",
            }
        })
    return examples


def generate_successful_db_queries(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = _random_service()
        table = random.choice(["users", "orders", "payments", "sessions", "inventory", "invoices"])
        rows = random.randint(1, 50)
        duration = random.uniform(1, 100)
        
        logs = [
            f"{_random_timestamp()} INFO [{service}] query SELECT * FROM {table} WHERE id=? rows={rows} duration={duration:.1f}ms",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 1,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "benign_db_query",
                    "region": _random_region(),
                    "host": _random_host(service),
                }
            },
            "expected_output": {
                "severity": "low",
                "disposition": "NO_ACTION",
                "confidence": 0.98,
                "summary": "Normal database query completed successfully with acceptable response time.",
                "suspected_root_cause": None,
                "next_steps": [],
                "ticket_title": "",
                "ticket_body": "",
            }
        })
    return examples


# ============================================================
# LOW SEVERITY INCIDENT GENERATORS
# ============================================================

def generate_low_severity_examples(count: int) -> list[dict]:
    """Generate low-severity incidents: transient errors, single occurrences, cosmetic issues."""
    examples = []
    
    low_patterns = [
        ("transient_timeout", generate_transient_timeout),
        ("transient_connection_error", generate_transient_connection_error),
        ("healthcheck_noise", generate_healthcheck_timeout_noise),
        ("single_4xx_error", generate_single_4xx_error),
        ("transient_cache_miss", generate_transient_cache_miss),
        ("slow_query_single", generate_slow_query_single),
    ]
    
    per_pattern = count // len(low_patterns)
    remainder = count % len(low_patterns)
    
    for pattern_name, generator in low_patterns:
        pattern_count = per_pattern + (1 if remainder > 0 else 0)
        remainder -= 1
        examples.extend(generator(pattern_count))
    
    return examples


def generate_transient_timeout(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = _random_service()
        dependency = random.choice(["recommendation-engine", "email-service", "analytics-pipeline"])
        
        logs = [
            f"{_random_timestamp()} WARN [{service}] request to {dependency} timed out after 5000ms attempt=1",
            f"{_random_timestamp()} INFO [{service}] retry to {dependency} succeeded duration=234ms",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 1,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "transient_timeout",
                    "region": _random_region(),
                    "host": _random_host(service),
                    "dependency": dependency,
                }
            },
            "expected_output": {
                "severity": "low",
                "disposition": "OBSERVE",
                "confidence": 0.75,
                "summary": f"Single transient timeout to {dependency} that recovered on retry. No sustained impact observed.",
                "suspected_root_cause": "Likely network hiccup or momentary dependency slowness that self-resolved.",
                "next_steps": [
                    "Monitor for recurrence pattern",
                    "Check if timeout threshold needs tuning",
                ],
                "ticket_title": f"Transient timeout to {dependency} (self-resolved)",
                "ticket_body": f"A single request from {service} to {dependency} timed out after 5s but succeeded on immediate retry with normal latency. This appears to be a transient issue with no sustained impact. Monitor for recurrence before investigating further.",
            }
        })
    return examples


def generate_transient_connection_error(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = _random_service()
        db_host = f"db-replica-{random.randint(1, 5)}"
        
        logs = [
            f"{_random_timestamp()} WARN [{service}] connection to {db_host}:5432 refused attempt=1",
            f"{_random_timestamp()} INFO [{service}] connection to {db_host}:5432 established",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 1,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "transient_connection_error",
                    "region": _random_region(),
                    "host": _random_host(service),
                    "db_host": db_host,
                }
            },
            "expected_output": {
                "severity": "low",
                "disposition": "OBSERVE",
                "confidence": 0.72,
                "summary": f"Single connection refusal from {db_host} that immediately recovered. Likely transient network issue.",
                "suspected_root_cause": "Transient network packet loss or database connection pool momentarily full.",
                "next_steps": [
                    "Monitor connection pool metrics",
                    "Check for pattern across replicas",
                ],
                "ticket_title": f"Transient connection error to {db_host}",
                "ticket_body": f"{service} experienced a single connection refusal to {db_host} that recovered immediately on retry. This is consistent with transient network conditions or momentary connection pool saturation. No user impact. Worth monitoring but no immediate action required.",
            }
        })
    return examples


def generate_healthcheck_timeout_noise(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = _random_service()
        monitor = "synthetic-monitor"
        probe_region = random.choice(["us-east", "eu-west", "ap-southeast"])
        
        logs = [
            f"{_random_timestamp()} WARN [{monitor}] synthetic health check timeout from {probe_region} probe no customer impact",
            f"{_random_timestamp()} INFO [{service}] health probe succeeded readiness probe ok traffic healthy",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 2,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "healthcheck_timeout_noise",
                    "region": _random_region(),
                    "probe_region": probe_region,
                }
            },
            "expected_output": {
                "severity": "low",
                "disposition": "NO_ACTION",
                "confidence": 0.85,
                "summary": f"Synthetic health check timeout from {probe_region} probe with no customer impact. Service remains healthy.",
                "suspected_root_cause": "External monitoring probe network latency or probe infrastructure issue, not actual service degradation.",
                "next_steps": [
                    "Suppress or tune synthetic monitor sensitivity",
                    "Verify probe infrastructure health",
                ],
                "ticket_title": f"Synthetic health check noise from {probe_region}",
                "ticket_body": f"External synthetic monitor from {probe_region} reported a timeout to {service}, but the service's own health probes show healthy status and production traffic is unaffected. This appears to be monitoring infrastructure noise rather than a real service issue. Consider adjusting probe timeout thresholds or suppressing alerts from flaky probes.",
            }
        })
    return examples


def generate_single_4xx_error(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = _random_service()
        endpoint = random.choice(["/api/users/me", "/api/checkout", "/api/orders"])
        status = random.choice([400, 401, 403, 404])
        
        logs = [
            f"{_random_timestamp()} WARN [{service}] GET {endpoint} {status} duration=45ms trace={_random_trace_id()} client_error=invalid_parameter",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 1,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "single_4xx_error",
                    "region": _random_region(),
                    "host": _random_host(service),
                }
            },
            "expected_output": {
                "severity": "low",
                "disposition": "NO_ACTION",
                "confidence": 0.80,
                "summary": f"Single {status} client error on {endpoint}. Likely invalid client request, not service issue.",
                "suspected_root_cause": "Invalid or malformed client request. Not indicative of service degradation.",
                "next_steps": [],
                "ticket_title": "",
                "ticket_body": "",
            }
        })
    return examples


def generate_transient_cache_miss(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = _random_service()
        key = f"{random.choice(['product', 'user', 'config'])}:{random.randint(1000, 9999)}"
        
        logs = [
            f"{_random_timestamp()} WARN [{service}] redis GET key={key} MISS fallback_to_db=true",
            f"{_random_timestamp()} INFO [{service}] cache warm-up completed key={key}",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 2,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "transient_cache_miss",
                    "region": _random_region(),
                    "host": _random_host(service),
                }
            },
            "expected_output": {
                "severity": "low",
                "disposition": "NO_ACTION",
                "confidence": 0.88,
                "summary": f"Transient cache miss for {key} with successful DB fallback and re-warming. Normal cache behavior.",
                "suspected_root_cause": "Cache key expiry or eviction. Normal cache lifecycle.",
                "next_steps": [],
                "ticket_title": "",
                "ticket_body": "",
            }
        })
    return examples


def generate_slow_query_single(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = _random_service()
        table = random.choice(["orders", "users", "sessions"])
        duration = random.uniform(1000, 3000)
        
        logs = [
            f"{_random_timestamp()} WARN [{service}] slow query SELECT * FROM {table} duration={duration:.0f}ms rows=1",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 1,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "slow_query_single",
                    "region": _random_region(),
                    "host": _random_host(service),
                }
            },
            "expected_output": {
                "severity": "low",
                "disposition": "OBSERVE",
                "confidence": 0.70,
                "summary": f"Single slow query on {table} table ({duration:.0f}ms). May be one-off or transient database load.",
                "suspected_root_cause": "Likely transient database load spike or lock contention. Single occurrence.",
                "next_steps": [
                    "Monitor for sustained pattern",
                    "Check database metrics for load spikes",
                ],
                "ticket_title": f"Single slow query on {table}",
                "ticket_body": f"{service} logged a slow query against {table} with {duration:.0f}ms duration. This is a single occurrence and may be due to transient database load or a cold cache. Monitor for recurrence pattern before investigating query optimization.",
            }
        })
    return examples


# Continue in next message due to length...


# ============================================================
# MEDIUM SEVERITY INCIDENT GENERATORS
# ============================================================

def generate_medium_severity_examples(count: int) -> list[dict]:
    """Generate medium-severity incidents: meaningful degradation requiring investigation."""
    examples = []
    
    medium_patterns = [
        ("sustained_slow_queries", generate_sustained_slow_queries),
        ("elevated_4xx_rate", generate_elevated_4xx_rate),
        ("kafka_consumer_lag_moderate", generate_kafka_consumer_lag_moderate),
        ("cache_performance_degradation", generate_cache_performance_degradation),
        ("api_latency_increase", generate_api_latency_increase),
        ("queue_depth_elevated", generate_queue_depth_elevated),
    ]
    
    per_pattern = count // len(medium_patterns)
    remainder = count % len(medium_patterns)
    
    for pattern_name, generator in medium_patterns:
        pattern_count = per_pattern + (1 if remainder > 0 else 0)
        remainder -= 1
        examples.extend(generator(pattern_count))
    
    return examples


def generate_sustained_slow_queries(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = _random_service()
        table = random.choice(["orders", "payments", "inventory"])
        duration = random.uniform(2000, 5000)
        
        logs = [
            f"{_random_timestamp()} WARN [{service}] slow query SELECT * FROM {table} duration={duration:.0f}ms rows=150",
            f"{_random_timestamp()} WARN [{service}] slow query SELECT * FROM {table} duration={duration*0.9:.0f}ms rows=180",
            f"{_random_timestamp()} WARN [{service}] slow query SELECT * FROM {table} duration={duration*1.1:.0f}ms rows=142",
            f"{_random_timestamp()} INFO [{service}] query performance degraded avg_duration={duration:.0f}ms threshold_exceeded",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 12,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "sustained_slow_queries",
                    "region": _random_region(),
                    "host": _random_host(service),
                    "table": table,
                }
            },
            "expected_output": {
                "severity": "medium",
                "disposition": "NEEDS_DEV",
                "confidence": 0.78,
                "summary": f"Sustained slow query performance on {table} table averaging {duration:.0f}ms. Impacting {service} response times but not causing failures.",
                "suspected_root_cause": f"Possible missing index on {table}, query plan regression, or increased table size requiring optimization.",
                "next_steps": [
                    f"Review query execution plan for {table} queries",
                    "Check for missing indexes or outdated statistics",
                    "Consider query optimization or caching layer",
                    "Monitor database CPU and I/O metrics",
                ],
                "ticket_title": f"Sustained slow queries on {table} table",
                "ticket_body": f"{service} is experiencing sustained slow query performance against the {table} table, with queries averaging {duration:.0f}ms (well above normal baseline). This is causing elevated API response times but not yet causing request failures. Investigation needed to identify root cause (missing index, query plan issue, data volume growth) and implement optimization. Review recent schema changes or data migrations that may have impacted query performance.",
            }
        })
    return examples


def generate_elevated_4xx_rate(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = _random_service()
        endpoint = random.choice(["/api/checkout", "/api/orders", "/api/users"])
        
        logs = [
            f"{_random_timestamp()} WARN [{service}] GET {endpoint} 403 duration=45ms error=invalid_token",
            f"{_random_timestamp()} WARN [{service}] GET {endpoint} 403 duration=52ms error=invalid_token",
            f"{_random_timestamp()} WARN [{service}] GET {endpoint} 403 duration=48ms error=invalid_token",
            f"{_random_timestamp()} ERROR [{service}] elevated 403 rate on {endpoint} threshold_exceeded=true",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 25,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "elevated_4xx_rate",
                    "region": _random_region(),
                    "host": _random_host(service),
                }
            },
            "expected_output": {
                "severity": "medium",
                "disposition": "NEEDS_DEV",
                "confidence": 0.72,
                "summary": f"Elevated 403 error rate on {endpoint} endpoint. Token validation failures suggesting auth service issue or token expiry problem.",
                "suspected_root_cause": "Possible auth service degradation, expired signing keys, or clock skew causing token validation failures.",
                "next_steps": [
                    "Check auth service health and token validation metrics",
                    "Verify token signing key rotation hasn't caused issues",
                    "Review recent auth changes or deployments",
                    "Check for clock skew between services",
                ],
                "ticket_title": f"Elevated 403 rate on {endpoint}",
                "ticket_body": f"{service} is experiencing an elevated rate of 403 Forbidden errors on {endpoint}, suggesting token validation failures. This could indicate auth service degradation, issues with token signing key rotation, or clock skew between services. Customer impact is currently limited to authentication errors. Investigate auth service health, recent deployments, and token validation logic.",
            }
        })
    return examples


def generate_kafka_consumer_lag_moderate(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        # Add service variation
        service = random.choice([
            "notification-worker", "email-sender", "webhook-relay",
            "event-processor", "analytics-worker", "reporting-worker"
        ])
        # Add topic variation
        topic = random.choice([
            "order-events", "inventory-updates", "payment-notifications",
            "user-activity", "audit-logs", "notification-queue"
        ])
        # Vary lag values with realistic progression
        base_lag = random.randint(5000, 15000)
        lag_values = [
            _vary_count(base_lag, 0.4),
            _vary_count(int(base_lag * 1.5), 0.3),
            _vary_count(int(base_lag * 1.2), 0.3),
        ]
        
        # Vary log wording
        lag_verb = random.choice(["increasing", "growing", "not draining"])
        region = _random_region()
        host = _random_host(service)
        
        # Vary replica counts
        current_replicas = random.choice([2, 3, 4, 5])
        target_replicas = current_replicas + random.choice([2, 3, 4])
        
        logs = [
            f"{_random_timestamp()} WARN [{service}] kafka consumer lag topic={topic} partition=3 lag={lag_values[0]}",
            f"{_random_timestamp()} WARN [{service}] kafka consumer lag topic={topic} partition=3 lag={lag_values[1]}",
            f"{_random_timestamp()} WARN [{service}] kafka consumer lag {lag_verb} topic={topic} partition=3 lag={lag_values[2]}",
            f"{_random_timestamp()} INFO [{service}] consumer scaling triggered current_replicas={current_replicas} target_replicas={target_replicas}",
        ]
        
        # Vary summary description
        summary_templates = [
            f"Moderate Kafka consumer lag on {topic} topic (partition 3 showing {lag_values[2]} messages). Consumer scaling initiated.",
            f"Consumer lag detected on {topic} with {lag_values[2]} messages pending. Auto-scaling triggered to add capacity.",
            f"{service} falling behind on {topic} processing with {lag_values[2]} message lag. Scaling from {current_replicas} to {target_replicas} replicas.",
        ]
        
        # Vary ticket body
        ticket_body_templates = [
            f"{service} is experiencing moderate consumer lag on {topic} topic, with partition 3 lagging by {lag_values[2]} messages. Auto-scaling has been triggered to add consumer instances. This suggests consumer throughput is not keeping pace with producer rate. Monitor whether scaling resolves the lag, and investigate processing duration for slow operations. Check for partition key skew that may be overloading specific consumers.",
            f"Kafka consumer lag alert for {service} on {topic}. Current lag is {lag_values[2]} messages with scaling action initiated (from {current_replicas} to {target_replicas} replicas). Consumer processing rate is insufficient for current message volume. Investigate downstream dependencies, processing time per message, and partition distribution.",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 18,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "kafka_consumer_lag_moderate",
                    "region": region,
                    "host": host,
                    "topic": topic,
                }
            },
            "expected_output": {
                "severity": "medium",
                "disposition": "NEEDS_DEV",
                "confidence": round(random.uniform(0.72, 0.78), 2),
                "summary": random.choice(summary_templates),
                "suspected_root_cause": random.choice([
                    "Consumer throughput unable to keep pace with producer rate. May be due to slow downstream processing, partition skew, or insufficient consumer instances.",
                    "Processing rate per message slower than message arrival rate. Possible downstream dependency latency or resource contention.",
                    "Consumer instance count insufficient for current message volume. May need permanent scaling or processing optimization.",
                ]),
                "next_steps": [
                    "Monitor consumer scaling effectiveness",
                    f"Check {service} processing duration metrics",
                    "Review partition key distribution for skew",
                    "Verify no downstream dependency degradation"
                ],
                "ticket_title": f"Kafka consumer lag on {topic}",
                "ticket_body": random.choice(ticket_body_templates),
            }
        })
    return examples


def generate_cache_performance_degradation(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = _random_service()
        cache = "redis"
        
        logs = [
            f"{_random_timestamp()} WARN [{service}] {cache} response time elevated avg_latency=150ms p99=450ms",
            f"{_random_timestamp()} WARN [{service}] {cache} connection pool saturation active=95/100",
            f"{_random_timestamp()} ERROR [{service}] {cache} timeout on GET operations timeout_count=12",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 25,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "cache_performance_degradation",
                    "region": _random_region(),
                    "host": _random_host(service),
                }
            },
            "expected_output": {
                "severity": "medium",
                "disposition": "NEEDS_DEV",
                "confidence": 0.76,
                "summary": f"Redis cache performance degradation with elevated latency (avg 150ms, p99 450ms) and connection pool saturation. Causing timeout errors.",
                "suspected_root_cause": "Redis overload due to increased traffic, memory pressure, or slow operations. Connection pool may be undersized.",
                "next_steps": [
                    "Check Redis CPU and memory metrics",
                    "Review slow log for expensive operations",
                    "Consider increasing connection pool size",
                    "Evaluate cache eviction policy and memory limits",
                ],
                "ticket_title": "Redis cache performance degradation",
                "ticket_body": f"{service} is experiencing Redis cache performance issues with elevated latencies and connection pool saturation. Average latency is 150ms (normally <10ms) with p99 at 450ms. Connection pool is at 95% utilization causing timeouts. This suggests Redis is overloaded or experiencing slow operations. Investigate Redis metrics, slow log, and consider scaling cache cluster or tuning connection pool settings.",
            }
        })
    return examples


def generate_api_latency_increase(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = _random_service()
        endpoint = random.choice(["/api/checkout", "/api/search", "/api/recommendations"])
        baseline = random.uniform(50, 150)
        degraded = baseline * random.uniform(3, 6)
        
        logs = [
            f"{_random_timestamp()} WARN [{service}] latency alert endpoint={endpoint} p50={degraded:.0f}ms baseline={baseline:.0f}ms",
            f"{_random_timestamp()} INFO [{service}] GET {endpoint} 200 duration={degraded:.0f}ms trace={_random_trace_id()}",
            f"{_random_timestamp()} INFO [{service}] GET {endpoint} 200 duration={degraded*0.9:.0f}ms trace={_random_trace_id()}",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 50,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "api_latency_increase",
                    "region": _random_region(),
                    "host": _random_host(service),
                }
            },
            "expected_output": {
                "severity": "medium",
                "disposition": "NEEDS_ONCALL",
                "confidence": 0.74,
                "summary": f"API latency degradation on {endpoint}. P50 latency increased from {baseline:.0f}ms to {degraded:.0f}ms ({degraded/baseline:.1f}x baseline).",
                "suspected_root_cause": "Downstream dependency slowness, database query performance issue, or increased request complexity.",
                "next_steps": [
                    "Profile endpoint to identify slow operations",
                    "Check database and cache latency metrics",
                    "Review recent code changes to endpoint logic",
                    "Monitor downstream service health",
                ],
                "ticket_title": f"Latency increase on {endpoint}",
                "ticket_body": f"{service} {endpoint} is experiencing significant latency degradation. P50 latency has increased from {baseline:.0f}ms to {degraded:.0f}ms ({degraded/baseline:.1f}x baseline). Requests are still succeeding but user experience is degraded. Profile the endpoint to identify the bottleneck - likely database queries, downstream service calls, or increased computational complexity.",
            }
        })
    return examples


def generate_queue_depth_elevated(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        queue = random.choice(["email-queue", "webhook-queue", "notification-queue"])
        depth = random.randint(5000, 15000)
        
        logs = [
            f"{_random_timestamp()} WARN message-queue queue={queue} depth={depth} threshold_exceeded=true",
            f"{_random_timestamp()} INFO message-queue queue={queue} consumer_lag_seconds=120",
            f"{_random_timestamp()} WARN message-queue queue={queue} depth_increasing current={depth*1.1:.0f}",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": "message-queue",
                "count": 15,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "queue_depth_elevated",
                    "region": _random_region(),
                    "queue": queue,
                }
            },
            "expected_output": {
                "severity": "medium",
                "disposition": "NEEDS_DEV",
                "confidence": 0.77,
                "summary": f"Elevated queue depth on {queue} ({depth} messages, growing). Consumer lag at 120 seconds.",
                "suspected_root_cause": "Consumers not keeping pace with producers. May be due to slow consumer processing, insufficient consumer instances, or producer rate spike.",
                "next_steps": [
                    "Scale consumer instances",
                    "Check consumer processing duration",
                    "Review producer rate for spikes",
                    "Monitor dead-letter queue for failures",
                ],
                "ticket_title": f"Elevated queue depth on {queue}",
                "ticket_body": f"Message queue {queue} has elevated depth ({depth} messages and growing) with consumer lag at 120 seconds. This indicates consumers are not keeping pace with message production rate. Scale consumer instances and investigate consumer processing time for bottlenecks. Check if there's a producer rate spike or if consumers are experiencing degraded performance.",
            }
        })
    return examples


# ============================================================
# HIGH SEVERITY INCIDENT GENERATORS
# ============================================================

def generate_high_severity_examples(count: int) -> list[dict]:
    """Generate high-severity incidents: substantial service impact, sustained failures."""
    examples = []
    
    high_patterns = [
        ("db_connection_pool_exhausted", generate_db_pool_exhaustion),
        ("payment_gateway_degraded", generate_payment_gateway_degraded),
        ("auth_failure_cascade", generate_auth_failure_cascade),
        ("api_gateway_5xx_spike", generate_api_5xx_spike),
        ("deployment_regression", generate_deployment_regression),
        ("dependency_timeout_cascade", generate_dependency_timeout),
        ("tls_certificate_expired", generate_certificate_expiry),
    ]
    
    per_pattern = count // len(high_patterns)
    remainder = count % len(high_patterns)
    
    for pattern_name, generator in high_patterns:
        pattern_count = per_pattern + (1 if remainder > 0 else 0)
        remainder -= 1
        examples.extend(generator(pattern_count))
    
    return examples


def generate_db_pool_exhaustion(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = "checkout-api"
        db_host = "db-primary-1"
        
        logs = [
            f"{_random_timestamp()} ERROR [{service}] database connection pool exhausted all connections in use request queued",
            f"{_random_timestamp()} ERROR [{service}] too many connections from {service} checkout database timeout",
            f"{_random_timestamp()} ERROR payment-worker transaction retries exceeded after checkout database timeout",
            f"{_random_timestamp()} ERROR api-gateway 5xx rising checkout returned 500 from upstream service error",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 45,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "db_connection_pool_exhausted",
                    "region": _random_region(),
                    "host": _random_host(service),
                    "db_host": db_host,
                }
            },
            "expected_output": {
                "severity": "high",
                "disposition": "NEEDS_ONCALL",
                "confidence": 0.88,
                "summary": f"Database connection pool exhaustion on {service} causing checkout failures and cascading to payment-worker and api-gateway. Customer impact confirmed.",
                "suspected_root_cause": "Connection leak, missing connection release in error paths, or traffic spike exceeding pool capacity.",
                "next_steps": [
                    "Emergency: Restart affected service instances to release connections",
                    "Review recent code changes for connection leak",
                    "Check application connection pool configuration",
                    "Monitor database max_connections setting",
                    "Add connection leak detection",
                ],
                "ticket_title": "URGENT: DB connection pool exhaustion causing checkout failures",
                "ticket_body": f"{service} has exhausted its database connection pool causing cascading failures across checkout, payment processing, and API gateway (5xx errors). Customer checkout is currently failing. Immediate action needed to restart services and release connections. Investigation required for root cause: likely connection leak in error handling paths or undersized connection pool for current traffic. Review connection acquisition/release patterns in recent code changes.",
            }
        })
    return examples


def generate_payment_gateway_degraded(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        gateway = random.choice(["Stripe", "PayPal", "Braintree"])
        service = "payment-worker"
        
        logs = [
            f"{_random_timestamp()} ERROR [{service}] PaymentGatewayTimeout: {gateway} did not respond payment gateway timeout transaction aborted",
            f"{_random_timestamp()} ERROR [{service}] {gateway} unavailable transaction authorization failed payment retry limit exceeded",
            f"{_random_timestamp()} ERROR checkout-api {gateway} unavailable transaction authorization failed transaction aborted",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 38,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "payment_gateway_degraded",
                    "region": _random_region(),
                    "host": _random_host(service),
                    "gateway": gateway,
                }
            },
            "expected_output": {
                "severity": "high",
                "disposition": "NEEDS_ONCALL",
                "confidence": 0.90,
                "summary": f"{gateway} payment gateway experiencing severe degradation with widespread timeout and authorization failures. Customer payments failing.",
                "suspected_root_cause": f"{gateway} upstream provider outage or severe degradation. External dependency failure.",
                "next_steps": [
                    f"Check {gateway} status page for incidents",
                    "Failover to backup payment processor if available",
                    "Queue failed transactions for retry",
                    "Communicate customer impact to stakeholders",
                    "Monitor for recovery",
                ],
                "ticket_title": f"URGENT: {gateway} payment gateway outage",
                "ticket_body": f"{gateway} payment gateway is experiencing severe degradation with widespread timeouts and authorization failures. Customer payments are currently failing. This appears to be an upstream {gateway} provider issue. Check their status page and incident communications. If degradation persists, consider failing over to backup payment processor. Queue failed transactions for retry once {gateway} recovers.",
            }
        })
    return examples


def generate_auth_failure_cascade(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        logs = [
            f"{_random_timestamp()} ERROR [auth-service] authentication failure cascade auth token validation failed JWKS fetch failed",
            f"{_random_timestamp()} ERROR [session-api] session verification failed jwt validation failed across services 401 increase",
            f"{_random_timestamp()} ERROR [api-gateway] login failure spike token introspection timeout 403 increase",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": "auth-service",
                "count": 67,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "auth_failure_cascade",
                    "region": _random_region(),
                    "host": _random_host("auth-service"),
                }
            },
            "expected_output": {
                "severity": "high",
                "disposition": "NEEDS_ONCALL",
                "confidence": 0.87,
                "summary": "Authentication failure cascade across auth-service, session-api, and api-gateway. JWKS fetch failure causing widespread 401/403 errors.",
                "suspected_root_cause": "JWKS endpoint unreachable, expired TLS certificates for identity provider, or signing key rotation issue.",
                "next_steps": [
                    "Check identity provider / JWKS endpoint availability",
                    "Verify TLS certificates not expired",
                    "Review recent key rotation activities",
                    "Check for network connectivity issues to IdP",
                    "Consider emergency signing key rollback",
                ],
                "ticket_title": "URGENT: Auth failure cascade - JWKS fetch failing",
                "ticket_body": "Widespread authentication failures across multiple services due to JWKS fetch failures from auth-service. This is causing cascading 401/403 errors preventing user login and session validation. Root cause appears to be connectivity to identity provider JWKS endpoint or TLS certificate issue. Immediate investigation of identity provider health, network connectivity, and certificate validity required. Significant customer impact to authentication flows.",
            }
        })
    return examples


def generate_api_5xx_spike(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = "api-gateway"
        
        logs = [
            f"{_random_timestamp()} ERROR [{service}] api gateway 5xx spike gateway returned 502 upstream service error",
            f"{_random_timestamp()} ERROR [{service}] gateway returned 503 service unavailable edge 5xx rate elevated",
            f"{_random_timestamp()} ERROR [{service}] gateway returned 504 gateway timeout upstream service error 5xx rate above threshold",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 89,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "api_gateway_5xx_spike",
                    "region": _random_region(),
                    "host": _random_host(service),
                }
            },
            "expected_output": {
                "severity": "high",
                "disposition": "NEEDS_ONCALL",
                "confidence": 0.85,
                "summary": "API Gateway experiencing severe 5xx spike (502/503/504 errors) indicating upstream service failures. Customer-facing impact.",
                "suspected_root_cause": "Downstream service failures, overload, or connectivity issues causing gateway to return errors to clients.",
                "next_steps": [
                    "Identify which upstream services are failing",
                    "Check for related incidents in backend services",
                    "Review traffic patterns for sudden spike",
                    "Check circuit breaker and retry policies",
                    "Coordinate with backend service teams",
                ],
                "ticket_title": "URGENT: API Gateway 5xx spike",
                "ticket_body": "API Gateway is experiencing a severe 5xx error spike with mix of 502 Bad Gateway, 503 Service Unavailable, and 504 Gateway Timeout errors. This indicates upstream service failures or severe degradation. Customer-facing APIs are impacted. Investigate backend service health, identify failing dependencies, and coordinate resolution with service owners. Check for traffic spikes or deployment events that may have triggered cascading failures.",
            }
        })
    return examples


def generate_deployment_regression(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = "checkout-api"
        version = f"v{random.randint(2, 5)}.{random.randint(1, 30)}.{random.randint(0, 50)}"
        
        logs = [
            f"{_random_timestamp()} INFO [{service}] new deployment version {service} {version} feature flag enabled checkout-v2",
            f"{_random_timestamp()} ERROR [{service}] deployment regression error rate increased after deploy post deploy 5xx spike",
            f"{_random_timestamp()} ERROR [api-gateway] canary health degraded new release causing failures rollback candidate requires human approval",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 52,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "deployment_regression",
                    "region": _random_region(),
                    "deployment_version": version,
                }
            },
            "expected_output": {
                "severity": "high",
                "disposition": "NEEDS_ONCALL",
                "confidence": 0.89,
                "summary": f"Deployment regression detected: {service} {version} causing elevated 5xx error rate. Canary analysis shows failures correlated with new release.",
                "suspected_root_cause": f"Code regression in {version} release, feature flag misconfiguration, or incompatibility with production data/traffic patterns.",
                "next_steps": [
                    f"Immediate rollback of {service} {version} to previous stable version",
                    "Disable checkout-v2 feature flag if rollback insufficient",
                    "Capture error logs and stack traces for analysis",
                    "Review recent code changes in deployment",
                    "Post-incident: improve pre-prod testing coverage",
                ],
                "ticket_title": f"URGENT: Deployment regression - {service} {version}",
                "ticket_body": f"Production deployment of {service} {version} is causing a significant error rate regression. 5xx errors spiked immediately following deployment and feature flag enablement. Canary analysis confirms failures are correlated with the new release. Immediate rollback recommended. After rollback, review code changes, test coverage gaps, and consider staged rollout strategy for future deployments. Customer impact is ongoing until rollback completes.",
            }
        })
    return examples


def generate_dependency_timeout(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = _random_service()
        dependency = random.choice(["Salesforce", "HubSpot", "Shopify", "external-partner-api"])
        
        logs = [
            f"{_random_timestamp()} ERROR [integration-worker] vendor API timeout {dependency} external vendor timed out partner api did not respond",
            f"{_random_timestamp()} ERROR [integration-worker] {dependency} unavailable third-party rate limited vendor gateway timeout",
            f"{_random_timestamp()} ERROR [integration-worker] {dependency} third party api timeout fallback queue preserved",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": "integration-worker",
                "count": 34,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "vendor_api_timeout",
                    "region": _random_region(),
                    "host": _random_host("integration-worker"),
                    "dependency": dependency,
                }
            },
            "expected_output": {
                "severity": "high",
                "disposition": "NEEDS_DEV",
                "confidence": 0.82,
                "summary": f"Widespread timeout failures to {dependency} API. Integration jobs failing, data sync interrupted.",
                "suspected_root_cause": f"{dependency} upstream service degradation or outage. External dependency failure.",
                "next_steps": [
                    f"Check {dependency} status page",
                    "Verify fallback queue is preserving failed operations",
                    "Monitor for recovery and trigger retry",
                    f"Contact {dependency} support if outage persists",
                    "Review timeout thresholds and retry policies",
                ],
                "ticket_title": f"Vendor API outage: {dependency}",
                "ticket_body": f"Integration worker is experiencing widespread failures connecting to {dependency} API with timeouts and unavailable errors. This is impacting data synchronization and integration workflows. Appears to be upstream {dependency} service issue. Verify fallback queues are preserving operations for retry. Monitor {dependency} status page and contact support if outage persists. Once recovered, trigger batch retry of failed operations.",
            }
        })
    return examples


def generate_certificate_expiry(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        domain = random.choice(["api.example.com", "webhook.example.com", "partner-gateway.example.com"])
        service = "webhook-dispatcher"
        
        logs = [
            f"{_random_timestamp()} ERROR [{service}] x509: certificate has expired or is not yet valid for {domain}",
            f"{_random_timestamp()} ERROR [{service}] TLS handshake error from 41.199.8.101:59398: remote error: tls: bad certificate",
            f"{_random_timestamp()} ERROR nginx SSL_do_handshake() failed (SSL: error:0A000086:SSL routines::certificate verify failed)",
            f"{_random_timestamp()} WARN [{service}] client connections failing cert_expiry={domain}",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 28,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "tls_certificate_expired",
                    "region": _random_region(),
                    "host": _random_host(service),
                    "domain": domain,
                }
            },
            "expected_output": {
                "severity": "high",
                "disposition": "NEEDS_ONCALL",
                "confidence": 0.95,
                "summary": f"TLS certificate expired for {domain} causing widespread connection failures and handshake errors.",
                "suspected_root_cause": f"Certificate for {domain} expired and automatic renewal failed or was not configured.",
                "next_steps": [
                    f"Emergency: Manually renew certificate for {domain}",
                    "Deploy renewed certificate to affected services",
                    "Verify cert-manager or renewal automation",
                    "Set up expiry monitoring and alerting",
                    "Document certificate renewal process",
                ],
                "ticket_title": f"URGENT: TLS certificate expired for {domain}",
                "ticket_body": f"TLS certificate for {domain} has expired causing widespread TLS handshake failures and connection errors. Client connections are being rejected. This requires immediate manual certificate renewal and deployment. After incident resolution, investigate why automatic renewal failed and implement monitoring to prevent future expiry incidents. Customer impact: all services dependent on {domain} are currently unable to establish TLS connections.",
            }
        })
    return examples


# ============================================================
# CRITICAL SEVERITY INCIDENT GENERATORS
# ============================================================

def generate_critical_severity_examples(count: int) -> list[dict]:
    """Generate critical-severity incidents: broad outage, data/security risk, severe production impact."""
    examples = []
    
    critical_patterns = [
        ("memory_pressure_oom", generate_memory_oom),
        ("database_primary_down", generate_database_down),
        ("regional_outage", generate_regional_outage),
        ("data_corruption", generate_data_corruption),
        ("security_breach_detected", generate_security_breach),
    ]
    
    per_pattern = count // len(critical_patterns)
    remainder = count % len(critical_patterns)
    
    for pattern_name, generator in critical_patterns:
        pattern_count = per_pattern + (1 if remainder > 0 else 0)
        remainder -= 1
        examples.extend(generator(pattern_count))
    
    return examples


def generate_memory_oom(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = random.choice(["checkout-api", "order-orchestrator", "reporting-service"])
        pod = f"{service}-{random.randint(1000, 9999):04x}"
        
        logs = [
            f"{_random_timestamp()} WARN k8s event namespace=prod Pod {pod} memory usage 89% of limit",
            f"{_random_timestamp()} ERROR k8s event namespace=prod Pod {pod} container {service} OOMKilled, exit code 137",
            f"{_random_timestamp()} ERROR [{service}] worker process killed signal=SIGKILL host={pod}",
            f"{_random_timestamp()} WARN k8s event namespace=prod Pod {pod} restarted by ReplicaSet controller",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 15,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "memory_pressure_oom",
                    "region": _random_region(),
                    "pod": pod,
                }
            },
            "expected_output": {
                "severity": "critical",
                "disposition": "ESCALATE",
                "confidence": 0.92,
                "summary": f"OOM kill of {service} pod due to memory limit exceeded. Service capacity reduced, potential customer impact.",
                "suspected_root_cause": "Memory leak, inefficient memory usage, or undersized memory limit for workload. Possible unbounded cache growth or memory-intensive operation.",
                "next_steps": [
                    "Emergency: Increase memory limits to stabilize service",
                    "Capture heap dump if pod crashes again",
                    "Profile memory usage to identify leak",
                    "Review recent code changes for memory issues",
                    "Monitor remaining pods for memory trends",
                    "Scale horizontal replicas to compensate",
                ],
                "ticket_title": f"CRITICAL: OOM kill - {service} pod memory exhausted",
                "ticket_body": f"Pod {pod} was OOMKilled after exceeding memory limit (89% → 100% → kill). Service capacity is reduced and customer requests may be failing. This indicates memory leak or insufficient memory allocation. Immediate action: increase memory limits and scale replicas to restore capacity. Investigation: capture heap dumps, profile memory usage, review recent changes. This is a P0 incident with customer impact until service capacity is restored.",
            }
        })
    return examples


def generate_database_down(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        db_host = "db-primary.us-west-2.rds.amazonaws.com"
        
        logs = [
            f"{_random_timestamp()} ERROR [user-profile-api] could not connect to server: Connection refused host={db_host} port=5432",
            f"{_random_timestamp()} ERROR [user-profile-api] psycopg2.OperationalError: connection to server at \"{db_host}\" failed",
            f"{_random_timestamp()} WARN [user-profile-api] falling back to read replica replica-1",
            f"{_random_timestamp()} ERROR [user-profile-api] GET /api/v1/users/me 500 duration=1774.2ms trace={_random_trace_id()}",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": "user-profile-api",
                "count": 78,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "database_primary_down",
                    "region": "us-west-2",
                    "host": _random_host("user-profile-api"),
                    "db_host": db_host,
                }
            },
            "expected_output": {
                "severity": "critical",
                "disposition": "ESCALATE",
                "confidence": 0.96,
                "summary": f"Database primary ({db_host}) unreachable. Multiple services unable to write data. Read-only fallback active.",
                "suspected_root_cause": "Database primary failure, network partition, or infrastructure outage. RDS instance may have failed over or become unreachable.",
                "next_steps": [
                    "URGENT: Check RDS console for instance status",
                    "Verify database failover completed successfully",
                    "Update connection strings if primary IP changed",
                    "Assess data loss risk from failed writes",
                    "Communicate outage to all stakeholders",
                    "Post-incident: review HA configuration",
                ],
                "ticket_title": "CRITICAL: Database primary down - write operations failing",
                "ticket_body": f"Database primary {db_host} is unreachable causing widespread write failures across multiple services. Services have fallen back to read replicas but cannot persist any data. This is a P0 production outage with severe customer impact (unable to create/update data). Check RDS console immediately for instance status, verify automatic failover completed, and update application connection strings if primary endpoint changed. All write operations are currently failing.",
            }
        })
    return examples


def generate_regional_outage(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        region = random.choice(["us-east-1", "eu-west-1", "ap-southeast-1"])
        
        logs = [
            f"{_random_timestamp()} ERROR [region-monitor] AWS {region} experiencing elevated error rates across multiple services",
            f"{_random_timestamp()} ERROR [api-gateway] upstream connection failures region={region} service_count=15",
            f"{_random_timestamp()} ERROR [health-check] region {region} failing all health checks timeout_rate=95%",
            f"{_random_timestamp()} CRITICAL [ops] initiating emergency failover from {region} to backup region",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": "multi-region-orchestrator",
                "count": 120,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "regional_outage",
                    "region": region,
                }
            },
            "expected_output": {
                "severity": "critical",
                "disposition": "ESCALATE",
                "confidence": 0.94,
                "summary": f"Regional outage in {region}. Multiple services down, health checks failing. Emergency failover initiated.",
                "suspected_root_cause": f"AWS {region} infrastructure outage affecting multiple availability zones. Upstream provider incident.",
                "next_steps": [
                    "Monitor failover completion to backup region",
                    f"Check AWS {region} status dashboard",
                    "Verify DNS/routing updated to backup region",
                    "Assess data replication lag between regions",
                    "Communicate customer impact and ETA",
                    "Prepare rollback plan once region recovers",
                ],
                "ticket_title": f"CRITICAL: Regional outage - {region}",
                "ticket_body": f"Widespread outage in AWS {region} with 95% health check failure rate across all services. Emergency failover to backup region has been initiated. This is a P0 incident with significant customer impact for users served from {region}. Monitor AWS status dashboard for provider incident updates. Verify failover routing and DNS updates are propagating. Assess any data inconsistencies from replication lag. Maintain incident bridge until service fully restored.",
            }
        })
    return examples


def generate_data_corruption(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        service = random.choice(["ledger-api", "payment-reconciliation", "order-fulfillment"])
        
        logs = [
            f"{_random_timestamp()} CRITICAL [{service}] data integrity violation detected checksum_mismatch=true",
            f"{_random_timestamp()} ERROR [{service}] database constraint violation FOREIGN KEY constraint failed",
            f"{_random_timestamp()} CRITICAL [{service}] inconsistent state detected expected_total=10000 actual_total=9847",
            f"{_random_timestamp()} ERROR [{service}] emergency: stopping writes pending integrity check",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": service,
                "count": 8,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "data_corruption",
                    "region": _random_region(),
                    "host": _random_host(service),
                }
            },
            "expected_output": {
                "severity": "critical",
                "disposition": "ESCALATE",
                "confidence": 0.97,
                "summary": f"Data corruption detected in {service}. Checksum mismatches, constraint violations, and state inconsistencies. Write operations suspended.",
                "suspected_root_cause": "Possible database corruption, application bug causing data inconsistency, failed migration, or concurrent write race condition.",
                "next_steps": [
                    "IMMEDIATE: Suspend all write operations to prevent further corruption",
                    "Initiate data integrity audit",
                    "Identify scope of affected records",
                    "Restore from backup if corruption is widespread",
                    "Review recent schema changes and migrations",
                    "Engage database team for recovery plan",
                ],
                "ticket_title": "CRITICAL: Data corruption detected - writes suspended",
                "ticket_body": f"Data integrity violations detected in {service} including checksum mismatches and constraint failures. System has automatically suspended write operations to prevent further corruption. This is a P0 incident requiring immediate data integrity assessment. Scope affected records, determine if database restore is needed, and develop recovery plan. Do not resume writes until data integrity is verified. Customer impact: service is read-only until resolved.",
            }
        })
    return examples


def generate_security_breach(count: int) -> list[dict]:
    examples = []
    for _ in range(count):
        source_ip = f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        
        logs = [
            f"{_random_timestamp()} CRITICAL [security] SQL injection attempt detected ip={source_ip} blocked=true pattern=UNION_SELECT",
            f"{_random_timestamp()} CRITICAL [security] multiple authentication bypass attempts ip={source_ip} rate=150/min",
            f"{_random_timestamp()} CRITICAL [security] suspicious admin privilege escalation detected user=external_user action=GRANT_ADMIN",
            f"{_random_timestamp()} ERROR [security] WAF rules triggered blocking malicious traffic source={source_ip}",
        ]
        
        examples.append({
            "input": {
                "logs": logs,
                "environment": "prod",
                "service": "security-monitor",
                "count": 25,
                "related_incidents": [],
                "metadata": {
                    "incident_type": "security_breach_detected",
                    "source_ip": source_ip,
                }
            },
            "expected_output": {
                "severity": "critical",
                "disposition": "ESCALATE",
                "confidence": 0.98,
                "summary": f"Active security attack detected from {source_ip}. SQL injection, auth bypass attempts, and privilege escalation. WAF blocking.",
                "suspected_root_cause": f"Coordinated attack from {source_ip} attempting multiple exploit vectors including SQL injection and authentication bypass.",
                "next_steps": [
                    "IMMEDIATE: Verify IP is blocked at firewall level",
                    "Audit all recent admin privilege changes",
                    "Review access logs for successful breaches",
                    "Engage security incident response team",
                    "Preserve logs for forensic analysis",
                    "Check for data exfiltration",
                    "Notify security stakeholders per incident response plan",
                ],
                "ticket_title": "CRITICAL: Active security attack detected",
                "ticket_body": f"Active coordinated security attack detected from {source_ip} with multiple exploit attempts: SQL injection, authentication bypass (150 attempts/min), and suspicious privilege escalation. WAF is blocking but this requires immediate security team engagement. Verify attacker is fully blocked, audit for any successful breaches, preserve forensic evidence, and follow security incident response procedures. This is a P0 security incident.",
            }
        })
    return examples


# ============================================================
# MAIN GENERATION ORCHESTRATION
# ============================================================

def generate_full_dataset(
    benign_count: int = 400,
    low_count: int = 400,
    medium_count: int = 600,
    high_count: int = 500,
    critical_count: int = 300,
) -> tuple[list[dict], GenerationStats]:
    """Generate complete balanced training dataset."""
    
    print(f"Generating training dataset...")
    print(f"  Benign:   {benign_count}")
    print(f"  Low:      {low_count}")
    print(f"  Medium:   {medium_count}")
    print(f"  High:     {high_count}")
    print(f"  Critical: {critical_count}")
    print(f"  Total:    {benign_count + low_count + medium_count + high_count + critical_count}")
    print()
    
    stats = GenerationStats()
    examples = []
    
    # Generate each severity category
    print("Generating benign examples...")
    benign_examples = generate_benign_examples(benign_count)
    for ex in benign_examples:
        stats.record(ex)
    examples.extend(benign_examples)
    
    print("Generating low severity examples...")
    low_examples = generate_low_severity_examples(low_count)
    for ex in low_examples:
        stats.record(ex)
    examples.extend(low_examples)
    
    print("Generating medium severity examples...")
    medium_examples = generate_medium_severity_examples(medium_count)
    for ex in medium_examples:
        stats.record(ex)
    examples.extend(medium_examples)
    
    print("Generating high severity examples...")
    high_examples = generate_high_severity_examples(high_count)
    for ex in high_examples:
        stats.record(ex)
    examples.extend(high_examples)
    
    print("Generating critical severity examples...")
    critical_examples = generate_critical_severity_examples(critical_count)
    for ex in critical_examples:
        stats.record(ex)
    examples.extend(critical_examples)
    
    # Shuffle to mix severities
    random.shuffle(examples)
    
    return examples, stats


def save_dataset(examples: list[dict], output_path: Path):
    """Save dataset in JSONL format."""
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            json.dump(example, f, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            f.write("\n")
    print(f"\nDataset saved to: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """Generate training dataset with correct schema and balanced distribution."""
    output_dir = Path(__file__).parent.parent.parent / "data"
    output_path = output_dir / "dataset_v2.jsonl"
    
    # Generate dataset with target distribution
    # Targeting ~2200 examples total
    examples, stats = generate_full_dataset(
        benign_count=400,    # ~18%
        low_count=440,       # ~20%
        medium_count=660,    # ~30%
        high_count=550,      # ~25%
        critical_count=150,  # ~7%
    )
    
    # Save to file
    save_dataset(examples, output_path)
    
    # Print report
    print(stats.report())
    
    # Validate schema
    print("\nValidating dataset schema...")
    from app.training.dataset_serializer import JsonLinesDatasetSerializer
    
    serializer = JsonLinesDatasetSerializer()
    try:
        serializer.validate(examples)
        print("✓ Schema validation PASSED")
    except Exception as e:
        print(f"✗ Schema validation FAILED: {e}")
        return 1
    
    print("\n" + "="*80)
    print("SUCCESS: Training dataset generated")
    print("="*80)
    return 0


if __name__ == "__main__":
    exit(main())
