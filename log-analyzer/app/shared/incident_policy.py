"""
Canonical IncidentLens ML Policy and Prompt Configuration

This module is the SINGLE SOURCE OF TRUTH for:
- Decision policy (severity and disposition definitions)
- System and user prompts for training and inference
- Output schema
- Grounding rules

Both training and inference MUST import from this module to ensure consistency.
"""

import json
from typing import Any

# Policy version identifier for model artifact metadata
POLICY_VERSION = "impact-v4.1-compact-grounded"

# Valid severity levels (NO "benign" - use low/NO_ACTION instead)
SEVERITIES = frozenset(["low", "medium", "high", "critical"])

# Valid disposition levels
DISPOSITIONS = frozenset(["NO_ACTION", "OBSERVE", "NEEDS_DEV", "NEEDS_ONCALL", "ESCALATE"])

# Required output fields for validation
REQUIRED_OUTPUT_FIELDS = frozenset([
    "severity",
    "disposition",
    "confidence",
    "summary",
    "suspected_root_cause",
    "next_steps",
    "ticket_title",
    "ticket_body",
])

# Canonical output schema
OUTPUT_SCHEMA = {
    "severity": "low|medium|high|critical",
    "disposition": "NO_ACTION|OBSERVE|NEEDS_DEV|NEEDS_ONCALL|ESCALATE",
    "confidence": 0.0,
    "summary": "...",
    "suspected_root_cause": "...",
    "next_steps": ["..."],
    "ticket_title": "...",
    "ticket_body": "...",
}

# System prompt for training and inference
SYSTEM_PROMPT = (
    "You are IncidentLens. Analyze the supplied backend logs "
    "and return one grounded JSON object only."
)

# Complete policy text with severity, disposition, rules, and grounding constraints
POLICY_TEXT = f"""DECISION POLICY ({POLICY_VERSION})

SEVERITY:
- low: No meaningful production impact; benign or isolated transient behavior.
- medium: Limited/localized degradation; service remains broadly available.
- high: Material production impact, repeated live failures, major capacity loss/SLO breach, or urgent mitigation is warranted.
- critical: Major/widespread or critical-path failure, severe security/data-integrity impact, or coordinated emergency response is required.

DISPOSITION:
- NO_ACTION: Benign/non-incident only; recovery alone does not justify NO_ACTION.
- OBSERVE: Minor/self-healing event worth monitoring; no fix or urgent response is currently needed.
- NEEDS_DEV: Engineering fix/investigation is needed, but the urgent production condition is contained.
- NEEDS_ONCALL: Immediate/near-term operational ownership, verification, or mitigation is warranted, even after recovery.
- ESCALATE: Critical incident requiring coordinated urgent escalation or cross-team/security/data response.

DECISION RULES:
- Judge severity from the maximum impact supported by the full sequence; later recovery does not erase earlier impact.
- Choose disposition separately from severity based on required ownership and follow-up.
- NEEDS_DEV means contained + engineering follow-up; NEEDS_ONCALL means operational action/verification is still warranted.
- Use critical/ESCALATE only with multiple concrete high-impact indicators; not from one generic error.

GROUNDING RULES:
- State only facts supported by the supplied logs. Do not invent PagerDuty, incident command, failover, rollback, customer impact, data loss, outage scope, recovery state, or numeric impact.
- Do not infer customer impact from infrastructure/probe failure alone.
- Keep unproven root causes explicitly hypothetical.
- Do not invent P0/P1/P2 labels.
- next_steps must be a JSON array of strings only."""


_EVIDENCE_METADATA_FIELDS = frozenset({
    "host", "hostname", "pod", "container", "namespace", "region", "zone",
    "domain", "endpoint", "topic", "partition", "deployment_version", "trace_id",
    "request_id", "peer", "dependency",
})


def normalize_incident_input(value: dict[str, Any] | list[str] | str) -> dict[str, Any]:
    """Return the evidence contract shared by dataset training and inference.

    Synthetic dataset bookkeeping (for example ``incident_type`` and policy
    profile) is intentionally excluded: it would leak the answer into training
    and is not available from a production log stream.
    """
    if not isinstance(value, dict):
        value = {"logs": value if isinstance(value, list) else [str(value)]}
    logs = value.get("logs") or []
    if isinstance(logs, str):
        logs = [logs]
    logs = [str(line).strip() for line in logs if str(line).strip()]
    metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
    evidence_metadata = {
        key: metadata[key]
        for key in sorted(metadata)
        if key in _EVIDENCE_METADATA_FIELDS and metadata[key] not in (None, "", [], {})
    }
    related = value.get("related_incidents") if isinstance(value.get("related_incidents"), list) else []
    related = [
        {
            key: item[key]
            for key in ("source", "count", "severity", "disposition", "summary", "first_seen", "last_seen")
            if key in item and item[key] not in (None, "", [], {})
        }
        for item in related
        if isinstance(item, dict)
    ][:5]
    return {
        "service": str(value.get("service") or "unknown"),
        "environment": str(value.get("environment") or "unknown"),
        "observed_event_count": max(1, int(value.get("count") or len(logs) or 1)),
        "metadata": evidence_metadata,
        "related_incidents": related,
        "logs": logs,
    }


def build_user_prompt(incident_input: dict[str, Any] | list[str] | str) -> str:
    """Build the canonical evidence prompt for training and inference."""
    context = normalize_incident_input(incident_input)
    return (
        POLICY_TEXT + "\n\n"
        "OUTPUT JSON SCHEMA:\n"
        + json.dumps(OUTPUT_SCHEMA, ensure_ascii=False)
        + "\nBenign => low/NO_ACTION. next_steps must contain strings only.\n\n"
        "INCIDENT CONTEXT (observed facts):\n"
        + json.dumps(
            {key: value for key, value in context.items() if key != "logs"},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n\n"
        "LOGS:\n"
        + "\n".join(context["logs"])
    )


def build_training_messages(incident_input: dict[str, Any] | list[str] | str, expected_output: dict[str, Any]) -> list[dict[str, str]]:
    """
    Build complete message list for training examples.
    
    Args:
        incident_input: Canonical evidence object (or legacy log list)
        expected_output: Expected JSON output from training example
    
    Returns:
        List of message dicts for chat template
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(incident_input)},
        {
            "role": "assistant",
            "content": json.dumps(
                expected_output,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


def build_inference_messages(incident_input: dict[str, Any] | list[str] | str) -> list[dict[str, str]]:
    """
    Build message list for inference (no assistant response).
    
    Args:
        incident_input: Canonical evidence object (or legacy log list)
    
    Returns:
        List of message dicts for chat template
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(incident_input)},
    ]


def validate_output_schema(output: dict[str, Any]) -> tuple[bool, str | None]:
    """
    Validate that output conforms to the canonical schema.
    
    Args:
        output: Parsed JSON output to validate
    
    Returns:
        (is_valid, error_message) tuple
    """
    if not isinstance(output, dict):
        return False, "Output must be a JSON object"
    
    # Check required fields
    missing = REQUIRED_OUTPUT_FIELDS - set(output.keys())
    if missing:
        return False, f"Missing required fields: {sorted(missing)}"
    
    # Validate severity
    if output["severity"] not in SEVERITIES:
        return False, f"Invalid severity: {output['severity']!r}, must be one of {sorted(SEVERITIES)}"
    
    # Validate disposition
    if output["disposition"] not in DISPOSITIONS:
        return False, f"Invalid disposition: {output['disposition']!r}, must be one of {sorted(DISPOSITIONS)}"
    
    # Validate confidence
    if not isinstance(output["confidence"], (int, float)) or isinstance(output["confidence"], bool):
        return False, "confidence must be numeric"
    if not 0.0 <= float(output["confidence"]) <= 1.0:
        return False, "confidence must be in range [0.0, 1.0]"
    
    # Validate suspected_root_cause
    if output["suspected_root_cause"] is not None and not isinstance(output["suspected_root_cause"], str):
        return False, "suspected_root_cause must be string or null"
    
    # Validate next_steps
    if not isinstance(output["next_steps"], list):
        return False, "next_steps must be an array"
    if not all(isinstance(step, str) for step in output["next_steps"]):
        return False, "next_steps must contain only strings"
    
    # Validate string fields
    for field in ("summary", "ticket_title", "ticket_body"):
        if not isinstance(output[field], str):
            return False, f"{field} must be a string"
    
    return True, None


# Inference configuration (canonical for production)
INFERENCE_CONFIG = {
    "max_new_tokens": 768,
    "temperature": 0.0,
    "do_sample": False,  # Greedy decoding only
}


# Training configuration defaults (base model and LoRA)
TRAINING_DEFAULTS = {
    "base_model": "Qwen/Qwen3.5-4B",
    "use_4bit": "auto",  # Use QLoRA when CUDA + bitsandbytes available
    "trust_remote_code": False,
}
