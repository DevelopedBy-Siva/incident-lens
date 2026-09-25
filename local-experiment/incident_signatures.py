"""
Log normalization and signature generation for local evaluation.

This module DIRECTLY REUSES production logic from log-analyzer/app/data/signatures.py
to ensure evaluation clustering matches production behavior.
"""

import re
import hashlib
from typing import Optional


def normalize_message(message: str) -> str:
    """
    Normalize log messages by replacing volatile values with tokens.
    
    This is COPIED from production (log-analyzer/app/data/signatures.py)
    to ensure identical signature generation.
    """
    msg = message

    # Drop trailing explanatory clauses that often vary log-to-log
    msg = re.sub(r"\s[-—]\s.*$", "", msg)

    # UUIDs
    msg = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "UUID",
        msg,
        flags=re.IGNORECASE,
    )

    # infrastructure identifiers
    msg = re.sub(r"db-(primary|replica|analytics)-\d+", "db-host", msg)
    msg = re.sub(r"\b(stripe|paypal|braintree|adyen)\b", "payment-gateway", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\b(salesforce|hubspot|shopify|twilio|sendgrid)\b", "vendor-api", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\b(email|sms|webhook|export)-queue\b", "queue-name", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\.(pdf|csv|xlsx|zip|jpg)\b", ".fileext", msg, flags=re.IGNORECASE)

    # known ID patterns
    msg = re.sub(r"ORD-\d+", "ORD-N", msg)
    msg = re.sub(r"user_id=\d+", "user_id=N", msg)
    msg = re.sub(r"upload_\d+", "upload_N", msg)
    msg = re.sub(r"token=[0-9a-f]+", "token=HASH", msg)
    msg = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", msg)
    msg = re.sub(r"/tmp/\S+", "/tmp/FILE", msg)

    # floating point seconds e.g. "3.69s", "2.54s" — normalize before integers
    msg = re.sub(r"\d+\.\d+s\b", "N.Ns", msg)

    # memory sizes e.g. "1907MB", "2048MB" — normalize the number, keep MB
    msg = re.sub(r"\d+MB", "NMB", msg)

    # durations e.g. "5000ms", "30s" — normalize number, keep unit
    msg = re.sub(r"\d+ms\b", "Nms", msg)
    msg = re.sub(r"\d+s\b", "Ns", msg)

    # preserve HTTP 4xx/5xx status codes, replace all other standalone numbers
    msg = re.sub(r"\b(?![45]\d{2}\b)\d+\b", "N", msg)

    # normalize whitespace + lowercase
    msg = re.sub(r"\s+", " ", msg)
    msg = msg.lower().strip()

    return msg


def generate_signature(service: str, level: str, message: str, exception_type: Optional[str] = None) -> str:
    """
    Generate incident signature from log components.
    
    This is COPIED from production (log-analyzer/app/data/signatures.py)
    to ensure identical clustering behavior.
    
    Args:
        service: Source service name
        level: Log level (INFO, WARN, ERROR, etc.)
        message: Raw log message
        exception_type: Exception class name if present
    
    Returns:
        MD5 hash signature string
    """
    components = [
        service,
        level,
        normalize_message(message),
    ]
    if exception_type:
        components.append(exception_type)

    key = "|".join(components)
    return hashlib.md5(key.encode()).hexdigest()


def extract_exception_type(message: str) -> Optional[str]:
    """
    Extract exception type from log message if present.
    
    Common patterns:
    - "PaymentGatewayTimeout: message"
    - "NullPointerException at line 42"
    - "Error: DatabaseConnectionError"
    """
    # Pattern: ExceptionName followed by : or whitespace
    match = re.search(r'\b([A-Z][a-zA-Z0-9]*(?:Error|Exception|Timeout|Failure))\b', message)
    if match:
        return match.group(1)
    
    return None
