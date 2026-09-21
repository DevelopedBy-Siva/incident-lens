import re
import hashlib


def normalize_message(message: str) -> str:
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


def generate_signature(source: str, parsed_log) -> str:
    components = [
        source,
        parsed_log.level,
        normalize_message(parsed_log.message),
    ]
    if parsed_log.exception_type:
        components.append(parsed_log.exception_type)

    key = "|".join(components)
    return hashlib.md5(key.encode()).hexdigest()
