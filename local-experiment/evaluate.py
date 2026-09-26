#!/usr/bin/env python3
"""Local IncidentLens evaluator: model-only and end-to-end incident analysis."""

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import yaml
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM

SEVERITIES = ["low", "medium", "high", "critical"]
DISPOSITIONS = ["NO_ACTION", "OBSERVE", "NEEDS_DEV", "NEEDS_ONCALL", "ESCALATE"]
REQUIRED_FIELDS = [
    "severity",
    "disposition",
    "confidence",
    "summary",
    "suspected_root_cause",
    "next_steps",
    "ticket_title",
    "ticket_body",
]
SIGNAL_LEVELS = {"WARN", "ERROR", "CRITICAL", "FATAL"}
TIMESTAMP_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\s+"
    r"(?P<level>TRACE|DEBUG|INFO|WARN|ERROR|CRITICAL|FATAL)\s+(?P<body>.*)$"
)
BRACKET_SOURCE_RE = re.compile(r"^\[([^\]]+)\]")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:/-]{2,}")
STOPWORDS = {
    "the", "and", "for", "from", "with", "into", "after", "before", "this", "that",
    "status", "namespace", "prod", "info", "warn", "error", "critical", "fatal", "event",
    "http", "https", "com", "api", "v1", "duration", "trace", "host", "region", "attempt",
    "service", "worker", "deployment", "container", "pod", "ops", "k8s", "true", "false",
}

SYSTEM_PROMPT = (
    "You are IncidentLens. Analyze the supplied backend logs and return one grounded JSON object only."
)

LEGACY_IMPACT_V2 = (
    "DECISION POLICY (impact-v2):\n"
    "Judge the incident from the full sequence, including the worst impact observed.\n"
    "Recovery or mitigation does not erase earlier impact.\n"
    "NO_ACTION is only for genuinely benign/non-incident sequences.\n"
)


def render_policy(cfg: Dict[str, Any], requested_version: Optional[str] = None) -> Tuple[str, str]:
    if requested_version == "legacy-v1":
        return "legacy-v1", ""
    if requested_version == "impact-v2":
        return "impact-v2", LEGACY_IMPACT_V2

    policy = cfg.get("policy", {}) or {}
    version = str(requested_version or policy.get("version", "impact-v4.1-compact-grounded"))
    configured_version = str(policy.get("version", "impact-v4.1-compact-grounded"))
    if version != configured_version:
        raise ValueError(
            f"Policy {version!r} is not defined in config. Available: legacy-v1, impact-v2, {configured_version}"
        )
    severity = policy.get("severity", {}) or {}
    disposition = policy.get("disposition", {}) or {}
    rules = list(policy.get("rules", []) or [])
    grounding = list(policy.get("grounding_rules", []) or [])
    parts = [f"DECISION POLICY ({version})", "", "SEVERITY:"]
    for label in SEVERITIES:
        parts.append(f"- {label}: {severity.get(label, '')}")
    parts += ["", "DISPOSITION:"]
    for label in DISPOSITIONS:
        parts.append(f"- {label}: {disposition.get(label, '')}")
    if rules:
        parts += ["", "DECISION RULES:"] + [f"- {x}" for x in rules]
    if grounding:
        parts += ["", "GROUNDING RULES:"] + [f"- {x}" for x in grounding]
    return version, "\n".join(parts).strip()


def build_user_prompt(logs: List[str], policy_text: str = "") -> str:
    schema = {
        "severity": "low|medium|high|critical",
        "disposition": "NO_ACTION|OBSERVE|NEEDS_DEV|NEEDS_ONCALL|ESCALATE",
        "confidence": 0.0,
        "summary": "...",
        "suspected_root_cause": "...",
        "next_steps": ["..."],
        "ticket_title": "...",
        "ticket_body": "...",
    }
    return (
        "Analyze the following backend log sequence.\n\n"
        + (policy_text + "\n\n" if policy_text else "")
        + "Return exactly ONE JSON object using this schema:\n"
        + json.dumps(schema, ensure_ascii=False)
        + "\n\nIf the sequence is genuinely benign, use severity=low and disposition=NO_ACTION.\n"
        + "Use concise evidence-grounded language. Do not claim impact not supported by the logs.\n"
        + "next_steps MUST be a JSON array of quoted strings only; do not place objects or key-value syntax inside it.\n"
        + "Do not output anything outside the JSON object.\n\nLOGS:\n"
        + "\n".join(logs)
    )


def load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def nested(cfg: Dict[str, Any], section: str, key: str, default: Any) -> Any:
    return cfg.get(section, {}).get(key, default)


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def apply_chat_template(tokenizer, messages, add_generation_prompt: bool, tokenize: bool):
    kwargs = dict(tokenize=tokenize, add_generation_prompt=add_generation_prompt)
    if tokenize:
        kwargs["return_dict"] = False
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def bnb_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


def choose_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def boolish(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Expected true/false/auto, got {value!r}")


def load_model(adapter_path: str, use_4bit: Any, trust_remote_code: bool):
    peft_cfg = PeftConfig.from_pretrained(adapter_path)
    base_model = peft_cfg.base_model_name_or_path
    dtype = choose_dtype()
    requested = boolish(use_4bit)
    can_4bit = torch.cuda.is_available() and bnb_available()
    quantized = can_4bit if requested == "auto" else bool(requested)
    if quantized and not can_4bit:
        raise RuntimeError("4-bit requested but CUDA + bitsandbytes are not both available")

    kwargs: Dict[str, Any] = {"dtype": dtype, "trust_remote_code": trust_remote_code}
    if quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": torch.cuda.current_device()}

    base = Qwen3_5ForCausalLM.from_pretrained(base_model, **kwargs)
    model = PeftModel.from_pretrained(base, adapter_path)
    if not quantized:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    model.eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=trust_remote_code)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, base_model, quantized, dtype


def record_tokens(text: str) -> set:
    tokens = set()
    for raw in TOKEN_RE.findall(text.lower()):
        token = raw.strip(".,:;/\\()[]{}<>\"'")
        if token in STOPWORDS or len(token) < 3:
            continue
        if token.startswith("trace=") or token.startswith("span_id="):
            continue
        if re.fullmatch(r"[0-9a-f]{12,}", token):
            continue
        token = re.sub(r"\b\d+(?:\.\d+)?(?:ms|s|mb|gb|gi|%)?\b", "<num>", token)
        if token != "<num>":
            tokens.add(token)
    return tokens


def bracket_source(body: str) -> Optional[str]:
    m = BRACKET_SOURCE_RE.match(body)
    return m.group(1).lower() if m else None


def relation_keys(body: str) -> set:
    """Small set of explicit resource names used only to relate nearby log lines."""
    text = body.lower()
    keys = set()
    src = bracket_source(body)
    if src:
        keys.add(src)
    keys.update(re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*-(?:service|api|worker|gateway|processor|dispatcher|ingest|producer)\b", text))
    keys.update(re.findall(r"\b(?:service|queue|topic|index)=([^\s,]+)", text))
    keys.update(re.findall(r"\b[a-z0-9.-]+\.(?:internal|example\.com|amazonaws\.com)\b", text))
    for pod in re.findall(r"\b([a-z0-9-]+-[0-9a-f]{8,10}-[a-z0-9]{4,6})\b", text):
        base = re.sub(r"-[0-9a-f]{8,10}-[a-z0-9]{4,6}$", "", pod)
        if base:
            keys.add(base)
    return keys


def read_log_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.rstrip("\n")
            m = TIMESTAMP_RE.match(line)
            if m:
                if current is not None:
                    records.append(current)
                current = {
                    "timestamp": parse_time(m.group("ts")),
                    "level": m.group("level"),
                    "body": m.group("body"),
                    "lines": [line],
                    "line_number": line_number,
                }
                current["source"] = bracket_source(current["body"])
                current["tokens"] = record_tokens(current["body"])
                current["keys"] = relation_keys(current["body"])
            elif current is not None:
                current["lines"].append(line)
                current["tokens"].update(record_tokens(line))
                current["keys"].update(relation_keys(line))
        if current is not None:
            records.append(current)
    if not records:
        raise ValueError(f"No timestamped log records found in {path}")
    records.sort(key=lambda x: x["timestamp"])
    return records


def similarity(record: Dict[str, Any], group: Dict[str, Any]) -> float:
    rt, gt = record["tokens"], group["tokens"]
    shared, union = rt & gt, rt | gt
    score = min(len(shared), 4) * 0.30
    if union:
        score += 1.5 * (len(shared) / len(union))
    if record["keys"] & group["keys"]:
        score += 3.0
    return score


def group_incidents(
    records: List[Dict[str, Any]],
    max_gap_seconds: float,
    info_context_seconds: float,
    signal_match_threshold: float,
    info_match_threshold: float,
    min_shared_tokens: int,
) -> List[Dict[str, Any]]:
    """Simple grouping: signal records seed/extend groups; matching INFO adds nearby context."""
    groups: List[Dict[str, Any]] = []

    for record in records:
        if record["level"] not in SIGNAL_LEVELS:
            continue
        best: Optional[Tuple[float, Dict[str, Any]]] = None
        for group in groups:
            gap = (record["timestamp"] - group["last_signal"]).total_seconds()
            if gap < 0 or gap > max_gap_seconds:
                continue
            shared = len(record["tokens"] & group["tokens"])
            key_match = bool(record["keys"] & group["keys"])
            score = similarity(record, group)
            if not key_match and shared < max(2, min_shared_tokens):
                continue
            if score >= signal_match_threshold and (best is None or score > best[0]):
                best = (score, group)

        if best is None:
            groups.append(
                {
                    "records": [record],
                    "signal_records": [record],
                    "start": record["timestamp"],
                    "end": record["timestamp"],
                    "last_signal": record["timestamp"],
                    "tokens": set(record["tokens"]),
                    "keys": set(record["keys"]),
                }
            )
        else:
            group = best[1]
            group["records"].append(record)
            group["signal_records"].append(record)
            group["last_signal"] = record["timestamp"]
            group["end"] = record["timestamp"]
            group["tokens"].update(record["tokens"])
            group["keys"].update(record["keys"])

    # Add only nearby INFO/DEBUG context that is lexically related to an already formed signal group.
    for record in records:
        if record["level"] in SIGNAL_LEVELS:
            continue
        best: Optional[Tuple[float, Dict[str, Any]]] = None
        for group in groups:
            lower = group["start"].timestamp() - info_context_seconds
            upper = group["end"].timestamp() + info_context_seconds
            if not (lower <= record["timestamp"].timestamp() <= upper):
                continue
            shared = len(record["tokens"] & group["tokens"])
            key_match = bool(record["keys"] & group["keys"])
            if not key_match and shared < max(2, min_shared_tokens):
                continue
            score = similarity(record, group)
            if score >= info_match_threshold and (best is None or score > best[0]):
                best = (score, group)
        if best is not None:
            group = best[1]
            group["records"].append(record)
            group["tokens"].update(record["tokens"])
            group["keys"].update(record["keys"])

    output = []
    for idx, group in enumerate(sorted(groups, key=lambda g: g["start"]), 1):
        recs = sorted(group["records"], key=lambda r: r["timestamp"])
        output.append(
            {
                "id": f"group-{idx:03d}",
                "start_time": recs[0]["timestamp"],
                "end_time": recs[-1]["timestamp"],
                "records": recs,
                "logs": [line for r in recs for line in r["lines"]],
                "line_numbers": sorted({int(r["line_number"]) for r in recs}),
                "signal_count": len(group["signal_records"]),
            }
        )
    return output


def load_ground_truth(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    incidents = data.get("incidents") if isinstance(data, dict) else data
    if not isinstance(incidents, list):
        raise ValueError("Ground truth must be a list or an object with an 'incidents' list")

    normalized = []
    for i, item in enumerate(incidents, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Ground truth incident {i} must be an object")
        if "start_time" not in item or "end_time" not in item:
            raise ValueError(f"Ground truth incident {i} needs start_time and end_time")
        expected = item.get("expected_output", {}) if isinstance(item.get("expected_output"), dict) else {}
        merged_expected = dict(expected)
        for field in REQUIRED_FIELDS:
            if field in item:
                merged_expected[field] = item[field]
        normalized.append(
            {
                "id": item.get("id", f"incident-{i:03d}"),
                "start_time": parse_time(item["start_time"]),
                "end_time": parse_time(item["end_time"]),
                "expected": merged_expected,
                "log_line_numbers": [int(x) for x in item.get("log_line_numbers", [])] if isinstance(item.get("log_line_numbers"), list) else [],
                "raw": item,
            }
        )
    normalized.sort(key=lambda x: x["start_time"])
    return normalized


def logs_for_window(records: List[Dict[str, Any]], start: datetime, end: datetime) -> List[str]:
    return [line for r in records if start <= r["timestamp"] <= end for line in r["lines"]]


def logs_for_ground_truth(records: List[Dict[str, Any]], incident: Dict[str, Any]) -> List[str]:
    """Prefer explicitly curated record line numbers for model-only evaluation.

    Time windows remain the source of truth for end-to-end detection.  The optional
    log_line_numbers field prevents overlapping incidents from contaminating the
    model-only context with unrelated events that happened at the same time.
    """
    raw = incident.get("raw", {})
    selected = raw.get("log_line_numbers")
    if isinstance(selected, list) and selected:
        wanted = {int(x) for x in selected}
        logs = [
            line
            for r in records
            if r.get("line_number") in wanted
            for line in r["lines"]
        ]
        if logs:
            return logs
    return logs_for_window(records, incident["start_time"], incident["end_time"])


def validate_output(obj: Any) -> List[str]:
    errors = []
    if not isinstance(obj, dict):
        return ["top-level output is not an object"]
    for field in REQUIRED_FIELDS:
        if field not in obj:
            errors.append(f"missing field: {field}")
    if errors:
        return errors
    if obj["severity"] not in SEVERITIES:
        errors.append("invalid severity")
    if obj["disposition"] not in DISPOSITIONS:
        errors.append("invalid disposition")
    confidence = obj["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        errors.append("confidence must be numeric in [0,1]")
    if not isinstance(obj["summary"], str):
        errors.append("summary must be a string")
    if obj["suspected_root_cause"] is not None and not isinstance(obj["suspected_root_cause"], str):
        errors.append("suspected_root_cause must be string or null")
    if not isinstance(obj["next_steps"], list) or not all(isinstance(x, str) for x in obj["next_steps"]):
        errors.append("next_steps must be a list of strings")
    if not isinstance(obj["ticket_title"], str):
        errors.append("ticket_title must be a string")
    if not isinstance(obj["ticket_body"], str):
        errors.append("ticket_body must be a string")
    return errors


def parse_model_output(raw: str) -> Tuple[Optional[Dict[str, Any]], bool, List[str]]:
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError as e:
        return None, False, [f"JSON parse error: {e}"]
    errors = validate_output(parsed)
    return parsed if isinstance(parsed, dict) else None, True, errors


def run_inference(model, tokenizer, logs: List[str], max_new_tokens: int, policy_text: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(logs, policy_text)},
    ]
    prompt = apply_chat_template(tokenizer, messages, True, False)
    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - started
    new_tokens = generated[0, encoded["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    parsed, valid_json, validation_errors = parse_model_output(raw)
    return {
        "raw_model_output": raw,
        "parsed_output": parsed,
        "valid_json": valid_json,
        "validation_errors": validation_errors,
        "schema_valid": valid_json and not validation_errors,
        "latency_seconds": latency,
    }


def print_case(index: int, case_id: str, start: datetime, end: datetime, logs: List[str]) -> None:
    print(f"\nIncident {index} ({case_id})")
    print("-" * 72)
    print(f"Start: {iso(start)}")
    print(f"End:   {iso(end)}")
    print(f"Logs:  {len(logs)}")
    print()
    for line in logs:
        print(line)
    print()


def print_prediction(result: Dict[str, Any]) -> None:
    print("Model prediction:")
    if result["parsed_output"] is not None:
        print(json.dumps(result["parsed_output"], indent=2, ensure_ascii=False))
    else:
        print(result["raw_model_output"])
    if result["validation_errors"]:
        print("Validation errors:", "; ".join(result["validation_errors"]))
    print(f"Latency: {result['latency_seconds']:.3f}s")


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def classification_metrics(expected: List[str], predicted: List[str], classes: List[str]) -> Dict[str, Any]:
    if not expected:
        return {
            "count": 0,
            "accuracy": None,
            "macro_f1": None,
            "macro_f1_all_classes": None,
            "macro_f1_supported_classes": None,
            "per_class": {},
            "confusion_matrix": {},
        }
    labels = classes + (["__INVALID__"] if "__INVALID__" in predicted else [])
    confusion = {e: {p: 0 for p in labels} for e in classes}
    correct = 0
    for e, p in zip(expected, predicted):
        if e not in classes:
            continue
        pp = p if p in labels else "__INVALID__"
        if "__INVALID__" not in confusion[e] and pp == "__INVALID__":
            for row in confusion.values():
                row["__INVALID__"] = 0
        confusion[e][pp] += 1
        correct += int(e == p)

    per_class = {}
    all_f1s, supported_f1s = [], []
    for cls in classes:
        tp = sum(1 for e, p in zip(expected, predicted) if e == cls and p == cls)
        fp = sum(1 for e, p in zip(expected, predicted) if e != cls and p == cls)
        fn = sum(1 for e, p in zip(expected, predicted) if e == cls and p != cls)
        support = sum(e == cls for e in expected)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        all_f1s.append(f1)
        if support > 0:
            supported_f1s.append(f1)
    macro_all = sum(all_f1s) / len(all_f1s) if all_f1s else None
    macro_supported = sum(supported_f1s) / len(supported_f1s) if supported_f1s else None
    return {
        "count": len(expected),
        "accuracy": correct / len(expected),
        "macro_f1": macro_all,  # backwards-compatible meaning
        "macro_f1_all_classes": macro_all,
        "macro_f1_supported_classes": macro_supported,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def field_complete(out: Optional[Dict[str, Any]], field: str) -> bool:
    if not isinstance(out, dict) or field not in out:
        return False
    value = out[field]
    disposition = out.get("disposition")
    if field == "next_steps":
        return isinstance(value, list) and (bool(value) or disposition == "NO_ACTION")
    if field == "suspected_root_cause":
        return (isinstance(value, str) and bool(value.strip())) or (value is None and disposition == "NO_ACTION")
    if field in {"ticket_title", "ticket_body"}:
        return isinstance(value, str) and (bool(value.strip()) or disposition == "NO_ACTION")
    if field == "summary":
        return isinstance(value, str) and bool(value.strip())
    return value is not None


def field_nonempty(out: Optional[Dict[str, Any]], field: str) -> bool:
    """Measure actual non-empty content, independent of NO_ACTION allowances."""
    if not isinstance(out, dict) or field not in out:
        return False
    value = out[field]
    if field == "next_steps":
        return isinstance(value, list) and bool(value) and all(isinstance(x, str) and bool(x.strip()) for x in value)
    if field == "suspected_root_cause":
        return isinstance(value, str) and bool(value.strip())
    if field in {"summary", "ticket_title", "ticket_body"}:
        return isinstance(value, str) and bool(value.strip())
    return value is not None


def classification_diagnostics(
    expected: List[str],
    predicted: List[str],
    confidences: List[Optional[float]],
    classes: List[str],
) -> Dict[str, Any]:
    """Expose label bias and confidence on correct vs incorrect decisions."""
    expected_dist = {c: sum(x == c for x in expected) for c in classes}
    predicted_dist = {c: sum(x == c for x in predicted) for c in classes}
    invalid_pred = sum(x not in classes for x in predicted)
    if invalid_pred:
        predicted_dist["__INVALID__"] = invalid_pred

    rank = {c: i for i, c in enumerate(classes)}
    under = correct = over = invalid = 0
    correct_conf: List[float] = []
    incorrect_conf: List[float] = []
    all_conf: List[float] = []
    for e, p, conf in zip(expected, predicted, confidences):
        is_correct = e == p
        if p not in rank or e not in rank:
            invalid += 1
        elif rank[p] < rank[e]:
            under += 1
        elif rank[p] > rank[e]:
            over += 1
        else:
            correct += 1
        if isinstance(conf, (int, float)) and not isinstance(conf, bool):
            c = float(conf)
            all_conf.append(c)
            (correct_conf if is_correct else incorrect_conf).append(c)

    return {
        "expected_distribution": expected_dist,
        "predicted_distribution": predicted_dist,
        "direction": {
            "underclassified": under,
            "correct": correct,
            "overclassified": over,
            "invalid": invalid,
        },
        "confidence": {
            "mean_all": statistics.fmean(all_conf) if all_conf else None,
            "mean_correct": statistics.fmean(correct_conf) if correct_conf else None,
            "mean_incorrect": statistics.fmean(incorrect_conf) if incorrect_conf else None,
            "correct_count": len(correct_conf),
            "incorrect_count": len(incorrect_conf),
        },
    }


def structured_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(predictions)
    if not n:
        return {}
    valid_json = sum(bool(p["valid_json"]) for p in predictions)
    schema_valid = sum(bool(p["schema_valid"]) for p in predictions)
    parsed = [p.get("parsed_output") for p in predictions]
    required_done = sum(sum(field_complete(out, f) for f in REQUIRED_FIELDS) for out in parsed)
    metrics = {
        "valid_json_pct": 100.0 * valid_json / n,
        "parse_failure_pct": 100.0 * (n - valid_json) / n,
        "schema_valid_pct": 100.0 * schema_valid / n,
        "required_field_completion_pct": 100.0 * required_done / (n * len(REQUIRED_FIELDS)),
    }
    names = {
        "summary": "summary_completion_pct",
        "suspected_root_cause": "root_cause_completion_pct",
        "next_steps": "next_steps_completion_pct",
        "ticket_title": "ticket_title_completion_pct",
        "ticket_body": "ticket_body_completion_pct",
    }
    for field, key in names.items():
        metrics[key] = 100.0 * sum(field_complete(out, field) for out in parsed) / n

    nonempty_names = {
        "summary": "non_empty_summary_pct",
        "suspected_root_cause": "non_empty_root_cause_pct",
        "next_steps": "non_empty_next_steps_pct",
        "ticket_title": "non_empty_ticket_title_pct",
        "ticket_body": "non_empty_ticket_body_pct",
    }
    for field, key in nonempty_names.items():
        metrics[key] = 100.0 * sum(field_nonempty(out, field) for out in parsed) / n
    return metrics


PAGER_CLAIM_RE = re.compile(r"\bpagerduty\b|\bpaged\b", re.I)
COMMANDER_CLAIM_RE = re.compile(r"\bincident commander\b", re.I)
BRIDGE_CLAIM_RE = re.compile(r"\bincident bridge\b", re.I)
FAILOVER_CLAIM_RE = re.compile(r"\bfail(?:ing)? over\b|\bfailover\b", re.I)
ROLLBACK_CLAIM_RE = re.compile(r"\brollback\b|\brolled back\b", re.I)
RECOVERY_CLAIM_RE = re.compile(r"\b(recovered|recovery confirmed|resolved|returned to baseline|normalized|restored)\b", re.I)
MEMORY_CLAIM_RE = re.compile(r"\boom\b|oomkilled|out[- ]of[- ]memory|memory pressure|memory leak|heap dump", re.I)
CRASHLOOP_CLAIM_RE = re.compile(r"\bcrashloopbackoff\b", re.I)
DATA_CLAIM_RE = re.compile(r"\bdata loss\b|\bdata corruption\b|corrupt", re.I)
TLS_CLAIM_RE = re.compile(r"\btls\b|certificate expir", re.I)
DNS_CLAIM_RE = re.compile(r"\bdns\b|getaddrinfo|enotfound", re.I)
P_LEVEL_RE = re.compile(r"\bP[0-3]\b", re.I)
CUSTOMER_CLAIM_RE = re.compile(
    r"\b(customer impact|customers? (?:were|are|unable|affected)|customer-facing (?:impact|failure|outage|operations)|widespread customer)\b",
    re.I,
)
WIDESPREAD_CLAIM_RE = re.compile(r"\bwidespread\b|\bmajor outage\b|\bbusiness-critical (?:failure|outage)\b", re.I)


def _observed_output_text(out: Dict[str, Any]) -> str:
    body = str(out.get("ticket_body", ""))
    # Do not judge the hypothesis section as an observed-fact claim.
    body = re.split(r"(?i)suspected root cause:", body, maxsplit=1)[0]
    return " ".join([str(out.get("summary", "")), str(out.get("ticket_title", "")), body])


def _customer_evidence(logs: str) -> bool:
    # Liveness/startup probe 503s are not customer evidence by themselves.
    request_5xx = re.search(r"\b(?:GET|POST|PUT|DELETE|PATCH)\s+\S+\s+5\d\d\b", logs, re.I)
    return bool(
        request_5xx
        or re.search(r"affected_requests=|payment authorization failed|failed requests?|customer", logs, re.I)
        or re.search(r"error_rate=\d+%", logs, re.I)
    )


def _widespread_evidence(logs: str) -> bool:
    rates = [int(x) for x in re.findall(r"error_rate=(\d+)%", logs, re.I)]
    affected = [int(x) for x in re.findall(r"affected_requests=(\d+)", logs, re.I)]
    return bool(
        (rates and max(rates) >= 50)
        or (affected and max(affected) >= 1000)
        or re.search(r"critical_path_unavailable=true|multi_service_failure=true|widespread_impact=true|business_critical=true", logs, re.I)
    )


def unsupported_claim_violations(log_lines: Sequence[str], out: Optional[Dict[str, Any]]) -> List[str]:
    if not out:
        return []
    logs = "\n".join(log_lines)
    text = _observed_output_text(out)
    violations: List[str] = []

    rules = [
        ("pagerduty_not_in_logs", PAGER_CLAIM_RE, re.compile(r"\bpagerduty\b", re.I)),
        ("incident_commander_not_in_logs", COMMANDER_CLAIM_RE, re.compile(r"\bincident commander\b", re.I)),
        ("incident_bridge_not_in_logs", BRIDGE_CLAIM_RE, re.compile(r"\bincident bridge\b", re.I)),
        ("failover_not_in_logs", FAILOVER_CLAIM_RE, re.compile(r"\bfail(?:ing)? over\b|\bfailover\b", re.I)),
        ("rollback_not_in_logs", ROLLBACK_CLAIM_RE, re.compile(r"\brollback\b|\brolled back\b", re.I)),
        ("recovery_not_in_logs", RECOVERY_CLAIM_RE, re.compile(r"\b(recovered|recovery confirmed|resolved|returned to baseline|normalized|restored|healthy again|draining)\b", re.I)),
        ("memory_oom_not_in_logs", MEMORY_CLAIM_RE, re.compile(r"oomkilled|\boom\b|out[- ]of[- ]memory|memory pressure|memory usage|rss=|heap", re.I)),
        ("crashloop_not_in_logs", CRASHLOOP_CLAIM_RE, re.compile(r"\bcrashloopbackoff\b", re.I)),
        ("data_integrity_not_in_logs", DATA_CLAIM_RE, re.compile(r"data loss|data corruption|corrupt|data integrity|checksum_mismatch|inconsistent state|constraint violation", re.I)),
        ("tls_not_in_logs", TLS_CLAIM_RE, re.compile(r"\btls\b|certificate|x509|ssl", re.I)),
        ("dns_not_in_logs", DNS_CLAIM_RE, re.compile(r"\bdns\b|getaddrinfo|enotfound|lookup .* no such host|coredns", re.I)),
    ]
    for name, claim_re, evidence_re in rules:
        if claim_re.search(text) and not evidence_re.search(logs):
            violations.append(name)

    if CUSTOMER_CLAIM_RE.search(text) and not _customer_evidence(logs):
        violations.append("customer_impact_not_supported")
    if WIDESPREAD_CLAIM_RE.search(text) and not _widespread_evidence(logs):
        violations.append("widespread_impact_not_supported")
    if P_LEVEL_RE.search(text):
        violations.append("invented_p_level")

    # Numeric percentages in observed claims should come directly from logs.
    claimed_pct = set(re.findall(r"\b(\d{1,3})%\b", text))
    logged_pct = set(re.findall(r"\b(\d{1,3})%\b", logs))
    if claimed_pct - logged_pct:
        violations.append("percentage_not_in_logs")
    return sorted(set(violations))


def grounding_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Conservative claim-evidence diagnostics; not a full semantic factuality score."""
    flagged_ids: List[str] = []
    by_type: Counter = Counter()
    details: List[Dict[str, Any]] = []
    for p in predictions:
        violations = unsupported_claim_violations(p.get("logs", []), p.get("parsed_output"))
        if not violations:
            continue
        incident_id = str(p.get("incident_id"))
        flagged_ids.append(incident_id)
        for v in violations:
            by_type[v] += 1
        details.append({"incident_id": incident_id, "violations": violations})
    n = len(predictions)
    return {
        "heuristic_name": "unsupported_observed_claims_v2",
        "flagged_count": len(flagged_ids),
        "flagged_pct": 100.0 * len(flagged_ids) / n if n else 0.0,
        "flagged_incident_ids": flagged_ids,
        "violations_by_type": dict(by_type),
        "details": details,
        "note": "Conservative heuristic; it checks several concrete claim/evidence mismatches but is not a full semantic factuality score.",
    }


def performance_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    values = [float(p["latency_seconds"]) for p in predictions]
    return {
        "mean_seconds": statistics.fmean(values) if values else None,
        "p50_seconds": percentile(values, 0.50),
        "p95_seconds": percentile(values, 0.95),
        "p99_seconds": percentile(values, 0.99),
    }


def overlap_ratio(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> float:
    if a_end <= a_start:
        a_end = a_start + timedelta(seconds=1)
    if b_end <= b_start:
        b_end = b_start + timedelta(seconds=1)
    overlap = max(0.0, (min(a_end, b_end) - max(a_start, b_start)).total_seconds())
    if overlap <= 0:
        return 0.0
    return overlap / min((a_end - a_start).total_seconds(), (b_end - b_start).total_seconds())


def line_overlap_f1(gt_lines: Sequence[int], pred_lines: Sequence[int]) -> float:
    a, b = set(int(x) for x in gt_lines), set(int(x) for x in pred_lines)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    precision = inter / len(b)
    recall = inter / len(a)
    return 2 * precision * recall / (precision + recall)


def detection_match(
    gt: List[Dict[str, Any]], predictions: List[Dict[str, Any]],
    line_threshold: float, temporal_threshold: float,
) -> Tuple[List[Tuple[int, int, float, str]], List[int], List[int]]:
    candidates = []
    for gi, g in enumerate(gt):
        gt_lines = g.get("log_line_numbers") or []
        for pi, p in enumerate(predictions):
            pred_lines = p.get("line_numbers") or []
            if gt_lines and pred_lines:
                score = line_overlap_f1(gt_lines, pred_lines)
                method = "line_f1"
                threshold = line_threshold
            else:
                score = overlap_ratio(g["start_time"], g["end_time"], p["start_time_dt"], p["end_time_dt"])
                method = "temporal_overlap"
                threshold = temporal_threshold
            if score >= threshold:
                candidates.append((score, gi, pi, method))
    candidates.sort(reverse=True, key=lambda x: x[0])
    used_g, used_p, matches = set(), set(), []
    for score, gi, pi, method in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi, score, method))
    return matches, [i for i in range(len(gt)) if i not in used_g], [i for i in range(len(predictions)) if i not in used_p]


def detection_metrics(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def text_token_f1(expected: Any, predicted: Any) -> Optional[float]:
    if expected is None:
        return None
    if isinstance(expected, list):
        expected = " ".join(str(x) for x in expected)
    if isinstance(predicted, list):
        predicted = " ".join(str(x) for x in predicted)
    if not isinstance(expected, str) or not isinstance(predicted, str):
        return 0.0
    a = Counter(re.findall(r"[a-z0-9]+", expected.lower()))
    b = Counter(re.findall(r"[a-z0-9]+", predicted.lower()))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    common = sum((a & b).values())
    precision = common / sum(b.values())
    recall = common / sum(a.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def content_error_types(expected: Dict[str, Any], predicted: Optional[Dict[str, Any]], threshold: float) -> List[str]:
    if not predicted:
        return []
    mapping = {
        "suspected_root_cause": "poor root cause",
        "summary": "poor summary",
        "next_steps": "poor next steps",
    }
    errors = []
    for field, label in mapping.items():
        if field not in expected or expected[field] in (None, "", []):
            continue
        score = text_token_f1(expected[field], predicted.get(field))
        if score is not None and score < threshold:
            errors.append(label)
    return errors


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if not k.endswith("_dt")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def run_model_only(
    model, tokenizer, records, gt, max_new_tokens: int, poor_text_f1_threshold: float, policy_version: str, policy_text: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    predictions, errors = [], []
    severity_expected, severity_pred, severity_conf = [], [], []
    disp_expected, disp_pred, disp_conf = [], [], []

    for i, incident in enumerate(gt, 1):
        logs = logs_for_ground_truth(records, incident)
        print_case(i, incident["id"], incident["start_time"], incident["end_time"], logs)
        result = run_inference(model, tokenizer, logs, max_new_tokens, policy_text)
        print_prediction(result)
        row = {
            "mode": "model-only",
            "policy_version": policy_version,
            "incident_id": incident["id"],
            "start_time": iso(incident["start_time"]),
            "end_time": iso(incident["end_time"]),
            "logs": logs,
            "expected_result": incident["expected"],
            **result,
        }
        predictions.append(row)

        predicted = result["parsed_output"] if result["schema_valid"] else None
        expected = incident["expected"]
        error_types = []
        if not result["valid_json"] or result["validation_errors"]:
            error_types.append("malformed output")
        if expected.get("severity") in SEVERITIES:
            severity_expected.append(expected["severity"])
            severity_pred.append(predicted.get("severity", "__INVALID__") if predicted else "__INVALID__")
            severity_conf.append(predicted.get("confidence") if predicted else None)
            if not predicted or predicted.get("severity") != expected["severity"]:
                error_types.append("wrong severity")
        if expected.get("disposition") in DISPOSITIONS:
            disp_expected.append(expected["disposition"])
            disp_pred.append(predicted.get("disposition", "__INVALID__") if predicted else "__INVALID__")
            disp_conf.append(predicted.get("confidence") if predicted else None)
            if not predicted or predicted.get("disposition") != expected["disposition"]:
                error_types.append("wrong disposition")
        error_types.extend(content_error_types(expected, predicted, poor_text_f1_threshold))
        claim_violations = unsupported_claim_violations(logs, predicted)
        if claim_violations:
            error_types.append("unsupported claim")
        if error_types:
            errors.append(
                {
                    "incident_id": incident["id"],
                    "error_types": sorted(set(error_types)),
                    "expected_result": expected,
                    "predicted_result": predicted,
                    "logs_given_to_model": logs,
                    "raw_model_response": result["raw_model_output"],
                    "parsed_response": result["parsed_output"],
                    "unsupported_claim_violations": claim_violations,
                    "latency_seconds": result["latency_seconds"],
                }
            )

    severity_metrics = classification_metrics(severity_expected, severity_pred, SEVERITIES)
    severity_metrics.update(classification_diagnostics(severity_expected, severity_pred, severity_conf, SEVERITIES))
    disposition_metrics = classification_metrics(disp_expected, disp_pred, DISPOSITIONS)
    disposition_metrics.update(classification_diagnostics(disp_expected, disp_pred, disp_conf, DISPOSITIONS))

    metrics = {
        "mode": "model-only",
        "policy_version": policy_version,
        "ground_truth_incidents": len(gt),
        "incident_detection": {
            "applicable": False,
            "reason": "Model-only mode uses exact ground-truth windows, so grouping/detection is intentionally bypassed.",
        },
        "severity": severity_metrics,
        "disposition": disposition_metrics,
        "structured_output": structured_metrics(predictions),
        "grounding": grounding_metrics(predictions),
        "performance": performance_metrics(predictions),
    }
    return predictions, errors, metrics


def run_end_to_end(
    model,
    tokenizer,
    records,
    gt,
    max_new_tokens: int,
    line_match_threshold: float,
    temporal_match_threshold: float,
    poor_text_f1_threshold: float,
    grouping_cfg: Dict[str, Any],
    policy_version: str,
    policy_text: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    groups = group_incidents(records, **grouping_cfg)
    all_predictions = []
    for i, group in enumerate(groups, 1):
        print_case(i, group["id"], group["start_time"], group["end_time"], group["logs"])
        result = run_inference(model, tokenizer, group["logs"], max_new_tokens, policy_text)
        print_prediction(result)
        all_predictions.append(
            {
                "mode": "end-to-end",
                "policy_version": policy_version,
                "incident_id": group["id"],
                "start_time": iso(group["start_time"]),
                "end_time": iso(group["end_time"]),
                "start_time_dt": group["start_time"],
                "end_time_dt": group["end_time"],
                "logs": group["logs"],
                "line_numbers": group.get("line_numbers", []),
                "signal_count": group["signal_count"],
                **result,
            }
        )

    meaningful = [
        p for p in all_predictions
        if p["schema_valid"] and p["parsed_output"].get("disposition") != "NO_ACTION"
    ]
    true_gt = [g for g in gt if g["expected"].get("disposition") != "NO_ACTION"]
    matches, missed_gt, false_pred = detection_match(true_gt, meaningful, line_match_threshold, temporal_match_threshold)

    severity_expected, severity_pred, severity_conf = [], [], []
    disp_expected, disp_pred, disp_conf = [], [], []
    errors = []

    for gi, pi, score, match_method in matches:
        g = true_gt[gi]
        p = meaningful[pi]
        expected = g["expected"]
        predicted = p["parsed_output"]
        error_types = []
        if expected.get("severity") in SEVERITIES:
            severity_expected.append(expected["severity"])
            severity_pred.append(predicted.get("severity", "__INVALID__"))
            severity_conf.append(predicted.get("confidence"))
            if predicted.get("severity") != expected["severity"]:
                error_types.append("wrong severity")
        if expected.get("disposition") in DISPOSITIONS:
            disp_expected.append(expected["disposition"])
            disp_pred.append(predicted.get("disposition", "__INVALID__"))
            disp_conf.append(predicted.get("confidence"))
            if predicted.get("disposition") != expected["disposition"]:
                error_types.append("wrong disposition")
        error_types.extend(content_error_types(expected, predicted, poor_text_f1_threshold))
        claim_violations = unsupported_claim_violations(p["logs"], predicted)
        if claim_violations:
            error_types.append("unsupported claim")
        if error_types:
            errors.append(
                {
                    "incident_id": g["id"],
                    "matched_prediction_id": p["incident_id"],
                    "match_score": score,
                    "match_method": match_method,
                    "error_types": sorted(set(error_types)),
                    "expected_result": expected,
                    "predicted_result": predicted,
                    "logs_given_to_model": p["logs"],
                    "raw_model_response": p["raw_model_output"],
                    "parsed_response": p["parsed_output"],
                    "unsupported_claim_violations": claim_violations,
                    "latency_seconds": p["latency_seconds"],
                }
            )

    for gi in missed_gt:
        g = true_gt[gi]
        best = None
        for p in all_predictions:
            if g.get("log_line_numbers") and p.get("line_numbers"):
                score = line_overlap_f1(g["log_line_numbers"], p["line_numbers"])
            else:
                score = overlap_ratio(g["start_time"], g["end_time"], p["start_time_dt"], p["end_time_dt"])
            if best is None or score > best[0]:
                best = (score, p)
        p = best[1] if best and best[0] > 0 else None
        etypes = ["missed incident"]
        if p and (not p["valid_json"] or p["validation_errors"]):
            etypes.append("malformed output")
        errors.append(
            {
                "incident_id": g["id"],
                "error_types": etypes,
                "expected_result": g["expected"],
                "predicted_result": p["parsed_output"] if p else None,
                "ground_truth_logs": logs_for_ground_truth(records, g),
                "logs_given_to_model": p["logs"] if p else [],
                "raw_model_response": p["raw_model_output"] if p else None,
                "parsed_response": p["parsed_output"] if p else None,
                "latency_seconds": p["latency_seconds"] if p else None,
            }
        )

    for pi in false_pred:
        p = meaningful[pi]
        errors.append(
            {
                "incident_id": p["incident_id"],
                "error_types": ["false positive"],
                "expected_result": None,
                "predicted_result": p["parsed_output"],
                "logs_given_to_model": p["logs"],
                "raw_model_response": p["raw_model_output"],
                "parsed_response": p["parsed_output"],
                "latency_seconds": p["latency_seconds"],
            }
        )

    # Malformed candidate outputs are useful even when they do not become a predicted incident.
    already = {(e.get("incident_id"), "malformed output") for e in errors if "malformed output" in e["error_types"]}
    for p in all_predictions:
        if not p["valid_json"] or p["validation_errors"]:
            key = (p["incident_id"], "malformed output")
            if key not in already:
                errors.append(
                    {
                        "incident_id": p["incident_id"],
                        "error_types": ["malformed output"],
                        "expected_result": None,
                        "predicted_result": p["parsed_output"],
                        "logs_given_to_model": p["logs"],
                        "raw_model_response": p["raw_model_output"],
                        "parsed_response": p["parsed_output"],
                        "latency_seconds": p["latency_seconds"],
                    }
                )

    severity_metrics = classification_metrics(severity_expected, severity_pred, SEVERITIES)
    severity_metrics.update(classification_diagnostics(severity_expected, severity_pred, severity_conf, SEVERITIES))
    disposition_metrics = classification_metrics(disp_expected, disp_pred, DISPOSITIONS)
    disposition_metrics.update(classification_diagnostics(disp_expected, disp_pred, disp_conf, DISPOSITIONS))

    metrics = {
        "mode": "end-to-end",
        "policy_version": policy_version,
        "ground_truth_incidents": len(true_gt),
        "candidate_windows": len(groups),
        "predicted_meaningful_incidents": len(meaningful),
        "incident_detection": detection_metrics(len(matches), len(false_pred), len(missed_gt)),
        "severity": severity_metrics,
        "disposition": disposition_metrics,
        "structured_output": structured_metrics(all_predictions),
        "grounding": grounding_metrics(all_predictions),
        "performance": performance_metrics(all_predictions),
    }
    return all_predictions, errors, metrics


def make_summary(metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
    s = metrics.get("structured_output", {})
    p = metrics.get("performance", {})
    sev = metrics.get("severity", {})
    disp = metrics.get("disposition", {})
    lines = [
        "# IncidentLens Evaluation Summary",
        "",
        f"- Mode: `{metrics.get('mode')}`",
        f"- Adapter: `{config.get('adapter')}`",
        f"- Log file: `{config.get('logs')}`",
        f"- Ground truth: `{config.get('ground_truth')}`",
        f"- Decision policy: `{config.get('policy_version')}`",
        "",
    ]
    det = metrics.get("incident_detection", {})
    if det.get("applicable") is False:
        lines += ["## Incident detection", "", f"Not applicable: {det.get('reason')}", ""]
    else:
        lines += [
            "## Incident detection",
            "",
            f"- TP / FP / FN: {det.get('tp')} / {det.get('fp')} / {det.get('fn')}",
            f"- Precision: {det.get('precision', 0):.4f}",
            f"- Recall: {det.get('recall', 0):.4f}",
            f"- F1: {det.get('f1', 0):.4f}",
            "",
        ]
    lines += [
        "## Analysis quality",
        "",
        f"- Severity accuracy: {sev.get('accuracy')}",
        f"- Severity macro F1 (all classes): {sev.get('macro_f1_all_classes')}"
        , f"- Severity macro F1 (supported classes): {sev.get('macro_f1_supported_classes')}",
        f"- Severity expected distribution: {sev.get('expected_distribution')}",
        f"- Severity predicted distribution: {sev.get('predicted_distribution')}",
        f"- Severity under/correct/over/invalid: {sev.get('direction')}",
        f"- Severity confidence correct/incorrect: {sev.get('confidence', {}).get('mean_correct')} / {sev.get('confidence', {}).get('mean_incorrect')}",
        f"- Disposition accuracy: {disp.get('accuracy')}",
        f"- Disposition macro F1 (all classes): {disp.get('macro_f1_all_classes')}"
        , f"- Disposition macro F1 (supported classes): {disp.get('macro_f1_supported_classes')}",
        f"- Disposition expected distribution: {disp.get('expected_distribution')}",
        f"- Disposition predicted distribution: {disp.get('predicted_distribution')}",
        f"- Disposition under/correct/over/invalid: {disp.get('direction')}",
        f"- Disposition confidence correct/incorrect: {disp.get('confidence', {}).get('mean_correct')} / {disp.get('confidence', {}).get('mean_incorrect')}",
        "",
        "## Structured output",
        "",
        f"- Valid JSON: {s.get('valid_json_pct')}%",
        f"- Parse failures: {s.get('parse_failure_pct')}%",
        f"- Schema valid: {s.get('schema_valid_pct')}%",
        f"- Required-field completion: {s.get('required_field_completion_pct')}%",
        f"- Non-empty summary: {s.get('non_empty_summary_pct')}%",
        f"- Non-empty root cause: {s.get('non_empty_root_cause_pct')}%",
        f"- Non-empty next steps: {s.get('non_empty_next_steps_pct')}%",
        f"- Non-empty ticket title: {s.get('non_empty_ticket_title_pct')}%",
        f"- Non-empty ticket body: {s.get('non_empty_ticket_body_pct')}%",
        "",
        "## Grounding heuristic",
        "",
        f"- Unsupported observed-claim flags: {metrics.get('grounding', {}).get('flagged_count')} ({metrics.get('grounding', {}).get('flagged_pct')}%)",
        f"- Violations by type: {metrics.get('grounding', {}).get('violations_by_type')}",
        "- This is a conservative heuristic; review flagged outputs manually.",
        "",
        "## Performance",
        "",
        f"- Mean: {p.get('mean_seconds')} s",
        f"- P50: {p.get('p50_seconds')} s",
        f"- P95: {p.get('p95_seconds')} s",
        f"- P99: {p.get('p99_seconds')} s",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a local IncidentLens LoRA adapter")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--adapter", required=True)
    p.add_argument("--logs")
    p.add_argument("--ground-truth")
    p.add_argument("--output")
    p.add_argument("--mode", choices=["model-only", "end-to-end"])
    p.add_argument("--use-4bit", choices=["auto", "true", "false"])
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--line-match-threshold", type=float)
    p.add_argument("--temporal-match-threshold", type=float)
    p.add_argument("--max-gap-seconds", type=float)
    p.add_argument("--info-context-seconds", type=float)
    p.add_argument("--policy-version")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    eval_cfg = cfg.get("evaluation", {})
    group_cfg_file = eval_cfg.get("grouping", {})

    logs_path = args.logs or eval_cfg.get("logs", "evaluation.log")
    gt_path = args.ground_truth or eval_cfg.get("ground_truth", "ground_truth.json")
    mode = args.mode or eval_cfg.get("mode", "end-to-end")
    output_arg = args.output or eval_cfg.get("output")
    output = Path(output_arg or f"results/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output.mkdir(parents=True, exist_ok=True)

    artifacts = [output / "evaluation_metrics.json", output / "predictions.jsonl", output / "error_analysis.json", output / "summary.md"]
    existing = [str(p) for p in artifacts if p.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing evaluation artifacts: " + ", ".join(existing))

    use_4bit = args.use_4bit or nested(cfg, "model", "use_4bit", "auto")
    trust_remote_code = bool(nested(cfg, "model", "trust_remote_code", False))
    max_new_tokens = args.max_new_tokens or nested(cfg, "inference", "max_new_tokens", 768)
    detection_cfg = eval_cfg.get("detection", {}) or {}
    line_match_threshold = args.line_match_threshold if args.line_match_threshold is not None else float(detection_cfg.get("line_overlap_f1_threshold", 0.30))
    temporal_match_threshold = args.temporal_match_threshold if args.temporal_match_threshold is not None else float(detection_cfg.get("temporal_overlap_threshold", 0.30))
    poor_text_f1_threshold = float(eval_cfg.get("poor_text_f1_threshold", 0.40))
    requested_policy = args.policy_version or eval_cfg.get("policy_version")
    policy_version, policy_text = render_policy(cfg, requested_policy)
    grouping_cfg = {
        "max_gap_seconds": args.max_gap_seconds if args.max_gap_seconds is not None else float(group_cfg_file.get("max_gap_seconds", 300)),
        "info_context_seconds": args.info_context_seconds if args.info_context_seconds is not None else float(group_cfg_file.get("info_context_seconds", 75)),
        "signal_match_threshold": float(group_cfg_file.get("signal_match_threshold", 1.25)),
        "info_match_threshold": float(group_cfg_file.get("info_match_threshold", 1.75)),
        "min_shared_tokens": int(group_cfg_file.get("min_shared_tokens", 1)),
    }

    records = read_log_records(logs_path)
    gt = load_ground_truth(gt_path)
    model, tokenizer, base_model, quantized, dtype = load_model(args.adapter, use_4bit, trust_remote_code)

    run_config = {
        "adapter": args.adapter,
        "base_model": base_model,
        "logs": logs_path,
        "ground_truth": gt_path,
        "output": str(output),
        "mode": mode,
        "max_new_tokens": max_new_tokens,
        "use_4bit": use_4bit,
        "quantized_4bit": quantized,
        "dtype": str(dtype).replace("torch.", ""),
        "line_match_threshold": line_match_threshold,
        "temporal_match_threshold": temporal_match_threshold,
        "poor_text_f1_threshold": poor_text_f1_threshold,
        "policy_version": policy_version,
        "grouping": grouping_cfg,
    }

    if mode == "model-only":
        predictions, errors, metrics = run_model_only(
            model, tokenizer, records, gt, max_new_tokens, poor_text_f1_threshold, policy_version, policy_text
        )
    else:
        predictions, errors, metrics = run_end_to_end(
            model,
            tokenizer,
            records,
            gt,
            max_new_tokens,
            line_match_threshold,
            temporal_match_threshold,
            poor_text_f1_threshold,
            grouping_cfg,
            policy_version,
            policy_text,
        )

    with (output / "evaluation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_jsonl(output / "predictions.jsonl", predictions)
    with (output / "error_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    config_target = output / "config.json"
    if config_target.exists():
        config_target = output / "evaluation_config.json"
    with config_target.open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    with (output / "summary.md").open("w", encoding="utf-8") as f:
        f.write(make_summary(metrics, run_config))

    print("\n" + "=" * 72)
    print("Evaluation complete")
    print(json.dumps(metrics, indent=2))
    print(f"Predictions:    {output / 'predictions.jsonl'}")
    print(f"Error analysis: {output / 'error_analysis.json'}")
    print(f"Summary:        {output / 'summary.md'}")


if __name__ == "__main__":
    main()
