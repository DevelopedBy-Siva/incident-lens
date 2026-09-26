#!/usr/bin/env python3
"""Local QLoRA/LoRA fine-tuning for the IncidentLens incident-analysis model."""

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen3_5ForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

SEVERITIES = {"low", "medium", "high", "critical"}
DISPOSITIONS = {"NO_ACTION", "OBSERVE", "NEEDS_DEV", "NEEDS_ONCALL", "ESCALATE"}
REQUIRED_OUTPUT_FIELDS = {
    "severity",
    "disposition",
    "confidence",
    "summary",
    "suspected_root_cause",
    "next_steps",
    "ticket_title",
    "ticket_body",
}

ACTIVE_POLICY_VERSION = "impact-v4.1-compact-grounded"
ACTIVE_POLICY_TEXT = ""

SYSTEM_PROMPT = (
    "You are IncidentLens. Analyze the supplied backend logs and return one grounded JSON object only."
)


def render_policy(cfg: Dict[str, Any]) -> Tuple[str, str]:
    policy = cfg.get("policy", {}) or {}
    version = str(policy.get("version", "impact-v4.1-compact-grounded"))
    severity = policy.get("severity", {}) or {}
    disposition = policy.get("disposition", {}) or {}
    rules = list(policy.get("rules", []) or [])
    grounding = list(policy.get("grounding_rules", []) or [])

    parts = [f"DECISION POLICY ({version})", "", "SEVERITY:"]
    for label in ("low", "medium", "high", "critical"):
        parts.append(f"- {label}: {severity.get(label, '')}")
    parts += ["", "DISPOSITION:"]
    for label in ("NO_ACTION", "OBSERVE", "NEEDS_DEV", "NEEDS_ONCALL", "ESCALATE"):
        parts.append(f"- {label}: {disposition.get(label, '')}")
    if rules:
        parts += ["", "DECISION RULES:"] + [f"- {x}" for x in rules]
    if grounding:
        parts += ["", "GROUNDING RULES:"] + [f"- {x}" for x in grounding]
    return version, "\n".join(parts).strip()


def build_user_prompt(logs: List[str]) -> str:
    return build_user_prompt_from_text("\n".join(logs))


def build_user_prompt_from_text(log_text: str) -> str:
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
        (ACTIVE_POLICY_TEXT + "\n\n" if ACTIVE_POLICY_TEXT else "")
        + "OUTPUT JSON SCHEMA:\n"
        + json.dumps(schema, ensure_ascii=False)
        + "\nBenign => low/NO_ACTION. next_steps must contain strings only.\n\nLOGS:\n"
        + log_text
    )


def load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def nested(cfg: Dict[str, Any], section: str, key: str, default: Any) -> Any:
    return cfg.get(section, {}).get(key, default)


def boolish(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    if text == "auto":
        return "auto"
    raise ValueError(f"Expected true/false/auto, got {value!r}")


def parse_target_modules(value: Any) -> Union[str, List[str]]:
    if value is None:
        return "all-linear"
    if isinstance(value, list):
        modules = [str(x).strip() for x in value if str(x).strip()]
        if not modules:
            raise ValueError("lora_target_modules list cannot be empty")
        return modules
    text = str(value).strip()
    if text == "all-linear":
        return text
    modules = [x.strip() for x in text.split(",") if x.strip()]
    if not modules:
        raise ValueError("lora_target_modules cannot be empty")
    return modules


def validate_example(obj: Dict[str, Any], line_no: int) -> None:
    if set(obj.keys()) != {"input", "expected_output"}:
        raise ValueError(f"Line {line_no}: expected top-level keys input and expected_output")
    inp = obj["input"]
    out = obj["expected_output"]
    if not isinstance(inp, dict) or not isinstance(out, dict):
        raise ValueError(f"Line {line_no}: input and expected_output must be objects")
    logs = inp.get("logs")
    if not isinstance(logs, list) or not logs or not all(isinstance(x, str) and x.strip() for x in logs):
        raise ValueError(f"Line {line_no}: input.logs must be a non-empty list of strings")
    missing = REQUIRED_OUTPUT_FIELDS - set(out)
    if missing:
        raise ValueError(f"Line {line_no}: missing output fields: {sorted(missing)}")
    if out["severity"] not in SEVERITIES:
        raise ValueError(f"Line {line_no}: invalid severity {out['severity']!r}")
    if out["disposition"] not in DISPOSITIONS:
        raise ValueError(f"Line {line_no}: invalid disposition {out['disposition']!r}")
    if not isinstance(out["confidence"], (int, float)) or isinstance(out["confidence"], bool):
        raise ValueError(f"Line {line_no}: confidence must be numeric")
    if not 0.0 <= float(out["confidence"]) <= 1.0:
        raise ValueError(f"Line {line_no}: confidence must be in [0, 1]")
    if out["suspected_root_cause"] is not None and not isinstance(out["suspected_root_cause"], str):
        raise ValueError(f"Line {line_no}: suspected_root_cause must be string or null")
    if not isinstance(out["next_steps"], list) or not all(isinstance(x, str) for x in out["next_steps"]):
        raise ValueError(f"Line {line_no}: next_steps must be a list of strings")
    for field in ("summary", "ticket_title", "ticket_body"):
        if not isinstance(out[field], str):
            raise ValueError(f"Line {line_no}: {field} must be a string")


def load_examples(path: str) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_no}: invalid JSON: {e}") from e
            validate_example(obj, line_no)
            examples.append(obj)
    if len(examples) < 2:
        raise ValueError("Need at least 2 training examples")
    return examples


def scenario_group_key(example: Dict[str, Any], normalize_source: bool = True) -> str:
    metadata = example.get("input", {}).get("metadata", {}) or {}
    explicit = metadata.get("scenario_group_id")
    if explicit:
        return str(explicit)
    return normalize_scenario(example, normalize_source=normalize_source)


def normalize_scenario(example: Dict[str, Any], normalize_source: bool = True) -> str:
    """Create a deterministic fingerprint that collapses obvious synthetic slot variations."""
    text = "\n".join(example["input"]["logs"]).lower()
    replacements = (
        (ISO_TS_RE, "<ts>"),
        (UUID_RE, "<uuid>"),
        (IP_RE, "<ip>"),
        (HEX_RE, "<hex>"),
        (VERSION_RE, "<version>"),
        (API_PATH_RE, "<endpoint>"),
        (NUMBER_RE, "<num>"),
    )
    for pattern, replacement in replacements:
        text = pattern.sub(replacement, text)
    if normalize_source:
        text = BRACKET_SOURCE_RE.sub("[<source>]", text)
    text = re.sub(r"\s+", " ", text).strip()
    incident_type = str(example["input"].get("metadata", {}).get("incident_type", "__unknown__"))
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{incident_type}:{digest}"


def random_split(
    examples: List[Dict[str, Any]], validation_split: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    indices = list(range(len(examples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_n = max(1, int(round(len(indices) * validation_split)))
    val_n = min(val_n, len(indices) - 1)
    val_idx = set(indices[:val_n])
    train = [x for i, x in enumerate(examples) if i not in val_idx]
    val = [x for i, x in enumerate(examples) if i in val_idx]
    stats = {
        "strategy": "random",
        "requested_validation_fraction": validation_split,
        "actual_validation_fraction": len(val) / len(examples),
        "train_examples": len(train),
        "validation_examples": len(val),
    }
    return train, val, stats


def group_aware_split(
    examples: List[Dict[str, Any]],
    validation_split: float,
    seed: int,
    normalize_source: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Split entire normalized scenario groups together.

    Constraint: never move the final scenario group of an incident_type into validation.
    If an incident_type has only one clean scenario group, it remains train-only rather
    than leaking near-duplicate variants across train and validation.
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    group_type: Dict[str, str] = {}
    type_group_counts: Counter = Counter()

    for idx, example in enumerate(examples):
        key = scenario_group_key(example, normalize_source=normalize_source)
        groups[key].append(idx)
        incident_type = str(example["input"].get("metadata", {}).get("incident_type", "__unknown__"))
        group_type[key] = incident_type

    for key in groups:
        type_group_counts[group_type[key]] += 1

    target_val = max(1, int(round(len(examples) * validation_split)))
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)

    selected_groups: set = set()
    selected_per_type: Counter = Counter()
    val_count = 0

    # One transparent pass over shuffled groups. Whole groups are selected only when
    # doing so keeps at least one group of that incident_type in training.
    for key, indices in items:
        if val_count >= target_val:
            break
        incident_type = group_type[key]
        if selected_per_type[incident_type] >= type_group_counts[incident_type] - 1:
            continue

        before = abs(target_val - val_count)
        after = abs(target_val - (val_count + len(indices)))
        # While far below the target, accept groups for diversity. Near the target,
        # only accept a group when it improves the requested split size.
        if val_count < target_val * 0.85 or after <= before:
            selected_groups.add(key)
            selected_per_type[incident_type] += 1
            val_count += len(indices)

    # Fallback for unusual tiny datasets.
    if not selected_groups:
        candidates = [
            (key, idxs)
            for key, idxs in items
            if type_group_counts[group_type[key]] > 1
        ]
        if not candidates:
            raise ValueError(
                "Group-aware validation split is impossible: every incident type has only one scenario group. "
                "Use --split-strategy random or provide more scenario templates."
            )
        key, indices = min(candidates, key=lambda kv: len(kv[1]))
        selected_groups.add(key)
        selected_per_type[group_type[key]] += 1
        val_count = len(indices)

    val_indices = {
        idx
        for key in selected_groups
        for idx in groups[key]
    }
    train = [x for i, x in enumerate(examples) if i not in val_indices]
    val = [x for i, x in enumerate(examples) if i in val_indices]

    train_groups = {
        scenario_group_key(x, normalize_source=normalize_source) for x in train
    }
    val_groups = {
        scenario_group_key(x, normalize_source=normalize_source) for x in val
    }
    overlap = train_groups & val_groups
    if overlap:
        raise RuntimeError(f"Group-aware split failed: {len(overlap)} scenario groups overlap")

    train_types = {
        str(x["input"].get("metadata", {}).get("incident_type", "__unknown__")) for x in train
    }
    val_types = {
        str(x["input"].get("metadata", {}).get("incident_type", "__unknown__")) for x in val
    }
    all_types = train_types | val_types

    stats = {
        "strategy": "group-aware",
        "normalize_source": normalize_source,
        "requested_validation_fraction": validation_split,
        "actual_validation_fraction": len(val) / len(examples),
        "target_validation_examples": target_val,
        "train_examples": len(train),
        "validation_examples": len(val),
        "scenario_groups_total": len(groups),
        "train_scenario_groups": len(train_groups),
        "validation_scenario_groups": len(val_groups),
        "overlapping_scenario_groups": len(overlap),
        "incident_types_total": len(all_types),
        "incident_types_in_train": len(train_types),
        "incident_types_in_validation": len(val_types),
        "incident_types_without_clean_validation_group": sorted(all_types - val_types),
    }
    return train, val, stats


RECOVERY_RE = re.compile(r"\b(recovered|recovery|restored|normalized|stabilized|healthy|draining|resolved|rollback complete|mitigation succeeded)\b", re.I)
LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)")


def label_counts(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    sev = Counter(x["expected_output"]["severity"] for x in examples)
    disp = Counter(x["expected_output"]["disposition"] for x in examples)
    pair = Counter((x["expected_output"]["severity"], x["expected_output"]["disposition"]) for x in examples)
    return {
        "severity": dict(sev),
        "disposition": dict(disp),
        "severity_disposition": {f"{a}|{b}": n for (a, b), n in sorted(pair.items())},
    }


def dataset_audit(examples: List[Dict[str, Any]], normalize_source: bool = True) -> Dict[str, Any]:
    by_type_pairs: Dict[str, set] = defaultdict(set)
    groups = set()
    recovery_pairs = Counter()
    chronological = 0
    parseable = 0
    target_counter = Counter()
    profile_counts = Counter()
    target_grounding_violations = Counter()
    non_benign_summary_counter = Counter()

    for ex in examples:
        inp, out = ex["input"], ex["expected_output"]
        md = inp.get("metadata", {}) or {}
        it = str(md.get("incident_type", "__unknown__"))
        pair = (out["severity"], out["disposition"])
        by_type_pairs[it].add(pair)
        groups.add(scenario_group_key(ex, normalize_source=normalize_source))
        if md.get("policy_profile"):
            profile_counts[str(md["policy_profile"])] += 1
        for violation in md.get("target_grounding_violations", []) or []:
            target_grounding_violations[str(violation)] += 1
        if RECOVERY_RE.search("\n".join(inp.get("logs", []))):
            recovery_pairs[pair] += 1
        target_counter[json.dumps(out, sort_keys=True, ensure_ascii=False)] += 1

        times = []
        for log in inp.get("logs", []):
            m = LOG_TS_RE.match(log)
            if m:
                try:
                    times.append(datetime.fromisoformat(m.group(1).replace("Z", "+00:00")))
                except ValueError:
                    pass
        if times:
            parseable += 1
            if all(a <= b for a, b in zip(times, times[1:])):
                chronological += 1

    benign_like = {
        k for k in by_type_pairs
        if k.startswith("benign_") or k in {"transient_cache_miss", "single_4xx_error", "healthcheck_timeout_noise"}
    }
    non_benign = set(by_type_pairs) - benign_like
    non_benign_multi = sum(len(by_type_pairs[k]) > 1 for k in non_benign)
    for ex in examples:
        it = str((ex["input"].get("metadata", {}) or {}).get("incident_type", "__unknown__"))
        if it in non_benign:
            non_benign_summary_counter[str(ex["expected_output"].get("summary", ""))] += 1
    non_benign_duplicate_summaries = sum(n - 1 for n in non_benign_summary_counter.values() if n > 1)
    high_urgency_recovered = sum(
        n for (sev, disp), n in recovery_pairs.items()
        if sev in {"high", "critical"} or disp in {"NEEDS_ONCALL", "ESCALATE"}
    )
    duplicate_examples = sum(n - 1 for n in target_counter.values() if n > 1)

    audit = {
        "examples": len(examples),
        **label_counts(examples),
        "scenario_groups": len(groups),
        "incident_types": len(by_type_pairs),
        "incident_types_with_multiple_label_pairs": sum(len(v) > 1 for v in by_type_pairs.values()),
        "non_benign_incident_types": len(non_benign),
        "non_benign_incident_types_with_multiple_label_pairs": non_benign_multi,
        "non_benign_multilabel_type_fraction": (non_benign_multi / len(non_benign)) if non_benign else 1.0,
        "incident_type_pair_counts": {k: len(v) for k, v in sorted(by_type_pairs.items())},
        "recovered_examples_by_pair": {f"{a}|{b}": n for (a, b), n in sorted(recovery_pairs.items())},
        "recovered_high_urgency_examples": high_urgency_recovered,
        "timestamp_parseable_examples": parseable,
        "chronological_examples": chronological,
        "chronological_pct": 100.0 * chronological / parseable if parseable else 100.0,
        "unique_exact_targets": len(target_counter),
        "duplicate_exact_target_examples": duplicate_examples,
        "duplicate_exact_target_pct": 100.0 * duplicate_examples / len(examples) if examples else 0.0,
        "target_grounding_violation_count": sum(target_grounding_violations.values()),
        "target_grounding_violations": dict(target_grounding_violations),
        "unique_non_benign_summaries": len(non_benign_summary_counter),
        "non_benign_duplicate_summary_examples": non_benign_duplicate_summaries,
        "non_benign_duplicate_summary_pct": (100.0 * non_benign_duplicate_summaries / sum(non_benign_summary_counter.values())) if non_benign_summary_counter else 0.0,
        "policy_profiles": dict(profile_counts),
    }
    return audit


def enforce_quality_gates(audit: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    q = ((cfg.get("training", {}) or {}).get("audit", {}) or {})
    if not bool(q.get("fail_on_quality_gate", True)):
        return
    failures = []
    if audit["chronological_pct"] < float(q.get("min_chronological_pct", 99.0)):
        failures.append(f"chronological_pct={audit['chronological_pct']:.2f}")
    if bool(q.get("require_all_severities", True)):
        missing = SEVERITIES - set(audit["severity"])
        if missing:
            failures.append(f"missing severities={sorted(missing)}")
    if bool(q.get("require_all_dispositions", True)):
        missing = DISPOSITIONS - set(audit["disposition"])
        if missing:
            failures.append(f"missing dispositions={sorted(missing)}")
    if bool(q.get("require_recovered_high_urgency_examples", True)) and audit["recovered_high_urgency_examples"] <= 0:
        failures.append("no recovered high-urgency examples")
    min_multi = float(q.get("min_non_benign_multilabel_type_fraction", 0.75))
    if audit["non_benign_multilabel_type_fraction"] < min_multi:
        failures.append(
            f"non_benign_multilabel_type_fraction={audit['non_benign_multilabel_type_fraction']:.3f} < {min_multi:.3f}"
        )
    if bool(q.get("require_zero_target_grounding_violations", True)) and audit.get("target_grounding_violation_count", 0) != 0:
        failures.append(f"target_grounding_violation_count={audit.get('target_grounding_violation_count')}")
    max_dup_target = float(q.get("max_duplicate_exact_target_pct", 15.0))
    if audit.get("duplicate_exact_target_pct", 0.0) > max_dup_target:
        failures.append(f"duplicate_exact_target_pct={audit['duplicate_exact_target_pct']:.2f} > {max_dup_target:.2f}")
    max_dup_summary = float(q.get("max_non_benign_duplicate_summary_pct", 10.0))
    if audit.get("non_benign_duplicate_summary_pct", 0.0) > max_dup_summary:
        failures.append(f"non_benign_duplicate_summary_pct={audit['non_benign_duplicate_summary_pct']:.2f} > {max_dup_summary:.2f}")
    if failures:
        raise ValueError("Dataset quality gate failed: " + "; ".join(failures))


def group_stratified_split(
    examples: List[Dict[str, Any]], validation_split: float, seed: int, normalize_source: bool
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Select whole scenario groups while approximating labels and broad type coverage."""
    groups: Dict[str, List[int]] = defaultdict(list)
    group_type: Dict[str, str] = {}
    for i, ex in enumerate(examples):
        g = scenario_group_key(ex, normalize_source=normalize_source)
        groups[g].append(i)
        group_type[g] = str(ex["input"].get("metadata", {}).get("incident_type", "__unknown__"))

    type_groups: Dict[str, set] = defaultdict(set)
    for g, t in group_type.items():
        type_groups[t].add(g)

    target_n = max(1, round(len(examples) * validation_split))
    full_sev = Counter(x["expected_output"]["severity"] for x in examples)
    full_disp = Counter(x["expected_output"]["disposition"] for x in examples)
    full_pair = Counter((x["expected_output"]["severity"], x["expected_output"]["disposition"]) for x in examples)
    target_sev = {k: full_sev[k] * validation_split for k in SEVERITIES}
    target_disp = {k: full_disp[k] * validation_split for k in DISPOSITIONS}
    target_pair = {k: full_pair[k] * validation_split for k in full_pair}

    rng = random.Random(seed)
    remaining = list(groups)
    rng.shuffle(remaining)
    selected = set()
    selected_by_type = Counter()
    val_n = 0
    cur_sev = Counter()
    cur_disp = Counter()
    cur_pair = Counter()

    def group_counts(g: str):
        sev = Counter(examples[i]["expected_output"]["severity"] for i in groups[g])
        disp = Counter(examples[i]["expected_output"]["disposition"] for i in groups[g])
        pair = Counter((examples[i]["expected_output"]["severity"], examples[i]["expected_output"]["disposition"]) for i in groups[g])
        return sev, disp, pair

    cached = {g: group_counts(g) for g in remaining}

    def can_select(g: str) -> bool:
        t = group_type[g]
        return selected_by_type[t] < len(type_groups[t]) - 1

    def score_after(g: str) -> float:
        size = len(groups[g])
        new_n = val_n + size
        sev_g, disp_g, pair_g = cached[g]
        size_err = abs(new_n - target_n) / max(target_n, 1)
        sev_err = sum(abs((cur_sev[k] + sev_g[k]) - target_sev[k]) / max(target_sev[k], 1.0) for k in SEVERITIES)
        disp_err = sum(abs((cur_disp[k] + disp_g[k]) - target_disp[k]) / max(target_disp[k], 1.0) for k in DISPOSITIONS)
        pair_err = sum(abs((cur_pair[k] + pair_g[k]) - target_pair[k]) / max(target_pair[k], 1.0) for k in full_pair)
        missing_bonus = 0.0
        for k in SEVERITIES:
            if cur_sev[k] == 0 and sev_g[k] > 0:
                missing_bonus -= 1.5
        for k in DISPOSITIONS:
            if cur_disp[k] == 0 and disp_g[k] > 0:
                missing_bonus -= 1.5
        for k in full_pair:
            if cur_pair[k] == 0 and pair_g[k] > 0:
                missing_bonus -= 1.0
        t = group_type[g]
        type_bonus = -0.75 if selected_by_type[t] == 0 and len(type_groups[t]) > 1 else 0.0
        return 2.0 * size_err + sev_err + disp_err + 0.75 * pair_err + missing_bonus + type_bonus

    while val_n < target_n and remaining:
        candidates = [g for g in remaining if can_select(g)]
        if not candidates:
            break
        best = min(candidates, key=score_after)
        selected.add(best)
        remaining.remove(best)
        selected_by_type[group_type[best]] += 1
        sev_g, disp_g, pair_g = cached[best]
        cur_sev.update(sev_g)
        cur_disp.update(disp_g)
        cur_pair.update(pair_g)
        val_n += len(groups[best])
        if (
            val_n >= target_n * 0.95
            and all(cur_sev[k] > 0 for k in SEVERITIES)
            and all(cur_disp[k] > 0 for k in DISPOSITIONS)
            and all(cur_pair[k] > 0 for k in full_pair)
        ):
            break

    # Post-pass 1: guarantee every observed label pair is represented when possible.
    for pair_key in full_pair:
        if cur_pair[pair_key] > 0:
            continue
        candidates = []
        for g in remaining:
            if not can_select(g):
                continue
            if cached[g][2][pair_key] > 0:
                candidates.append(g)
        if candidates:
            g = min(candidates, key=lambda x: len(groups[x]))
            selected.add(g); remaining.remove(g); selected_by_type[group_type[g]] += 1
            sev_g, disp_g, pair_g = cached[g]
            cur_sev.update(sev_g); cur_disp.update(disp_g); cur_pair.update(pair_g); val_n += len(groups[g])

    # Post-pass 2: broaden incident-type coverage without allowing extreme split growth.
    max_val_n = max(target_n, int(round(len(examples) * min(validation_split * 1.35, 0.15))))
    missing_types = [t for t in sorted(type_groups) if selected_by_type[t] == 0 and len(type_groups[t]) > 1]
    for t in missing_types:
        if val_n >= max_val_n:
            break
        candidates = [g for g in remaining if group_type[g] == t and can_select(g)]
        if not candidates:
            continue
        g = min(candidates, key=lambda x: len(groups[x]))
        if val_n + len(groups[g]) > max_val_n:
            continue
        selected.add(g); remaining.remove(g); selected_by_type[t] += 1
        sev_g, disp_g, pair_g = cached[g]
        cur_sev.update(sev_g); cur_disp.update(disp_g); cur_pair.update(pair_g); val_n += len(groups[g])

    if not selected:
        raise ValueError("Could not create group-stratified validation split")

    val_indices = {i for g in selected for i in groups[g]}
    train = [x for i, x in enumerate(examples) if i not in val_indices]
    val = [x for i, x in enumerate(examples) if i in val_indices]
    train_groups = {scenario_group_key(x, normalize_source) for x in train}
    val_groups = {scenario_group_key(x, normalize_source) for x in val}
    overlap = train_groups & val_groups
    if overlap:
        raise RuntimeError(f"Group-stratified split leaked {len(overlap)} groups")

    train_types = {str(x["input"].get("metadata", {}).get("incident_type", "__unknown__")) for x in train}
    val_types = {str(x["input"].get("metadata", {}).get("incident_type", "__unknown__")) for x in val}
    stats = {
        "strategy": "group-stratified",
        "requested_validation_fraction": validation_split,
        "actual_validation_fraction": len(val) / len(examples),
        "target_validation_examples": target_n,
        "train_examples": len(train),
        "validation_examples": len(val),
        "scenario_groups_total": len(groups),
        "train_scenario_groups": len(train_groups),
        "validation_scenario_groups": len(val_groups),
        "overlapping_scenario_groups": len(overlap),
        "incident_types_total": len(train_types | val_types),
        "incident_types_in_train": len(train_types),
        "incident_types_in_validation": len(val_types),
        "incident_types_without_clean_validation_group": sorted((train_types | val_types) - val_types),
        "train_labels": label_counts(train),
        "validation_labels": label_counts(val),
    }
    return train, val, stats


def apply_chat_template(tokenizer, messages, add_generation_prompt: bool, tokenize: bool):
    # Transformers 5.x may default return_dict=True. We need a plain token list.
    kwargs: Dict[str, Any] = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
    }
    if tokenize:
        kwargs["return_dict"] = False
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def render_ids(tokenizer, log_text: str, target: str) -> Tuple[List[int], List[int]]:
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt_from_text(log_text)},
    ]
    full_messages = prompt_messages + [{"role": "assistant", "content": target}]
    prompt_ids = list(apply_chat_template(tokenizer, prompt_messages, True, True))
    full_ids = list(apply_chat_template(tokenizer, full_messages, False, True))

    if full_ids[: len(prompt_ids)] != prompt_ids:
        # Conservative fallback for chat templates whose tokenized generation prompt
        # is not returned as an exact prefix in one pass.
        prompt_text = apply_chat_template(tokenizer, prompt_messages, True, False)
        full_text = apply_chat_template(tokenizer, full_messages, False, False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(
                "Could not identify the assistant-token boundary from the model chat template."
            )
    return list(prompt_ids), list(full_ids)


def truncate_log_text(tokenizer, text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= token_budget:
        return text

    marker = "\n[... log context truncated ...]\n"
    marker_ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
    if token_budget <= len(marker_ids) + 4:
        return tokenizer.decode(ids[-token_budget:], skip_special_tokens=True)

    usable = token_budget - len(marker_ids)
    head_n = usable // 2
    tail_n = usable - head_n
    kept = ids[:head_n] + marker_ids + ids[-tail_n:]
    return tokenizer.decode(kept, skip_special_tokens=True, clean_up_tokenization_spaces=False)


class IncidentDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_length: int, name: str):
        self.rows: List[Dict[str, List[int]]] = []
        original_lengths: List[int] = []
        final_lengths: List[int] = []
        target_lengths: List[int] = []
        truncated_logs = 0

        for idx, example in enumerate(examples):
            logs_text = "\n".join(example["input"]["logs"])
            target = json.dumps(example["expected_output"], ensure_ascii=False, separators=(",", ":"))

            prompt_ids, full_ids = render_ids(tokenizer, logs_text, target)
            original_len = len(full_ids)
            original_target = full_ids[len(prompt_ids):]
            if not original_target:
                raise ValueError(f"{name} example {idx}: assistant target tokenization is empty")

            if len(full_ids) > max_length:
                empty_prompt_ids, empty_full_ids = render_ids(tokenizer, "", target)
                empty_target = empty_full_ids[len(empty_prompt_ids):]
                # The assistant segment should not depend on the log contents.
                target_ids = original_target if original_target else empty_target
                fixed_tokens = len(empty_prompt_ids)
                available_for_logs = max_length - fixed_tokens - len(target_ids)
                if available_for_logs <= 0:
                    raise ValueError(
                        f"{name} example {idx}: max_sequence_length={max_length} cannot fit the fixed prompt "
                        f"plus the complete assistant target ({fixed_tokens + len(target_ids)} tokens). "
                        "Increase max_sequence_length or shorten the target text."
                    )

                budget = available_for_logs
                while True:
                    shortened_logs = truncate_log_text(tokenizer, logs_text, budget)
                    prompt_ids, full_ids = render_ids(tokenizer, shortened_logs, target)
                    if len(full_ids) <= max_length:
                        break
                    overflow = len(full_ids) - max_length
                    budget -= overflow + 8
                    if budget <= 0:
                        raise ValueError(
                            f"{name} example {idx}: unable to fit prompt while preserving the complete target."
                        )

                final_target = full_ids[len(prompt_ids):]
                if final_target != original_target:
                    raise RuntimeError(
                        f"{name} example {idx}: target tokens changed while truncating log context; refusing to train."
                    )
                truncated_logs += 1

            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            if not any(x != -100 for x in labels):
                raise ValueError(f"{name} example {idx}: no assistant tokens remain after encoding")

            self.rows.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels,
                }
            )
            original_lengths.append(original_len)
            final_lengths.append(len(full_ids))
            target_lengths.append(len(full_ids) - len(prompt_ids))

        if not self.rows:
            raise ValueError(f"{name}: no usable examples")

        self.stats = {
            "examples": len(self.rows),
            "max_sequence_length": max_length,
            "log_context_truncated_examples": truncated_logs,
            "log_context_truncated_percent": 100.0 * truncated_logs / len(self.rows),
            "target_truncated_examples": 0,
            "original_token_length_max": max(original_lengths),
            "original_token_length_mean": sum(original_lengths) / len(original_lengths),
            "final_token_length_max": max(final_lengths),
            "assistant_target_token_length_max": max(target_lengths),
            "assistant_target_token_length_mean": sum(target_lengths) / len(target_lengths),
        }
        print(
            f"{name} encoding: {len(self.rows)} examples, "
            f"log-context truncated={truncated_logs} ({self.stats['log_context_truncated_percent']:.1f}%), "
            f"target truncated=0, max final tokens={self.stats['final_token_length_max']}"
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.rows[index]


@dataclass
class CausalCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id
        batch_ids, batch_mask, batch_labels = [], [], []
        for row in features:
            pad = max_len - len(row["input_ids"])
            batch_ids.append(row["input_ids"] + [pad_id] * pad)
            batch_mask.append(row["attention_mask"] + [0] * pad)
            batch_labels.append(row["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


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


def build_model(model_name: str, use_4bit: Any, trust_remote_code: bool, gradient_checkpointing: bool):
    dtype = choose_dtype()
    requested = boolish(use_4bit)
    can_4bit = torch.cuda.is_available() and bnb_available()
    quantized = can_4bit if requested == "auto" else bool(requested)
    if quantized and not can_4bit:
        raise RuntimeError("4-bit requested but CUDA + bitsandbytes are not both available")

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "dtype": dtype,
    }
    if quantized:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = {"": torch.cuda.current_device()}

    model = Qwen3_5ForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.config.use_cache = False
    if quantized:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model, quantized, dtype


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune IncidentLens locally with LoRA/QLoRA")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--data")
    p.add_argument("--output")
    p.add_argument("--model")
    p.add_argument("--use-4bit", choices=["auto", "true", "false"])
    p.add_argument("--lora-r", type=int)
    p.add_argument("--lora-alpha", type=int)
    p.add_argument("--lora-dropout", type=float)
    p.add_argument(
        "--lora-target-modules",
        help="all-linear or comma-separated modules, e.g. q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--epochs", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--gradient-accumulation-steps", type=int)
    p.add_argument("--max-sequence-length", type=int)
    p.add_argument("--validation-split", type=float)
    p.add_argument("--split-strategy", choices=["group-stratified", "group-aware", "random"])
    p.add_argument("--group-normalize-source", choices=["true", "false"])
    p.add_argument("--seed", type=int)
    p.add_argument("--audit-only", action="store_true")
    return p.parse_args()


def main() -> None:
    global ACTIVE_POLICY_VERSION, ACTIVE_POLICY_TEXT
    args = parse_args()
    cfg = load_yaml(args.config)
    ACTIVE_POLICY_VERSION, ACTIVE_POLICY_TEXT = render_policy(cfg)

    data_path = args.data or nested(cfg, "training", "data", "dataset_v4.jsonl")
    target_modules_raw = (
        args.lora_target_modules if args.lora_target_modules is not None
        else nested(cfg, "training", "lora_target_modules", "all-linear")
    )
    values = {
        "model_name": args.model or nested(cfg, "model", "name", "Qwen/Qwen3.5-4B"),
        "use_4bit": args.use_4bit or nested(cfg, "model", "use_4bit", "auto"),
        "trust_remote_code": bool(nested(cfg, "model", "trust_remote_code", False)),
        "lora_r": args.lora_r if args.lora_r is not None else nested(cfg, "training", "lora_r", 8),
        "lora_alpha": args.lora_alpha if args.lora_alpha is not None else nested(cfg, "training", "lora_alpha", 16),
        "lora_dropout": args.lora_dropout if args.lora_dropout is not None else nested(cfg, "training", "lora_dropout", 0.05),
        "lora_target_modules": parse_target_modules(target_modules_raw),
        "learning_rate": args.learning_rate if args.learning_rate is not None else nested(cfg, "training", "learning_rate", 1e-4),
        "epochs": args.epochs if args.epochs is not None else nested(cfg, "training", "epochs", 1),
        "batch_size": args.batch_size if args.batch_size is not None else nested(cfg, "training", "batch_size", 1),
        "gradient_accumulation_steps": args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else nested(cfg, "training", "gradient_accumulation_steps", 1),
        "max_sequence_length": args.max_sequence_length if args.max_sequence_length is not None else nested(cfg, "training", "max_sequence_length", 1024),
        "validation_split": args.validation_split if args.validation_split is not None else nested(cfg, "training", "validation_split", 0.1),
        "split_strategy": args.split_strategy or nested(cfg, "training", "split_strategy", "group-stratified"),
        "group_normalize_source": boolish(args.group_normalize_source if args.group_normalize_source is not None else nested(cfg, "training", "group_normalize_source", True)),
        "seed": args.seed if args.seed is not None else nested(cfg, "training", "seed", 42),
        "logging_steps": int(nested(cfg, "training", "logging_steps", 10)),
        "gradient_checkpointing": bool(nested(cfg, "training", "gradient_checkpointing", True)),
        "policy_version": ACTIVE_POLICY_VERSION,
    }
    if not 0 < values["validation_split"] < 1:
        raise ValueError("validation_split must be between 0 and 1")

    set_seed(values["seed"])
    random.seed(values["seed"])
    examples = load_examples(data_path)
    audit = dataset_audit(examples, normalize_source=bool(values["group_normalize_source"]))
    enforce_quality_gates(audit, cfg)

    print(f"Dataset audit: {audit['examples']} examples, {audit['scenario_groups']} scenario groups")
    print(f"Labels severity={audit['severity']} disposition={audit['disposition']}")
    print(
        "Non-benign incident types with multiple label pairs: "
        f"{audit['non_benign_incident_types_with_multiple_label_pairs']}/{audit['non_benign_incident_types']}"
    )
    print(
        f"Chronological={audit['chronological_pct']:.1f}% | "
        f"recovered high-urgency examples={audit['recovered_high_urgency_examples']} | "
        f"duplicate exact targets={audit['duplicate_exact_target_pct']:.1f}% | "
        f"non-benign duplicate summaries={audit['non_benign_duplicate_summary_pct']:.1f}% | "
        f"target-grounding violations={audit['target_grounding_violation_count']}"
    )
    if args.audit_only:
        print(json.dumps(audit, indent=2))
        return

    output_arg = args.output or nested(cfg, "training", "output", None)
    output_dir = Path(output_arg or f"results/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if values["split_strategy"] == "group-stratified":
        train_examples, val_examples, split_stats = group_stratified_split(
            examples, values["validation_split"], values["seed"], bool(values["group_normalize_source"])
        )
    elif values["split_strategy"] == "group-aware":
        train_examples, val_examples, split_stats = group_aware_split(
            examples, values["validation_split"], values["seed"], bool(values["group_normalize_source"])
        )
    elif values["split_strategy"] == "random":
        train_examples, val_examples, split_stats = random_split(examples, values["validation_split"], values["seed"])
    else:
        raise ValueError(f"Unknown split_strategy: {values['split_strategy']}")

    print(f"Loaded {len(examples)} examples: train={len(train_examples)} validation={len(val_examples)}")
    print(f"Split: {split_stats['strategy']} | actual validation={split_stats['actual_validation_fraction']:.3f}")
    if "scenario_groups_total" in split_stats:
        print(
            "Scenario groups: "
            f"total={split_stats['scenario_groups_total']} train={split_stats['train_scenario_groups']} "
            f"validation={split_stats['validation_scenario_groups']} overlap={split_stats['overlapping_scenario_groups']}"
        )
        print(
            "Incident types represented in validation: "
            f"{split_stats['incident_types_in_validation']}/{split_stats['incident_types_total']}"
        )
    if split_stats.get("validation_labels"):
        print("Validation labels:", split_stats["validation_labels"])

    tokenizer = AutoTokenizer.from_pretrained(values["model_name"], trust_remote_code=values["trust_remote_code"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = IncidentDataset(train_examples, tokenizer, values["max_sequence_length"], name="train")
    val_ds = IncidentDataset(val_examples, tokenizer, values["max_sequence_length"], name="validation")

    model, quantized, dtype = build_model(
        values["model_name"], values["use_4bit"], values["trust_remote_code"], values["gradient_checkpointing"]
    )
    lora = LoraConfig(
        r=values["lora_r"], lora_alpha=values["lora_alpha"], lora_dropout=values["lora_dropout"],
        bias="none", task_type="CAUSAL_LM", target_modules=values["lora_target_modules"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_state"),
        num_train_epochs=values["epochs"],
        per_device_train_batch_size=values["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=values["gradient_accumulation_steps"],
        learning_rate=values["learning_rate"],
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=values["logging_steps"],
        report_to="none",
        seed=values["seed"],
        data_seed=values["seed"],
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        gradient_checkpointing=values["gradient_checkpointing"],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=CausalCollator(tokenizer),
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    train_loss = float(train_result.metrics.get("train_loss", math.nan))
    val_loss = float(eval_metrics.get("eval_loss", math.nan))
    print(f"Training loss:   {train_loss:.6f}")
    print(f"Validation loss: {val_loss:.6f}")

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    encoding = {"train": train_ds.stats, "validation": val_ds.stats}
    saved_config = {
        **values,
        "data": str(data_path),
        "output": str(output_dir),
        "quantized_4bit": quantized,
        "dtype": str(dtype).replace("torch.", ""),
        "train_examples": len(train_ds),
        "validation_examples": len(val_ds),
        "dataset_audit": audit,
        "split": split_stats,
        "encoding": encoding,
    }
    metrics = {
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "eval_runtime_seconds": eval_metrics.get("eval_runtime"),
        "policy_version": ACTIVE_POLICY_VERSION,
        "dataset_audit": audit,
        "split": split_stats,
        "encoding": encoding,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(saved_config, f, indent=2)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "dataset_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print(f"Saved adapter: {adapter_dir}")


if __name__ == "__main__":
    main()
