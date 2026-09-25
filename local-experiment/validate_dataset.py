#!/usr/bin/env python3
"""
Validate training dataset quality and report statistics.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


def normalize_for_duplicate_detection(example: dict) -> str:
    """Normalize example to detect template duplicates."""
    ex_str = json.dumps(example, sort_keys=True)
    
    # Remove variable elements
    ex_str = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z', 'TIMESTAMP', ex_str)
    ex_str = re.sub(r'-\w{4}', '-XXXX', ex_str)
    ex_str = re.sub(r'trace=\w+', 'trace=TRACE', ex_str)
    ex_str = re.sub(r'req_\w+', 'req_REQ', ex_str)
    ex_str = re.sub(r'host=\S+', 'host=HOST', ex_str)
    ex_str = re.sub(r'uptime=\d+s', 'uptime=XXXs', ex_str)
    ex_str = re.sub(r'duration=[\d.]+ms', 'duration=XXms', ex_str)
    ex_str = re.sub(r'lag=\d+', 'lag=XXX', ex_str)
    ex_str = re.sub(r'p\d{2}=\d+ms', 'pXX=XXms', ex_str)
    ex_str = re.sub(r'baseline=\d+ms', 'baseline=XXms', ex_str)
    ex_str = re.sub(r'depth=\d+', 'depth=XXX', ex_str)
    ex_str = re.sub(r'items=\d+', 'items=XXX', ex_str)
    ex_str = re.sub(r'batch=\d+', 'batch=XXX', ex_str)
    ex_str = re.sub(r'replicas=\d+', 'replicas=X', ex_str)
    ex_str = re.sub(r'v\d+\.\d+\.\d+', 'vX.X.X', ex_str)
    
    return hashlib.sha256(ex_str.encode()).hexdigest()


def validate_schema(examples: list[dict]) -> tuple[bool, list[str]]:
    """Validate canonical 8-field schema."""
    errors = []
    valid_severities = {"low", "medium", "high", "critical"}
    valid_dispositions = {"NO_ACTION", "OBSERVE", "NEEDS_DEV", "NEEDS_ONCALL", "ESCALATE"}
    
    required_output_fields = {
        "severity", "disposition", "confidence", "summary",
        "suspected_root_cause", "next_steps", "ticket_title", "ticket_body"
    }
    
    for i, example in enumerate(examples, 1):
        # Check structure
        if "input" not in example or "expected_output" not in example:
            errors.append(f"Record {i}: missing 'input' or 'expected_output'")
            continue
        
        output = example["expected_output"]
        
        # Check required fields present
        missing = required_output_fields - set(output.keys())
        if missing:
            errors.append(f"Record {i}: missing fields {missing}")
        
        # Validate severity
        severity = output.get("severity", "")
        if severity == "benign":
            errors.append(f"Record {i}: INVALID severity 'benign' - use 'low' with NO_ACTION instead")
        elif severity not in valid_severities:
            errors.append(f"Record {i}: invalid severity '{severity}' - must be {valid_severities}")
        
        # Validate disposition
        disposition = output.get("disposition", "")
        if disposition not in valid_dispositions:
            errors.append(f"Record {i}: invalid disposition '{disposition}' - must be {valid_dispositions}")
        
        # Validate confidence
        confidence = output.get("confidence")
        if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
            errors.append(f"Record {i}: confidence must be float in [0, 1], got {confidence}")
        
        # Validate next_steps
        if not isinstance(output.get("next_steps"), list):
            errors.append(f"Record {i}: next_steps must be a list")
        
        # Validate actionable incidents have details
        if disposition in {"NEEDS_DEV", "NEEDS_ONCALL", "ESCALATE"}:
            if not output.get("ticket_title", "").strip():
                errors.append(f"Record {i}: actionable incident ({disposition}) missing ticket_title")
            if not output.get("ticket_body", "").strip():
                errors.append(f"Record {i}: actionable incident ({disposition}) missing ticket_body")
            if not output.get("next_steps"):
                errors.append(f"Record {i}: actionable incident ({disposition}) missing next_steps")
    
    return len(errors) == 0, errors


def analyze_dataset(dataset_path: Path) -> dict:
    """Comprehensive dataset analysis."""
    examples = []
    
    with dataset_path.open('r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"ERROR: Line {line_num}: Invalid JSON - {e}")
                sys.exit(1)
    
    if not examples:
        print("ERROR: Dataset is empty")
        sys.exit(1)
    
    # Validate schema
    valid, errors = validate_schema(examples)
    
    # Statistics
    severity_counts = Counter()
    disposition_counts = Counter()
    incident_type_counts = Counter()
    log_counts = []
    
    no_action_benign = 0
    
    for example in examples:
        output = example["expected_output"]
        inp = example["input"]
        
        severity = output.get("severity", "unknown")
        disposition = output.get("disposition", "unknown")
        
        severity_counts[severity] += 1
        disposition_counts[disposition] += 1
        
        if severity == "low" and disposition == "NO_ACTION":
            no_action_benign += 1
        
        incident_type = inp.get("metadata", {}).get("incident_type", "unknown")
        incident_type_counts[incident_type] += 1
        
        log_counts.append(len(inp.get("logs", [])))
    
    # Duplicate detection
    exact_hashes = [hashlib.sha256(json.dumps(ex, sort_keys=True).encode()).hexdigest() for ex in examples]
    exact_counter = Counter(exact_hashes)
    exact_duplicates = sum(1 for count in exact_counter.values() if count > 1)
    
    normalized_hashes = [normalize_for_duplicate_detection(ex) for ex in examples]
    normalized_counter = Counter(normalized_hashes)
    
    return {
        "valid": valid,
        "errors": errors,
        "total": len(examples),
        "severity_counts": dict(severity_counts),
        "disposition_counts": dict(disposition_counts),
        "incident_type_counts": dict(incident_type_counts),
        "no_action_benign": no_action_benign,
        "log_counts": {
            "min": min(log_counts) if log_counts else 0,
            "max": max(log_counts) if log_counts else 0,
            "avg": sum(log_counts) / len(log_counts) if log_counts else 0,
            "single_log": sum(1 for x in log_counts if x == 1),
            "multi_log_2_3": sum(1 for x in log_counts if 2 <= x <= 3),
            "multi_log_4plus": sum(1 for x in log_counts if x >= 4),
        },
        "duplicates": {
            "exact": exact_duplicates,
            "unique_templates": len(normalized_counter),
            "most_common_template_count": normalized_counter.most_common(1)[0][1] if normalized_counter else 0,
        }
    }


def print_report(stats: dict):
    """Print human-readable validation report."""
    print("=" * 80)
    print("DATASET VALIDATION REPORT")
    print("=" * 80)
    
    if not stats["valid"]:
        print("\n❌ VALIDATION FAILED")
        print(f"\nErrors found: {len(stats['errors'])}")
        for error in stats["errors"][:10]:
            print(f"  • {error}")
        if len(stats["errors"]) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
        print("\n" + "=" * 80)
        return
    
    print("\n✅ SCHEMA VALIDATION PASSED")
    
    print(f"\nTotal Records: {stats['total']}")
    
    print("\nSeverity Distribution:")
    total = stats['total']
    for sev in ['low', 'medium', 'high', 'critical']:
        count = stats['severity_counts'].get(sev, 0)
        pct = 100 * count / total if total else 0
        print(f"  {sev:10s} {count:5d} ({pct:5.1f}%)")
    
    print(f"\n  No-action/benign (low+NO_ACTION): {stats['no_action_benign']} ({100*stats['no_action_benign']/total:.1f}%)")
    
    print("\nDisposition Distribution:")
    for disp in sorted(stats['disposition_counts'].keys()):
        count = stats['disposition_counts'][disp]
        pct = 100 * count / total if total else 0
        print(f"  {disp:15s} {count:5d} ({pct:5.1f}%)")
    
    print("\nTop 10 Incident Types:")
    sorted_types = sorted(stats['incident_type_counts'].items(), key=lambda x: x[1], reverse=True)
    for inc_type, count in sorted_types[:10]:
        pct = 100 * count / total if total else 0
        print(f"  {inc_type:45s} {count:4d} ({pct:5.1f}%)")
    
    if len(sorted_types) > 10:
        print(f"  ... and {len(sorted_types) - 10} more types")
    
    log_stats = stats['log_counts']
    print("\nLog Sequence Statistics:")
    print(f"  Minimum:      {log_stats['min']}")
    print(f"  Maximum:      {log_stats['max']}")
    print(f"  Average:      {log_stats['avg']:.1f}")
    print(f"  Single log:   {log_stats['single_log']} ({100*log_stats['single_log']/total:.1f}%)")
    print(f"  2-3 logs:     {log_stats['multi_log_2_3']} ({100*log_stats['multi_log_2_3']/total:.1f}%)")
    print(f"  4+ logs:      {log_stats['multi_log_4plus']} ({100*log_stats['multi_log_4plus']/total:.1f}%)")
    
    dup_stats = stats['duplicates']
    print("\nDuplicate Analysis:")
    print(f"  Exact duplicates: {dup_stats['exact']}")
    print(f"  Unique templates: {dup_stats['unique_templates']}")
    print(f"  Most repeated template: {dup_stats['most_common_template_count']} occurrences")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Validate training dataset")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/train.jsonl"),
        help="Path to training dataset (default: data/train.jsonl)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON"
    )
    
    args = parser.parse_args()
    
    if not args.dataset.exists():
        print(f"ERROR: Dataset not found: {args.dataset}")
        sys.exit(1)
    
    stats = analyze_dataset(args.dataset)
    
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print_report(stats)
    
    sys.exit(0 if stats["valid"] else 1)


if __name__ == "__main__":
    main()
