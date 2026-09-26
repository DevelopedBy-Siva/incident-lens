"""
Validate training dataset.

Checks schema, distributions, and data quality.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from data_loader import load_training_data, validate_training_data


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def check_for_duplicates(examples) -> dict:
    """Check for duplicate or near-duplicate examples."""
    seen_prompts = {}
    duplicates = []
    
    for idx, example in enumerate(examples):
        # Create a simple hash of the logs
        log_hash = hashlib.md5(
            "".join(sorted(example.logs)).encode()
        ).hexdigest()
        
        if log_hash in seen_prompts:
            duplicates.append({
                "example_index": idx,
                "duplicate_of": seen_prompts[log_hash],
                "service": example.service,
            })
        else:
            seen_prompts[log_hash] = idx
    
    return {
        "duplicate_count": len(duplicates),
        "duplicates": duplicates[:10],  # Show first 10
    }


def analyze_dataset_balance(examples) -> dict:
    """Analyze dataset balance and potential issues."""
    analysis = {
        "total": len(examples),
        "by_service": {},
        "by_severity": {},
        "by_disposition": {},
        "benign_vs_actionable": {},
        "field_completeness": {},
    }
    
    # Count by service
    for ex in examples:
        service = ex.service
        analysis["by_service"][service] = analysis["by_service"].get(service, 0) + 1
    
    # Count by severity
    for ex in examples:
        sev = ex.severity
        analysis["by_severity"][sev] = analysis["by_severity"].get(sev, 0) + 1
    
    # Count by disposition
    for ex in examples:
        disp = ex.disposition
        analysis["by_disposition"][disp] = analysis["by_disposition"].get(disp, 0) + 1
    
    # Benign vs actionable
    benign_count = sum(
        1 for ex in examples
        if ex.disposition in ["NO_ACTION", "OBSERVE"]
    )
    actionable_count = len(examples) - benign_count
    
    analysis["benign_vs_actionable"] = {
        "benign": benign_count,
        "actionable": actionable_count,
        "ratio": benign_count / len(examples) if examples else 0,
    }
    
    # Field completeness
    fields = ["summary", "suspected_root_cause", "next_steps", "ticket_title", "ticket_body"]
    for field in fields:
        non_empty = 0
        for ex in examples:
            value = getattr(ex, field, None)
            if value and value not in ["", []]:
                non_empty += 1
        analysis["field_completeness"][field] = non_empty / len(examples) if examples else 0
    
    return analysis


def validate(args):
    """Run validation."""
    dataset_path = Path(args.dataset)
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Dataset Validation")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"{'='*60}\n")
    
    # Compute hash
    print("Computing dataset hash...")
    file_hash = compute_file_hash(dataset_path)
    print(f"  SHA256: {file_hash}\n")
    
    # Load data
    print("Loading dataset...")
    try:
        examples = load_training_data(dataset_path)
        print(f"  Loaded {len(examples)} examples\n")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        sys.exit(1)
    
    # Validate schema
    print("Validating schema...")
    report = validate_training_data(examples)
    
    print(f"\nValidation Results:")
    print(f"  Total: {report['total_examples']}")
    print(f"  Valid: {report['valid_examples']} ✓")
    print(f"  Invalid: {report['invalid_examples']} {'✗' if report['invalid_examples'] > 0 else ''}")
    
    if report['invalid_examples'] > 0:
        print(f"\n⚠️  Validation Errors:")
        for error in report['errors'][:10]:  # Show first 10
            print(f"    Example {error['example_index']}: {error['errors']}")
        if len(report['errors']) > 10:
            print(f"    ... and {len(report['errors']) - 10} more")
    
    print(f"\nSeverity Distribution:")
    for sev, count in sorted(report['severity_distribution'].items()):
        pct = count / report['total_examples'] * 100
        print(f"  {sev:8s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nDisposition Distribution:")
    for disp, count in sorted(report['disposition_distribution'].items()):
        pct = count / report['total_examples'] * 100
        print(f"  {disp:12s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nBenign vs Actionable:")
    print(f"  Benign (NO_ACTION/OBSERVE): {report['benign_count']}")
    print(f"  Actionable: {report['actionable_count']}")
    
    # Check for duplicates
    print(f"\nChecking for duplicates...")
    dup_report = check_for_duplicates(examples)
    if dup_report['duplicate_count'] > 0:
        print(f"  ⚠️  Found {dup_report['duplicate_count']} potential duplicates")
    else:
        print(f"  No duplicates found ✓")
    
    # Analyze balance
    print(f"\nDataset Balance Analysis:")
    balance = analyze_dataset_balance(examples)
    
    print(f"  Services: {len(balance['by_service'])}")
    if len(balance['by_service']) <= 10:
        for service, count in sorted(balance['by_service'].items(), key=lambda x: -x[1])[:10]:
            print(f"    {service}: {count}")
    
    print(f"\n  Field Completeness:")
    for field, rate in balance['field_completeness'].items():
        print(f"    {field:20s}: {rate*100:5.1f}%")
    
    # Summary
    print(f"\n{'='*60}")
    if report['invalid_examples'] == 0:
        print("✓ Dataset validation PASSED")
        print(f"{'='*60}\n")
        sys.exit(0)
    else:
        print(f"✗ Dataset validation FAILED ({report['invalid_examples']} invalid examples)")
        print(f"{'='*60}\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Validate training dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="../data/dataset_v2.jsonl",
        help="Path to training dataset",
    )
    
    args = parser.parse_args()
    validate(args)


if __name__ == "__main__":
    main()
