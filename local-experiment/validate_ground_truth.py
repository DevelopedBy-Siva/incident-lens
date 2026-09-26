"""
Validate ground truth file.

Ensures ground truth is properly structured for evaluation.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_ground_truth(gt_data: dict) -> list:
    """Validate ground truth structure and return errors."""
    errors = []
    
    # Check metadata
    if "metadata" not in gt_data:
        errors.append("Missing 'metadata' section")
    
    # Check incidents
    if "incidents" not in gt_data:
        errors.append("Missing 'incidents' section")
        return errors
    
    incidents = gt_data["incidents"]
    if not isinstance(incidents, list):
        errors.append("'incidents' must be a list")
        return errors
    
    # Validate each incident
    required_fields = [
        "incident_id",
        "service",
        "severity",
        "disposition",
        "start_time",
        "end_time",
    ]
    
    valid_severities = ["low", "medium", "high", "critical"]
    valid_dispositions = ["NO_ACTION", "OBSERVE", "NEEDS_DEV", "NEEDS_ONCALL", "ESCALATE"]
    
    for idx, incident in enumerate(incidents):
        incident_errors = []
        
        # Check required fields
        for field in required_fields:
            if field not in incident:
                incident_errors.append(f"Missing field: {field}")
        
        # Validate severity
        if "severity" in incident:
            sev = incident["severity"].lower()
            if sev not in valid_severities:
                incident_errors.append(f"Invalid severity: {incident['severity']}")
        
        # Validate disposition
        if "disposition" in incident:
            disp = incident["disposition"].upper()
            if disp not in valid_dispositions:
                incident_errors.append(f"Invalid disposition: {incident['disposition']}")
        
        # Validate timestamps
        for time_field in ["start_time", "end_time"]:
            if time_field in incident:
                try:
                    ts = incident[time_field]
                    if ts.endswith('Z'):
                        ts = ts[:-1] + '+00:00'
                    datetime.fromisoformat(ts)
                except Exception as e:
                    incident_errors.append(f"Invalid {time_field}: {e}")
        
        if incident_errors:
            errors.append(f"Incident {idx} ({incident.get('incident_id', 'unknown')}): {incident_errors}")
    
    return errors


def analyze_ground_truth(gt_data: dict) -> dict:
    """Analyze ground truth coverage."""
    incidents = gt_data.get("incidents", [])
    
    analysis = {
        "total_incidents": len(incidents),
        "by_service": {},
        "by_severity": {},
        "by_disposition": {},
        "incident_types": {},
    }
    
    for incident in incidents:
        # By service
        service = incident.get("service", "unknown")
        analysis["by_service"][service] = analysis["by_service"].get(service, 0) + 1
        
        # By severity
        severity = incident.get("severity", "unknown").lower()
        analysis["by_severity"][severity] = analysis["by_severity"].get(severity, 0) + 1
        
        # By disposition
        disposition = incident.get("disposition", "unknown").upper()
        analysis["by_disposition"][disposition] = analysis["by_disposition"].get(disposition, 0) + 1
        
        # By incident type
        inc_type = incident.get("incident_type", "unknown")
        analysis["incident_types"][inc_type] = analysis["incident_types"].get(inc_type, 0) + 1
    
    return analysis


def validate(args):
    """Run validation."""
    gt_path = Path(args.ground_truth)
    
    if not gt_path.exists():
        print(f"❌ Ground truth not found: {gt_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Ground Truth Validation")
    print(f"{'='*60}")
    print(f"Ground truth: {gt_path}")
    print(f"{'='*60}\n")
    
    # Compute hash
    print("Computing ground truth hash...")
    file_hash = compute_file_hash(gt_path)
    print(f"  SHA256: {file_hash}\n")
    
    # Load data
    print("Loading ground truth...")
    try:
        with open(gt_path, "r") as f:
            gt_data = json.load(f)
        print(f"  Loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load ground truth: {e}")
        sys.exit(1)
    
    # Validate structure
    print("Validating structure...")
    errors = validate_ground_truth(gt_data)
    
    if errors:
        print(f"  ✗ Found {len(errors)} validation errors:\n")
        for error in errors[:10]:
            print(f"    {error}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")
        print(f"\n{'='*60}")
        print("✗ Ground truth validation FAILED")
        print(f"{'='*60}\n")
        sys.exit(1)
    else:
        print(f"  ✓ Structure valid\n")
    
    # Analyze coverage
    print("Analyzing coverage...")
    analysis = analyze_ground_truth(gt_data)
    
    print(f"\nTotal incidents: {analysis['total_incidents']}")
    
    print(f"\nBy Service:")
    for service, count in sorted(analysis['by_service'].items()):
        print(f"  {service}: {count}")
    
    print(f"\nBy Severity:")
    for sev, count in sorted(analysis['by_severity'].items()):
        print(f"  {sev}: {count}")
    
    print(f"\nBy Disposition:")
    for disp, count in sorted(analysis['by_disposition'].items()):
        print(f"  {disp}: {count}")
    
    print(f"\nBy Incident Type:")
    for inc_type, count in sorted(analysis['incident_types'].items()):
        print(f"  {inc_type}: {count}")
    
    # Check metadata
    metadata = gt_data.get("metadata", {})
    if metadata:
        print(f"\nMetadata:")
        print(f"  Description: {metadata.get('description', 'N/A')}")
        print(f"  Total log lines: {metadata.get('total_log_lines', 'N/A')}")
        print(f"  Clustering method: {metadata.get('clustering_method', 'N/A')}")
    
    print(f"\n{'='*60}")
    print("✓ Ground truth validation PASSED")
    print(f"{'='*60}\n")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Validate ground truth file")
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="ground_truth/evaluation.json",
        help="Path to ground truth JSON",
    )
    
    args = parser.parse_args()
    validate(args)


if __name__ == "__main__":
    main()
