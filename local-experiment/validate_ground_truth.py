#!/usr/bin/env python3
"""
Validate ground truth against evaluation log.

This script verifies that:
1. Ground truth incidents can be found in the log
2. Incident boundaries are reasonable
3. Log line markers exist in the logs
4. No temporal overlaps for same-service incidents
"""

import json
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict


def load_ground_truth(gt_path: Path) -> dict:
    """Load ground truth JSON."""
    with gt_path.open() as f:
        return json.load(f)


def load_evaluation_logs(log_path: Path) -> list[dict]:
    """Load and parse evaluation logs."""
    logs = []
    with log_path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse: TIMESTAMP LEVEL [service] message
            match = re.match(r'(\S+)\s+(\w+)\s+\[([^\]]+)\]\s+(.+)', line)
            if match:
                timestamp_str, level, service, message = match.groups()
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    logs.append({
                        'line_num': line_num,
                        'timestamp': timestamp,
                        'level': level,
                        'service': service,
                        'message': message,
                        'raw': line
                    })
                except ValueError:
                    print(f"⚠️  Line {line_num}: Failed to parse timestamp: {timestamp_str}")
    
    return logs


def validate_incident_in_logs(incident: dict, logs: list[dict]) -> dict:
    """Validate that incident can be found in logs."""
    start_time = datetime.fromisoformat(incident['start_time'].replace('Z', '+00:00'))
    end_time = datetime.fromisoformat(incident['end_time'].replace('Z', '+00:00'))
    service = incident['service']
    
    # Find logs in incident time window and service
    matching_logs = [
        log for log in logs
        if log['service'] == service
        and start_time <= log['timestamp'] <= end_time
    ]
    
    # Check log line markers
    found_markers = []
    missing_markers = []
    for marker in incident.get('log_line_markers', []):
        marker_found = any(marker.lower() in log['message'].lower() for log in matching_logs)
        if marker_found:
            found_markers.append(marker)
        else:
            missing_markers.append(marker)
    
    return {
        'logs_found': len(matching_logs),
        'expected_count': incident['log_count'],
        'count_match': len(matching_logs) == incident['log_count'],
        'found_markers': found_markers,
        'missing_markers': missing_markers,
        'all_markers_found': len(missing_markers) == 0
    }


def check_temporal_overlaps(incidents: list[dict]) -> list[dict]:
    """Check for temporal overlaps in same-service incidents."""
    service_incidents = defaultdict(list)
    
    for incident in incidents:
        service_incidents[incident['service']].append(incident)
    
    overlaps = []
    for service, service_incs in service_incidents.items():
        if len(service_incs) <= 1:
            continue
        
        # Sort by start time
        service_incs.sort(key=lambda x: x['start_time'])
        
        for i in range(len(service_incs) - 1):
            inc1 = service_incs[i]
            inc2 = service_incs[i + 1]
            
            end1 = datetime.fromisoformat(inc1['end_time'].replace('Z', '+00:00'))
            start2 = datetime.fromisoformat(inc2['start_time'].replace('Z', '+00:00'))
            
            if end1 >= start2:
                overlaps.append({
                    'service': service,
                    'incident1': inc1['incident_id'],
                    'incident2': inc2['incident_id'],
                    'overlap_seconds': (end1 - start2).total_seconds()
                })
    
    return overlaps


def main():
    base_dir = Path(__file__).parent
    gt_path = base_dir / "ground_truth" / "evaluation.json"
    log_path = base_dir / "data" / "evaluation.log"
    
    print("=" * 80)
    print("GROUND TRUTH VALIDATION")
    print("=" * 80)
    
    # Load ground truth
    print(f"\n📋 Loading ground truth: {gt_path}")
    gt = load_ground_truth(gt_path)
    
    print(f"   Total incidents: {len(gt['incidents'])}")
    print(f"   Evaluation log: {gt['metadata']['evaluation_log']}")
    
    # Load logs
    print(f"\n📄 Loading evaluation logs: {log_path}")
    logs = load_evaluation_logs(log_path)
    print(f"   Parsed {len(logs)} log lines")
    
    # Validate each incident
    print(f"\n🔍 Validating incidents against logs...")
    print()
    
    all_valid = True
    for incident in gt['incidents']:
        inc_id = incident['incident_id']
        inc_type = incident['incident_type']
        service = incident['service']
        severity = incident['severity']
        
        print(f"  {inc_id}: {inc_type}")
        print(f"    Service: {service} | Severity: {severity}")
        
        validation = validate_incident_in_logs(incident, logs)
        
        # Check log count
        if validation['count_match']:
            print(f"    ✅ Log count: {validation['logs_found']}/{validation['expected_count']}")
        else:
            print(f"    ❌ Log count mismatch: found {validation['logs_found']}, expected {validation['expected_count']}")
            all_valid = False
        
        # Check markers
        if validation['all_markers_found']:
            print(f"    ✅ All log markers found: {len(validation['found_markers'])}")
        else:
            print(f"    ⚠️  Missing markers: {validation['missing_markers']}")
            all_valid = False
        
        print()
    
    # Check for temporal overlaps
    print(f"\n⏰ Checking for temporal overlaps...")
    overlaps = check_temporal_overlaps(gt['incidents'])
    
    if not overlaps:
        print("   ✅ No temporal overlaps detected")
    else:
        print(f"   ⚠️  Found {len(overlaps)} temporal overlaps:")
        for overlap in overlaps:
            print(f"      {overlap['service']}: {overlap['incident1']} overlaps {overlap['incident2']} by {overlap['overlap_seconds']}s")
            all_valid = False
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"\nGround Truth Statistics:")
    print(f"  Total incidents: {len(gt['incidents'])}")
    print(f"  Services covered: {len(gt['service_incident_mapping'])}")
    print(f"  Multi-incident services: notification-worker (2 incidents)")
    
    print(f"\nSeverity Distribution:")
    for severity, count in gt['severity_distribution'].items():
        print(f"  {severity}: {count}")
    
    print(f"\nDisposition Distribution:")
    for disposition, count in gt['disposition_distribution'].items():
        print(f"  {disposition}: {count}")
    
    print()
    if all_valid:
        print("✅ Ground truth validation PASSED")
        print("   All incidents verified against evaluation log")
        return 0
    else:
        print("❌ Ground truth validation FAILED")
        print("   Fix issues above before running evaluation")
        return 1


if __name__ == "__main__":
    exit(main())
