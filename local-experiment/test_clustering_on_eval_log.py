#!/usr/bin/env python3
"""
Test that clustering produces reasonable incidents from evaluation.log.

This validates that the signature-based clustering correctly identifies
distinct incidents before we run the full evaluation.
"""

import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from incident_signatures import generate_signature, extract_exception_type
from incident_clustering import ParsedLog, cluster_logs


def parse_log_line(line: str) -> ParsedLog:
    """Parse a log line into ParsedLog structure."""
    # Parse: TIMESTAMP LEVEL [service] message
    match = re.match(r'(\S+)\s+(\w+)\s+\[([^\]]+)\]\s+(.+)', line)
    if not match:
        raise ValueError(f"Cannot parse line: {line}")
    
    timestamp_str, level, service, message = match.groups()
    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    exception_type = extract_exception_type(message)
    
    return ParsedLog(
        timestamp=timestamp,
        level=level,
        service=service,
        message=message,
        exception_type=exception_type,
        raw=line
    )


def main():
    base_dir = Path(__file__).parent
    log_path = base_dir / "data" / "evaluation.log"
    
    print("=" * 80)
    print("CLUSTERING TEST ON EVALUATION.LOG")
    print("=" * 80)
    
    # Load and parse logs
    print(f"\n📄 Loading logs: {log_path}")
    parsed_logs = []
    
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            try:
                parsed_log = parse_log_line(line)
                parsed_logs.append(parsed_log)
            except ValueError as e:
                print(f"   ⚠️  {e}")
    
    print(f"   Parsed {len(parsed_logs)} log lines")
    
    # Cluster logs
    print(f"\n🔗 Clustering logs...")
    incidents = cluster_logs(parsed_logs, time_window_minutes=2)
    
    print(f"   Created {len(incidents)} incidents")
    
    # Analyze incidents
    print(f"\n📊 Incident Analysis:")
    print()
    
    services_with_multiple = defaultdict(int)
    for incident in incidents:
        services_with_multiple[incident.service] += 1
    
    # Sort incidents by time
    incidents.sort(key=lambda x: x.first_seen)
    
    for i, incident in enumerate(incidents, 1):
        duration = (incident.last_seen - incident.first_seen).total_seconds()
        
        # Truncate signature for display
        sig_preview = incident.signature[:12]
        
        # Preview first sample
        sample_preview = incident.sample_lines[0][:80] if incident.sample_lines else "(no samples)"
        
        print(f"  Incident {i:2d}: {incident.service}")
        print(f"    Signature: {sig_preview}...")
        print(f"    Count: {incident.count} logs | Duration: {duration:.0f}s")
        print(f"    Time: {incident.first_seen.strftime('%H:%M:%S')} - {incident.last_seen.strftime('%H:%M:%S')}")
        print(f"    Sample: {sample_preview}")
        print()
    
    # Statistics
    print("=" * 80)
    print("CLUSTERING STATISTICS")
    print("=" * 80)
    
    print(f"\nTotal Incidents: {len(incidents)}")
    print(f"Unique Services: {len(set(inc.service for inc in incidents))}")
    print(f"Unique Signatures: {len(set(inc.signature for inc in incidents))}")
    
    print(f"\nIncidents per Service:")
    for service, count in sorted(services_with_multiple.items(), key=lambda x: -x[1]):
        print(f"  {service}: {count}")
    
    print(f"\nServices with Multiple Incidents:")
    multi = [svc for svc, cnt in services_with_multiple.items() if cnt > 1]
    if multi:
        for svc in multi:
            print(f"  ✅ {svc}: {services_with_multiple[svc]} incidents")
    else:
        print("  (none)")
    
    print(f"\nIncident Size Distribution:")
    size_buckets = defaultdict(int)
    for incident in incidents:
        if incident.count == 1:
            size_buckets["1 log"] += 1
        elif incident.count <= 5:
            size_buckets["2-5 logs"] += 1
        elif incident.count <= 10:
            size_buckets["6-10 logs"] += 1
        else:
            size_buckets["11+ logs"] += 1
    
    for bucket, count in sorted(size_buckets.items()):
        print(f"  {bucket}: {count} incidents")
    
    # Validation
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    
    validation_passed = True
    
    # Check: notification-worker should have multiple incidents
    nw_count = services_with_multiple.get("notification-worker", 0)
    if nw_count >= 2:
        print(f"✅ notification-worker has {nw_count} incidents (expected >= 2)")
    else:
        print(f"❌ notification-worker has {nw_count} incidents (expected >= 2)")
        validation_passed = False
    
    # Check: Should have more incidents than services
    unique_services = len(set(inc.service for inc in incidents))
    if len(incidents) > unique_services:
        print(f"✅ {len(incidents)} incidents > {unique_services} services (signature-based clustering working)")
    else:
        print(f"❌ {len(incidents)} incidents <= {unique_services} services (looks like service-based clustering!)")
        validation_passed = False
    
    # Check: Reasonable number of incidents (expecting ~8-15 depending on clustering)
    if 7 <= len(incidents) <= 20:
        print(f"✅ {len(incidents)} incidents (reasonable range 7-20)")
    else:
        print(f"⚠️  {len(incidents)} incidents (expected 7-20, may need investigation)")
    
    print()
    if validation_passed:
        print("✅ Clustering validation PASSED")
        print("   Signature-based clustering is working correctly")
        return 0
    else:
        print("❌ Clustering validation FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
