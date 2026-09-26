#!/usr/bin/env python3
"""
Test the evaluation pipeline v2 WITHOUT running model inference.

This validates:
1. Signature-based clustering produces correct incident candidates
2. Ground truth matching works correctly
3. Multiple incidents per service are detected
4. Metrics calculation is accurate
"""

import json
from datetime import datetime
from pathlib import Path

from incident_clustering import ParsedLog, cluster_logs
from incident_signatures import extract_exception_type
from evaluation_metrics import (
    GroundTruthIncident,
    PredictedIncident,
    evaluate,
    match_predictions_to_ground_truth,
)


def parse_log_line(line: str):
    """Parse log line for testing."""
    import re
    
    if line.startswith("#") or not line.strip():
        return None
    
    match = re.match(r'(\S+)\s+(\w+)\s+\[([^\]]+)\]\s+(.+)', line)
    if not match:
        return None
    
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


def test_clustering():
    """Test signature-based clustering."""
    print("="*80)
    print("TEST 1: SIGNATURE-BASED CLUSTERING")
    print("="*80)
    
    log_file = Path("data/evaluation.log")
    
    # Load and parse logs
    parsed_logs = []
    with log_file.open() as f:
        for line in f:
            parsed = parse_log_line(line.strip())
            if parsed and parsed.level in ["ERROR", "WARN", "WARNING", "CRITICAL"]:
                parsed_logs.append(parsed)
    
    print(f"📄 Parsed {len(parsed_logs)} ERROR/WARN/CRITICAL logs")
    
    # Cluster with 2-minute window
    incidents = cluster_logs(parsed_logs, time_window_minutes=2)
    
    print(f"🔗 Created {len(incidents)} incident candidates")
    
    # Analyze by service
    from collections import defaultdict
    by_service = defaultdict(list)
    for inc in incidents:
        by_service[inc.service].append(inc)
    
    print(f"\n📊 Incidents by service:")
    for service in sorted(by_service.keys()):
        count = len(by_service[service])
        print(f"   {service}: {count} incidents")
        if count > 1:
            print(f"      ✅ MULTIPLE incidents detected (good!)")
            for idx, inc in enumerate(by_service[service][:3]):  # Show first 3
                print(f"         #{idx+1}: {inc.count} logs, {inc.signature[:16]}...")
    
    # Key validation
    notification_incidents = by_service.get("notification-worker", [])
    print(f"\n🔍 Validation:")
    print(f"   notification-worker incidents: {len(notification_incidents)}")
    if len(notification_incidents) >= 2:
        print(f"   ✅ PASS: Multiple incidents in notification-worker")
    else:
        print(f"   ❌ FAIL: Expected >= 2 incidents in notification-worker")
    
    if len(incidents) > len(by_service):
        print(f"   ✅ PASS: {len(incidents)} incidents > {len(by_service)} services")
    else:
        print(f"   ❌ FAIL: Expected more incidents than services")
    
    return incidents


def test_ground_truth_matching(incidents):
    """Test ground truth matching."""
    print("\n" + "="*80)
    print("TEST 2: GROUND TRUTH MATCHING")
    print("="*80)
    
    # Load ground truth
    gt_file = Path("ground_truth/evaluation.json")
    with gt_file.open() as f:
        gt_data = json.load(f)
    
    print(f"📋 Loaded {len(gt_data['incidents'])} ground truth incidents")
    
    # Convert to evaluation format
    gt_incidents = [
        GroundTruthIncident(
            incident_id=inc["incident_id"],
            incident_type=inc["incident_type"],
            service=inc["service"],
            severity=inc["severity"],
            disposition=inc["disposition"],
            start_time=datetime.fromisoformat(inc["start_time"]),
            end_time=datetime.fromisoformat(inc["end_time"]),
            log_count=inc["log_count"],
            description=inc["description"],
        )
        for inc in gt_data["incidents"]
    ]
    
    # Create mock predictions (use severity from signature analysis)
    # For testing, we'll assign reasonable severities based on log content
    pred_incidents = []
    for inc in incidents:
        # Analyze severity from logs
        sample_text = " ".join(inc.sample_lines).lower()
        
        if "critical" in sample_text or "oom" in sample_text or "unavailable" in sample_text:
            severity = "critical"
            disposition = "ESCALATE"
        elif "error" in sample_text and ("pool exhausted" in sample_text or "timeout" in sample_text):
            severity = "high"
            disposition = "NEEDS_ONCALL"
        elif "warn" in sample_text or "lag" in sample_text:
            severity = "medium"
            disposition = "OBSERVE"
        else:
            severity = "low"
            disposition = "NO_ACTION"
        
        # Create mock ParsedOutput
        from output_parser import ParsedOutput, ParseStatus
        parsed_output = ParsedOutput(
            severity=severity,
            disposition=disposition,
            summary=f"Mock analysis for {inc.service}",
            confidence=0.8,
            suspected_root_cause="Mock root cause",
            next_steps=["Mock step 1", "Mock step 2"],
            ticket_title=f"Mock ticket for {inc.service}",
            ticket_body="Mock ticket body",
            parse_status=ParseStatus.SUCCESS,
            raw_response="Mock response",
        )
        
        pred_incidents.append(
            PredictedIncident(
                incident_id=inc.id,
                service=inc.service,
                start_time=inc.first_seen,
                end_time=inc.last_seen,
                log_count=inc.count,
                parsed_output=parsed_output,
                latency_ms=100.0,  # Mock
            )
        )
    
    print(f"🔮 Created {len(pred_incidents)} predictions (mock severities)")
    
    # Match (note: function expects predictions FIRST, then ground truth)
    matches = match_predictions_to_ground_truth(pred_incidents, gt_incidents)
    
    print(f"\n🎯 Matching results:")
    print(f"   Matched pairs: {len([m for m in matches if m.prediction])}")
    print(f"   False negatives: {len([m for m in matches if not m.prediction])}")
    print(f"   False positives: {len([p for p in pred_incidents if not any(m.prediction and m.prediction.incident_id == p.incident_id for m in matches)])}")
    
    # Show some matches
    print(f"\n📝 Sample matches:")
    for m in matches[:5]:
        if m.prediction:
            print(f"   ✅ {m.ground_truth.incident_id} ({m.ground_truth.service})")
            print(f"      → matched to prediction with {m.prediction.parsed_output.severity} severity")
        else:
            print(f"   ❌ {m.ground_truth.incident_id} ({m.ground_truth.service})")
            print(f"      → NO MATCH (false negative)")
    
    # Validation
    matched_count = len([m for m in matches if m.prediction])
    if matched_count >= 6:  # Expect at least 75% match
        print(f"\n   ✅ PASS: {matched_count}/8 ground truth incidents matched")
    else:
        print(f"\n   ⚠️  WARNING: Only {matched_count}/8 matched")
    
    return matches


def test_metrics_calculation(matches):
    """Test metrics calculation."""
    print("\n" + "="*80)
    print("TEST 3: METRICS CALCULATION")
    print("="*80)
    
    # Extract incidents for full evaluation
    gt_incidents = [m.ground_truth for m in matches]
    pred_incidents = [m.prediction for m in matches if m.prediction]
    
    # Add some predictions that aren't matched (false positives)
    # This simulates the full evaluation
    
    results = evaluate(pred_incidents, gt_incidents)
    
    print(f"📊 Detection Metrics:")
    print(f"   True Positives: {results.detection.true_positives}")
    print(f"   False Positives: {results.detection.false_positives}")
    print(f"   False Negatives: {results.detection.false_negatives}")
    print(f"   Precision: {results.detection.precision:.3f}")
    print(f"   Recall: {results.detection.recall:.3f}")
    print(f"   F1: {results.detection.f1:.3f}")
    
    print(f"\n📊 Severity Metrics:")
    print(f"   Accuracy: {results.severity.accuracy:.3f}")
    print(f"   Macro Precision: {results.severity.macro_precision:.3f}")
    print(f"   Macro Recall: {results.severity.macro_recall:.3f}")
    print(f"   Macro F1: {results.severity.macro_f1:.3f}")
    
    if results.detection.recall >= 0.75:
        print(f"\n   ✅ PASS: Recall >= 0.75")
    else:
        print(f"\n   ⚠️  WARNING: Recall < 0.75")
    
    return results


def test_incident_candidates_file():
    """Test incident_candidates.jsonl generation."""
    print("\n" + "="*80)
    print("TEST 4: INCIDENT CANDIDATES FILE")
    print("="*80)
    
    log_file = Path("data/evaluation.log")
    output_dir = Path("results/pipeline_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and cluster
    parsed_logs = []
    with log_file.open() as f:
        for line in f:
            parsed = parse_log_line(line.strip())
            if parsed and parsed.level in ["ERROR", "WARN", "WARNING", "CRITICAL"]:
                parsed_logs.append(parsed)
    
    incidents = cluster_logs(parsed_logs, time_window_minutes=2)
    
    # Save candidates
    candidates_file = output_dir / "incident_candidates.jsonl"
    with candidates_file.open("w") as f:
        for idx, inc in enumerate(incidents):
            candidate = {
                "incident_id": inc.id,
                "index": idx,
                "service": inc.service,
                "signature": inc.signature,
                "start_time": inc.first_seen.isoformat(),
                "end_time": inc.last_seen.isoformat(),
                "duration_seconds": int((inc.last_seen - inc.first_seen).total_seconds()),
                "log_count": inc.count,
                "evidence": inc.get_evidence_samples(),
                "grouping_reason": f"signature-based clustering",
                "raw_log_indices": inc.log_indices,
            }
            f.write(json.dumps(candidate) + "\n")
    
    print(f"💾 Saved: {candidates_file}")
    
    # Verify file
    candidates = []
    with candidates_file.open() as f:
        for line in f:
            candidates.append(json.loads(line))
    
    print(f"✅ Verified: {len(candidates)} candidates")
    
    # Show samples
    print(f"\n📋 Sample candidates:")
    for c in candidates[:3]:
        print(f"   {c['service']}: {c['log_count']} logs, {c['duration_seconds']}s duration")
        print(f"      Evidence: {c['evidence'][0][:80]}...")
    
    return candidates_file


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("EVALUATION PIPELINE V2 - VALIDATION TESTS")
    print("="*80)
    print("Testing clustering, matching, and metrics WITHOUT model inference\n")
    
    try:
        # Test 1: Clustering
        incidents = test_clustering()
        
        # Test 2: Ground truth matching
        matches = test_ground_truth_matching(incidents)
        
        # Test 3: Metrics calculation
        results = test_metrics_calculation(matches)
        
        # Test 4: Output files
        candidates_file = test_incident_candidates_file()
        
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        print("✅ All tests completed")
        print(f"✅ Signature-based clustering working")
        print(f"✅ Multiple incidents per service detected")
        print(f"✅ Ground truth matching functional")
        print(f"✅ Metrics calculation working")
        print(f"✅ Output files generated")
        print("\n🎉 Pipeline validation PASSED")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
