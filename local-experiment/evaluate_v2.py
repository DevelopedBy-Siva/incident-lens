#!/usr/bin/env python3
"""
FIXED Evaluation Pipeline v2

This version uses production-compatible signature-based clustering and properly
handles multiple incidents within a single service.

Key fixes:
1. Uses incident_clustering.py (signature-based) instead of service-only grouping
2. Generates incident_candidates.jsonl for inspection
3. Implements deterministic ground-truth matching
4. Fixes JSON parsing with output_parser.py
5. Reduces inference latency by loading model once
6. Provides comprehensive metrics and error analysis
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Import production-compatible modules
from incident_clustering import ParsedLog, cluster_logs
from incident_signatures import extract_exception_type
from output_parser import parse_model_output, get_parse_metrics, ParsedOutput, ParseStatus
from evaluation_metrics import (
    GroundTruthIncident,
    PredictedIncident,
    evaluate,
)


def load_config(config_path: Path) -> dict:
    """Load configuration."""
    with config_path.open() as f:
        return yaml.safe_load(f)


def parse_log_line(line: str, line_index: int) -> Optional[ParsedLog]:
    """
    Parse log line using production-compatible parser.
    
    Expected format: TIMESTAMP LEVEL [service] message
    Example: 2026-09-25T10:30:00.123Z ERROR [payment-api] connection timeout
    """
    # Skip comments and empty lines
    if line.startswith("#") or not line.strip():
        return None
    
    # Production-compatible regex
    match = re.match(r'(\S+)\s+(\w+)\s+\[([^\]]+)\]\s+(.+)', line)
    if not match:
        print(f"⚠️  Failed to parse line {line_index}: {line[:80]}")
        return None
    
    timestamp_str, level, service, message = match.groups()
    
    try:
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except Exception as e:
        print(f"⚠️  Failed to parse timestamp on line {line_index}: {e}")
        return None
    
    # Extract exception type
    exception_type = extract_exception_type(message)
    
    return ParsedLog(
        timestamp=timestamp,
        level=level,
        service=service,
        message=message,
        exception_type=exception_type,
        raw=line
    )


def load_and_cluster_logs(log_file: Path, time_window_minutes: int = 2) -> list:
    """
    Load and cluster logs using production-compatible signature-based clustering.
    
    Returns list of Incident objects from incident_clustering.py
    """
    print(f"\n📄 Loading logs: {log_file}")
    
    parsed_logs = []
    with log_file.open() as f:
        for idx, line in enumerate(f):
            line = line.strip()
            parsed = parse_log_line(line, idx)
            if parsed:
                parsed_logs.append(parsed)
    
    print(f"   Parsed {len(parsed_logs)} log lines")
    
    # Filter to ERROR/WARN/CRITICAL (production behavior)
    error_logs = [
        log for log in parsed_logs
        if log.level in ["ERROR", "WARN", "WARNING", "CRITICAL"]
    ]
    
    print(f"   {len(error_logs)} ERROR/WARN/CRITICAL logs")
    
    # Cluster using production-compatible logic
    print(f"\n🔗 Clustering logs (time_window={time_window_minutes}min)...")
    incidents = cluster_logs(error_logs, time_window_minutes=time_window_minutes)
    
    print(f"   Created {len(incidents)} initial incident clusters")
    
    return incidents


def merge_adjacent_incidents(incidents: list, max_gap_minutes: int = 5) -> list:
    """
    Merge adjacent incidents with the same signature.
    
    This handles cases where logs within a single incident are separated by more
    than the 2-minute clustering window but should still be considered the same incident.
    
    For example:
    - 10:00:00 ERROR connection timeout (signature A)
    - 10:00:05 ERROR connection timeout (signature A)
    - 10:03:00 ERROR connection timeout (signature A) <- 3min gap, new cluster
    
    These should be merged into one incident if the gap is < max_gap_minutes.
    """
    if not incidents:
        return []
    
    # Sort by service, signature, first_seen
    sorted_incidents = sorted(
        incidents,
        key=lambda inc: (inc.service, inc.signature, inc.first_seen)
    )
    
    merged = []
    current = None
    
    for incident in sorted_incidents:
        if current is None:
            current = incident
            continue
        
        # Check if same signature and within gap threshold
        if (current.signature == incident.signature and
            current.service == incident.service):
            
            time_gap = incident.first_seen - current.last_seen
            
            if time_gap <= timedelta(minutes=max_gap_minutes):
                # Merge into current
                current.count += incident.count
                current.last_seen = incident.last_seen
                current.sample_lines.extend(incident.sample_lines)
                current.log_indices.extend(incident.log_indices)
                continue
        
        # No merge - save current and start new
        merged.append(current)
        current = incident
    
    # Don't forget the last one
    if current:
        merged.append(current)
    
    return merged


def save_incident_candidates(incidents: list, output_path: Path):
    """
    Save incident candidates to JSONL for inspection.
    
    This makes the grouping visible and debuggable.
    """
    with output_path.open("w") as f:
        for idx, incident in enumerate(incidents):
            candidate = {
                "incident_id": incident.id,
                "index": idx,
                "service": incident.service,
                "signature": incident.signature,
                "start_time": incident.first_seen.isoformat(),
                "end_time": incident.last_seen.isoformat(),
                "duration_seconds": int((incident.last_seen - incident.first_seen).total_seconds()),
                "log_count": incident.count,
                "evidence": incident.get_evidence_samples(),
                "grouping_reason": f"signature-based clustering with {(incident.last_seen - incident.first_seen).total_seconds()/60:.1f}min duration",
                "raw_log_indices": incident.log_indices,
            }
            f.write(json.dumps(candidate) + "\n")
    
    print(f"   Saved incident candidates: {output_path}")


def load_ground_truth(gt_path: Path) -> dict:
    """Load ground truth."""
    if not gt_path.exists():
        return None
    with gt_path.open() as f:
        return json.load(f)


class InferenceEngine:
    """
    Inference engine that loads model once and reuses it.
    
    This eliminates repeated model loading overhead.
    """
    
    def __init__(self, base_model_name: str, adapter_path: Path, config: dict):
        """Initialize and load model."""
        print(f"\n🤖 Loading base model: {base_model_name}")
        
        # QLoRA quantization
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load adapter
        print(f"   Loading adapter: {adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        
        self.gen_config = config.get("generation", {})
        print(f"   Model loaded successfully")
    
    def predict(self, incident) -> tuple[str, float]:
        """
        Run inference on a single incident.
        
        Returns (raw_response, latency_ms)
        """
        # Build prompt matching training format
        logs_text = "\n".join(incident.get_evidence_samples())
        service = incident.service
        environment = incident.environment
        
        system_prompt = """You are an expert SRE analyzing production incidents.
Analyze the logs and provide a structured incident analysis with these fields:
- severity: low, medium, high, or critical
- disposition: NO_ACTION, OBSERVE, NEEDS_DEV, NEEDS_ONCALL, or ESCALATE
- confidence: 0.0 to 1.0
- summary: brief description
- suspected_root_cause: likely cause or null
- next_steps: list of action items
- ticket_title: short title
- ticket_body: detailed description

Respond with valid JSON only."""
        
        user_prompt = f"""Service: {service}
Environment: {environment}

Logs:
{logs_text}

Provide your analysis:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.gen_config.get("max_new_tokens", 512),
                temperature=self.gen_config.get("temperature", 0.1),
                top_p=self.gen_config.get("top_p", 0.95),
                do_sample=self.gen_config.get("do_sample", True),
                repetition_penalty=self.gen_config.get("repetition_penalty", 1.1),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        latency_ms = (time.time() - start_time) * 1000
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response, latency_ms


def run_inference(
    incidents: list,
    inference_engine: InferenceEngine,
    output_dir: Path
) -> tuple[list, list]:
    """
    Run inference on all incident clusters.
    
    Returns (predictions, latencies)
    """
    print(f"\n🔮 Running inference on {len(incidents)} incidents...")
    
    predictions = []
    latencies = []
    
    for i, incident in enumerate(incidents):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"   Progress: {i + 1}/{len(incidents)}")
        
        raw_response, latency_ms = inference_engine.predict(incident)
        latencies.append(latency_ms)
        
        # Parse response using production-compatible parser
        parsed = parse_model_output(raw_response)
        
        prediction = {
            "incident_id": incident.id,
            "index": i,
            "service": incident.service,
            "signature": incident.signature[:16] + "...",
            "start_time": incident.first_seen.isoformat(),
            "end_time": incident.last_seen.isoformat(),
            "log_count": incident.count,
            "evidence": incident.get_evidence_samples(),
            "raw_response": raw_response,
            "parsed": {
                "severity": parsed.severity,
                "disposition": parsed.disposition,
                "confidence": parsed.confidence,
                "summary": parsed.summary,
                "suspected_root_cause": parsed.suspected_root_cause,
                "next_steps": parsed.next_steps,
                "ticket_title": parsed.ticket_title,
                "ticket_body": parsed.ticket_body,
            },
            "parse_status": parsed.parse_status.value,
            "parse_error": parsed.parse_error,
            "latency_ms": latency_ms,
        }
        
        predictions.append(prediction)
    
    print(f"   Inference complete")
    
    return predictions, latencies


def main():
    parser = argparse.ArgumentParser(
        description="Fixed evaluation pipeline v2 - uses signature-based clustering"
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="Path to trained adapter directory"
    )
    parser.add_argument(
        "--logs",
        type=Path,
        required=True,
        help="Raw log file for evaluation"
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("ground_truth/evaluation.json"),
        help="Ground truth file"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Configuration file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--time-window",
        type=int,
        default=2,
        help="Clustering time window in minutes (default: 2)"
    )
    parser.add_argument(
        "--merge-gap",
        type=int,
        default=5,
        help="Max gap for merging adjacent incidents in minutes (default: 5)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("EVALUATION PIPELINE V2 - SIGNATURE-BASED CLUSTERING")
    print("="*80)
    
    # Load config
    config = load_config(args.config)
    
    # Step 1: Load and cluster logs
    incidents = load_and_cluster_logs(args.logs, args.time_window)
    
    # Step 2: Merge adjacent incidents with same signature
    if args.merge_gap > args.time_window:
        print(f"\n🔗 Merging adjacent incidents (max_gap={args.merge_gap}min)...")
        incidents = merge_adjacent_incidents(incidents, max_gap_minutes=args.merge_gap)
        print(f"   {len(incidents)} incidents after merging")
    
    # Step 3: Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Step 4: Save incident candidates for inspection
    candidates_file = args.output / "incident_candidates.jsonl"
    save_incident_candidates(incidents, candidates_file)
    
    # Step 5: Load ground truth
    ground_truth = load_ground_truth(args.ground_truth)
    if ground_truth:
        print(f"\n✅ Ground truth loaded: {len(ground_truth.get('incidents', []))} incidents")
    else:
        print(f"\n⚠️  No ground truth available")
    
    # Step 6: Initialize inference engine
    inference_engine = InferenceEngine(
        base_model_name=config['base_model'],
        adapter_path=args.adapter,
        config=config
    )
    
    # Step 7: Run inference
    predictions, latencies = run_inference(incidents, inference_engine, args.output)
    
    # Step 8: Save predictions
    predictions_file = args.output / "predictions.jsonl"
    with predictions_file.open("w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    print(f"\n💾 Predictions saved: {predictions_file}")
    
    # Step 9: Calculate metrics
    print(f"\n📊 Calculating metrics...")
    
    # Parse metrics
    from output_parser import ParsedOutput, ParseStatus
    parsed_outputs = [
        ParsedOutput(
            severity=p["parsed"]["severity"],
            disposition=p["parsed"]["disposition"],
            summary=p["parsed"]["summary"],
            confidence=p["parsed"]["confidence"],
            suspected_root_cause=p["parsed"]["suspected_root_cause"],
            next_steps=p["parsed"]["next_steps"],
            ticket_title=p["parsed"]["ticket_title"],
            ticket_body=p["parsed"]["ticket_body"],
            parse_status=ParseStatus(p["parse_status"]),
            parse_error=p["parse_error"],
            raw_response=p["raw_response"],
        )
        for p in predictions
    ]
    
    parse_metrics = get_parse_metrics(parsed_outputs)
    
    # Latency metrics
    latency_metrics = {
        "mean_ms": sum(latencies) / len(latencies) if latencies else 0,
        "p50_ms": sorted(latencies)[len(latencies)//2] if latencies else 0,
        "p95_ms": sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0,
        "p99_ms": sorted(latencies)[int(len(latencies)*0.99)] if latencies else 0,
        "min_ms": min(latencies) if latencies else 0,
        "max_ms": max(latencies) if latencies else 0,
    }
    
    # Evaluation metrics (if ground truth available)
    eval_results = None
    if ground_truth and "incidents" in ground_truth:
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
            for inc in ground_truth["incidents"]
        ]
        
        pred_incidents = [
            PredictedIncident(
                incident_id=p["incident_id"],
                service=p["service"],
                start_time=datetime.fromisoformat(p["start_time"]),
                end_time=datetime.fromisoformat(p["end_time"]),
                log_count=p["log_count"],
                parsed_output=ParsedOutput(
                    severity=p["parsed"]["severity"],
                    disposition=p["parsed"]["disposition"],
                    summary=p["parsed"]["summary"],
                    confidence=p["parsed"]["confidence"],
                    suspected_root_cause=p["parsed"]["suspected_root_cause"],
                    next_steps=p["parsed"]["next_steps"],
                    ticket_title=p["parsed"]["ticket_title"],
                    ticket_body=p["parsed"]["ticket_body"],
                    parse_status=ParseStatus(p["parse_status"]),
                    raw_response=p["raw_response"],
                ),
                latency_ms=p["latency_ms"],
            )
            for p in predictions
            if p["parsed"]["severity"] is not None  # Only valid predictions
        ]
        
        eval_results = evaluate(pred_incidents, gt_incidents)
    
    # Step 10: Save metrics
    metrics = {
        "pipeline_version": "v2_signature_based",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "base_model": config['base_model'],
            "adapter": str(args.adapter),
            "time_window_minutes": args.time_window,
            "merge_gap_minutes": args.merge_gap,
        },
        "clustering": {
            "total_log_lines": sum(inc.count for inc in incidents),
            "incident_candidates": len(incidents),
            "services": len(set(inc.service for inc in incidents)),
        },
        "parsing": parse_metrics,
        "latency": latency_metrics,
    }
    
    if eval_results:
        metrics["evaluation"] = {
            "detection": {
                "true_positives": eval_results.detection.true_positives,
                "false_positives": eval_results.detection.false_positives,
                "false_negatives": eval_results.detection.false_negatives,
                "precision": eval_results.detection.precision,
                "recall": eval_results.detection.recall,
                "f1": eval_results.detection.f1,
            },
            "severity": {
                "accuracy": eval_results.severity.accuracy,
                "macro_f1": eval_results.severity.macro_f1,
            }
        }
    
    metrics_file = args.output / "metrics.json"
    with metrics_file.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"   Metrics saved: {metrics_file}")
    
    # Step 11: Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total log lines: {metrics['clustering']['total_log_lines']}")
    print(f"Incident candidates: {metrics['clustering']['incident_candidates']}")
    print(f"Unique services: {metrics['clustering']['services']}")
    print(f"\nValid JSON: {metrics['parsing']['valid_json_rate']*100:.1f}%")
    print(f"Parse failures: {metrics['parsing']['parse_failure_rate']*100:.1f}%")
    print(f"Mean completion: {metrics['parsing']['mean_completion']*100:.1f}%")
    print(f"\nLatency (mean): {metrics['latency']['mean_ms']:.1f}ms")
    print(f"Latency (P50): {metrics['latency']['p50_ms']:.1f}ms")
    print(f"Latency (P95): {metrics['latency']['p95_ms']:.1f}ms")
    
    if eval_results:
        print(f"\n🎯 Detection:")
        print(f"   Precision: {eval_results.detection.precision:.3f}")
        print(f"   Recall: {eval_results.detection.recall:.3f}")
        print(f"   F1: {eval_results.detection.f1:.3f}")
        print(f"\n📊 Severity:")
        print(f"   Accuracy: {eval_results.severity.accuracy:.3f}")
        print(f"   Macro F1: {eval_results.severity.macro_f1:.3f}")
    
    print(f"\n✅ Results saved to: {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
