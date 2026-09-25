#!/usr/bin/env python3
"""
Evaluate trained adapter on UNSEEN evaluation log file.

IMPORTANT: This must NOT use the training data. It processes raw log files
that were never seen during training.

NOTE: This is a simplified evaluation harness. For production-style
log clustering and investigation, use the full log-analyzer pipeline.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "log-analyzer"))


def load_config(config_path: Path) -> dict:
    """Load configuration."""
    with config_path.open() as f:
        return yaml.safe_load(f)


def load_raw_logs(log_file: Path) -> list[str]:
    """Load raw log file (NOT training JSONL format)."""
    with log_file.open() as f:
        return [line.strip() for line in f if line.strip()]


def parse_log_line(line: str) -> dict:
    """
    Simple log parser to extract timestamp, level, service, message.
    
    Expected format: TIMESTAMP LEVEL [service] message
    Example: 2026-09-25T10:30:00.123Z ERROR [payment-api] connection timeout
    """
    import re
    
    # Try to parse structured log
    match = re.match(r'(\S+)\s+(\w+)\s+\[([^\]]+)\]\s+(.+)', line)
    if match:
        timestamp_str, level, service, message = match.groups()
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            timestamp = None
        
        return {
            "timestamp": timestamp,
            "level": level,
            "service": service,
            "message": message,
            "raw": line
        }
    
    # Fallback: treat as unstructured
    return {
        "timestamp": None,
        "level": "INFO",
        "service": "unknown",
        "message": line,
        "raw": line
    }


def cluster_logs_by_service(parsed_logs: list[dict]) -> dict[str, list[dict]]:
    """
    Group logs by service.
    
    This is a SIMPLIFIED clustering. Production uses more sophisticated
    incident detection and grouping.
    """
    clusters = defaultdict(list)
    for log in parsed_logs:
        service = log.get("service", "unknown")
        clusters[service].append(log)
    return dict(clusters)


def create_incident_from_cluster(cluster_logs: list[dict]) -> dict:
    """
    Create an incident representation from clustered logs.
    
    This is SIMPLIFIED. Production uses signature detection,
    time-window grouping, and related incident linking.
    """
    # Sort by timestamp if available
    sorted_logs = sorted(
        cluster_logs,
        key=lambda x: x.get("timestamp") or datetime.min
    )
    
    # Extract service
    service = sorted_logs[0].get("service", "unknown")
    
    # Get raw log lines
    log_lines = [log["raw"] for log in sorted_logs]
    
    return {
        "service": service,
        "environment": "prod",  # Assume prod for evaluation
        "logs": log_lines,
        "count": len(log_lines)
    }


def load_ground_truth(gt_path: Path) -> dict:
    """Load ground truth if available."""
    if not gt_path.exists():
        return None
    with gt_path.open() as f:
        return json.load(f)


def run_inference(adapter_path: Path, incidents: list[dict], config: dict) -> tuple[list[dict], list[float]]:
    """
    Run inference on incident clusters using the trained adapter.
    
    This processes incidents (groups of related logs), not individual log lines.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    
    print(f"\nLoading base model: {config['base_model']}")
    
    # QLoRA quantization config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        config['base_model'],
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load adapter
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'], trust_remote_code=True)
    
    gen_config = config.get("generation", {})
    
    predictions = []
    latencies = []
    
    print(f"\nRunning inference on {len(incidents)} incident clusters...")
    
    for i, incident in enumerate(incidents):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(incidents)}")
        
        # Build prompt (matches training format)
        logs_text = "\n".join(incident["logs"])
        service = incident["service"]
        environment = incident.get("environment", "prod")
        
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
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_config.get("max_new_tokens", 512),
                temperature=gen_config.get("temperature", 0.1),
                top_p=gen_config.get("top_p", 0.95),
                do_sample=gen_config.get("do_sample", True),
                repetition_penalty=gen_config.get("repetition_penalty", 1.1),
                pad_token_id=tokenizer.eos_token_id,
            )
        
        latency_ms = (time.time() - start_time) * 1000
        latencies.append(latency_ms)
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Parse JSON response
        import re
        try:
            # Extract JSON
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
            else:
                parsed = None
        except Exception:
            parsed = None
        
        predictions.append({
            "index": i,
            "service": service,
            "logs": incident["logs"],
            "log_count": len(incident["logs"]),
            "raw_response": response,
            "parsed": parsed,
            "latency_ms": latency_ms,
        })
    
    return predictions, latencies


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained adapter on UNSEEN evaluation logs")
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
        help="Raw log file for evaluation (NOT the training JSONL!)"
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("ground_truth/evaluation.json"),
        help="Ground truth file (optional)"
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
    
    args = parser.parse_args()
    
    # Verify we're not accidentally using training data
    if "train.jsonl" in str(args.logs):
        print("❌ ERROR: --logs points to training data (train.jsonl)!")
        print("   Evaluation MUST use a separate, unseen log file")
        print("   Example: --logs data/evaluation.log")
        sys.exit(1)
    
    # Load config
    config = load_config(args.config)
    
    # Load and parse raw logs
    print(f"Loading raw evaluation logs: {args.logs}")
    if not args.logs.exists():
        print(f"ERROR: Log file not found: {args.logs}")
        sys.exit(1)
    
    raw_logs = load_raw_logs(args.logs)
    print(f"  Loaded {len(raw_logs)} log lines")
    
    # Parse logs
    print("  Parsing log lines...")
    parsed_logs = [parse_log_line(line) for line in raw_logs]
    
    # Cluster logs by service (simplified incident detection)
    print("  Clustering logs by service...")
    clusters = cluster_logs_by_service(parsed_logs)
    
    # Create incident representations
    incidents = []
    for service, logs in clusters.items():
        incident = create_incident_from_cluster(logs)
        incidents.append(incident)
    
    print(f"  Created {len(incidents)} incident clusters from {len(clusters)} services")
    
    # Load ground truth
    ground_truth = load_ground_truth(args.ground_truth)
    if ground_truth:
        print(f"  Ground truth loaded: {args.ground_truth}")
    else:
        print(f"  No ground truth available - metrics will be limited")
    
    # Run inference on incidents (NOT individual log lines)
    predictions, latencies = run_inference(args.adapter, incidents, config)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    predictions_file = args.output / "predictions.jsonl"
    with predictions_file.open("w") as f:
        for pred in predictions:
            json.dump(pred, f)
            f.write("\n")
    
    print(f"\nPredictions saved: {predictions_file}")
    
    # Calculate metrics if ground truth available
    if ground_truth and "incidents" in ground_truth:
        print("\nCalculating metrics against ground truth...")
        
        # For now, if ground truth is provided, we need to match predictions to ground truth
        # This is simplified - production would have more sophisticated matching
        print("WARNING: Ground truth matching is simplified in this harness")
        print("         For production-quality metrics, use the full log-analyzer pipeline")
        
        # Save a basic summary
        summary = {
            "total_log_lines": len(raw_logs),
            "incident_clusters": len(incidents),
            "predictions_made": len(predictions),
            "valid_json_responses": sum(1 for p in predictions if p["parsed"] is not None),
            "latency": {
                "mean_ms": sum(latencies) / len(latencies) if latencies else 0,
                "p50_ms": sorted(latencies)[len(latencies)//2] if latencies else 0,
                "p95_ms": sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0,
            }
        }
        
        with (args.output / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary:")
        print(f"  Total log lines: {summary['total_log_lines']}")
        print(f"  Incident clusters: {summary['incident_clusters']}")
        print(f"  Valid JSON responses: {summary['valid_json_responses']}/{len(predictions)}")
        print(f"  Mean latency: {summary['latency']['mean_ms']:.1f}ms")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
