"""
Evaluation script for trained incident analysis models.

Pipeline:
1. Load evaluation log
2. Group by service
3. Detect incident candidates (using production clustering)
4. Run model inference on each candidate
5. Parse and validate outputs
6. Compare against ground truth
7. Generate metrics and error analysis
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from log_parser import load_evaluation_log, group_logs_by_service_with_metadata
from incident_detector import detect_all_incidents, summarize_candidates
from metrics import compute_metrics, load_ground_truth
from train import SYSTEM_PROMPT


class ModelInference:
    """Handles model loading and inference."""
    
    def __init__(self, base_model: str, adapter_path: Optional[str] = None):
        """Initialize model and tokenizer."""
        print(f"Loading tokenizer from {base_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading base model {base_model}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        if adapter_path:
            print(f"Loading adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
        
        self.model.eval()
        print("Model ready")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 800,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        """Generate response from prompt."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response


def build_incident_prompt(candidate) -> str:
    """Build prompt for incident candidate (matching production format)."""
    # Format evidence like production
    evidence_lines = ["=== Sample log lines ==="]
    for log in candidate.sample_lines[:10]:  # Limit to 10 samples
        evidence_lines.append(f"  {log}")
    
    evidence_lines.append("\n=== Related open incidents ===")
    evidence_lines.append("  (none)")  # Local eval doesn't have related incidents
    
    evidence_context = "\n".join(evidence_lines)
    
    prompt = f"""Analyze this incident:

Source: {candidate.service} | Environment: {candidate.environment}
Count: {candidate.count} | First seen: {candidate.first_seen.strftime('%Y-%m-%d %H:%M:%S')} | Last seen: {candidate.last_seen.strftime('%Y-%m-%d %H:%M:%S')}

{evidence_context}

Provide your analysis as a JSON object."""
    
    return prompt


def parse_model_output(raw_output: str) -> Dict[str, Any]:
    """
    Parse model output to extract JSON.
    
    Returns dict with:
        - success: bool
        - parsed: dict or None
        - error: str or None
        - raw: original output
    """
    import re
    
    result = {
        "success": False,
        "parsed": None,
        "error": None,
        "raw": raw_output,
    }
    
    # Try to extract JSON from response
    # Handle markdown code blocks
    cleaned = re.sub(r"```(?:json)?", "", raw_output).strip()
    
    # Try to find JSON object
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not json_match:
        result["error"] = "No JSON object found in output"
        return result
    
    json_str = json_match.group(0)
    
    try:
        parsed = json.loads(json_str)
        
        # Validate required fields
        required = [
            "severity",
            "disposition",
            "confidence",
            "summary",
            "next_steps",
            "ticket_title",
            "ticket_body",
        ]
        
        missing = [f for f in required if f not in parsed]
        if missing:
            result["error"] = f"Missing required fields: {missing}"
            return result
        
        result["success"] = True
        result["parsed"] = parsed
        return result
        
    except json.JSONDecodeError as e:
        result["error"] = f"JSON decode error: {e}"
        return result


def evaluate_model(
    model: ModelInference,
    candidates: List,
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run model inference on all candidates.
    
    Returns dict with predictions and metadata.
    """
    generation_config = config.get("generation", {})
    
    predictions = []
    inference_times = []
    
    print(f"\n{'='*60}")
    print(f"Running inference on {len(candidates)} candidates...")
    print(f"{'='*60}\n")
    
    for idx, candidate in enumerate(candidates):
        print(f"[{idx+1}/{len(candidates)}] {candidate.incident_id} ({candidate.service})...", end=" ")
        
        prompt = build_incident_prompt(candidate)
        
        t0 = time.time()
        try:
            raw_output = model.generate(
                prompt,
                max_new_tokens=generation_config.get("max_new_tokens", 800),
                temperature=generation_config.get("temperature", 0.2),
                top_p=generation_config.get("top_p", 0.9),
            )
            inference_time_ms = int((time.time() - t0) * 1000)
            inference_times.append(inference_time_ms)
            
            # Parse output
            parse_result = parse_model_output(raw_output)
            
            prediction = {
                "incident_id": candidate.incident_id,
                "service": candidate.service,
                "environment": candidate.environment,
                "first_seen": candidate.first_seen.isoformat(),
                "last_seen": candidate.last_seen.isoformat(),
                "count": candidate.count,
                "sample_logs": candidate.sample_lines[:3],  # First 3 for inspection
                "raw_output": raw_output,
                "parse_success": parse_result["success"],
                "parse_error": parse_result["error"],
                "parsed_output": parse_result["parsed"],
                "inference_time_ms": inference_time_ms,
            }
            
            predictions.append(prediction)
            
            status = "✓" if parse_result["success"] else "✗"
            print(f"{status} ({inference_time_ms}ms)")
            
        except Exception as e:
            print(f"✗ ERROR: {e}")
            predictions.append({
                "incident_id": candidate.incident_id,
                "service": candidate.service,
                "error": str(e),
                "parse_success": False,
            })
    
    # Compute statistics
    successful_parses = sum(1 for p in predictions if p.get("parse_success", False))
    parse_rate = successful_parses / len(predictions) if predictions else 0.0
    
    stats = {
        "total_candidates": len(candidates),
        "successful_predictions": successful_parses,
        "failed_predictions": len(predictions) - successful_parses,
        "parse_success_rate": parse_rate,
        "inference_times_ms": {
            "mean": sum(inference_times) / len(inference_times) if inference_times else 0,
            "min": min(inference_times) if inference_times else 0,
            "max": max(inference_times) if inference_times else 0,
            "p50": sorted(inference_times)[len(inference_times)//2] if inference_times else 0,
            "p95": sorted(inference_times)[int(len(inference_times)*0.95)] if inference_times else 0,
        }
    }
    
    print(f"\n{'='*60}")
    print(f"Inference complete:")
    print(f"  Successful: {successful_parses}/{len(predictions)} ({parse_rate:.1%})")
    print(f"  Mean latency: {stats['inference_times_ms']['mean']:.0f}ms")
    print(f"  P95 latency: {stats['inference_times_ms']['p95']:.0f}ms")
    print(f"{'='*60}\n")
    
    return {
        "predictions": predictions,
        "statistics": stats,
    }


def evaluate(args):
    """Main evaluation pipeline."""
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"IncidentLens Evaluation")
    print(f"{'='*60}")
    print(f"Evaluation log: {args.logs}")
    print(f"Ground truth: {args.ground_truth}")
    print(f"Adapter: {args.adapter or 'None (base model only)'}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Step 1: Load and group logs
    print("Step 1: Loading evaluation logs...")
    log_lines = load_evaluation_log(Path(args.logs))
    print(f"  Loaded {len(log_lines)} log lines")
    
    print("\nStep 2: Grouping logs by service...")
    service_logs = group_logs_by_service_with_metadata(log_lines)
    print(f"  Found {len(service_logs)} services: {', '.join(sorted(service_logs.keys()))}")
    
    # Step 3: Detect incident candidates
    print("\nStep 3: Detecting incident candidates...")
    candidates = detect_all_incidents(service_logs, environment="prod")
    summary = summarize_candidates(candidates)
    print(f"  Detected {summary['total_candidates']} incident candidates")
    for service, count in summary['candidates_by_service'].items():
        print(f"    {service}: {count}")
    
    # Save candidates
    candidates_path = output_dir / "incident_candidates.jsonl"
    with open(candidates_path, "w") as f:
        for c in candidates:
            f.write(json.dumps(c.to_dict()) + "\n")
    print(f"\n  Saved candidates to {candidates_path}")
    
    # Step 4: Load model
    print("\nStep 4: Loading model...")
    base_model = config["model"]["base_model"]
    model = ModelInference(base_model, args.adapter)
    
    # Step 5: Run inference
    print("\nStep 5: Running model inference...")
    results = evaluate_model(model, candidates, config, output_dir)
    
    # Save predictions
    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved predictions to {predictions_path}")
    
    # Step 6: Compute metrics
    if args.ground_truth:
        print("\nStep 6: Computing metrics against ground truth...")
        ground_truth = load_ground_truth(Path(args.ground_truth))
        
        metrics_result = compute_metrics(
            predictions=results["predictions"],
            ground_truth=ground_truth,
            candidates=candidates,
        )
        
        # Save metrics
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_result, f, indent=2)
        print(f"Saved metrics to {metrics_path}")
        
        # Save error analysis
        error_analysis_path = output_dir / "error_analysis.json"
        with open(error_analysis_path, "w") as f:
            json.dump(metrics_result.get("error_analysis", {}), f, indent=2)
        print(f"Saved error analysis to {error_analysis_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("Metrics Summary:")
        print(f"{'='*60}")
        
        det = metrics_result.get("detection", {})
        print(f"\nIncident Detection:")
        print(f"  Precision: {det.get('precision', 0):.3f}")
        print(f"  Recall: {det.get('recall', 0):.3f}")
        print(f"  F1: {det.get('f1', 0):.3f}")
        print(f"  TP: {det.get('true_positives', 0)}")
        print(f"  FP: {det.get('false_positives', 0)}")
        print(f"  FN: {det.get('false_negatives', 0)}")
        
        sev = metrics_result.get("severity", {})
        print(f"\nSeverity Classification:")
        print(f"  Accuracy: {sev.get('accuracy', 0):.3f}")
        print(f"  Macro F1: {sev.get('macro_f1', 0):.3f}")
        
        disp = metrics_result.get("disposition", {})
        print(f"\nDisposition Classification:")
        print(f"  Accuracy: {disp.get('accuracy', 0):.3f}")
        print(f"  Macro F1: {disp.get('macro_f1', 0):.3f}")
        
        struct = metrics_result.get("structured_output", {})
        print(f"\nStructured Output:")
        print(f"  Valid JSON rate: {struct.get('valid_json_rate', 0):.3f}")
        
        print(f"\n{'='*60}\n")
    
    # Save metadata
    metadata = {
        "evaluation_started": datetime.now().isoformat(),
        "evaluation_log": str(args.logs),
        "ground_truth": str(args.ground_truth) if args.ground_truth else None,
        "adapter": str(args.adapter) if args.adapter else None,
        "base_model": config["model"]["base_model"],
        "candidates_detected": len(candidates),
        "config": config,
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Evaluation complete. Results in {output_dir}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Evaluate incident analysis model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--logs",
        type=str,
        required=True,
        help="Path to evaluation log file",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        help="Path to ground truth JSON",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        help="Path to trained adapter (optional, uses base model if not provided)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    if not Path(args.logs).exists():
        print(f"❌ Evaluation log not found: {args.logs}")
        sys.exit(1)
    
    if args.ground_truth and not Path(args.ground_truth).exists():
        print(f"❌ Ground truth not found: {args.ground_truth}")
        sys.exit(1)
    
    if not Path(args.config).exists():
        print(f"❌ Config not found: {args.config}")
        sys.exit(1)
    
    evaluate(args)


if __name__ == "__main__":
    main()
