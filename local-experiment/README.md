# IncidentLens Local Experiment Harness

A simple, focused training and evaluation framework for the IncidentLens incident analysis model.

## What is IncidentLens?

**IncidentLens** is an AI-powered incident response system that analyzes production log streams and provides structured incident analysis.

The core task:

- Given production-style logs for a service
- Recognize meaningful operational incidents
- Distinguish incidents from normal production noise
- Assess severity (low/medium/high/critical)
- Determine appropriate operational disposition (NO_ACTION/OBSERVE/NEEDS_DEV/NEEDS_ONCALL/ESCALATE)
- Summarize what happened, infer root cause, recommend next steps

## What We're Training

A small language model (Qwen2.5-3B-Instruct) fine-tuned with QLoRA to analyze candidate incident sequences from service logs and produce structured incident analysis.

**One model invocation represents:**
"Analyze this candidate incident sequence from this service."

The model receives:

- Service name and environment
- Incident metadata (count, timestamps)
- Sample log lines from the incident window
- Related incidents (if any)

The model produces:

```json
{
  "severity": "low|medium|high|critical",
  "disposition": "NO_ACTION|OBSERVE|NEEDS_DEV|NEEDS_ONCALL|ESCALATE",
  "confidence": 0.0-1.0,
  "summary": "2-3 sentence summary",
  "suspected_root_cause": "explanation or null",
  "next_steps": ["action1", "action2", "action3"],
  "ticket_title": "concise title",
  "ticket_body": "detailed description"
}
```

## What We're Evaluating

Whether the fine-tuned model can:

1. **Detect incidents** - Distinguish real incidents from normal noise
2. **Classify severity correctly** - Critical vs high vs medium vs low
3. **Recommend appropriate disposition** - Escalate, notify on-call, create ticket, observe, or no action
4. **Produce valid structured output** - Complete, parseable JSON with all required fields
5. **Provide useful analysis** - Meaningful summaries, root causes, and next steps

## Architecture

### Pipeline Overview

```
Training Data (dataset_v2.jsonl)
         ↓
    Validation
         ↓
   QLoRA Fine-tuning
         ↓
      Adapter

Evaluation Log (evaluation.log)
         ↓
   Group by Service
         ↓
  Detect Incident Windows (production clustering)
         ↓
   Generate Candidates
         ↓
    Model Inference
         ↓
   Structured Output
         ↓
 Compare vs Ground Truth
         ↓
      Metrics
```

### Key Design Principles

1. **Uses Production Logic**: Reuses production parser, signatures, and clustering (2-minute time windows)
2. **Stage-Specific Metrics**: Separates candidate detection quality from model analysis quality
3. **Runs Locally**: No AWS, S3, EC2, PostgreSQL, or Docker required
4. **Simple**: Focused on the actual IncidentLens task, not generic log clustering research

## Directory Structure

```
local-experiment/
├── config.yaml                    # Model, LoRA, training, generation config
├── data/
│   └── evaluation.log            # Evaluation log stream
├── ground_truth/
│   └── evaluation.json           # Known incidents for evaluation
├── results/                       # Experiment outputs (gitignored)
│
├── data_loader.py                # Training data loader & validation
├── log_parser.py                 # Log parsing (reuses production parser)
├── incident_detector.py          # Incident windowing (production clustering)
├── metrics.py                    # Metrics computation & error analysis
│
├── train.py                      # QLoRA fine-tuning script
├── evaluate.py                   # Evaluation pipeline
├── validate_dataset.py           # Dataset validation
├── validate_ground_truth.py      # Ground truth validation
├── run_experiment.py             # End-to-end orchestrator
├── compare.py                    # Compare two experiments
│
├── test_experiment.py            # Comprehensive tests
└── README.md                     # This file
```

## Quick Start

### Prerequisites

```bash
# Python 3.10+
# GPU recommended for training (CPU works but slow)

# Install dependencies
pip install -r requirements.txt
```

### 1. Validate Data

```bash
# Validate training dataset
python validate_dataset.py --dataset ../data/dataset_v2.jsonl

# Validate ground truth
python validate_ground_truth.py --ground-truth ground_truth/evaluation.json
```

### 2. Run Full Experiment

```bash
python run_experiment.py \
  --dataset ../data/dataset_v2.jsonl \
  --evaluation-log data/evaluation.log \
  --ground-truth ground_truth/evaluation.json \
  --name baseline_experiment
```

This will:

1. Validate dataset and ground truth
2. Train model with QLoRA
3. Evaluate on unseen logs
4. Generate metrics and error analysis

Results saved to: `results/baseline_experiment/`

### 3. View Results

```bash
# View metrics
cat results/baseline_experiment/metrics.json

# View error analysis
cat results/baseline_experiment/evaluation/eval_*/error_analysis.json

# View incident candidates (debugging)
cat results/baseline_experiment/evaluation/eval_*/incident_candidates.jsonl
```

### 4. Compare Experiments

```bash
python compare.py results/baseline_experiment results/improved_experiment
```

## Individual Commands

### Train Only

```bash
python train.py \
  --config config.yaml \
  --dataset ../data/dataset_v2.jsonl \
  --output results
```

Output: `results/train_TIMESTAMP/adapter/`

### Evaluate Only

```bash
python evaluate.py \
  --config config.yaml \
  --logs data/evaluation.log \
  --ground-truth ground_truth/evaluation.json \
  --adapter results/train_TIMESTAMP/adapter \
  --output results
```

### Evaluate Base Model (No Fine-tuning)

```bash
python evaluate.py \
  --config config.yaml \
  --logs data/evaluation.log \
  --ground-truth ground_truth/evaluation.json \
  --output results
```

Note: Omit `--adapter` to evaluate base model without fine-tuning.

## Configuration

Edit `config.yaml` to adjust:

```yaml
model:
  base_model: "Qwen/Qwen2.5-3B-Instruct" # Change model

lora:
  r: 16 # LoRA rank
  alpha: 32 # LoRA alpha
  dropout: 0.05

training:
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4

generation:
  max_new_tokens: 800
  temperature: 0.2
```

## Metrics Explained

### Detection Metrics

- **Precision**: Of incidents predicted, what % were real?
- **Recall**: Of real incidents, what % were detected?
- **F1**: Harmonic mean of precision and recall

### Severity Metrics

- **Accuracy**: % of matched incidents with correct severity
- **Macro F1**: Average F1 across severity classes

### Disposition Metrics

- **Accuracy**: % of matched incidents with correct disposition
- **Macro F1**: Average F1 across disposition classes

### Structured Output Metrics

- **Valid JSON Rate**: % of predictions that produced parseable JSON
- **Field Completion**: % of predictions with non-empty required fields

### Stage-Specific Metrics

**Candidate Detection**: Did we identify the right incident windows?

- Uses production clustering (signature-based, 2-minute windows)
- Separate from model quality

**Model Analysis**: Given correct incident window, did model classify correctly?

- Severity/disposition accuracy
- Structured output validity

**End-to-End**: Complete pipeline performance

- Detection × Analysis

## Ground Truth

Ground truth (`ground_truth/evaluation.json`) describes known incidents in the evaluation log:

```json
{
  "incidents": [
    {
      "incident_id": "gt-incident-001",
      "service": "payment-api",
      "severity": "high",
      "disposition": "NEEDS_ONCALL",
      "start_time": "2026-09-25T10:00:00Z",
      "end_time": "2026-09-25T10:02:45Z",
      "description": "Database connection pool exhaustion",
      "log_line_markers": ["connection pool exhausted"]
    }
  ]
}
```

Predictions are matched to ground truth using:

- Service name match
- Temporal overlap ≥ 50%
- Best match per ground truth incident

## Training Data

Training data (`../data/dataset_v2.jsonl`) contains examples with:

```json
{
  "input": {
    "service": "checkout-api",
    "environment": "prod",
    "count": 52,
    "logs": ["...", "..."],
    "related_incidents": []
  },
  "expected_output": {
    "severity": "high",
    "disposition": "NEEDS_ONCALL",
    "confidence": 0.89,
    "summary": "...",
    "suspected_root_cause": "...",
    "next_steps": ["...", "..."],
    "ticket_title": "...",
    "ticket_body": "..."
  }
}
```

2200 examples covering:

- Various severities (low/medium/high/critical)
- All dispositions (NO_ACTION through ESCALATE)
- Different incident types (OOM, DB errors, deployments, latency, etc.)
- Benign examples (normal operations that should be NO_ACTION)

## Testing

```bash
# Run all tests
python test_experiment.py

# Or with pytest directly
pytest test_experiment.py -v
```

Tests cover:

- Data loading and validation
- Log parsing and service grouping
- Incident detection and clustering
- Metrics computation
- Temporal overlap matching
- No AWS/database dependencies

## Production Alignment

This experiment harness closely follows production IncidentLens:

| Component            | Production                    | Experiment         |
| -------------------- | ----------------------------- | ------------------ |
| Log Parser           | `app/data/parser.py`          | Imported directly  |
| Signature Generation | `app/data/signatures.py`      | Imported directly  |
| Clustering           | 2-min window, signature-based | Same algorithm     |
| System Prompt        | `app/serving/investigator.py` | Matching structure |
| Output Schema        | `IncidentAnalysis` model      | Identical fields   |

## Troubleshooting

### Out of Memory During Training

Reduce batch size in `config.yaml`:

```yaml
training:
  batch_size: 2 # Reduce from 4
  gradient_accumulation_steps: 8 # Increase to maintain effective batch size
```

### Parse Failures During Evaluation

Check:

1. Model output format (look at `predictions.json` → `raw_output`)
2. Generation parameters (try lower `temperature`)
3. `max_new_tokens` (may need more tokens for full JSON)

### Low Detection Recall

Possible causes:

1. Incident windowing not matching ground truth boundaries
2. Check `incident_candidates.jsonl` to see what windows were detected
3. Production clustering may group differently than expected

### Low Severity/Disposition Accuracy

Model may need:

1. More training data for rare classes
2. Better prompt engineering
3. Larger base model
4. More training epochs

## Limitations

1. **Local-only**: This is a development/research harness, not production
2. **Simplified context**: No real-time related incidents, runbooks, or root-cause chaining
3. **Static evaluation**: Uses pre-generated logs, not live streams
4. **No cross-service reasoning**: Each service analyzed independently

## FAQ

**Q: Why not use the existing production training pipeline?**  
A: Production training requires AWS infrastructure. This runs locally for fast iteration.

**Q: Why signature-based clustering instead of embeddings?**  
A: That's what production uses. This experiment matches production behavior.

**Q: Can I use a different base model?**  
A: Yes, edit `config.yaml`. Tested with Qwen2.5-3B and Qwen2.5-7B.

**Q: How long does training take?**  
A: ~30-60 minutes on a single GPU (RTX 3090 or similar) for 3 epochs on 2200 examples.

**Q: Can I run without GPU?**  
A: Yes, but training will be very slow. Evaluation is feasible on CPU.

**Q: What if I want to add more training data?**  
A: Add to `dataset_v2.jsonl` following the schema, then run `validate_dataset.py`.

**Q: How do I know if my model is better?**  
A: Use `compare.py` to see metric deltas. Focus on detection F1, severity accuracy, and disposition accuracy.

## Next Steps

After running experiments:

1. **Review error analysis** - Understand false positives and false negatives
2. **Iterate on training data** - Add examples for failure modes
3. **Tune hyperparameters** - Adjust LoRA rank, learning rate, epochs
4. **Try different models** - Test Qwen2.5-7B or Llama-3.2-3B
5. **Compare results** - Use `compare.py` to quantify improvements

## Contributing

When modifying the experiment harness:

1. Keep it simple - no unnecessary abstractions
2. Match production behavior - import production modules when possible
3. Run tests - `python test_experiment.py`
4. Validate data - run validation scripts before committing data changes
5. Document changes - update this README

## License

See main repository LICENSE.
