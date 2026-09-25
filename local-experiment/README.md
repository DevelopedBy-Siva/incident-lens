# Local Experiment Workspace

**Isolated QLoRA training and evaluation for IncidentLens incident detection models.**

This workspace allows iterating on datasets, prompts, and hyperparameters on a local high-GPU machine without repeatedly paying for AWS EC2 training. It reuses production IncidentLens training components but operates completely offline without AWS/S3/Datadog/database dependencies.

## Requirements

- Linux machine with NVIDIA GPU (CUDA-capable)
- Python 3.10+
- ~24GB GPU VRAM (for Qwen2.5-3B with QLoRA)
- ~50GB disk space

## Installation

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install IncidentLens Dependencies

The experiment harness reuses production training components from the parent `log-analyzer` directory. Ensure the parent environment is available or install:

```bash
pip install -e ../log-analyzer
```

## Directory Structure

```
local-experiment/
├── config.yaml              # Experiment configuration
├── requirements.txt         # Python dependencies
├── validate_dataset.py      # Dataset quality validation
├── train.py                 # QLoRA training script
├── evaluate.py              # Inference and evaluation
├── metrics.py               # Metric calculation
├── compare.py               # Compare experiment runs
├── run_experiment.py        # Full pipeline orchestrator
├── README.md                # This file
│
├── data/                    # Place your data here
│   ├── train.jsonl          # Training dataset (YOU provide)
│   └── evaluation.log       # Evaluation logs (optional, YOU provide)
│
├── ground_truth/            # Optional ground truth for evaluation
│   └── evaluation.json      # Expected incidents (optional, YOU provide)
│
└── results/                 # Experiment outputs (auto-generated)
    └── 20260925_120000/     # Timestamped experiment
        ├── adapter/         # Trained LoRA adapter
        ├── metadata.json    # Experiment metadata
        ├── training_result.json
        ├── predictions.jsonl
        ├── metrics.json
        ├── error_analysis.json
        └── report.txt
```

## Data Preparation

### ⚠️ CRITICAL: Training and Evaluation Data MUST Be Separate

**Training data** and **evaluation data** must be completely separate to get valid metrics:

- **Training:** `data/train.jsonl` (structured JSONL with expected outputs)
- **Evaluation:** `data/evaluation.log` (raw log file, NEVER seen during training)

### Training Dataset

Place your training dataset as `data/train.jsonl`. Each line must be a JSON object with:

```json
{
  "input": {
    "logs": ["2024-01-01T00:00:00Z ERROR service crashed", "..."],
    "service": "payment-api",
    "environment": "prod",
    "count": 5,
    "metadata": {"incident_type": "crash", "region": "us-east-1"}
  },
  "expected_output": {
    "severity": "high",
    "disposition": "NEEDS_ONCALL",
    "confidence": 0.85,
    "summary": "Service crash detected...",
    "suspected_root_cause": "Memory leak...",
    "next_steps": ["Restart service", "Check memory metrics"],
    "ticket_title": "Payment API crash",
    "ticket_body": "Detailed investigation needed..."
  }
}
```

**Schema Requirements:**
- `severity`: `low`, `medium`, `high`, or `critical` (NOT "benign")
- `disposition`: `NO_ACTION`, `OBSERVE`, `NEEDS_DEV`, `NEEDS_ONCALL`, or `ESCALATE`
- `confidence`: float between 0.0 and 1.0
- All 8 fields required: severity, disposition, confidence, summary, suspected_root_cause, next_steps, ticket_title, ticket_body

Benign/no-incident examples should use:
```json
{
  "severity": "low",
  "disposition": "NO_ACTION",
  "next_steps": [],
  "ticket_title": "",
  "ticket_body": ""
}
```

### Evaluation Data

**Evaluation MUST use unseen raw log files**, not the training JSONL:

**File:** `data/evaluation.log`

**Format:** Raw log lines (one per line), similar to production logs:

```
2026-09-25T10:30:00.123Z ERROR [payment-api] connection timeout to database
2026-09-25T10:30:01.456Z WARN [payment-api] retry attempt 1 failed
2026-09-25T10:30:02.789Z ERROR [payment-api] transaction aborted after 3 retries
2026-09-25T10:35:00.000Z INFO [user-service] heartbeat ok
```

**Expected format:** `TIMESTAMP LEVEL [service] message`

The evaluation script will:
1. Parse raw log lines
2. Cluster logs by service (simplified incident detection)
3. Run inference on each incident cluster
4. Generate predictions

**DO NOT:**
- ❌ Use training JSONL for evaluation
- ❌ Include evaluation logs in training data
- ❌ Evaluate on the same data you trained on

### Ground Truth (Optional)

If you want quantitative metrics, provide:

**File:** `ground_truth/evaluation.json`

**Format:**
```json
{
  "incidents": [
    {
      "id": "incident-001",
      "service": "payment-api",
      "severity": "high",
      "disposition": "NEEDS_ONCALL",
      "start_time": "2026-09-25T10:30:00Z",
      "end_time": "2026-09-25T10:35:00Z"
    }
  ]
}
```

If no ground truth is provided:
- Evaluation will still run inference
- Predictions will be saved
- Basic stats will be reported (log count, cluster count, latency)
- But no precision/recall/F1 metrics

## Configuration

Edit `config.yaml` to adjust:

### Model Selection

```yaml
base_model: "Qwen/Qwen2.5-3B-Instruct"  # or "Qwen/Qwen2.5-7B-Instruct"
```

### LoRA Hyperparameters

```yaml
lora:
  rank: 16         # LoRA rank (8, 16, 32, 64)
  alpha: 32        # LoRA alpha (typically 2x rank)
  dropout: 0.05    # LoRA dropout
```

### Training Parameters

```yaml
training:
  learning_rate: 0.0002
  num_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8  # Effective batch size = 1 * 8 = 8
  max_seq_length: 4096
```

### Generation Settings

```yaml
generation:
  max_new_tokens: 512
  temperature: 0.1    # Lower = more deterministic
  top_p: 0.95
  repetition_penalty: 1.1
```

## Usage

### Validate Training Dataset

Before training, validate your training dataset:

```bash
python validate_dataset.py --dataset data/train.jsonl
```

This checks:
- Schema validity (8 required fields)
- No "benign" severity values
- Severity/disposition vocabularies
- Confidence ranges
- Actionable incidents have required details
- Duplicate analysis
- Log sequence statistics

### Train Adapter

Train a LoRA adapter on the training dataset:

```bash
python train.py \
  --config config.yaml \
  --dataset data/train.jsonl \
  --output-dir results
```

This will:
1. Validate the dataset
2. Print system/GPU information
3. Load base model with QLoRA quantization
4. Fine-tune using LoRA
5. Save adapter to `results/<timestamp>/adapter/`
6. Save metadata and metrics

**Training takes ~30-60 minutes** for 2200 examples on a single A100 GPU (3 epochs).

**IMPORTANT:** This only uses `data/train.jsonl`. Evaluation data is NOT touched during training.

### Evaluate Adapter on Unseen Logs

Evaluate a trained adapter on **raw unseen log files**:

```bash
python evaluate.py \
  --adapter results/20260925_120000/adapter \
  --logs data/evaluation.log \
  --ground-truth ground_truth/evaluation.json \
  --config config.yaml \
  --output results/20260925_120000
```

**CRITICAL:** The `--logs` parameter must point to a **raw log file**, NOT the training JSONL!

This will:
1. Load raw log file (NOT training data)
2. Parse log lines
3. Cluster logs by service (simplified incident detection)
4. Run inference on incident clusters
5. Save predictions and summary

**The evaluation script has a safety check** - it will refuse to run if you accidentally point to `train.jsonl`.

### Run Complete Experiment

Run the full pipeline (validate → train → evaluate):

```bash
python run_experiment.py \
  --config config.yaml \
  --dataset data/train.jsonl \
  --evaluation-log data/evaluation.log \
  --ground-truth ground_truth/evaluation.json
```

Or evaluate an existing adapter without retraining:

```bash
python run_experiment.py \
  --config config.yaml \
  --dataset data/train.jsonl \
  --evaluation-log data/evaluation.log \
  --skip-training \
  --adapter results/20260925_120000/adapter
```

**Safety:** The script will fail if training and evaluation files are the same.

### Compare Experiments

Compare multiple experiment runs:

```bash
python compare.py \
  results/20260925_120000 \
  results/20260925_130000 \
  results/20260925_140000
```

This shows:
- Detection F1, severity accuracy, disposition F1
- Inference latency
- Configuration differences between runs

## Interpreting Results

### Metrics

Key metrics from `results/<experiment>/metrics.json`:

**Detection (binary: incident vs no-incident)**
- Precision: fraction of predicted incidents that were real
- Recall: fraction of real incidents that were detected
- F1: harmonic mean of precision and recall
- TP/FP/FN/TN counts

**Severity Classification**
- Accuracy: fraction of correct severity predictions
- Macro F1: average F1 across all severity classes
- Per-class precision/recall/F1

**Disposition Classification**
- Accuracy: fraction of correct disposition predictions
- Macro F1: average F1 across all dispositions

**Structured Output**
- Valid JSON rate: fraction of responses that parsed successfully
- Field completion rates: how often each field is present

**Latency**
- Mean, P50, P95, P99 inference time in milliseconds

### Error Analysis

Inspect `results/<experiment>/error_analysis.json`:

**False Positives**
- Logs where model predicted incident but ground truth was benign
- Check for pattern: are specific log types triggering false alarms?

**False Negatives**
- Logs where model missed a real incident
- Check for pattern: are specific incident types being missed?

**Severity Errors**
- Incidents where severity was wrong (e.g., predicted medium but was high)
- Check if errors are systematic (e.g., consistently under-predicting severity)

### Predictions

Review `results/<experiment>/predictions.jsonl`:
- Raw model outputs
- Parsed structured responses
- Per-example latency

## Experiment Workflow

### Baseline Run

1. Start with production config (default `config.yaml`)
2. Run baseline experiment:
   ```bash
   python run_experiment.py --dataset data/train.jsonl
   ```
3. Record baseline metrics

### Iterate on Hyperparameters

Create variant configs:

```bash
cp config.yaml config_rank32.yaml
# Edit config_rank32.yaml: set lora.rank = 32

python run_experiment.py --config config_rank32.yaml --dataset data/train.jsonl
```

Compare:
```bash
python compare.py results/baseline results/rank32
```

### Iterate on Dataset

If metrics show specific errors:
1. Fix dataset generator
2. Regenerate dataset
3. Retrain:
   ```bash
   python run_experiment.py --dataset data/train_v3.jsonl
   ```

### Test Prompt Changes

To test prompt changes, you'll need to modify the system prompt in production code (`app/training/lora_trainer.py`) since the local experiment reuses that component.

## Reproducibility

Each experiment saves:
- `metadata.json`: Git SHA, dataset SHA256, config, environment versions
- `training_result.json`: Training metrics and adapter path
- `config.yaml` snapshot (via metadata)

To reproduce an experiment:
1. Check out the same Git SHA
2. Use the same dataset (verify SHA256)
3. Use the same config
4. Run training with same random seed (set in `config.yaml` → `data.random_seed`)

## Production Integration

Once you've identified a better model/dataset/config locally:

1. **Update production dataset** (`../data/dataset_v2.jsonl`)
2. **Update training config** if needed
3. **Trigger production retraining** on AWS EC2:
   ```bash
   cd ../log-analyzer
   # Follow production training workflow
   ```
4. **Create new AMI** with updated adapter
5. **Deploy to production**

## Limitations

This local workspace is **simplified** compared to production:

### What It Does
✅ Uses production training code (TransformersPeftTrainingEngine)  
✅ Uses production LoRA profiles and hyperparameters  
✅ Trains identical adapters to production  
✅ Evaluates with same inference prompt/parsing  

### What It Doesn't Do
❌ Does not connect to production PostgreSQL/Neon  
❌ Does not upload to S3  
❌ Does not launch/terminate EC2  
❌ Does not send Datadog logs  
❌ Does not perform production-style log clustering/investigation  

For **full production-style evaluation** with clustering and multi-turn investigation, use the `log-analyzer` pipeline directly.

## Troubleshooting

### Out of Memory

If training fails with OOM:
- Reduce `per_device_train_batch_size` (try 1)
- Reduce `max_seq_length` (try 2048)
- Use smaller base model (Qwen2.5-1.5B)
- Increase `gradient_accumulation_steps` to compensate

### Slow Training

- Enable CUDA: check `torch.cuda.is_available()`
- Use smaller dataset for iteration
- Reduce `num_epochs`

### Invalid JSON Responses

If model produces unparseable JSON:
- Increase training data
- Add more JSON examples
- Adjust `temperature` (lower = more deterministic)
- Check prompt formatting

### Poor Metrics

- Check dataset quality with `validate_dataset.py`
- Verify ground truth matches expected format
- Inspect error analysis for systematic issues
- Try increasing training epochs or learning rate

## Files Reference

| File | Purpose |
|------|---------|
| `config.yaml` | Experiment hyperparameters |
| `requirements.txt` | Python dependencies |
| `validate_dataset.py` | Dataset quality checks |
| `train.py` | QLoRA training |
| `evaluate.py` | Inference and evaluation |
| `metrics.py` | Metric calculation library |
| `compare.py` | Multi-experiment comparison |
| `run_experiment.py` | Full pipeline orchestrator |
| `data/train.jsonl` | Training dataset (you provide) |
| `ground_truth/evaluation.json` | Ground truth (optional) |
| `results/<timestamp>/` | Experiment outputs |

## Support

For questions about:
- **Dataset format**: See production `data/dataset_v2.jsonl` and generator
- **Training issues**: Check production `app/training/lora_trainer.py`
- **Inference format**: Check production `app/serving/investigator.py`
- **Metrics**: Review `metrics.py` implementation

## License

Same as IncidentLens parent project.
