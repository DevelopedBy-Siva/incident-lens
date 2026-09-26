# Quick Start Guide

## Prerequisites

```bash
# Ensure you're in the virtual environment
source ../.venv/bin/activate  # or appropriate path

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

## 1. Validate Your Data (Recommended)

```bash
# Validate training dataset (2200 examples)
python validate_dataset.py --dataset ../data/dataset_v2.jsonl

# Validate ground truth (8 incidents)
python validate_ground_truth.py --ground-truth ground_truth/evaluation.json
```

Expected output:

- ✓ Dataset validation PASSED (2200 valid examples)
- ✓ Ground truth validation PASSED (8 incidents)

## 2. Run Tests (Optional)

```bash
python -m pytest test_experiment.py -v
```

Expected: 22 tests pass

## 3. Run a Baseline Experiment

```bash
# Full pipeline: validate → train → evaluate
python run_experiment.py \
  --dataset ../data/dataset_v2.jsonl \
  --evaluation-log data/evaluation.log \
  --ground-truth ground_truth/evaluation.json \
  --name baseline_qwen3b
```

This will:

1. Validate dataset and ground truth
2. Train with QLoRA (3 epochs, ~30-60 min on GPU)
3. Evaluate on unseen logs
4. Generate metrics and error analysis

Results saved to: `results/baseline_qwen3b/`

## 4. View Results

```bash
# Quick metrics summary
cat results/baseline_qwen3b/metrics.json | jq '.summary'

# Full metrics
cat results/baseline_qwen3b/metrics.json | jq '.'

# Error analysis
cat results/baseline_qwen3b/evaluation/eval_*/error_analysis.json | jq '.false_positives, .false_negatives'

# Incident candidates (for debugging)
head -5 results/baseline_qwen3b/evaluation/eval_*/incident_candidates.jsonl
```

## 5. Iterate and Compare

```bash
# Modify config.yaml (e.g., increase epochs, change model)
# Run another experiment
python run_experiment.py \
  --dataset ../data/dataset_v2.jsonl \
  --evaluation-log data/evaluation.log \
  --ground-truth ground_truth/evaluation.json \
  --name improved_qwen3b

# Compare results
python compare.py results/baseline_qwen3b results/improved_qwen3b
```

## Common Workflows

### Evaluate Base Model Without Fine-tuning

```bash
python evaluate.py \
  --logs data/evaluation.log \
  --ground-truth ground_truth/evaluation.json \
  --output results/base_model_only
```

### Re-evaluate Existing Adapter

```bash
python run_experiment.py \
  --skip-training \
  --adapter results/baseline_qwen3b/training/train_*/adapter \
  --evaluation-log data/evaluation.log \
  --ground-truth ground_truth/evaluation.json \
  --name baseline_reeval
```

### Train Only (No Evaluation)

```bash
python train.py \
  --dataset ../data/dataset_v2.jsonl \
  --output results/train_only
```

## Key Files to Inspect

After running an experiment:

```
results/baseline_qwen3b/
├── experiment_metadata.json       # Experiment config & timestamps
├── metrics.json                   # All metrics (copied for easy access)
├── training/
│   └── train_TIMESTAMP/
│       ├── adapter/               # Trained LoRA adapter
│       ├── metadata.json          # Training metadata
│       └── checkpoint-*/          # Training checkpoints
└── evaluation/
    └── eval_TIMESTAMP/
        ├── incident_candidates.jsonl  # Detected incident windows
        ├── predictions.json           # Model predictions & raw outputs
        ├── metrics.json               # Evaluation metrics
        ├── error_analysis.json        # FP/FN/mismatches
        └── metadata.json              # Evaluation metadata
```

## Troubleshooting

### Out of Memory During Training

Edit `config.yaml`:

```yaml
training:
  batch_size: 2 # Reduce from 4
  gradient_accumulation_steps: 8 # Increase to maintain effective batch size
```

### Low Parse Rate

Check raw outputs:

```bash
cat results/baseline_qwen3b/evaluation/eval_*/predictions.json | \
  jq '.predictions[] | select(.parse_success == false) | .raw_output' | head -20
```

Try adjusting generation config in `config.yaml`:

```yaml
generation:
  max_new_tokens: 1000 # Increase from 800
  temperature: 0.1 # Decrease from 0.2 for more deterministic output
```

### Adapter Not Found

Make sure training completed successfully:

```bash
ls -la results/baseline_qwen3b/training/train_*/adapter/
```

Should contain: `adapter_model.safetensors`, `adapter_config.json`, `tokenizer*`

## Expected Metrics (Rough Baseline)

For reference, a reasonable baseline with Qwen2.5-3B on this dataset:

- **Detection F1**: 0.7 - 0.9 (depends on windowing quality)
- **Severity Accuracy**: 0.6 - 0.8
- **Disposition Accuracy**: 0.5 - 0.7
- **Valid JSON Rate**: 0.9 - 1.0

Your results may vary based on:

- Training duration
- Model size
- Generation parameters
- Random seed

## Next Steps

1. **Analyze errors**: Look at `error_analysis.json` to understand failure modes
2. **Check candidates**: Inspect `incident_candidates.jsonl` to see if windows are correct
3. **Iterate on config**: Adjust hyperparameters in `config.yaml`
4. **Try different models**: Change `base_model` in `config.yaml`
5. **Compare experiments**: Use `compare.py` to track improvements

## Need Help?

- Read the full [README.md](README.md) for detailed documentation
- Check test cases in `test_experiment.py` for examples
- Review production code in `../log-analyzer/app/data/` and `../log-analyzer/app/serving/`
