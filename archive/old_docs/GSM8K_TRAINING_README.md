# GSM8K Finetuning with Learning Rate Sweep

Complete pipeline for finetuning all pretrained models on GSM8K with learning rate sweep.

## Quick Start

```bash
cd /scratch/gpfs/ARORA/xd7812/optimizers/modded-nanogpt

# Submit all 32 jobs (8 models × 4 LRs)
./submit_jobs.sh

# Or submit manually
sbatch run_gsm8k_finetuning.sbatch
```

## Job Configuration

### Models (8 total)
- **AdamW pretrained** (4 models):
  - 130M: seeds 1, 8
  - 300M: seeds 1, 8

- **Muon pretrained** (4 models):
  - 130M: seeds 1, 8
  - 300M: seeds 1, 8

### Learning Rates (4 total)
- `5e-6` - Very low (conservative)
- `1e-5` - Low
- `3e-5` - Medium
- `1e-4` - Medium-high

### Total Jobs
8 models × 4 LRs = **32 jobs**

## Resources per Job

- **GPU**: 1× GPU
- **RAM**: 64GB
- **Time**: 24 hours
- **Partition**: pli-c
- **Account**: pli

## Directory Structure

```
modded-nanogpt/
├── run_gsm8k_finetuning.sbatch    # Main SLURM script
├── submit_jobs.sh                  # Helper submission script
├── analyze_results.py              # Results analysis script
├── logs/
│   └── gsm8k_<job_id>_<task_id>.{out,err}
└── outputs/
    └── gsm8k/
        └── <model_name>/
            └── lr_<lr>/
                ├── checkpoint_step_*.pt
                └── final_model.pt
```

## Monitoring Jobs

### Check job status
```bash
# All jobs
squeue -u $USER

# Specific job array
squeue -u $USER | grep <job_id>

# Count running/pending
squeue -u $USER -t RUNNING | wc -l
squeue -u $USER -t PENDING | wc -l
```

### View logs
```bash
# Latest output
tail -f logs/gsm8k_<job_id>_0.out

# Search for specific model
grep "Model: muon_130m_1" logs/gsm8k_*.out

# Check for errors
grep -i "error\|fail" logs/gsm8k_*.err

# Check completion
grep "Training Completed Successfully" logs/gsm8k_*.out | wc -l
```

### Cancel jobs
```bash
# Cancel all jobs in the array
scancel <job_id>

# Cancel specific task
scancel <job_id>_<task_id>

# Cancel by model name (requires parsing)
for f in logs/gsm8k_*_*.out; do
  if grep -q "Model: muon_130m_1" "$f"; then
    task_id=$(basename "$f" .out | cut -d_ -f3)
    scancel ${job_id}_${task_id}
  fi
done
```

## Analyzing Results

### Run analysis
```bash
python analyze_results.py
```

This generates:
- `gsm8k_results_raw.csv` - All runs with metrics
- `gsm8k_results_best.csv` - Best LR per model
- `gsm8k_results_averaged.csv` - Averaged across seeds

### Manual inspection
```bash
# Check final accuracies
grep "Final GSM8K Accuracy" logs/gsm8k_*.out

# Find best result
grep "Final GSM8K Accuracy" logs/gsm8k_*.out | sort -k4 -n | tail -5

# Compare Muon vs AdamW for 130M
grep "Final GSM8K Accuracy" logs/gsm8k_*.out | grep "130m"
```

## Task ID Mapping

Task IDs are assigned as: `task_id = model_idx * 4 + lr_idx`

### Models (model_idx)
```
0: adamw_130m_1
1: adamw_130m_8
2: adamw_300m_1
3: adamw_300m_8
4: muon_130m_1
5: muon_130m_8
6: muon_300m_1
7: muon_300m_8
```

### Learning Rates (lr_idx)
```
0: 5e-6
1: 1e-5
2: 3e-5
3: 1e-4
```

### Examples
- Task 0: `adamw_130m_1` with `lr=5e-6`
- Task 3: `adamw_130m_1` with `lr=1e-4`
- Task 4: `adamw_130m_8` with `lr=5e-6`
- Task 16: `muon_130m_1` with `lr=5e-6`
- Task 31: `muon_300m_8` with `lr=1e-4`

## Troubleshooting

### Job fails immediately
- Check checkpoint path exists: `ls ../checkpoints/<model_name>`
- Check conda environment: `conda activate muon`
- Check GPU availability: `nvidia-smi`

### Out of memory
- Reduce batch size in training scripts (default: 4)
- Check model size vs GPU memory
- 130M/300M: should fit on 1 GPU
- 1.2B: may need larger GPU or gradient checkpointing

### Dataset not found
- Ensure GSM8K is cached: run download script first
- Check offline mode variables are set
- Verify: `ls ~/.cache/huggingface/datasets/openai___gsm8k/`

### NaN loss
- Lower learning rate (try 1e-5 or 3e-5)
- Check gradient clipping (default: 1.0)
- Verify checkpoint loaded correctly

## Expected Results

Based on typical GSM8K finetuning:

### Baseline Accuracies (random init)
- 130M: ~5-10%
- 300M: ~10-20%
- 1.2B: ~20-30%

### After Finetuning (rough estimates)
- 130M: ~30-40%
- 300M: ~40-50%
- 1.2B: ~50-60%

**Note**: Muon vs AdamW difference is the key research question!

## Next Steps After Training

1. **Analyze results**: `python analyze_results.py`
2. **Select best LR** per model size
3. **Compare Muon vs AdamW**: accuracy, convergence speed
4. **Plot learning curves**: use logs to extract per-step metrics
5. **Statistical significance**: t-test across 3 seeds

## Files Reference

- `train_llama_muon_single_gpu.py` - Muon training script
- `train_llama_adamw_single_gpu.py` - AdamW training script
- `eval_utils.py` - GSM8K evaluation (number extraction)
- `llama_model.py` - Llama model implementation
- `qa_data_utils.py` - QA data loading utilities
