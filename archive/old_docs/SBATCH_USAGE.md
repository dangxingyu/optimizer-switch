# SBATCH Scripts Usage Guide - Moonlight Muon

## New Script: run_gsm8k_moonlight_300m.sbatch

### Quick Start

```bash
# Submit the job array
sbatch run_gsm8k_moonlight_300m.sbatch

# Check job status
squeue -u $USER

# Monitor specific job output
tail -f logs/gsm8k_moonlight_300m_<JOB_ID>_<ARRAY_ID>.out
```

## Job Configuration

### Job Array Layout

**Total jobs**: 10 (2 base models × 5 learning rates)

**Array indices**: 0-9

| Array ID | Base Model | LR | Description |
|----------|-----------|-----|-------------|
| 0 | adamw_300m_8 | 1e-5 | Very conservative |
| 1 | adamw_300m_8 | 3e-5 | Conservative |
| 2 | adamw_300m_8 | 1e-4 | Moderate (recommended) |
| 3 | adamw_300m_8 | 3e-4 | Moderate-aggressive |
| 4 | adamw_300m_8 | 1e-3 | Aggressive |
| 5 | muon_300m_8 | 1e-5 | Very conservative |
| 6 | muon_300m_8 | 3e-5 | Conservative |
| 7 | muon_300m_8 | 1e-4 | Moderate (recommended) |
| 8 | muon_300m_8 | 3e-4 | Moderate-aggressive |
| 9 | muon_300m_8 | 1e-3 | Aggressive |

### Learning Rate Selection

**Why these LRs?**

Moonlight is ~180x more aggressive than NorMuon:
- Standard momentum: 20x larger gradient contribution
- LR scaling: ~9x larger effective LR

**Previous NorMuon LRs**: 5e-6, 1e-5, 3e-5, 1e-4
**Moonlight LRs**: 1e-5, 3e-5, 1e-4, 3e-4, 1e-3

**Expected best**: 1e-4 or 3e-4

## Resource Requirements

- **Nodes**: 1
- **GPUs**: 1 per job
- **Memory**: 64GB
- **Time**: 4 hours max
- **Partition**: pli-c
- **Account**: pli

## Output Structure

```
outputs/gsm8k_moonlight/
├── adamw_300m_8/
│   ├── lr_1e-5/
│   │   ├── training_log.csv
│   │   ├── checkpoint_epoch_1.pt
│   │   └── final_model.pt
│   ├── lr_3e-5/
│   ├── lr_1e-4/
│   ├── lr_3e-4/
│   └── lr_1e-3/
└── muon_300m_8/
    ├── lr_1e-5/
    ├── lr_3e-5/
    ├── lr_1e-4/
    ├── lr_3e-4/
    └── lr_1e-3/
```

## Log Files

```
logs/
├── gsm8k_moonlight_300m_<JOB_ID>_0.out  # adamw_300m_8, lr=1e-5
├── gsm8k_moonlight_300m_<JOB_ID>_1.out  # adamw_300m_8, lr=3e-5
├── ...
└── gsm8k_moonlight_300m_<JOB_ID>_9.out  # muon_300m_8, lr=1e-3
```

## Monitoring Jobs

### Check all jobs

```bash
squeue -u $USER
```

### Check specific job array

```bash
squeue -u $USER -j <JOB_ID>
```

### Monitor live output

```bash
# Replace <JOB_ID> with actual job ID
tail -f logs/gsm8k_moonlight_300m_<JOB_ID>_2.out  # Array task 2
```

### Check all outputs

```bash
# See last 10 lines of all job outputs
tail logs/gsm8k_moonlight_300m_*.out
```

## Analyzing Results

### Quick loss comparison

```bash
# Extract final losses from all jobs
for i in {0..9}; do
    log="logs/gsm8k_moonlight_300m_*_${i}.out"
    if ls $log 2>/dev/null; then
        echo -n "Task $i: "
        grep "final loss" $log | tail -1
    fi
done
```

### Compare training logs

```bash
# List all completed training logs
find outputs/gsm8k_moonlight -name "training_log.csv"

# Compare final training loss for each LR
for lr in 1e-5 3e-5 1e-4 3e-4 1e-3; do
    echo "LR $lr:"
    for model in adamw_300m_8 muon_300m_8; do
        log="outputs/gsm8k_moonlight/${model}/lr_${lr}/training_log.csv"
        if [ -f "$log" ]; then
            final_loss=$(tail -1 "$log" | cut -d',' -f2)
            echo "  ${model}: ${final_loss}"
        fi
    done
done
```

### Check for failures

```bash
# Look for error messages
grep -i "error\|fail\|nan\|inf" logs/gsm8k_moonlight_300m_*.out
```

## Canceling Jobs

### Cancel entire array

```bash
scancel <JOB_ID>
```

### Cancel specific array task

```bash
scancel <JOB_ID>_<ARRAY_TASK_ID>
```

### Cancel all your jobs

```bash
scancel -u $USER
```

## Running Subset of Jobs

If you only want to test specific configurations:

### Test only one base model

Edit the script:
```bash
# Comment out one base model
BASE_MODELS=(
    "adamw_300m_8"
    # "muon_300m_8"
)

# Update array size: 1 model × 5 LRs = 5 jobs
#SBATCH --array=0-4
```

### Test only specific LRs

Edit the script:
```bash
# Only test moderate LRs
LRS=(
    "1e-4"
    "3e-4"
)

# Update array size: 2 models × 2 LRs = 4 jobs
#SBATCH --array=0-3
```

## Comparison with Previous Tests

### Original run_gsm8k_test_300m.sbatch

- **Jobs**: 16 (2 models × 2 optimizers × 4 LRs)
- **Optimizers**: AdamW + basic Muon (failed)
- **LRs**: 5e-6, 1e-5, 3e-5, 1e-4

### New run_gsm8k_moonlight_300m.sbatch

- **Jobs**: 10 (2 models × 5 LRs)
- **Optimizer**: Moonlight Muon only
- **LRs**: 1e-5, 3e-5, 1e-4, 3e-4, 1e-3 (adjusted for Moonlight)
- **Implementation**: Newton-Schulz 5 + standard momentum

## Expected Outcomes

### Success Criteria

✅ **Minimum**: Accuracy > 2.3% (better than basic Muon)
✅ **Target**: Accuracy > 10% (approaching AdamW's 13.5%)
✅ **Ideal**: Accuracy > 13% (matching or beating AdamW)

### What to Check

1. **Training loss**: Should decrease smoothly
2. **No NaN/Inf**: All jobs complete without errors
3. **Best LR**: Likely 1e-4 or 3e-4
4. **Model comparison**: Which base model (adamw vs muon) finetunes better

## Troubleshooting

### Job won't start

- Check partition availability: `sinfo -p pli-c`
- Check account limits: `sacctmgr show assoc user=$USER`

### Job fails immediately

- Check checkpoint exists: `ls ../checkpoints/adamw_300m_8`
- Check conda environment: `conda activate muon`
- Review error log: `cat logs/gsm8k_moonlight_300m_*_0.err`

### Out of memory

- Reduce batch size in training script
- Request more memory: `#SBATCH --mem=128G`

### Training diverges (NaN/Inf)

- LR too high, use smaller value
- Check gradient norm in logs
- Enable gradient clipping (already enabled by default)

## Next Steps After Jobs Complete

1. **Identify best LR**:
   ```bash
   # Compare all results
   python generate_accuracy_table.py
   ```

2. **Full training with best LR**:
   ```bash
   # Run complete training (no max_steps limit)
   sbatch run_gsm8k_moonlight_full.sbatch  # Create this if needed
   ```

3. **Compare to baseline**:
   - Previous Muon: ~2.3%
   - AdamW: ~13.5%
   - Moonlight: ? (hopefully >10%)

---

**Created**: 2025-11-19
**For**: Moonlight Muon GSM8K finetuning tests
**Script**: run_gsm8k_moonlight_300m.sbatch
