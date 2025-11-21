# Comprehensive GSM8K Finetuning Sweep

## Overview

This comprehensive sweep tests **all 130M and 300M_1 models** with **both AdamW and Moonlight Muon** optimizers across a wide range of learning rates.

## Experiment Setup

### Base Models (8 total)

**130M Models:**
- adamw_130m_1, adamw_130m_2, adamw_130m_8
- muon_130m_1, muon_130m_2, muon_130m_8

**300M Models:**
- adamw_300m_1, muon_300m_1

### Finetuning Optimizers (2 total)

1. **AdamW** - Standard AdamW finetuning
2. **Moonlight Muon** - Simplified Muon with Newton-Schulz 5 orthogonalization

### Learning Rates (6 total)

- 1e-5
- 3e-5
- 1e-4
- 3e-4
- 1e-3
- 2e-3 (new, added based on 300m_8 results)

### Total Experiments

**96 jobs** = 8 base models × 2 optimizers × 6 learning rates

## Training Configuration

- **Max Steps:** 500
- **Eval Interval:** 250 (3 evaluations per run)
- **Batch Size:** 8
- **Gradient Accumulation:** 4 (effective batch size = 32)
- **Dataset:** GSM8K (math word problems)
- **GPU:** Single A100 80GB per job
- **Time Limit:** 4 hours per job

## Usage

### Submit All Jobs

```bash
cd /scratch/gpfs/ARORA/xd7812/optimizers/modded-nanogpt
sbatch run_gsm8k_comprehensive_sweep.sbatch
```

This will launch all 96 jobs in a SLURM array (array indices 0-95).

### Check Status

```bash
# Check running jobs
squeue -u $USER

# Check specific job array
squeue -j <JOB_ID>

# Count completed jobs
ls logs/gsm8k_sweep_*_*.out | wc -l
```

### Monitor Progress

```bash
# View logs for a specific task
tail -f logs/gsm8k_sweep_<JOB_ID>_<TASK_ID>.out

# Check for errors
grep -r "Error" logs/gsm8k_sweep_*.err

# View latest results
tail logs/gsm8k_sweep_*_*.out | grep "GSM8K Accuracy"
```

### Cancel Jobs

```bash
# Cancel all sweep jobs
scancel -u $USER -n gsm8k_sweep

# Cancel specific job array
scancel <JOB_ID>

# Cancel specific task in array
scancel <JOB_ID>_<TASK_ID>
```

## Output Structure

Results are organized by base model, optimizer, and learning rate:

```
outputs/gsm8k_sweep/
├── adamw_130m_1/
│   ├── adamw/
│   │   ├── lr_1e-5/
│   │   ├── lr_3e-5/
│   │   ├── lr_1e-4/
│   │   ├── lr_3e-4/
│   │   ├── lr_1e-3/
│   │   └── lr_2e-3/
│   └── moonlight_muon/
│       ├── lr_1e-5/
│       ├── lr_3e-5/
│       ├── lr_1e-4/
│       ├── lr_3e-4/
│       ├── lr_1e-3/
│       └── lr_2e-3/
├── adamw_130m_2/
│   └── ...
└── ...
```

Each experiment directory contains:
- `training_log.csv` - Step-by-step training metrics
- `config.json` - Training configuration
- `final_checkpoint.pt` - Final model weights

## Expected Results

Based on previous 300m_8 experiments:

### AdamW Baseline
- Best LR typically: 1e-4 to 3e-4
- Expected accuracy: 10-12% for 300M, 8-10% for 130M

### Moonlight Muon
- Best LR typically: 3e-4 to 1e-3
- Expected accuracy: 12-14% for 300M, 10-12% for 130M
- Should outperform AdamW by 1-2pp

### Key Questions to Answer

1. Does Moonlight Muon consistently outperform AdamW across all model sizes?
2. Is 2e-3 learning rate viable for Moonlight Muon?
3. How do results scale between 130M and 300M models?
4. Does base model (AdamW vs Muon pretrained) affect finetuning performance?

## Analysis Scripts

After jobs complete, use these scripts to analyze results:

```bash
# Extract all results into summary table
python3 analyze_comprehensive_sweep.py

# Plot training curves
python3 plot_comprehensive_sweep.py

# Compare optimizers
python3 compare_optimizers.py
```

## Previous Results (300m_8 models)

For reference, previous experiments on 300m_8 models achieved:

| Base Model | Optimizer | Best LR | Accuracy |
|:-----------|:----------|:--------|:---------|
| adamw_300m_8 | AdamW | 1e-4 | 5.08% |
| adamw_300m_8 | Moonlight | 1e-3 | 13.67% |
| muon_300m_8 | AdamW | 1e-4 | 12.11% |
| muon_300m_8 | Moonlight | 3e-4 | 13.28% |

**Key Finding:** Moonlight Muon achieved best results, with 8.59pp improvement on adamw_300m_8 and 1.17pp on muon_300m_8.

## Notes

- All checkpoints verified to exist before job submission
- Jobs are independent and can be run in any order
- Failed jobs can be resubmitted individually
- Results are automatically saved to persistent storage
