# Testing Scripts for Moonlight Muon

## Available Scripts

### 1. Quick Smoke Test (Login Node)

**Script**: `test_muon_single_gpu.sh`

```bash
./test_muon_single_gpu.sh
```

**Purpose**: Fast sanity check on login node
**Dataset**: 64 samples, 50 steps
**Time**: ~2-3 minutes
**Use case**: Verify basic functionality before submitting jobs

**What it tests:**
- Script runs without errors
- Loss decreases over steps
- No NaN/Inf values

---

### 2. Single Job Test

**Script**: `test_single_job.sh`

```bash
./test_single_job.sh
```

**Purpose**: Test one specific configuration more thoroughly
**Dataset**: 128 samples, 100 steps
**Current config**:
- Base model: `muon_300m_8`
- Optimizer: Moonlight Muon
- Learning rate: `1e-4` (updated for Moonlight)

**Use case**: Validate a specific setup before full training

**To customize:**
Edit the script and change:
```bash
BASE_MODEL="muon_300m_8"      # or adamw_300m_8, adamw_130m_1, etc.
LR="1e-4"                      # try 1e-5, 3e-5, 1e-4, 3e-4, 1e-3
```

---

### 3. Learning Rate Sweep (NEW - RECOMMENDED)

**Script**: `test_moonlight_lr_sweep.sh`

```bash
./test_moonlight_lr_sweep.sh
```

**Purpose**: Find optimal learning rate for Moonlight Muon
**Dataset**: 128 samples, 100 steps per LR
**Learning rates tested**: `1e-5, 3e-5, 1e-4, 3e-4, 1e-3`
**Time**: ~15-20 minutes total

**Why this is important:**
- Moonlight is ~180x more aggressive than NorMuon
- Previous LRs (like 0.002) will likely be too high
- Need to find the sweet spot

**Output:**
```
outputs/moonlight_lr_sweep/muon_300m_8/
├── lr_1e-5/
│   └── training_log.csv
├── lr_3e-5/
│   └── training_log.csv
├── lr_1e-4/
│   └── training_log.csv
├── lr_3e-4/
│   └── training_log.csv
└── lr_1e-3/
    └── training_log.csv
```

**After running, compare results:**
```bash
# Quick comparison
for lr in 1e-5 3e-5 1e-4 3e-4 1e-3; do
    echo -n "LR $lr: "
    tail -1 outputs/moonlight_lr_sweep/muon_300m_8/lr_$lr/training_log.csv | cut -d',' -f2
done
```

**Choose the LR with:**
- ✅ Lowest final training loss
- ✅ No NaN/Inf values
- ✅ Smooth loss curve (check training_log.csv)

---

## Direct Python Usage

If you want more control, call the Python script directly:

```bash
# Minimal example
python3 train_llama_muon_single_gpu.py \
  --checkpoint_path ../checkpoints/muon_300m_8 \
  --lr 1e-4 \
  --max_steps 100

# Full example
python3 train_llama_muon_single_gpu.py \
  --checkpoint_path ../checkpoints/muon_300m_8 \
  --lr 1e-4 \
  --max_train_samples 500 \
  --max_steps 500 \
  --output_dir outputs/my_test
```

**Available arguments:**
- `--checkpoint_path`: Path to model checkpoint (required)
- `--lr`: Learning rate (default: 0.002, but try smaller like 1e-4)
- `--max_train_samples`: Limit training samples (-1 for all)
- `--max_steps`: Max training steps (-1 for full epochs)
- `--output_dir`: Output directory (default: "output")

---

## Understanding Moonlight vs NorMuon

### Key Difference: Aggressiveness

**NorMuon (previous):**
- Lerp momentum: `grad_contribution = 0.05 * grad`
- LR scaling: `max(1, d_out/d_in)**0.5 ≈ 1.0-2.0`
- Combined: Conservative

**Moonlight (current):**
- Standard momentum: `grad_contribution = 1.0 * grad` (20x larger!)
- LR scaling: `0.2 * sqrt(max(d_out, d_in)) ≈ 9.0` (9x larger!)
- Combined: **~180x more aggressive**

### LR Recommendations

| Previous Setup | Moonlight Equivalent | Recommendation |
|----------------|---------------------|----------------|
| NorMuon @ 0.002 | 0.002 * 180 = 0.36 | Way too high! |
| NorMuon @ 0.002 | Try 1e-4 to 1e-3 | Good starting range |

**Safe starting points:**
- `1e-4`: Conservative (recommended first try)
- `3e-4`: Moderate
- `1e-3`: Aggressive (may be too high)

---

## Workflow Recommendation

### Step 1: Quick Smoke Test
```bash
./test_muon_single_gpu.sh
```
✅ Verify: Script runs, loss decreases

### Step 2: LR Sweep
```bash
./test_moonlight_lr_sweep.sh
```
✅ Find optimal LR from {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}

### Step 3: Full Training
Using the best LR from step 2:
```bash
python3 train_llama_muon_single_gpu.py \
  --checkpoint_path ../checkpoints/muon_300m_8 \
  --lr <BEST_LR> \
  --output_dir outputs/final_training
```

### Step 4: Evaluation
Compare results to baseline:
- Previous Muon: ~2.3% accuracy (failed)
- AdamW: 13.5% accuracy (worked)
- **Target**: >10% to demonstrate improvement

---

## Troubleshooting

### Loss is NaN/Inf
**Cause**: Learning rate too high
**Fix**: Use smaller LR (try 1e-5 or 3e-5)

### Loss not decreasing
**Cause**: Learning rate too low OR too high
**Fix**: Run LR sweep to find sweet spot

### Script fails to import modules
**Cause**: Missing dependencies
**Fix**:
```bash
pip install torch transformers datasets
```

### Checkpoint not found
**Cause**: Wrong path
**Fix**: Verify checkpoint exists:
```bash
ls -la ../checkpoints/
```

---

## Expected Performance

### Previous Results (for reference)

| Base Model | Optimizer | LR | Accuracy | Status |
|------------|-----------|----|----|--------|
| muon_300m_8 | Basic Muon | 1e-5, 3e-5, 1e-4 | ~2.3% | ❌ Failed |
| adamw_300m_8 | AdamW | 1e-5 | 13.5% | ✅ Works |

### Moonlight Target

**Goal**: Accuracy > 10% (better than basic Muon, approaching AdamW)

**Hypothesis**:
- Moonlight's standard momentum + Nesterov should be more effective than lerp
- Proper LR tuning is critical due to increased aggressiveness
- Should perform between basic Muon (~2%) and AdamW (~13%)

---

## Files Overview

```
modded-nanogpt/
├── train_llama_muon_single_gpu.py      # Main training script (Moonlight)
├── test_muon_single_gpu.sh             # Quick smoke test (updated)
├── test_single_job.sh                  # Single config test (updated)
├── test_moonlight_lr_sweep.sh          # LR sweep (NEW)
├── MOONLIGHT_MIGRATION_COMPLETE.md     # Migration details
├── MOONLIGHT_VS_NORMUON.md             # Comparison
└── TESTING_SCRIPTS.md                  # This file
```

---

**Last Updated**: 2025-11-19
**Implementation**: Moonlight Muon (Newton-Schulz 5 + standard momentum)
