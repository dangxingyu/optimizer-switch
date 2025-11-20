# GSM8K Fine-tuning Quick Start

## Ready to Run!

All configurations are set and checkpoints verified.

### Submit All Jobs (32 total)

```bash
cd /scratch/gpfs/ARORA/xd7812/optimizers/modded-nanogpt
sbatch run_gsm8k_finetuning.sbatch
```

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View logs (replace JOB_ID with actual job ID)
tail -f logs/gsm8k_<JOB_ID>_*.out

# Check specific task
tail -f logs/gsm8k_<JOB_ID>_0.out  # Task 0: adamw_130m_1, lr=5e-6
```

### Quick Reference

**Configuration:**
- Batch size: 32
- Warmup: 50 steps
- Eval interval: 50 steps
- Time limit: ~2 hours per job
- Learning rates: 5e-6, 1e-5, 3e-5, 1e-4

**Models (8):**
- adamw_130m_1, adamw_130m_8
- adamw_300m_1, adamw_300m_8
- muon_130m_1, muon_130m_8
- muon_300m_1, muon_300m_8

**Total:** 8 models × 4 LRs = 32 jobs

### After Jobs Complete

```bash
# Analyze results
python analyze_results.py

# View best results
grep "Final GSM8K Accuracy" logs/gsm8k_*.out | sort -k4 -n | tail -10
```

### Outputs

Results saved to:
- `outputs/gsm8k/<model_name>/lr_<lr>/`
- `logs/gsm8k_<job_id>_<task_id>.out`

### Task ID Mapping

- Task 0-3: adamw_130m_1 (lr: 5e-6, 1e-5, 3e-5, 1e-4)
- Task 4-7: adamw_130m_8
- Task 8-11: adamw_300m_1
- Task 12-15: adamw_300m_8
- Task 16-19: muon_130m_1
- Task 20-23: muon_130m_8
- Task 24-27: muon_300m_1
- Task 28-31: muon_300m_8

See [GSM8K_TRAINING_README.md](GSM8K_TRAINING_README.md) for detailed documentation.
