#!/usr/bin/env python3
"""
Show the complete experiment plan for the comprehensive sweep
"""

BASE_MODELS = [
    "adamw_130m_1", "adamw_130m_2", "adamw_130m_8",
    "muon_130m_1", "muon_130m_2", "muon_130m_8",
    "adamw_300m_1", "muon_300m_1"
]

OPTIMIZERS = ["adamw", "moonlight_muon"]
LRS = ["1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "2e-3"]

NUM_LRS = len(LRS)
NUM_OPTS = len(OPTIMIZERS)

print("\n" + "="*80)
print("Comprehensive GSM8K Finetuning Sweep Plan")
print("="*80)
print()
print(f"Base Models: {len(BASE_MODELS)}")
print(f"Optimizers: {len(OPTIMIZERS)}")
print(f"Learning Rates: {len(LRS)}")
print(f"Total Jobs: {len(BASE_MODELS) * len(OPTIMIZERS) * len(LRS)}")
print()

# Show task mapping
print("Task Mapping (first 20 tasks):")
print("-" * 80)
print(f"{'Task ID':<10} {'Base Model':<20} {'Optimizer':<18} {'LR':<10}")
print("-" * 80)

for task_id in range(min(20, len(BASE_MODELS) * len(OPTIMIZERS) * len(LRS))):
    lr_idx = task_id % NUM_LRS
    opt_idx = (task_id // NUM_LRS) % NUM_OPTS
    base_idx = task_id // (NUM_LRS * NUM_OPTS)

    base_model = BASE_MODELS[base_idx]
    optimizer = OPTIMIZERS[opt_idx]
    lr = LRS[lr_idx]

    print(f"{task_id:<10} {base_model:<20} {optimizer:<18} {lr:<10}")

print("...")
print()

# Show summary by base model
print("Summary by Base Model:")
print("-" * 80)
for base_model in BASE_MODELS:
    print(f"\n{base_model}:")
    for optimizer in OPTIMIZERS:
        lrs_str = ", ".join(LRS)
        print(f"  {optimizer}: {lrs_str}")

print()
print("="*80)
print()
print("To submit all jobs:")
print("  sbatch run_gsm8k_comprehensive_sweep.sbatch")
print()
print("To check job status:")
print("  squeue -u $USER")
print()
print("To cancel all jobs:")
print("  scancel -u $USER -n gsm8k_sweep")
print()
print("="*80)
