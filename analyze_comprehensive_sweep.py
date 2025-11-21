#!/usr/bin/env python3
"""
Analyze results from comprehensive sweep
Extract all accuracies and create summary tables
"""

import os
import re
import pandas as pd

BASE_MODELS = [
    "adamw_130m_1", "adamw_130m_2", "adamw_130m_8",
    "muon_130m_1", "muon_130m_2", "muon_130m_8",
    "adamw_300m_1", "muon_300m_1"
]

OPTIMIZERS = ["adamw", "moonlight_muon"]
LRS = ["1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "2e-3"]

# Extract results from logs
log_dir = "logs"
results = {}

print("Extracting results from logs...")
print("="*80)

for base_model in BASE_MODELS:
    results[base_model] = {}
    for optimizer in OPTIMIZERS:
        results[base_model][optimizer] = {}
        for lr in LRS:
            # Find corresponding log file
            # Task mapping: array_id = base_idx * 12 + opt_idx * 6 + lr_idx
            base_idx = BASE_MODELS.index(base_model)
            opt_idx = OPTIMIZERS.index(optimizer)
            lr_idx = LRS.index(lr)
            task_id = base_idx * 12 + opt_idx * 6 + lr_idx

            # Find log file (job ID may vary)
            log_pattern = f"{log_dir}/gsm8k_sweep_*_{task_id}.out"
            import glob
            log_files = glob.glob(log_pattern)

            if not log_files:
                results[base_model][optimizer][lr] = None
                print(f"⚠ Missing: {base_model} / {optimizer} / {lr} (task {task_id})")
                continue

            log_file = log_files[0]  # Take first match

            # Extract accuracies
            with open(log_file, 'r') as f:
                content = f.read()

            step_pattern = r'Step \d+: GSM8K Accuracy = ([\d.]+)'
            final_pattern = r'Final GSM8K Accuracy: ([\d.]+)'

            step_matches = re.findall(step_pattern, content)
            final_match = re.search(final_pattern, content)

            accs = [float(m) * 100 for m in step_matches]
            if final_match:
                accs.append(float(final_match.group(1)) * 100)

            results[base_model][optimizer][lr] = accs if accs else None

            if accs:
                print(f"✓ {base_model} / {optimizer} / {lr}: {max(accs):.2f}%")
            else:
                print(f"✗ Failed: {base_model} / {optimizer} / {lr}")

print()
print("="*80)
print()

# Create summary table: best result for each base_model × optimizer
print("Summary Table: Best Results")
print("="*80)
print()
print(f"| {'Base Model':<20} | {'AdamW Best':<15} | {'Moonlight Best':<15} | {'Winner':<12} |")
print(f"|{'-'*20}|{'-'*15}|{'-'*15}|{'-'*12}|")

for base_model in BASE_MODELS:
    # Find best AdamW
    adamw_best = 0
    adamw_best_lr = None
    for lr in LRS:
        accs = results[base_model]["adamw"].get(lr)
        if accs:
            max_acc = max(accs)
            if max_acc > adamw_best:
                adamw_best = max_acc
                adamw_best_lr = lr

    # Find best Moonlight
    moonlight_best = 0
    moonlight_best_lr = None
    for lr in LRS:
        accs = results[base_model]["moonlight_muon"].get(lr)
        if accs:
            max_acc = max(accs)
            if max_acc > moonlight_best:
                moonlight_best = max_acc
                moonlight_best_lr = lr

    adamw_str = f"{adamw_best:.2f}% ({adamw_best_lr})" if adamw_best_lr else "N/A"
    moonlight_str = f"{moonlight_best:.2f}% ({moonlight_best_lr})" if moonlight_best_lr else "N/A"

    if moonlight_best > adamw_best:
        winner = "Moonlight"
    elif adamw_best > moonlight_best:
        winner = "AdamW"
    else:
        winner = "Tie"

    print(f"| {base_model:<20} | {adamw_str:<15} | {moonlight_str:<15} | {winner:<12} |")

print()
print("="*80)
print()

# Find overall best
print("Overall Best Results:")
print("-"*80)

all_results = []
for base_model in BASE_MODELS:
    for optimizer in OPTIMIZERS:
        for lr in LRS:
            accs = results[base_model][optimizer].get(lr)
            if accs:
                all_results.append((base_model, optimizer, lr, max(accs)))

all_results.sort(key=lambda x: x[3], reverse=True)

print(f"| {'Rank':<6} | {'Base Model':<20} | {'Optimizer':<16} | {'LR':<8} | {'Accuracy':<10} |")
print(f"|{'-'*6}|{'-'*20}|{'-'*16}|{'-'*8}|{'-'*10}|")

for i, (base_model, optimizer, lr, acc) in enumerate(all_results[:20], 1):
    print(f"| {i:<6} | {base_model:<20} | {optimizer:<16} | {lr:<8} | {acc:.2f}%{' '*5} |")

print()
print("="*80)

# Save detailed results to CSV
df_rows = []
for base_model in BASE_MODELS:
    for optimizer in OPTIMIZERS:
        for lr in LRS:
            accs = results[base_model][optimizer].get(lr)
            if accs:
                df_rows.append({
                    'base_model': base_model,
                    'optimizer': optimizer,
                    'lr': lr,
                    'step1_acc': accs[0] if len(accs) > 0 else None,
                    'step2_acc': accs[1] if len(accs) > 1 else None,
                    'final_acc': accs[2] if len(accs) > 2 else None,
                    'best_acc': max(accs)
                })

df = pd.DataFrame(df_rows)
df.to_csv('comprehensive_sweep_results.csv', index=False)
print(f"\n✓ Saved detailed results to: comprehensive_sweep_results.csv")
print()
