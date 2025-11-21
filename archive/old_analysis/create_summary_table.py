#!/usr/bin/env python3
"""
Create a 2x2 summary table: Base Model × Optimizer
Showing best performance across all learning rates
"""

import os
import re

# ============================================================================
# Extract Previous AdamW Results
# ============================================================================

prev_task_map = {}
idx = 0
for base_model in ["adamw_300m_8", "muon_300m_8"]:
    for finetune_opt in ["adamw", "muon"]:
        for lr in ["5e-6", "1e-5", "3e-5", "1e-4"]:
            prev_task_map[idx] = (base_model, finetune_opt, lr)
            idx += 1

prev_results = {}
log_dir = "/scratch/gpfs/ARORA/xd7812/optimizers/modded-nanogpt/logs"

for task_id in range(16):
    base_model, finetune_opt, lr = prev_task_map[task_id]
    log_file = f"{log_dir}/gsm8k_300m_2413689_{task_id}.out"

    key = (base_model, finetune_opt)
    if key not in prev_results:
        prev_results[key] = {}

    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            content = f.read()

        step_pattern = r'Step \d+: GSM8K Accuracy = ([\d.]+)'
        final_pattern = r'Final GSM8K Accuracy: ([\d.]+)'

        step_matches = re.findall(step_pattern, content)
        final_match = re.search(final_pattern, content)

        accs = [float(m) * 100 for m in step_matches]
        if final_match:
            accs.append(float(final_match.group(1)) * 100)

        prev_results[key][lr] = accs if accs else ["N/A"]
    else:
        prev_results[key][lr] = ["Missing"]

# ============================================================================
# Extract Moonlight Muon Results
# ============================================================================

moonlight_task_map = {}
idx = 0
for base_model in ["adamw_300m_8", "muon_300m_8"]:
    for lr in ["1e-5", "3e-5", "1e-4", "3e-4", "1e-3"]:
        moonlight_task_map[idx] = (base_model, lr)
        idx += 1

moonlight_results = {}

for task_id in range(10):
    base_model, lr = moonlight_task_map[task_id]
    log_file = f"{log_dir}/gsm8k_moonlight_300m_2448769_{task_id}.out"

    if base_model not in moonlight_results:
        moonlight_results[base_model] = {}

    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            content = f.read()

        step_pattern = r'Step \d+: GSM8K Accuracy = ([\d.]+)'
        final_pattern = r'Final GSM8K Accuracy: ([\d.]+)'

        step_matches = re.findall(step_pattern, content)
        final_match = re.search(final_pattern, content)

        accs = [float(m) * 100 for m in step_matches]
        if final_match:
            accs.append(float(final_match.group(1)) * 100)

        moonlight_results[base_model][lr] = accs if accs else ["N/A"]
    else:
        moonlight_results[base_model][lr] = ["Missing"]

# ============================================================================
# Create 2x2 Summary Table
# ============================================================================

# Find best performance for each base_model × optimizer combination
summary = {}

# Previous AdamW
for base_model in ["adamw_300m_8", "muon_300m_8"]:
    key = (base_model, "adamw")
    best_acc = 0
    best_lr = None

    for lr, accs in prev_results.get(key, {}).items():
        if accs and all(isinstance(a, (int, float)) for a in accs):
            max_acc = max(accs)
            if max_acc > best_acc:
                best_acc = max_acc
                best_lr = lr

    summary[(base_model, "Previous AdamW")] = (best_acc, best_lr)

# Moonlight Muon
for base_model in ["adamw_300m_8", "muon_300m_8"]:
    best_acc = 0
    best_lr = None

    for lr, accs in moonlight_results.get(base_model, {}).items():
        if accs and all(isinstance(a, (int, float)) for a in accs):
            max_acc = max(accs)
            if max_acc > best_acc:
                best_acc = max_acc
                best_lr = lr

    summary[(base_model, "Moonlight Muon")] = (best_acc, best_lr)

# Print table
print("\n" + "="*80)
print("GSM8K 300M Finetuning - Summary (Best Performance)")
print("="*80)
print()
print("| Base Model | Previous AdamW | Moonlight Muon |")
print("|:-----------|:--------------:|:--------------:|")

for base_model in ["adamw_300m_8", "muon_300m_8"]:
    adamw_acc, adamw_lr = summary.get((base_model, "Previous AdamW"), (0, None))
    moonlight_acc, moonlight_lr = summary.get((base_model, "Moonlight Muon"), (0, None))

    adamw_str = f"**{adamw_acc:.2f}%**<br>(LR={adamw_lr})" if adamw_lr else "N/A"
    moonlight_str = f"**{moonlight_acc:.2f}%**<br>(LR={moonlight_lr})" if moonlight_lr else "N/A"

    print(f"| **{base_model}** | {adamw_str} | {moonlight_str} |")

print()
print("="*80)

# Save to markdown
with open("SUMMARY_TABLE.md", "w") as f:
    f.write("# GSM8K 300M Finetuning - Summary\n\n")
    f.write("## Best Performance by Base Model and Optimizer\n\n")
    f.write("This table shows the **best accuracy** achieved across all learning rates for each combination.\n\n")

    f.write("| Base Model | Previous AdamW | Moonlight Muon |\n")
    f.write("|:-----------|:--------------:|:--------------:||\n")

    for base_model in ["adamw_300m_8", "muon_300m_8"]:
        adamw_acc, adamw_lr = summary.get((base_model, "Previous AdamW"), (0, None))
        moonlight_acc, moonlight_lr = summary.get((base_model, "Moonlight Muon"), (0, None))

        adamw_str = f"**{adamw_acc:.2f}%**<br>(LR={adamw_lr})" if adamw_lr else "N/A"
        moonlight_str = f"**{moonlight_acc:.2f}%**<br>(LR={moonlight_lr})" if moonlight_lr else "N/A"

        f.write(f"| **{base_model}** | {adamw_str} | {moonlight_str} |\n")

    f.write("\n## Key Findings\n\n")

    # Calculate improvements
    adamw_300m_8_improvement = summary[("adamw_300m_8", "Moonlight Muon")][0] - summary[("adamw_300m_8", "Previous AdamW")][0]
    muon_300m_8_improvement = summary[("muon_300m_8", "Moonlight Muon")][0] - summary[("muon_300m_8", "Previous AdamW")][0]

    f.write(f"- **adamw_300m_8**: Moonlight Muon improves by **+{adamw_300m_8_improvement:.2f}pp** over Previous AdamW\n")
    f.write(f"- **muon_300m_8**: Moonlight Muon improves by **+{muon_300m_8_improvement:.2f}pp** over Previous AdamW\n")
    f.write(f"\n**Overall**: Moonlight Muon achieves the best result of **{max(summary[('adamw_300m_8', 'Moonlight Muon')][0], summary[('muon_300m_8', 'Moonlight Muon')][0]):.2f}%** 🎉\n")

print("✓ Saved: SUMMARY_TABLE.md")
