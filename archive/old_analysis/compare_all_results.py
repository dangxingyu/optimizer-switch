#!/usr/bin/env python3
"""
Compare all GSM8K results: Previous AdamW vs Moonlight Muon
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Extract Previous Results (from gsm8k_300m logs)
# ============================================================================

# Task mapping for previous runs (2413689)
# Layout: base_model (2) × finetune_opt (2) × lr (4) = 16
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

        # Find accuracies
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
# Extract Moonlight Results
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
# Create Comparison Table
# ============================================================================

print("\n" + "="*100)
print("GSM8K 300M Finetuning Results - Complete Comparison")
print("="*100)
print()

# Table 1: AdamW base model
print("### Base Model: adamw_300m_8")
print()
print(f"| LR | Previous AdamW | **Moonlight Muon** |")
print(f"|:---|:---:|:---:|")

all_lrs = ["5e-6", "1e-5", "3e-5", "1e-4", "3e-4", "1e-3"]
for lr in all_lrs:
    prev_adamw = prev_results.get(("adamw_300m_8", "adamw"), {}).get(lr, [])
    moonlight = moonlight_results.get("adamw_300m_8", {}).get(lr, [])

    # Format
    prev_adamw_str = ", ".join([f"{a:.2f}%" if isinstance(a, (int, float)) else str(a) for a in prev_adamw]) if prev_adamw else "N/A"
    moonlight_str = ", ".join([f"{a:.2f}%" if isinstance(a, (int, float)) else str(a) for a in moonlight]) if moonlight else "N/A"

    # Add best
    if prev_adamw and all(isinstance(a, (int, float)) for a in prev_adamw):
        prev_adamw_str += f" **(best: {max(prev_adamw):.2f}%)**"
    if moonlight and all(isinstance(a, (int, float)) for a in moonlight):
        moonlight_str += f" **(best: {max(moonlight):.2f}%)**"

    print(f"| **{lr}** | {prev_adamw_str} | {moonlight_str} |")

print()
print("### Base Model: muon_300m_8")
print()
print(f"| LR | Previous AdamW | **Moonlight Muon** |")
print(f"|:---|:---:|:---:|")

for lr in all_lrs:
    prev_adamw = prev_results.get(("muon_300m_8", "adamw"), {}).get(lr, [])
    moonlight = moonlight_results.get("muon_300m_8", {}).get(lr, [])

    prev_adamw_str = ", ".join([f"{a:.2f}%" if isinstance(a, (int, float)) else str(a) for a in prev_adamw]) if prev_adamw else "N/A"
    moonlight_str = ", ".join([f"{a:.2f}%" if isinstance(a, (int, float)) else str(a) for a in moonlight]) if moonlight else "N/A"

    if prev_adamw and all(isinstance(a, (int, float)) for a in prev_adamw):
        prev_adamw_str += f" **(best: {max(prev_adamw):.2f}%)**"
    if moonlight and all(isinstance(a, (int, float)) for a in moonlight):
        moonlight_str += f" **(best: {max(moonlight):.2f}%)**"

    print(f"| **{lr}** | {prev_adamw_str} | {moonlight_str} |")

# ============================================================================
# Overall Best Results
# ============================================================================

print()
print("### Overall Best Results")
print()
print("| Method | Base Model | LR | Best Accuracy |")
print("|:---|:---:|:---:|:---:|")

# Collect all best results
all_best = []

# Previous AdamW
for (base_model, opt), lrs in prev_results.items():
    if opt == "adamw":
        for lr, accs in lrs.items():
            if accs and all(isinstance(a, (int, float)) for a in accs):
                all_best.append(("Previous AdamW", base_model, lr, max(accs)))

# Moonlight Muon
for base_model, lrs in moonlight_results.items():
    for lr, accs in lrs.items():
        if accs and all(isinstance(a, (int, float)) for a in accs):
            all_best.append(("**Moonlight Muon**", base_model, lr, max(accs)))

# Sort by accuracy
all_best.sort(key=lambda x: x[3], reverse=True)

# Print top 10
for method, base_model, lr, acc in all_best[:10]:
    print(f"| {method} | {base_model} | {lr} | **{acc:.2f}%** |")

print()
print("="*100)

# Save to markdown
with open("COMPLETE_COMPARISON.md", "w") as f:
    f.write("# GSM8K 300M Finetuning - Complete Comparison\n\n")
    f.write("## Previous AdamW vs Moonlight Muon\n\n")

    f.write("### Base Model: adamw_300m_8\n\n")
    f.write("| LR | Previous AdamW | **Moonlight Muon** |\n")
    f.write("|:---|:---:|:---:|\n")

    for lr in all_lrs:
        prev_adamw = prev_results.get(("adamw_300m_8", "adamw"), {}).get(lr, [])
        moonlight = moonlight_results.get("adamw_300m_8", {}).get(lr, [])

        prev_adamw_str = ", ".join([f"{a:.2f}%" if isinstance(a, (int, float)) else str(a) for a in prev_adamw]) if prev_adamw else "N/A"
        moonlight_str = ", ".join([f"{a:.2f}%" if isinstance(a, (int, float)) else str(a) for a in moonlight]) if moonlight else "N/A"

        f.write(f"| **{lr}** | {prev_adamw_str} | {moonlight_str} |\n")

    f.write("\n### Base Model: muon_300m_8\n\n")
    f.write("| LR | Previous AdamW | **Moonlight Muon** |\n")
    f.write("|:---|:---:|:---:|\n")

    for lr in all_lrs:
        prev_adamw = prev_results.get(("muon_300m_8", "adamw"), {}).get(lr, [])
        moonlight = moonlight_results.get("muon_300m_8", {}).get(lr, [])

        prev_adamw_str = ", ".join([f"{a:.2f}%" if isinstance(a, (int, float)) else str(a) for a in prev_adamw]) if prev_adamw else "N/A"
        moonlight_str = ", ".join([f"{a:.2f}%" if isinstance(a, (int, float)) else str(a) for a in moonlight]) if moonlight else "N/A"

        f.write(f"| **{lr}** | {prev_adamw_str} | {moonlight_str} |\n")

print("✓ Saved: COMPLETE_COMPARISON.md")
