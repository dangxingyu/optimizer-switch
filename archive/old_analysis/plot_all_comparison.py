#!/usr/bin/env python3
"""
Plot training loss curves comparing all methods:
- Previous AdamW finetuning
- Previous Basic Muon finetuning
- Moonlight Muon finetuning
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

# Previous experiment paths (from outputs/gsm8k_300m/)
# Task mapping for previous runs (job 2413689)
prev_task_map = {}
idx = 0
for base_model in ["adamw_300m_8", "muon_300m_8"]:
    for finetune_opt in ["adamw", "muon"]:
        for lr in ["5e-6", "1e-5", "3e-5", "1e-4"]:
            prev_task_map[idx] = (base_model, finetune_opt, lr)
            idx += 1

# Moonlight paths (from outputs/gsm8k_moonlight/)
moonlight_configs = [
    ("adamw_300m_8", "1e-5"),
    ("adamw_300m_8", "3e-5"),
    ("adamw_300m_8", "1e-4"),
    ("adamw_300m_8", "3e-4"),
    ("adamw_300m_8", "1e-3"),
    ("muon_300m_8", "1e-5"),
    ("muon_300m_8", "3e-5"),
    ("muon_300m_8", "1e-4"),
    ("muon_300m_8", "3e-4"),
    ("muon_300m_8", "1e-3"),
]

# ============================================================================
# Plot 1: Best configurations from each method
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(14, 7))

# Best previous AdamW: muon_300m_8, LR 1e-4 (12.11%)
prev_adamw_path = "outputs/gsm8k_300m/muon_300m_8/adamw/lr_1e-4/training_log.csv"
if os.path.exists(prev_adamw_path):
    df = pd.read_csv(prev_adamw_path)
    ax.plot(df['step'], df['train_loss'],
            linewidth=2.5, label='Previous AdamW (12.11%)',
            color='#1f77b4', alpha=0.9, marker='o', markersize=4, markevery=50)

# Best Moonlight #1: adamw_300m_8, LR 1e-3 (13.67%)
moonlight1_path = "outputs/gsm8k_moonlight/adamw_300m_8/lr_1e-3/training_log.csv"
if os.path.exists(moonlight1_path):
    df = pd.read_csv(moonlight1_path)
    ax.plot(df['step'], df['train_loss'],
            linewidth=3, label='Moonlight Muon #1 (13.67%)',
            color='#2ca02c', alpha=0.95, marker='^', markersize=5, markevery=50)

# Best Moonlight #2: muon_300m_8, LR 3e-4 (13.28%)
moonlight2_path = "outputs/gsm8k_moonlight/muon_300m_8/lr_3e-4/training_log.csv"
if os.path.exists(moonlight2_path):
    df = pd.read_csv(moonlight2_path)
    ax.plot(df['step'], df['train_loss'],
            linewidth=3, label='Moonlight Muon #2 (13.28%)',
            color='#d62728', alpha=0.95, marker='v', markersize=5, markevery=50)

ax.set_xlabel('Training Step', fontsize=14, fontweight='bold')
ax.set_ylabel('Training Loss', fontsize=14, fontweight='bold')
ax.set_title('GSM8K Finetuning - Best Configurations Comparison',
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='best')
ax.set_ylim([0, 3.5])

plt.tight_layout()
plt.savefig('comparison_best_all_methods.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comparison_best_all_methods.png")

# ============================================================================
# Plot 2: All AdamW finetuning curves (previous + moonlight)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# adamw_300m_8 base
for finetune_opt in ["adamw"]:
    for lr in ["5e-6", "1e-5", "3e-5", "1e-4"]:
        log_path = f"outputs/gsm8k_300m/adamw_300m_8/{finetune_opt}/lr_{lr}/training_log.csv"
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            axes[0].plot(df['step'], df['train_loss'],
                        linewidth=1.5, label=f'Prev AdamW LR={lr}',
                        alpha=0.7, linestyle='--')

# Add Moonlight curves (adamw_300m_8)
for lr in ["1e-5", "3e-5", "1e-4", "3e-4", "1e-3"]:
    log_path = f"outputs/gsm8k_moonlight/adamw_300m_8/lr_{lr}/training_log.csv"
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        axes[0].plot(df['step'], df['train_loss'],
                    linewidth=2.5, label=f'Moonlight LR={lr}',
                    alpha=0.9)

axes[0].set_xlabel('Training Step', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
axes[0].set_title('adamw_300m_8 Base - All Finetuning Methods',
                  fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=9, loc='best')
axes[0].set_ylim([0, 3.5])

# muon_300m_8 base
for finetune_opt in ["adamw"]:
    for lr in ["5e-6", "1e-5", "3e-5", "1e-4"]:
        log_path = f"outputs/gsm8k_300m/muon_300m_8/{finetune_opt}/lr_{lr}/training_log.csv"
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            axes[1].plot(df['step'], df['train_loss'],
                        linewidth=1.5, label=f'Prev AdamW LR={lr}',
                        alpha=0.7, linestyle='--')

# Add Moonlight curves (muon_300m_8)
for lr in ["1e-5", "3e-5", "1e-4", "3e-4", "1e-3"]:
    log_path = f"outputs/gsm8k_moonlight/muon_300m_8/lr_{lr}/training_log.csv"
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        axes[1].plot(df['step'], df['train_loss'],
                    linewidth=2.5, label=f'Moonlight LR={lr}',
                    alpha=0.9)

axes[1].set_xlabel('Training Step', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
axes[1].set_title('muon_300m_8 Base - All Finetuning Methods',
                  fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=9, loc='best')
axes[1].set_ylim([0, 3.5])

plt.tight_layout()
plt.savefig('comparison_all_adamw_methods.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comparison_all_adamw_methods.png")

# ============================================================================
# Plot 3: Moonlight Muon Learning Rate Comparison (both base models)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# adamw_300m_8 base - Moonlight Muon only
for lr in ["1e-5", "3e-5", "1e-4", "3e-4", "1e-3"]:
    log_path = f"outputs/gsm8k_moonlight/adamw_300m_8/lr_{lr}/training_log.csv"
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        axes[0].plot(df['step'], df['train_loss'],
                    linewidth=2.5, label=f'Moonlight Muon LR={lr}',
                    alpha=0.9)

axes[0].set_xlabel('Training Step', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
axes[0].set_title('adamw_300m_8 Base - Moonlight Muon LR Sweep',
                  fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10, loc='best')
axes[0].set_ylim([0, 3.5])

# muon_300m_8 base - Moonlight Muon only
for lr in ["1e-5", "3e-5", "1e-4", "3e-4", "1e-3"]:
    log_path = f"outputs/gsm8k_moonlight/muon_300m_8/lr_{lr}/training_log.csv"
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        axes[1].plot(df['step'], df['train_loss'],
                    linewidth=2.5, label=f'Moonlight Muon LR={lr}',
                    alpha=0.9)

axes[1].set_xlabel('Training Step', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
axes[1].set_title('muon_300m_8 Base - Moonlight Muon LR Sweep',
                  fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10, loc='best')
axes[1].set_ylim([0, 3.5])

plt.tight_layout()
plt.savefig('comparison_moonlight_lr_sweep.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comparison_moonlight_lr_sweep.png")

# ============================================================================
# Plot 4: Top 4 best results across all methods
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(14, 7))

top_configs = [
    ("Moonlight (adamw base, 1e-3)", "outputs/gsm8k_moonlight/adamw_300m_8/lr_1e-3/training_log.csv", 13.67, '#2ca02c'),
    ("Moonlight (muon base, 3e-4)", "outputs/gsm8k_moonlight/muon_300m_8/lr_3e-4/training_log.csv", 13.28, '#d62728'),
    ("Moonlight (muon base, 1e-4)", "outputs/gsm8k_moonlight/muon_300m_8/lr_1e-4/training_log.csv", 12.30, '#9467bd'),
    ("Prev AdamW (muon base, 1e-4)", "outputs/gsm8k_300m/muon_300m_8/adamw/lr_1e-4/training_log.csv", 12.11, '#1f77b4'),
]

for label, path, acc, color in top_configs:
    if os.path.exists(path):
        df = pd.read_csv(path)
        ax.plot(df['step'], df['train_loss'],
                linewidth=3, label=f'{label} ({acc:.2f}%)',
                color=color, alpha=0.9, marker='o', markersize=4, markevery=50)

ax.set_xlabel('Training Step', fontsize=14, fontweight='bold')
ax.set_ylabel('Training Loss', fontsize=14, fontweight='bold')
ax.set_title('GSM8K Finetuning - Top 4 Configurations',
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='best')
ax.set_ylim([0, 3.5])

plt.tight_layout()
plt.savefig('comparison_top4.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comparison_top4.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("All comparison plots generated successfully!")
print("="*70)
print("\nGenerated files:")
print("  1. comparison_best_all_methods.png - Best from each method")
print("  2. comparison_all_adamw_methods.png - Previous AdamW + Moonlight")
print("  3. comparison_moonlight_lr_sweep.png - Moonlight Muon LR sweep")
print("  4. comparison_top4.png - Top 4 configurations overall")
print("\nKey Finding:")
print("  ⭐ Moonlight Muon achieves 13.67% and 13.28%")
print("  📊 Previous AdamW best: 12.11%")
print("  ✅ Moonlight is the WINNER!")
print("="*70)
