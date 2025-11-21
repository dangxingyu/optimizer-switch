#!/usr/bin/env python3
"""
Plot training loss curves for all Moonlight Muon experiments
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration mapping
configs = [
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

# Colors for different LRs
lr_colors = {
    "1e-5": "#1f77b4",  # blue
    "3e-5": "#ff7f0e",  # orange
    "1e-4": "#2ca02c",  # green
    "3e-4": "#d62728",  # red
    "1e-3": "#9467bd",  # purple
}

# Line styles for different base models
model_styles = {
    "adamw_300m_8": "-",   # solid
    "muon_300m_8": "--",   # dashed
}

# Create figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot all curves
for base_model, lr in configs:
    log_path = f"outputs/gsm8k_moonlight/{base_model}/lr_{lr}/training_log.csv"

    if not os.path.exists(log_path):
        print(f"Warning: {log_path} not found")
        continue

    # Read training log
    df = pd.read_csv(log_path)

    # Extract step and loss
    steps = df['step'].values
    train_loss = df['train_loss'].values

    # Plot train loss
    color = lr_colors[lr]
    linestyle = model_styles[base_model]
    label = f"{base_model} LR={lr}"

    axes[0].plot(steps, train_loss,
                 color=color,
                 linestyle=linestyle,
                 linewidth=1.5,
                 label=label,
                 alpha=0.8)

# Customize top plot (all curves)
axes[0].set_xlabel('Training Step', fontsize=12)
axes[0].set_ylabel('Training Loss', fontsize=12)
axes[0].set_title('Moonlight Muon Training Loss - All Configurations', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
axes[0].set_ylim([0, max(train_loss) * 1.1])  # Set y-axis limit

# Plot by base model (separate subplots)
for idx, base_model in enumerate(["adamw_300m_8", "muon_300m_8"]):
    ax = axes[1] if idx == 0 else axes[1]

    for lr in ["1e-5", "3e-5", "1e-4", "3e-4", "1e-3"]:
        log_path = f"outputs/gsm8k_moonlight/{base_model}/lr_{lr}/training_log.csv"

        if not os.path.exists(log_path):
            continue

        df = pd.read_csv(log_path)
        steps = df['step'].values
        train_loss = df['train_loss'].values

        color = lr_colors[lr]
        linestyle = model_styles[base_model]
        label = f"{base_model} LR={lr}"

        axes[1].plot(steps, train_loss,
                     color=color,
                     linestyle=linestyle,
                     linewidth=2,
                     label=label,
                     alpha=0.8)

# Customize bottom plot (by model)
axes[1].set_xlabel('Training Step', fontsize=12)
axes[1].set_ylabel('Training Loss', fontsize=12)
axes[1].set_title('Moonlight Muon Training Loss - All LRs', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('moonlight_loss_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: moonlight_loss_curves.png")

# Create separate plots for each base model
for base_model in ["adamw_300m_8", "muon_300m_8"]:
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for lr in ["1e-5", "3e-5", "1e-4", "3e-4", "1e-3"]:
        log_path = f"outputs/gsm8k_moonlight/{base_model}/lr_{lr}/training_log.csv"

        if not os.path.exists(log_path):
            continue

        df = pd.read_csv(log_path)
        steps = df['step'].values
        train_loss = df['train_loss'].values

        color = lr_colors[lr]
        label = f"LR={lr}"

        ax.plot(steps, train_loss,
                color=color,
                linewidth=2.5,
                label=label,
                alpha=0.9,
                marker='o',
                markersize=3,
                markevery=50)

    ax.set_xlabel('Training Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=13, fontweight='bold')
    ax.set_title(f'Moonlight Muon Training Loss - {base_model}',
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')
    ax.set_ylim([0, 3.5])

    plt.tight_layout()
    filename = f'moonlight_loss_{base_model}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")

# Create comparison plot: best LRs only
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

best_configs = [
    ("adamw_300m_8", "1e-3", "AdamW Base + LR 1e-3 (13.67%)"),
    ("muon_300m_8", "3e-4", "Muon Base + LR 3e-4 (13.28%)"),
]

for base_model, lr, label in best_configs:
    log_path = f"outputs/gsm8k_moonlight/{base_model}/lr_{lr}/training_log.csv"

    if not os.path.exists(log_path):
        continue

    df = pd.read_csv(log_path)
    steps = df['step'].values
    train_loss = df['train_loss'].values

    ax.plot(steps, train_loss,
            linewidth=3,
            label=label,
            alpha=0.9,
            marker='o',
            markersize=4,
            markevery=50)

ax.set_xlabel('Training Step', fontsize=13, fontweight='bold')
ax.set_ylabel('Training Loss', fontsize=13, fontweight='bold')
ax.set_title('Moonlight Muon - Best Configurations Comparison',
             fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='best')
ax.set_ylim([0, 3.5])

plt.tight_layout()
plt.savefig('moonlight_loss_best.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: moonlight_loss_best.png")

print("\n" + "="*60)
print("All plots generated successfully!")
print("="*60)
print("\nGenerated files:")
print("  1. moonlight_loss_curves.png - All 10 configurations")
print("  2. moonlight_loss_adamw_300m_8.png - AdamW base model only")
print("  3. moonlight_loss_muon_300m_8.png - Muon base model only")
print("  4. moonlight_loss_best.png - Best 2 configurations")
