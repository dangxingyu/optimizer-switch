#!/usr/bin/env python3
"""
Plot GSM8K finetuning results for 300M models.
4 subplots (one per model combination), each with 4 learning rate curves.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("outputs/gsm8k")
LRS = ["5e-6", "1e-5", "3e-5", "1e-4"]

# Model combinations (base_model, finetune_optimizer)
MODELS = [
    ("adamw_300m_8", "adamw"),
    ("adamw_300m_8", "muon"),
    ("muon_300m_8", "adamw"),
    ("muon_300m_8", "muon"),
]

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Color palette for learning rates
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, (base_model, finetune_opt) in enumerate(MODELS):
    ax = axes[idx]

    # Title for this subplot
    title = f"{base_model.replace('_', ' ').title()} → Finetune with {finetune_opt.upper()}"
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Plot each learning rate
    for lr_idx, lr in enumerate(LRS):
        log_path = OUTPUT_DIR / f"{base_model}_{finetune_opt}" / f"lr_{lr}" / "training_log.csv"

        if not log_path.exists():
            print(f"⚠ Missing: {log_path}")
            continue

        # Read training log
        df = pd.read_csv(log_path)

        # Filter out incomplete rows (where val_loss might be in progress)
        # Keep only rows where step % 50 == 0 or step < max_step - 10
        max_step = df['step'].max()
        df = df[df['step'] <= max_step - 2].copy()

        # Plot training loss
        steps = df['step'].values
        train_loss = df['train_loss'].values

        # Get final loss for annotation
        final_loss = train_loss[-1]

        # Plot line
        label = f"LR={lr} (final={final_loss:.3f})"
        ax.plot(steps, train_loss,
                color=colors[lr_idx],
                linewidth=2,
                label=label,
                alpha=0.8)

        # Mark final point
        ax.scatter([steps[-1]], [final_loss],
                  color=colors[lr_idx],
                  s=100,
                  zorder=5,
                  edgecolors='white',
                  linewidths=1.5)

    # Styling
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

    # Set background
    ax.set_facecolor('#f8f9fa')

# Overall title
fig.suptitle('GSM8K Finetuning: 300M Models\nTraining Loss Curves',
             fontsize=16, fontweight='bold', y=0.995)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.985])

# Save figure
output_path = "gsm8k_300m_training_curves.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ Saved plot to: {output_path}")

# Also save as PDF for papers
pdf_path = "gsm8k_300m_training_curves.pdf"
plt.savefig(pdf_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ Saved plot to: {pdf_path}")

plt.close()

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for base_model, finetune_opt in MODELS:
    print(f"\n{base_model} → {finetune_opt.upper()} finetune:")
    print("-" * 60)

    for lr in LRS:
        log_path = OUTPUT_DIR / f"{base_model}_{finetune_opt}" / f"lr_{lr}" / "training_log.csv"

        if not log_path.exists():
            print(f"  LR={lr:>6}: MISSING")
            continue

        df = pd.read_csv(log_path)

        # Filter out last incomplete rows
        max_step = df['step'].max()
        df = df[df['step'] <= max_step - 2].copy()

        initial_loss = df['train_loss'].iloc[0]
        final_loss = df['train_loss'].iloc[-1]
        min_loss = df['train_loss'].min()
        improvement = initial_loss - final_loss

        print(f"  LR={lr:>6}: initial={initial_loss:.4f}, final={final_loss:.4f}, "
              f"min={min_loss:.4f}, improvement={improvement:+.4f}")

print("\n" + "="*80)
