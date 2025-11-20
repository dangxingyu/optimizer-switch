#!/usr/bin/env python3
"""
Plot 4x4 heatmap of GSM8K accuracies.
4 model combinations × 4 learning rates = 16 cells
"""

import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Configuration
LOG_DIR = Path("logs")
LOG_PATTERN = "gsm8k_300m_*.out"

# Model configurations
COMBOS = [
    ("adamw_300m_8", "adamw"),
    ("adamw_300m_8", "muon"),
    ("muon_300m_8", "adamw"),
    ("muon_300m_8", "muon"),
]
LRS = ["5e-6", "1e-5", "3e-5", "1e-4"]

def parse_log_file(log_path):
    """Parse a single log file to extract model config and accuracy."""
    with open(log_path, 'r') as f:
        content = f.read()

    # Extract configuration
    base_match = re.search(r'Base Model: (\w+)', content)
    finetune_match = re.search(r'Finetune Optimizer: (\w+)', content)
    lr_match = re.search(r'Learning Rate: ([\de-]+)', content)

    # Extract final accuracy
    acc_match = re.search(r'Final GSM8K Accuracy: ([\d.]+)', content)

    if not all([base_match, finetune_match, lr_match, acc_match]):
        return None

    return {
        'base_model': base_match.group(1),
        'finetune_opt': finetune_match.group(1),
        'lr': lr_match.group(1),
        'accuracy': float(acc_match.group(1)) * 100  # Convert to percentage
    }

def main():
    # Parse all log files
    results = []
    log_files = sorted(LOG_DIR.glob(LOG_PATTERN))

    for log_path in log_files:
        result = parse_log_file(log_path)
        if result:
            results.append(result)

    print(f"Parsed {len(results)} results\n")

    # Create 4x4 matrix
    data = np.zeros((len(COMBOS), len(LRS)))

    for i, (base_model, finetune_opt) in enumerate(COMBOS):
        for j, lr in enumerate(LRS):
            # Find matching result
            matching = [r for r in results
                       if r['base_model'] == base_model
                       and r['finetune_opt'] == finetune_opt
                       and r['lr'] == lr]

            if matching:
                data[i, j] = matching[0]['accuracy']

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create labels
    row_labels = [f"{base.replace('_', ' ').upper()}\n→ {ft.upper()}"
                  for base, ft in COMBOS]
    col_labels = [f"LR={lr}" for lr in LRS]

    # Create heatmap
    sns.heatmap(data,
                annot=True,
                fmt='.2f',
                cmap='YlOrRd',
                xticklabels=col_labels,
                yticklabels=row_labels,
                cbar_kws={'label': 'Accuracy (%)'},
                linewidths=2,
                linecolor='white',
                square=True,
                ax=ax,
                vmin=0,
                vmax=15)

    # Styling
    ax.set_title('GSM8K Finetuning Final Accuracy (%)\n300M Models, All Configurations',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Learning Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Base Model → Finetune Optimizer', fontsize=14, fontweight='bold')

    # Rotate labels
    plt.xticks(rotation=0, ha='center')
    plt.yticks(rotation=0)

    # Adjust layout
    plt.tight_layout()

    # Save
    output_path = "gsm8k_300m_accuracy_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved heatmap to: {output_path}")

    pdf_path = "gsm8k_300m_accuracy_heatmap.pdf"
    plt.savefig(pdf_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved heatmap to: {pdf_path}")

    plt.close()

    # Print summary
    print("\n" + "=" * 80)
    print("KEY OBSERVATIONS")
    print("=" * 80)

    # Find best overall
    best_idx = np.argmax(data)
    best_i, best_j = np.unravel_index(best_idx, data.shape)
    best_combo = COMBOS[best_i]
    best_lr = LRS[best_j]
    best_acc = data[best_i, best_j]

    print(f"\n🏆 Best Result: {best_acc:.2f}%")
    print(f"   Config: {best_combo[0]} → {best_combo[1].upper()} @ LR={best_lr}")

    # Compare by finetune optimizer
    print(f"\n📊 Average by Finetune Optimizer:")
    adamw_ft_avg = data[[0, 2], :].mean()
    muon_ft_avg = data[[1, 3], :].mean()
    print(f"   AdamW finetune: {adamw_ft_avg:.2f}%")
    print(f"   Muon finetune:  {muon_ft_avg:.2f}%")

    # Compare by base model
    print(f"\n📊 Average by Base Model:")
    adamw_base_avg = data[0:2, :].mean()
    muon_base_avg = data[2:4, :].mean()
    print(f"   AdamW base: {adamw_base_avg:.2f}%")
    print(f"   Muon base:  {muon_base_avg:.2f}%")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
