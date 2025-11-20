#!/usr/bin/env python3
"""
Generate 2x2 accuracy table for GSM8K finetuning results.
Rows: Base models (adamw_300m_8, muon_300m_8)
Cols: Finetune optimizers (adamw, muon)
"""

import re
from pathlib import Path
import pandas as pd
import numpy as np

# Configuration
LOG_DIR = Path("logs")
LOG_PATTERN = "gsm8k_300m_*.out"

# Model configurations
BASE_MODELS = ["adamw_300m_8", "muon_300m_8"]
FINETUNE_OPTS = ["adamw", "muon"]
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
    acc_match = re.search(r'Final GSM8K Accuracy: ([\d.]+) \((\d+)/(\d+)\)', content)

    if not all([base_match, finetune_match, lr_match, acc_match]):
        return None

    return {
        'base_model': base_match.group(1),
        'finetune_opt': finetune_match.group(1),
        'lr': lr_match.group(1),
        'accuracy': float(acc_match.group(1)),
        'correct': int(acc_match.group(2)),
        'total': int(acc_match.group(3))
    }

def main():
    # Parse all log files
    results = []
    log_files = sorted(LOG_DIR.glob(LOG_PATTERN))

    print(f"Parsing {len(log_files)} log files...\n")

    for log_path in log_files:
        result = parse_log_file(log_path)
        if result:
            results.append(result)

    print(f"Successfully parsed {len(results)} results\n")

    # Group by (base_model, finetune_opt) and find best LR
    best_results = {}

    for result in results:
        key = (result['base_model'], result['finetune_opt'])

        if key not in best_results or result['accuracy'] > best_results[key]['accuracy']:
            best_results[key] = result

    # Create 2x2 table
    print("=" * 80)
    print("GSM8K FINETUNING RESULTS - FINAL ACCURACY (Best LR)")
    print("=" * 80)
    print()

    # Print as markdown table
    print("| Base Model \\ Finetune | AdamW | Muon |")
    print("|------------------------|-------|------|")

    for base_model in BASE_MODELS:
        base_display = base_model.replace('_', ' ').upper()
        row = f"| {base_display:22} |"

        for finetune_opt in FINETUNE_OPTS:
            key = (base_model, finetune_opt)
            if key in best_results:
                r = best_results[key]
                cell = f" {r['accuracy']:.2%} (LR={r['lr']}) "
                row += f" {cell:20} |"
            else:
                row += " N/A |"

        print(row)

    print()
    print("=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    print()

    for base_model in BASE_MODELS:
        for finetune_opt in FINETUNE_OPTS:
            print(f"\n{base_model} → {finetune_opt.upper()} finetune:")
            print("-" * 60)

            # Show all LRs for this combination
            combo_results = [r for r in results
                           if r['base_model'] == base_model
                           and r['finetune_opt'] == finetune_opt]

            combo_results.sort(key=lambda x: x['accuracy'], reverse=True)

            for r in combo_results:
                best_marker = " ★" if r == best_results.get((base_model, finetune_opt)) else "  "
                print(f"{best_marker} LR={r['lr']:>6}: {r['accuracy']:.2%} ({r['correct']}/{r['total']})")

    print("\n" + "=" * 80)

    # Save as CSV
    csv_data = []
    for base_model in BASE_MODELS:
        for finetune_opt in FINETUNE_OPTS:
            key = (base_model, finetune_opt)
            if key in best_results:
                r = best_results[key]
                csv_data.append({
                    'base_model': base_model,
                    'finetune_optimizer': finetune_opt,
                    'best_lr': r['lr'],
                    'accuracy': r['accuracy'],
                    'correct': r['correct'],
                    'total': r['total']
                })

    df = pd.DataFrame(csv_data)
    csv_path = "gsm8k_300m_accuracy_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved detailed results to: {csv_path}")

    # Create pivot table for heatmap
    pivot_data = []
    for base_model in BASE_MODELS:
        row = {'Base Model': base_model}
        for finetune_opt in FINETUNE_OPTS:
            key = (base_model, finetune_opt)
            if key in best_results:
                row[finetune_opt] = best_results[key]['accuracy']
            else:
                row[finetune_opt] = np.nan
        pivot_data.append(row)

    pivot_df = pd.DataFrame(pivot_data)
    pivot_df.set_index('Base Model', inplace=True)

    pivot_path = "gsm8k_300m_accuracy_pivot.csv"
    pivot_df.to_csv(pivot_path)
    print(f"✓ Saved pivot table to: {pivot_path}")

    print()

if __name__ == "__main__":
    main()
