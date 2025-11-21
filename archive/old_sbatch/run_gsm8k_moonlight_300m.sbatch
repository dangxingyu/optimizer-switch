#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=3:59:59
#SBATCH --partition=pli-c
#SBATCH --account=pli
#SBATCH --job-name=gsm8k_moonlight
#SBATCH --output=logs/gsm8k_moonlight_300m_%A_%a.out
#SBATCH --error=logs/gsm8k_moonlight_300m_%A_%a.err
#SBATCH --array=0-9  # 2 base models × 5 LRs = 10 jobs
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xd7812@princeton.edu

# ============================================================================
# GSM8K Finetuning Test - Moonlight Muon (300M models)
# ============================================================================
# NOTE: Using Moonlight's Muon implementation:
#   - Newton-Schulz 5 orthogonalization (simpler than Polar Express)
#   - Standard SGD momentum + Nesterov (20x more aggressive than lerp)
#   - Moonlight LR scaling: 0.2*sqrt(max(d_out,d_in)) (~9x larger)
#   - Combined: ~180x more aggressive than previous NorMuon
#
# Testing combinations:
# - Base models: adamw_300m_8, muon_300m_8
# - Finetune with: Moonlight Muon only
# - Learning rates: 1e-5, 3e-5, 1e-4, 3e-4, 1e-3 (adjusted for Moonlight)
# Total: 2 × 5 = 10 jobs
# ============================================================================

set -eo pipefail

# ============================================================================
# Configuration Arrays
# ============================================================================

# Base model configurations (pretrained models)
BASE_MODELS=(
    "adamw_300m_8"
    "muon_300m_8"
)

# Learning rates for Moonlight Muon sweep
# NOTE: These are smaller than previous Muon tests because Moonlight is ~180x more aggressive
# Previous NorMuon used: 5e-6, 1e-5, 3e-5, 1e-4
# Moonlight: 1e-5, 3e-5, 1e-4, 3e-4, 1e-3
LRS=(
    "1e-5"   # Very conservative
    "3e-5"   # Conservative
    "1e-4"   # Moderate (RECOMMENDED)
    "3e-4"   # Moderate-aggressive
    "1e-3"   # Aggressive
)

# Calculate indices from SLURM_ARRAY_TASK_ID
# Layout: base_model (2) × lr (5) = 10
LR_IDX=$((SLURM_ARRAY_TASK_ID % 5))
BASE_IDX=$((SLURM_ARRAY_TASK_ID / 5))

BASE_MODEL="${BASE_MODELS[$BASE_IDX]}"
LR="${LRS[$LR_IDX]}"

# Always use Moonlight Muon script
FINETUNE_OPT="moonlight_muon"
SCRIPT="train_llama_muon_single_gpu.py"

# Paths
CHECKPOINT_PATH="../checkpoints/${BASE_MODEL}"
OUTPUT_DIR="outputs/gsm8k_moonlight/${BASE_MODEL}/lr_${LR}"
LOG_DIR="logs"

# ============================================================================
# Environment Setup
# ============================================================================

echo "========================================="
echo "GSM8K Finetuning - Moonlight Muon"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Base Model: ${BASE_MODEL}"
echo "Finetune Optimizer: ${FINETUNE_OPT}"
echo "Learning Rate: ${LR}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Script: ${SCRIPT}"
echo "========================================="
echo "Implementation Details:"
echo "  - Orthogonalization: Newton-Schulz 5"
echo "  - Momentum: Standard SGD + Nesterov"
echo "  - LR Scaling: 0.2 * sqrt(max(d_out, d_in))"
echo "  - Combined Optimizer: Muon + AdamW"
echo "========================================="
echo ""

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Conda activation
echo "Activating conda environment..."
source ~/.bashrc
conda activate muon

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi -L
echo ""

# Verify checkpoint exists
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint directory not found: ${CHECKPOINT_PATH}"
    exit 1
fi

echo "Checkpoint verified: ${CHECKPOINT_PATH}"
echo ""

# Force offline mode - no network access
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "Offline mode: ENABLED"
echo ""

# ============================================================================
# Run Training
# ============================================================================

echo "Starting Moonlight Muon training..."
echo "Start time: $(date)"
echo ""

python -u "${SCRIPT}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --lr "${LR}" \
    --output_dir "${OUTPUT_DIR}"

TRAIN_EXIT_CODE=$?

echo ""
echo "========================================="
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo "Training Completed Successfully"

    # Print training summary
    if [ -f "${OUTPUT_DIR}/training_log.csv" ]; then
        echo ""
        echo "Training Summary:"
        echo "  First 5 steps:"
        head -6 "${OUTPUT_DIR}/training_log.csv" | tail -5
        echo "  Last 5 steps:"
        tail -5 "${OUTPUT_DIR}/training_log.csv"
    fi
else
    echo "Training Failed with exit code: ${TRAIN_EXIT_CODE}"
fi
echo "========================================="
echo "End time: $(date)"
echo "========================================="

exit ${TRAIN_EXIT_CODE}
