#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=3:59:59
#SBATCH --partition=pli-c
#SBATCH --account=pli
#SBATCH --job-name=gsm8k_300m
#SBATCH --output=logs/gsm8k_300m_%A_%a.out
#SBATCH --error=logs/gsm8k_300m_%A_%a.err
#SBATCH --array=0-15  # 2 base models × 2 finetune optimizers × 4 LRs = 16 jobs
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xd7812@princeton.edu

# ============================================================================
# GSM8K Finetuning Test - 300M_8 models only
# ============================================================================
# Testing all combinations:
# - Base models: adamw_300m_8, muon_300m_8
# - Finetune with: AdamW, Muon
# - Learning rates: 5e-6, 1e-5, 3e-5, 1e-4
# Total: 2 × 2 × 4 = 16 jobs
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

# Finetune optimizers
FINETUNE_OPTS=(
    "adamw"
    "muon"
)

# Learning rates for sweep
LRS=(
    "5e-6"
    "1e-5"
    "3e-5"
    "1e-4"
)

# Calculate indices from SLURM_ARRAY_TASK_ID
# Layout: base_model (2) × finetune_opt (2) × lr (4) = 16
LR_IDX=$((SLURM_ARRAY_TASK_ID % 4))
TEMP=$((SLURM_ARRAY_TASK_ID / 4))
FINETUNE_IDX=$((TEMP % 2))
BASE_IDX=$((TEMP / 2))

BASE_MODEL="${BASE_MODELS[$BASE_IDX]}"
FINETUNE_OPT="${FINETUNE_OPTS[$FINETUNE_IDX]}"
LR="${LRS[$LR_IDX]}"

# Select training script based on finetune optimizer
if [[ "$FINETUNE_OPT" == "adamw" ]]; then
    SCRIPT="train_llama_adamw_single_gpu.py"
elif [[ "$FINETUNE_OPT" == "muon" ]]; then
    SCRIPT="train_llama_muon_single_gpu.py"
else
    echo "ERROR: Unknown finetune optimizer: $FINETUNE_OPT"
    exit 1
fi

# Paths
CHECKPOINT_PATH="../checkpoints/${BASE_MODEL}"
OUTPUT_DIR="outputs/gsm8k/${BASE_MODEL}_${FINETUNE_OPT}/lr_${LR}"
LOG_DIR="logs"

# ============================================================================
# Environment Setup
# ============================================================================

echo "========================================="
echo "GSM8K Finetuning - 300M Test"
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

echo "Starting training..."
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
else
    echo "Training Failed with exit code: ${TRAIN_EXIT_CODE}"
fi
echo "========================================="
echo "End time: $(date)"
echo "========================================="

exit ${TRAIN_EXIT_CODE}
