#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=3:59:59
#SBATCH --partition=pli-c
#SBATCH --account=pli
#SBATCH --job-name=gsm8k_ft
#SBATCH --output=logs/gsm8k_%A_%a.out
#SBATCH --error=logs/gsm8k_%A_%a.err
#SBATCH --array=0-31  # 8 models × 4 LRs = 32 jobs
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xd7812@princeton.edu

# ============================================================================
# GSM8K Finetuning with Muon and AdamW
# ============================================================================
# This script finetunes selected pretrained models (AdamW and Muon) on GSM8K
# with learning rate sweep.
#
# Job array explanation:
# - 8 models: 130m_{1,8}, 300m_{1,8} for both AdamW and Muon
# - 4 learning rates: 5e-6, 1e-5, 3e-5, 1e-4
# - Total: 8 × 4 = 32 jobs
# ============================================================================

set -eo pipefail

# ============================================================================
# Configuration Arrays
# ============================================================================

# Model configurations (optimizer_size_seed)
MODELS=(
    "adamw_130m_1"
    "adamw_130m_8"
    "adamw_300m_1"
    "adamw_300m_8"
    "muon_130m_1"
    "muon_130m_8"
    "muon_300m_1"
    "muon_300m_8"
)

# Learning rates for sweep
LRS=(
    "5e-6"
    "1e-5"
    "3e-5"
    "1e-4"
)

# Calculate model index and LR index from SLURM_ARRAY_TASK_ID
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 4))
LR_IDX=$((SLURM_ARRAY_TASK_ID % 4))

MODEL_NAME="${MODELS[$MODEL_IDX]}"
LR="${LRS[$LR_IDX]}"

# Extract optimizer type from model name
if [[ "$MODEL_NAME" == adamw_* ]]; then
    OPTIMIZER="adamw"
    SCRIPT="train_llama_adamw_single_gpu.py"
elif [[ "$MODEL_NAME" == muon_* ]]; then
    OPTIMIZER="muon"
    SCRIPT="train_llama_muon_single_gpu.py"
else
    echo "ERROR: Unknown optimizer in model name: $MODEL_NAME"
    exit 1
fi

# Paths
CHECKPOINT_PATH="../checkpoints/${MODEL_NAME}"
OUTPUT_DIR="outputs/gsm8k/${MODEL_NAME}/lr_${LR}"
LOG_DIR="logs"

# ============================================================================
# Environment Setup
# ============================================================================

echo "========================================="
echo "GSM8K Finetuning"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Model: ${MODEL_NAME}"
echo "Optimizer: ${OPTIMIZER}"
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
