#!/bin/bash
#SBATCH --job-name=gsm8k_sweep
#SBATCH --output=logs/gsm8k_sweep_%A_%a.out
#SBATCH --error=logs/gsm8k_sweep_%A_%a.err
#SBATCH --array=0-95
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80

################################################################################
# Comprehensive GSM8K Finetuning Sweep
#
# Testing ALL 130m and 300m_1 models with BOTH optimizers
#
# Base Models (8):
#   - adamw_130m_1, adamw_130m_2, adamw_130m_8
#   - muon_130m_1, muon_130m_2, muon_130m_8
#   - adamw_300m_1, muon_300m_1
#
# Optimizers (2):
#   - adamw: Standard AdamW finetuning
#   - moonlight_muon: Moonlight Muon implementation
#
# Learning Rates (6):
#   - 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 2e-3
#
# Total Jobs: 8 × 2 × 6 = 96
#
# Task mapping: array_id = base_idx * 12 + opt_idx * 6 + lr_idx
################################################################################

set -e

# Configuration
BASE_MODELS=(
    "adamw_130m_1" "adamw_130m_8"
    "muon_130m_1" "muon_130m_8"
    "adamw_300m_1" "muon_300m_1"
)

OPTIMIZERS=("adamw" "moonlight_muon")
LRS=("1e-5" "3e-5" "1e-4" "3e-4" "1e-3" "2e-3")

# Decode array task ID
NUM_LRS=${#LRS[@]}
NUM_OPTS=${#OPTIMIZERS[@]}

LR_IDX=$((SLURM_ARRAY_TASK_ID % NUM_LRS))
OPT_IDX=$(((SLURM_ARRAY_TASK_ID / NUM_LRS) % NUM_OPTS))
BASE_IDX=$((SLURM_ARRAY_TASK_ID / (NUM_LRS * NUM_OPTS)))

BASE_MODEL=${BASE_MODELS[$BASE_IDX]}
OPTIMIZER=${OPTIMIZERS[$OPT_IDX]}
LR=${LRS[$LR_IDX]}

echo "=========================================="
echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Base Model: $BASE_MODEL"
echo "Optimizer: $OPTIMIZER"
echo "Learning Rate: $LR"
echo "=========================================="

# Activate conda environment
source ~/.bashrc
conda activate llm

# Set paths
SCRIPT_DIR="/scratch/gpfs/ARORA/xd7812/optimizers/modded-nanogpt"
CHECKPOINT_BASE="/scratch/gpfs/ARORA/xd7812/optimizers/checkpoints"
OUTPUT_BASE="$SCRIPT_DIR/outputs/gsm8k_sweep"

# Create output directory
OUTPUT_DIR="$OUTPUT_BASE/${BASE_MODEL}/${OPTIMIZER}/lr_${LR}"
mkdir -p "$OUTPUT_DIR"

# Training parameters
MAX_STEPS=500
EVAL_INTERVAL=250
BATCH_SIZE=8
GRAD_ACCUM=4

echo ""
echo "Configuration:"
echo "  Script: $SCRIPT_DIR/train_llama_muon_single_gpu.py"
echo "  Checkpoint: $CHECKPOINT_BASE/$BASE_MODEL"
echo "  Output: $OUTPUT_DIR"
echo "  Max Steps: $MAX_STEPS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Grad Accum: $GRAD_ACCUM"
echo ""

# Run training
cd "$SCRIPT_DIR"

python3 train_llama_muon_single_gpu.py \
    --checkpoint_path "$CHECKPOINT_BASE/$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --finetune_opt "$OPTIMIZER" \
    --finetune_lr "$LR" \
    --max_steps "$MAX_STEPS" \
    --eval_interval "$EVAL_INTERVAL" \
    --batch_size "$BATCH_SIZE" \
    --grad_accumulation_steps "$GRAD_ACCUM" \
    --device "cuda"

echo ""
echo "=========================================="
echo "Training completed successfully!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
