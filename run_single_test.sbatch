#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --partition=pli-c
#SBATCH --account=pli
#SBATCH --job-name=gsm8k_test
#SBATCH --output=logs/gsm8k_single_test.out
#SBATCH --error=logs/gsm8k_single_test.err

# Single job test: muon_300m_8 + muon finetune + lr=1e-5

set -eo pipefail

BASE_MODEL="muon_300m_8"
FINETUNE_OPT="muon"
LR="1e-5"
SCRIPT="train_llama_muon_single_gpu.py"

CHECKPOINT_PATH="../checkpoints/${BASE_MODEL}"
OUTPUT_DIR="outputs/gsm8k_test/${BASE_MODEL}_${FINETUNE_OPT}/lr_${LR}"
LOG_DIR="logs"

echo "========================================="
echo "Single Job Test"
echo "========================================="
echo "Base Model: ${BASE_MODEL}"
echo "Finetune Optimizer: ${FINETUNE_OPT}"
echo "Learning Rate: ${LR}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "========================================="
echo ""

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

source ~/.bashrc
conda activate muon

nvidia-smi -L
echo ""

if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "Starting test with 256 samples, 200 steps..."
echo "Start time: $(date)"
echo ""

python -u "${SCRIPT}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --lr "${LR}" \
    --max_train_samples 256 \
    --max_steps 200 \
    --output_dir "${OUTPUT_DIR}"

EXIT_CODE=$?

echo ""
echo "========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Test Completed Successfully"
else
    echo "✗ Test Failed: ${EXIT_CODE}"
fi
echo "End time: $(date)"
echo "========================================="

exit ${EXIT_CODE}
