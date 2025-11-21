#!/bin/bash
# Test a single job configuration before running the full sweep
# This will test: muon_300m_8 base model + Moonlight Muon finetune
#
# NOTE: Now using Moonlight's Muon implementation:
#   - Newton-Schulz 5 (simpler than Polar Express)
#   - Standard SGD momentum + Nesterov (20x more aggressive than lerp)
#   - Moonlight LR scaling: 0.2*sqrt(max(d_out,d_in)) (~9x larger for typical layers)
#   - Combined: ~180x more aggressive than previous NorMuon
#
# RECOMMENDED: Start with smaller LR than NorMuon (try 1e-4, 3e-4, 1e-3)

set -e

BASE_MODEL="muon_300m_8"
FINETUNE_OPT="moonlight_muon"
LR="1e-4"  # Updated: Moonlight is more aggressive, use smaller LR
SCRIPT="train_llama_muon_single_gpu.py"

CHECKPOINT_PATH="../checkpoints/${BASE_MODEL}"
OUTPUT_DIR="outputs/gsm8k_test/${BASE_MODEL}_${FINETUNE_OPT}/lr_${LR}"

echo "========================================="
echo "Single Job Test"
echo "========================================="
echo "Base Model: ${BASE_MODEL}"
echo "Finetune Optimizer: ${FINETUNE_OPT}"
echo "Learning Rate: ${LR}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Script: ${SCRIPT}"
echo "========================================="
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Verify checkpoint exists
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint directory not found: ${CHECKPOINT_PATH}"
    exit 1
fi

echo "Running with reduced samples for quick test..."
echo ""

# Run with limited samples and steps for quick validation
python3 "${SCRIPT}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --lr "${LR}" \
    --max_train_samples 128 \
    --max_steps 100 \
    --output_dir "${OUTPUT_DIR}"

EXIT_CODE=$?

echo ""
echo "========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Test Completed Successfully"
    echo ""
    echo "Check the training log to verify loss is decreasing:"
    echo "  tail -20 ${OUTPUT_DIR}/training_log.csv"
else
    echo "✗ Test Failed with exit code: ${EXIT_CODE}"
fi
echo "========================================="

exit ${EXIT_CODE}
