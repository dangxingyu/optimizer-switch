#!/bin/bash
# Test Moonlight Muon with multiple learning rates to find optimal value
#
# Since Moonlight is ~180x more aggressive than NorMuon (20x momentum + 9x LR scaling),
# we need to test smaller learning rates.
#
# This script runs quick tests (128 samples, 100 steps) with different LRs:
#   - 1e-5: Very conservative (like previous attempts)
#   - 3e-5: Conservative
#   - 1e-4: Moderate (RECOMMENDED START)
#   - 3e-4: Moderate-aggressive
#   - 1e-3: Aggressive (may be too high)

set -e

BASE_MODEL="muon_300m_8"
CHECKPOINT_PATH="../checkpoints/${BASE_MODEL}"
SCRIPT="train_llama_muon_single_gpu.py"

# Verify checkpoint exists
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint directory not found: ${CHECKPOINT_PATH}"
    exit 1
fi

echo "========================================="
echo "Moonlight Muon Learning Rate Sweep"
echo "========================================="
echo "Base Model: ${BASE_MODEL}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Quick test: 128 samples, 100 steps each"
echo "========================================="
echo ""

# Array of learning rates to test
LRS=("1e-5" "3e-5" "1e-4" "3e-4" "1e-3")

for LR in "${LRS[@]}"; do
    OUTPUT_DIR="outputs/moonlight_lr_sweep/${BASE_MODEL}/lr_${LR}"

    echo ""
    echo "========================================"
    echo "Testing LR: ${LR}"
    echo "Output: ${OUTPUT_DIR}"
    echo "========================================"

    # Create output directory
    mkdir -p "${OUTPUT_DIR}"

    # Run training
    python3 "${SCRIPT}" \
        --checkpoint_path "${CHECKPOINT_PATH}" \
        --lr "${LR}" \
        --max_train_samples 128 \
        --max_steps 100 \
        --output_dir "${OUTPUT_DIR}"

    EXIT_CODE=$?

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "✓ LR ${LR} completed"

        # Extract final loss
        if [ -f "${OUTPUT_DIR}/training_log.csv" ]; then
            FINAL_LOSS=$(tail -1 "${OUTPUT_DIR}/training_log.csv" | cut -d',' -f2)
            echo "  Final training loss: ${FINAL_LOSS}"
        fi
    else
        echo "✗ LR ${LR} failed with exit code: ${EXIT_CODE}"
    fi
done

echo ""
echo "========================================="
echo "LR Sweep Complete"
echo "========================================="
echo ""
echo "Compare results:"
echo "  grep 'Step 100' outputs/moonlight_lr_sweep/${BASE_MODEL}/*/training_log.csv"
echo ""
echo "Or view losses manually:"
for LR in "${LRS[@]}"; do
    LOG_FILE="outputs/moonlight_lr_sweep/${BASE_MODEL}/lr_${LR}/training_log.csv"
    if [ -f "${LOG_FILE}" ]; then
        FINAL_LOSS=$(tail -1 "${LOG_FILE}" | cut -d',' -f2)
        echo "  LR ${LR}: final loss = ${FINAL_LOSS}"
    fi
done
echo ""
echo "RECOMMENDATION: Choose the LR with lowest final loss without NaN/Inf"
