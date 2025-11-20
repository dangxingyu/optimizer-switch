#!/bin/bash
# Quick test of Muon single-GPU implementation on login node

cd /scratch/gpfs/ARORA/xd7812/optimizers/modded-nanogpt

echo "Testing Muon single-GPU implementation..."
echo "Using 64 training samples, 50 max steps"
echo ""

python3 train_llama_muon_single_gpu.py \
  --checkpoint_path ../checkpoints/adamw_130m_1 \
  --max_train_samples 64 \
  --max_steps 50 \
  --output_dir ./test_output 2>&1 | tee test_muon.log

echo ""
echo "Test completed. Checking if loss decreased..."
echo ""

# Extract first and last loss values
python3 << 'EOF'
import re

with open('test_muon.log', 'r') as f:
    lines = f.readlines()

losses = []
for line in lines:
    # Match lines like "Step 1/50 | Loss: 2.5000"
    match = re.search(r'Step \d+/\d+ \| Loss: ([\d.]+)', line)
    if match:
        losses.append(float(match.group(1)))

if len(losses) >= 2:
    first_loss = losses[0]
    last_loss = losses[-1]
    print(f"First loss: {first_loss:.4f}")
    print(f"Last loss:  {last_loss:.4f}")
    print(f"Change:     {last_loss - first_loss:+.4f}")

    if last_loss < first_loss:
        print("\n✓ Loss decreased - Muon optimizer is working!")
    else:
        print("\n✗ Loss did not decrease - there may be a problem")
else:
    print("Could not extract loss values from log")
EOF
