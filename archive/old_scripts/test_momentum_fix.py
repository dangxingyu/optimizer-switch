#!/usr/bin/env python3
"""
Test that Muon momentum formula is now correct.
"""

import torch
import sys

# Import our fixed Muon
from muon_optimizer_complete import Muon

print("=" * 80)
print("Testing Muon Momentum Fix")
print("=" * 80)
print()

# Create a simple parameter
param = torch.randn(10, 10, requires_grad=True)

# Create optimizer
optimizer = Muon([param], lr=0.02, momentum=0.95)

# Simulate a training step
print("Step 1: Initial gradient")
param.grad = torch.ones_like(param)
optimizer.step()

# Check momentum buffer
state = optimizer.state[param]
buf = state['momentum_buffer']

print(f"  Gradient: {param.grad[0, 0].item():.4f}")
print(f"  Momentum buffer after step 1: {buf[0, 0].item():.4f}")
print(f"  Expected (0 * 0.95 + 1.0): {0 * 0.95 + 1.0:.4f}")
print()

# Step 2
print("Step 2: Another gradient")
param.grad = torch.ones_like(param) * 2.0
old_buf = buf[0, 0].item()
optimizer.step()
new_buf = state['momentum_buffer'][0, 0].item()

expected = old_buf * 0.95 + 2.0
print(f"  Gradient: {param.grad[0, 0].item():.4f}")
print(f"  Old momentum buffer: {old_buf:.4f}")
print(f"  New momentum buffer: {new_buf:.4f}")
print(f"  Expected ({old_buf:.4f} * 0.95 + 2.0): {expected:.4f}")
print(f"  Difference: {abs(new_buf - expected):.6f}")
print()

if abs(new_buf - expected) < 1e-5:
    print("✓ Momentum formula is CORRECT!")
    print("  buf = buf * momentum + grad")
else:
    print("✗ Momentum formula is still WRONG!")
    sys.exit(1)

print()
print("=" * 80)
print("Test PASSED - Muon momentum is fixed!")
print("=" * 80)
