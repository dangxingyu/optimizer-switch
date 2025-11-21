# Moonlight Muon Migration - Complete

## Summary

Successfully migrated from modded-nanogpt's NorMuon to Moonlight's Muon implementation in `train_llama_muon_single_gpu.py`.

## Changes Made

### 1. Replaced Orthogonalization Method ✅

**BEFORE (Polar Express):**
```python
def polar_express(G):
    # Complex implementation with 5 coefficient sets
    # Uses Triton kernels (XXT_kernel, ba_plus_cAA_kernel)
    # ~250 lines of code
```

**AFTER (Newton-Schulz 5):**
```python
@torch.compile
def zeropower_via_newtonschulz5(G, steps=5):
    # Simple quintic iteration
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # ... ~20 lines of code
```

**Benefits:**
- ✅ 90% less code (~20 lines vs ~250 lines)
- ✅ `@torch.compile` for JIT optimization
- ✅ Simpler, more maintainable
- ✅ No Triton dependency

### 2. Replaced Momentum Formula ✅

**BEFORE (Lerp - NorMuon):**
```python
momentum_buffer.lerp_(grad, 1 - momentum)
updated_grads = grad.lerp(momentum_buffer, momentum)
# Gradient contribution: 20x smaller
```

**AFTER (Standard SGD + Nesterov):**
```python
buf.mul_(momentum).add_(g)  # Standard SGD momentum
if nesterov:
    g = g.add(buf, alpha=momentum)  # Nesterov
else:
    g = buf
# Gradient contribution: full strength
```

**Benefits:**
- ✅ Standard momentum (20x more aggressive)
- ✅ Nesterov option (recommended for better convergence)
- ✅ Matches original Muon design

### 3. Changed LR Scaling Formula ✅

**BEFORE (NorMuon):**
```python
scale = max(1., d_out / d_in) ** 0.5
# (512, 2048): scale = 1.0
# (2048, 512): scale = 2.0
```

**AFTER (Moonlight):**
```python
scale = 0.2 * sqrt(max(d_out, d_in))
# (512, 2048): scale = 0.2 * sqrt(2048) = 9.05
# (2048, 512): scale = 0.2 * sqrt(2048) = 9.05
```

**Benefits:**
- ✅ More aggressive scaling (9x larger for (512, 2048))
- ✅ Symmetric (same for both matrix orientations)
- ✅ Proven in Moonlight's implementation

### 4. Removed NorMuon Features ✅

**Removed:**
- ❌ Second momentum buffer
- ❌ Adaptive step size
- ❌ Variance tracking
- ❌ Cautious weight decay

**Why:**
- NorMuon adds complexity that may not be necessary
- Moonlight's simpler approach is well-tested
- Easier to debug and understand

### 5. Combined Optimizer ✅

**BEFORE:**
```python
muon_optimizer = Muon(muon_params, ...)
adamw_optimizer = torch.optim.AdamW(adamw_params, ...)

# Training loop:
muon_optimizer.step()
adamw_optimizer.step()
muon_optimizer.zero_grad()
adamw_optimizer.zero_grad()
```

**AFTER:**
```python
optimizer = Muon(
    muon_params=muon_params,
    adamw_params=adamw_params,
    ...
)

# Training loop:
optimizer.step()
optimizer.zero_grad()
```

**Benefits:**
- ✅ Cleaner API
- ✅ Single optimizer handles both Muon and AdamW
- ✅ Simpler training loop
- ✅ Easier to manage learning rate schedules

### 6. Removed Triton Dependencies ✅

**Removed:**
- `_get_autotune_configs()`
- `XXT_kernel`
- `compute_XXT()`
- `ba_plus_cAA_kernel`
- `ba_plus_cAA()`
- `polar_express_coeffs`
- Triton imports

**Savings:**
- ~200 lines of Triton kernel code removed
- No triton import needed
- Simpler dependencies

## Feature Comparison

| Feature | NorMuon (Before) | Moonlight (After) |
|---------|-----------------|-------------------|
| **Orthogonalization** | Polar Express | Newton-Schulz 5 |
| **Code Size** | ~250 lines | ~20 lines |
| **JIT Compilation** | No | Yes (`@torch.compile`) |
| **Momentum** | Lerp (conservative) | Standard SGD + Nesterov |
| **LR Scaling** | `max(1, d_out/d_in)**0.5` | `0.2 * sqrt(max(d_out, d_in))` |
| **Adaptive Step Size** | Yes (complex) | No (simpler) |
| **Cautious WD** | Yes | No (standard WD) |
| **Combined Optimizer** | No | Yes |
| **Dependencies** | Triton required | No Triton |

## Migration Impact

### Code Structure
- **Lines removed**: ~200 (Triton kernels)
- **Lines simplified**: ~100 (optimizer class)
- **Net change**: Simpler, cleaner codebase

### Training Behavior Changes

1. **Momentum is 20x more aggressive**
   - Lerp: `grad_contribution = 0.05 * grad`
   - Standard: `grad_contribution = 1.0 * grad`
   - **Impact**: Faster learning, may need different hyperparameters

2. **LR scaling is different**
   - NorMuon (512, 2048): `scale = 1.0`
   - Moonlight (512, 2048): `scale = 9.05`
   - **Impact**: Effective LR is ~9x larger

3. **No adaptive step size**
   - NorMuon: Per-parameter normalization like Adam
   - Moonlight: Fixed step size
   - **Impact**: May be less stable on some tasks

4. **Standard weight decay**
   - NorMuon: Cautious (only when aligned)
   - Moonlight: Always applied
   - **Impact**: Standard behavior, proven to work

### Hyperparameter Recommendations

Since Moonlight is more aggressive (20x momentum + 9x LR scaling):

**Start with smaller base LR:**
- NorMuon used: `lr = 0.002`
- Moonlight: Try `lr = 0.0002` or `lr = 0.001`

**Or reduce weight decay:**
- NorMuon: `wd = 0.0`
- Moonlight: Try `wd = 0.01` or `wd = 0.1`

## Testing Plan

### Phase 1: Quick Smoke Test
```bash
python train_llama_muon_single_gpu.py \
  --checkpoint_path ../checkpoints/muon_300m_8 \
  --lr 0.001 \
  --max_steps 10 \
  --output_dir test_moonlight_quick
```

**Expected:** No crashes, gradients flow, loss decreases

### Phase 2: Short Training Run
```bash
python train_llama_muon_single_gpu.py \
  --checkpoint_path ../checkpoints/muon_300m_8 \
  --lr 0.001 \
  --max_steps 100 \
  --output_dir test_moonlight_100
```

**Expected:** Stable training, loss trends downward

### Phase 3: Full Comparison

Run multiple LRs to find optimal:
- `lr = 0.0001`
- `lr = 0.0003`
- `lr = 0.001`

Compare to baseline:
- Previous Muon: ~2.3% (failed)
- AdamW: 13.5% (worked)

**Target:** >10% accuracy to demonstrate improvement

## Rollback Plan

If Moonlight doesn't work, we can revert by:

1. **Keep NorMuon implementation**:
   - The previous NorMuon code is documented in `NORMUON_IMPLEMENTATION_COMPLETE.md`
   - Can restore from git history

2. **Try hybrid approach**:
   - Use Moonlight's Newton-Schulz + NorMuon's adaptive step size
   - Combine best of both worlds

## Files Modified

- `train_llama_muon_single_gpu.py` - Full Moonlight migration
  - Lines 40-88: Newton-Schulz 5 function
  - Lines 90-240: Muon class (combined optimizer)
  - Lines 490-500: Single optimizer setup
  - Lines 577-586: Single optimizer training loop

## Verification

Syntax check:
```bash
python3 -m py_compile train_llama_muon_single_gpu.py
```
✅ **PASSED** - No syntax errors

Import check:
```bash
python3 -c "from train_llama_muon_single_gpu import Muon, zeropower_via_newtonschulz5; print('OK')"
```
✅ **Expected to work**

## Why Moonlight?

1. **Simpler implementation** (~20 lines vs ~250 lines)
2. **No Triton dependency** (easier to maintain)
3. **Proven design** (matches original Muon paper)
4. **Standard momentum** (20x more aggressive than lerp)
5. **Combined optimizer** (cleaner API)
6. **JIT compilation** (`@torch.compile` for speed)

## What We Lose

1. **Adaptive step size** - No per-parameter normalization
2. **Cautious WD** - Standard WD always applied
3. **Conservative momentum** - Lerp was 20x smaller

**Tradeoff:** Simplicity + proven design vs advanced features

## Next Steps

1. ✅ Migration complete
2. ✅ Syntax verified
3. ⏳ Run smoke test (10 steps)
4. ⏳ Run short test (100 steps)
5. ⏳ Run full training with multiple LRs
6. ⏳ Compare results to baseline

---

**Status**: READY FOR TESTING

Created: 2025-11-19
