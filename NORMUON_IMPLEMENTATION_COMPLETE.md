# NorMuon Implementation - Complete

## Summary

Successfully updated `train_llama_muon_single_gpu.py` to use **NorMuon** (Normalized Muon) implementation that exactly matches `train_gpt.py` (world record code).

## Changes Made

### 1. Updated Muon Class (lines 315-426)

Replaced basic Muon with full NorMuon implementation including:

#### ✅ Lerp Momentum (train_gpt.py lines 555-556)
```python
momentum_buffer.lerp_(grad, 1 - group["momentum"])
updated_grads = grad.lerp(momentum_buffer, group["momentum"])
```
- **Status**: Already had this, kept it ✓

#### ✅ LR Scaling by Shape (train_gpt.py line 575)
```python
if p.ndim >= 2:
    scale = max(1., param_shape[-2] / param_shape[-1]) ** 0.5
else:
    scale = 1.0
state["param_lr"] = scale * getattr(p, "lr_mul", 1.0)
eff_lr = group["lr"] * state["param_lr"]
```
- **Status**: **ADDED** - was missing before ✓

#### ✅ Second Momentum Buffer (train_gpt.py lines 567-570)
```python
if "second_momentum_buffer" not in state:
    if param_shape[-2] >= param_shape[-1]:
        state["second_momentum_buffer"] = torch.zeros_like(v[..., :, :1])
    else:
        state["second_momentum_buffer"] = torch.zeros_like(v[..., :1, :])
```
- **Status**: **ADDED** - was missing before ✓

#### ✅ Adaptive Step Size (NorMuon) (train_gpt.py lines 596-602)
```python
# Track variance along one dimension
v_norm = v.norm(dim=(-2, -1), keepdim=True)
if param_shape[-2] >= param_shape[-1]:
    v_mean = v.square().mean(dim=-1, keepdim=True)
else:
    v_mean = v.square().mean(dim=-2, keepdim=True)

# Update second momentum
second_momentum_buffer.lerp_(v_mean.to(dtype=p.dtype), 1 - group["beta2"])

# Compute adaptive step size
step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
v.mul_(step_size)

# Renormalize to preserve magnitude
v_norm_new = v.norm(dim=(-2, -1), keepdim=True)
v.mul_(v_norm / v_norm_new.clamp_min_(1e-10))
```
- **Status**: **ADDED** - was missing before ✓
- **Impact**: This is the key NorMuon feature that provides adaptive per-parameter normalization

#### ✅ Cautious Weight Decay (train_gpt.py lines 609-611)
```python
if eff_wd > 0:
    mask = (v * p) >= 0  # Only when update and param have same sign
    v.addcmul_(p, (eff_wd * mask).to(p.dtype))
```
- **Status**: **ADDED** - was missing before ✓
- **Impact**: Prevents weight decay from interfering with optimization direction

#### ✅ Beta2 Parameter
```python
def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, beta2=0.95):
```
- **Status**: **ADDED** - required for NorMuon second momentum ✓

### 2. Updated Optimizer Initialization (line 675-681)

```python
muon_optimizer = Muon(
    muon_params,
    lr=config.muon_lr,
    momentum=config.muon_momentum,
    weight_decay=config.muon_weight_decay,
    beta2=0.95,  # NorMuon second momentum (for variance tracking)
)
```

## Before vs After Comparison

### BEFORE (Basic Muon)
```python
# No LR scaling
p.add_(update, alpha=-lr)

# No adaptive step size
update = polar_express(update_grad)

# Standard weight decay (always applied)
if weight_decay > 0:
    p.mul_(1 - lr * weight_decay)
```

**Problems:**
- ❌ No shape-based LR scaling → inconsistent effective LR
- ❌ No adaptive step size → too aggressive for finetuning
- ❌ Standard WD → hurts optimization direction

### AFTER (NorMuon)
```python
# LR scaling by shape
scale = max(1., d_out/d_in) ** 0.5
eff_lr = lr * scale

# Adaptive step size with variance tracking
v_mean = v.square().mean(dim=...)
second_momentum_buffer.lerp_(v_mean, 1 - beta2)
step_size = second_momentum_buffer.rsqrt_()
v.mul_(step_size)
v_norm_new = v.norm(...)
v.mul_(v_norm / v_norm_new)

# Cautious weight decay (only when aligned)
mask = (v * p) >= 0
v.addcmul_(p, (eff_wd * mask))
```

**Improvements:**
- ✅ Shape-based LR scaling → consistent effective LR
- ✅ Adaptive step size → stable finetuning like Adam
- ✅ Cautious WD → doesn't hurt optimization

## Feature Parity with train_gpt.py

| Feature | train_gpt.py | Our Implementation | Status |
|---------|-------------|-------------------|--------|
| Lerp momentum | ✅ (lines 555-556) | ✅ (lines 368-369) | **MATCH** |
| LR scaling | ✅ (line 575) | ✅ (lines 374-378) | **MATCH** |
| Second momentum | ✅ (lines 567-570) | ✅ (lines 393-398) | **MATCH** |
| Adaptive step size | ✅ (lines 596-602) | ✅ (lines 401-416) | **MATCH** |
| Cautious WD | ✅ (lines 609-611) | ✅ (lines 419-421) | **MATCH** |
| Beta2 parameter | ✅ | ✅ (line 335) | **MATCH** |
| Per-param lr_mul | ✅ | ✅ (line 377) | **MATCH** |
| Per-param wd_mul | ✅ | ✅ (line 378) | **MATCH** |

## Why This Should Fix Finetuning

### Previous Results (Basic Muon)
- **AdamW**: 13.5% accuracy ✅
- **Muon**: ~2.3% accuracy ❌ (barely better than random)

### Why AdamW Worked
- Has adaptive step size via RMSProp (second moment)
- Per-parameter normalization

### Why Basic Muon Failed
- No adaptive step size → too aggressive
- No LR scaling → inconsistent
- Standard WD → hurts optimization

### Why NorMuon Should Work
- ✅ Has adaptive step size (second momentum + variance normalization)
- ✅ Has LR scaling (shape-based)
- ✅ Has cautious WD (only when aligned)
- ✅ Proven to work (world record on train_gpt.py)

## Testing Recommendations

### 1. Quick Test (300M model, 100 steps)
```bash
python train_llama_muon_single_gpu.py \
  --checkpoint_path ../checkpoints/muon_300m_8 \
  --lr 1e-4 \
  --max_steps 100 \
  --output_dir test_normuon
```

### 2. Full Test (300M model, full training)
Run with multiple LRs to find optimal:
- LR 1e-5
- LR 3e-5
- LR 1e-4

Compare to previous Muon results (~2.3%) and AdamW results (13.5%).

### Expected Results

If implementation is correct:
- **Accuracy should improve significantly** from ~2.3% to >10%
- **Training should be stable** (no NaN/Inf)
- **Loss should decrease** steadily

If still not working:
- Check logs for NaN/Inf
- Verify gradient norms are reasonable
- Compare loss curves to AdamW

## Implementation Details

### Variance Tracking Dimension

The second momentum tracks variance along **one dimension**:
- If `d_out >= d_in`: track along **rows** (dim=-1)
- If `d_out < d_in`: track along **columns** (dim=-2)

This is different from Adam which tracks variance for each element.

### Magnitude Preservation

After scaling by adaptive step size, we renormalize to preserve the original magnitude:
```python
v_norm_original = v.norm(...)  # Before scaling
v.mul_(step_size)              # Scale
v_norm_new = v.norm(...)       # After scaling
v.mul_(v_norm_original / v_norm_new)  # Restore magnitude
```

This ensures the update magnitude is controlled by the optimizer, not by gradient variance.

### Cautious Weight Decay

Only applies decay when update and parameter have the **same sign**:
```python
mask = (v * p) >= 0  # Element-wise comparison
```

This prevents decay from pulling parameters in the wrong direction during finetuning.

## Code References

- **Our implementation**: [train_llama_muon_single_gpu.py lines 315-426](train_llama_muon_single_gpu.py#L315-L426)
- **Reference implementation**: [train_gpt.py lines 551-613](train_gpt.py#L551-L613)
- **NorMuon paper**: https://arxiv.org/pdf/2510.05491
- **Cautious WD paper**: https://arxiv.org/abs/2510.12402

## Verification

Syntax check:
```bash
python3 -m py_compile train_llama_muon_single_gpu.py
```
✅ **PASSED** - No syntax errors

Implementation review:
- ✅ All 8 features match train_gpt.py
- ✅ Line-by-line correspondence verified
- ✅ Comments reference exact train_gpt.py line numbers

---

**Status**: READY FOR TESTING

Created: 2025-11-19
