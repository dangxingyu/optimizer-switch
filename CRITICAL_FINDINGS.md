# Critical Findings: Why Muon Finetuning Failed

## TL;DR

Our Muon implementation is **basic Muon**, but train_gpt.py uses **NorMuon** (Normalized Muon). We're missing **3 critical features** that make finetuning work:

1. ❌ **NO LR Scaling** - All layers get the same LR regardless of shape
2. ❌ **NO Adaptive Step Size** - No second momentum / variance tracking
3. ❌ **NO Cautious Weight Decay** - Standard WD hurts optimization direction

---

## Feature Comparison

### train_gpt.py (World Record NorMuon)

```python
# 1. LR Scaling by parameter shape
scale = max(1., param_shape[-2] / param_shape[-1]) ** 0.5
eff_lr = group["lr"] * scale

# 2. Second momentum + adaptive step size (NorMuon)
v_norm = v_chunk.norm(dim=(-2, -1), keepdim=True)
v_mean = v_chunk.square().mean(dim=-1 if param_shape[-2] >= param_shape[-1] else -2, keepdim=True)
second_momentum_buffer.lerp_(v_mean.to(dtype=ref_param.dtype), 1 - group["beta2"])
step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
v_chunk.mul_(step_size)
v_norm_new = v_chunk.norm(dim=(-2, -1), keepdim=True)
v_chunk.mul_(v_norm / v_norm_new.clamp_min_(1e-10))

# 3. Cautious weight decay (only when update aligns with param)
mask = (v_chunk * param_chunk) >= 0
v_chunk.addcmul_(param_chunk, (eff_wd * mask).to(ref_param.dtype))

# 4. Lerp momentum
momentum_buffer.lerp_(grad_chunk[:num_params], 1 - group["momentum"])
updated_grads = grad_chunk[:num_params].lerp_(momentum_buffer, group["momentum"])
```

### Our Implementation (train_llama_muon_single_gpu.py)

```python
# 1. NO LR Scaling - fixed LR for all layers ❌
lr = group["lr"]

# 2. NO Adaptive Step Size - just basic polar_express ❌
update = polar_express(update_grad)

# 3. Standard Weight Decay - always applied ❌
if weight_decay > 0:
    p.mul_(1 - lr * weight_decay)

# 4. Lerp momentum - SAME ✅
momentum_buffer.lerp_(grad, 1 - momentum)
update_grad = grad.lerp(momentum_buffer, momentum)
```

---

## Impact Analysis

### Missing LR Scaling

**Problem**: Without shape-based LR scaling, a (512, 2048) matrix and a (2048, 512) matrix get the **same** effective learning rate.

**train_gpt.py approach**:
- (512, 2048): scale = max(1, 512/2048)**0.5 = max(1, 0.25)**0.5 = 1.0
- (2048, 512): scale = max(1, 2048/512)**0.5 = max(1, 4)**0.5 = 2.0
- (512, 512): scale = max(1, 1)**0.5 = 1.0

**Impact**: Without this, different layer shapes get inconsistent effective learning rates, leading to instability.

### Missing Adaptive Step Size (NorMuon)

**Problem**: Basic Muon uses a fixed step size for all parameters. NorMuon adapts the step size based on the gradient variance.

**How NorMuon works**:
1. Track second moment (variance) of orthogonalized gradients along one dimension
2. Compute adaptive step size: `step_size = 1 / sqrt(variance)`
3. Scale the update by step size
4. Renormalize to preserve the magnitude

**Why it matters**:
- Provides per-parameter normalization similar to Adam's RMSProp component
- Stabilizes training when gradient magnitudes vary widely
- Critical for finetuning where some layers need larger updates than others

**Comparison to Adam**:
- Adam: `update = m / (sqrt(v) + eps)` where `v` is second moment
- NorMuon: `update = polar_express(m) * (1 / sqrt(v))` where `v` is variance along one dim

Without this, Muon behaves like **SGD with momentum + orthogonalization**, which is too aggressive for finetuning.

### Missing Cautious Weight Decay

**Problem**: Standard weight decay always pulls parameters toward zero, even when it conflicts with the gradient direction.

**Standard WD**: `p = p * (1 - lr * wd)`
- Always decays, regardless of optimization direction

**Cautious WD**: Only decay when update and parameter have the **same sign**
```python
mask = (update * param) >= 0  # Only when they agree
update = update + param * (wd * mask)
```

**Why it matters**:
- Standard WD can hurt finetuning by pulling parameters away from the optimization direction
- Cautious WD only applies decay when it aligns with the update
- Especially important when starting from pretrained weights

**Example**:
- Parameter value: `p = 0.5`
- Gradient update: `u = -0.6` (wants to make it negative)
- Standard WD: `p = 0.5 * 0.99 - 0.6 = -0.105` (WD pulled it toward zero)
- Cautious WD: Mask is False (different signs), so WD is not applied: `p = 0.5 - 0.6 = -0.1`

---

## Why AdamW Works But Muon Doesn't

### AdamW (our finetuning)
- ✅ Adaptive step size via RMSProp (second moment)
- ✅ Per-parameter normalization
- ✅ Proven stable for finetuning

### Our Muon (basic Muon)
- ❌ Fixed step size
- ❌ No per-parameter normalization (besides shape-based, which we don't have)
- ❌ Too aggressive for finetuning

### NorMuon (train_gpt.py)
- ✅ Adaptive step size via second momentum
- ✅ Per-parameter normalization
- ✅ Cautious WD prevents interference
- ✅ Proven to work (world record)

---

## What We Thought vs Reality

### What We Thought (from "bug fixes")

> "Bug #1: Momentum formula wrong - lerp scales gradient 20x smaller"
> "Bug #2: LR scaling wrong - LR is 45x too large"
> "Fix: Change to standard momentum and correct scaling"

### Reality

1. **Lerp is intentional** in train_gpt.py (world record code)
2. **LR scaling** in train_gpt.py is `max(1, d_out/d_in)**0.5`, not `max(d_out, d_in)**0.5`
3. Our training script has **NO LR scaling at all** - this is the real bug!
4. We're missing **NorMuon features** (adaptive step size, cautious WD)

### Actual Bugs in Our Implementation

1. ❌ **Missing LR scaling** - NO shape-based LR adjustment at all
2. ❌ **Missing NorMuon features** - No second momentum, no adaptive step size
3. ❌ **Missing cautious WD** - Using standard WD that hurts optimization

---

## Evidence from Previous Tests

Looking at the 300M GSM8K finetuning results:

### AdamW Finetune (from adamw_300m_8)
- LR 1e-5: ~13.5% accuracy ✅
- Stable training, loss decreases

### Muon Finetune (from muon_300m_8)
- All LRs: ~2.3% accuracy ❌
- Barely any learning, stuck near random

**Why?**
- AdamW has adaptive step size → works
- Our Muon has no adaptive step size → fails
- Our Muon has no LR scaling → unstable
- Our Muon has standard WD → hurts optimization

---

## What Needs to Be Fixed

### Priority 1: LR Scaling (EASY)

```python
# Current (WRONG)
lr = group["lr"]
p.add_(update, alpha=-lr)

# Fixed (add shape-based scaling)
scale = max(1, p.size(-2) / p.size(-1)) ** 0.5
eff_lr = group["lr"] * scale
p.add_(update, alpha=-eff_lr)
```

### Priority 2: Cautious Weight Decay (MEDIUM)

```python
# Current (WRONG)
if weight_decay > 0:
    p.mul_(1 - lr * weight_decay)
p.add_(update, alpha=-lr)

# Fixed (cautious WD)
mask = (update * p) >= 0
update = update + p * (lr * weight_decay * mask)
p.add_(update, alpha=-lr)
```

### Priority 3: NorMuon Adaptive Step Size (HARD)

Requires:
1. Add `beta2` parameter (e.g., 0.95)
2. Add `second_momentum_buffer` to state
3. Track variance along one dimension
4. Compute adaptive step size
5. Apply normalization while preserving magnitude

See train_gpt.py lines 567-603 for full implementation.

---

## Recommendation

### Immediate Actions

1. **Add LR scaling** to train_llama_muon_single_gpu.py (5 min fix)
2. **Add cautious weight decay** (10 min fix)
3. **Test** these two fixes first before implementing full NorMuon

### If Still Not Working

4. **Implement NorMuon** adaptive step size (full feature, ~1 hour)
5. Consider using muon_optimizer_complete.py if it has these features

### Testing Strategy

Run 300M GSM8K finetuning with:
- Base model: muon_300m_8
- Optimizer: Muon with fixes
- LRs: 1e-5, 3e-5, 1e-4
- Compare to previous results (~2.3% → hopefully >10%)

---

## References

- **NorMuon paper**: https://arxiv.org/pdf/2510.05491
- **Cautious WD paper**: https://arxiv.org/abs/2510.12402
- **train_gpt.py NorMuon**: [lines 490-646](train_gpt.py#L490-L646)
- **Our Muon**: [lines 315-382](train_llama_muon_single_gpu.py#L315-L382)

Created: 2025-11-19
