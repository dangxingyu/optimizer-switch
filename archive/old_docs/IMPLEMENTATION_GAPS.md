# Implementation Gaps: train_llama_muon_single_gpu.py vs train_gpt.py (NorMuon)

## Executive Summary

Our current training script uses a **basic Muon** implementation, while train_gpt.py uses **NorMuon** (Normalized Muon with adaptive step size). There are several critical missing features.

---

## Gap #1: Missing NorMuon Variance Tracking ⚠️ CRITICAL

### What train_gpt.py has (lines 567-603):

```python
# Second momentum buffer for variance tracking
if "second_momentum_buffer" not in group:
    group["second_momentum_buffer"] = (torch.zeros_like(updated_grads[..., :, :1])
        if param_shape[-2] >= param_shape[-1] else torch.zeros_like(updated_grads[..., :1, :])
    )
second_momentum_buffer = group["second_momentum_buffer"]

# After polar_express orthogonalization:
v_chunk = polar_express(updated_grads)

# NorMuon adaptive step size (https://arxiv.org/pdf/2510.05491)
v_norm = v_chunk.norm(dim=(-2, -1), keepdim=True)
v_mean = v_chunk.square().mean(dim=-1 if param_shape[-2] >= param_shape[-1] else -2, keepdim=True)
second_momentum_buffer.lerp_(v_mean.to(dtype=ref_param.dtype), 1 - group["beta2"])
step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
v_chunk.mul_(step_size)
v_norm_new = v_chunk.norm(dim=(-2, -1), keepdim=True)
v_chunk.mul_(v_norm / v_norm_new.clamp_min_(1e-10))
```

### What we have (lines 363-380):

```python
# NO second momentum buffer
# NO variance tracking
# NO adaptive step size

# Just basic orthogonalization:
if p.ndim >= 2:
    update = polar_express(update_grad)
else:
    update = update_grad

# Direct parameter update
p.add_(update, alpha=-lr)
```

**Impact**: NorMuon's adaptive step size provides per-parameter normalization that can significantly improve stability and convergence, especially for finetuning.

---

## Gap #2: Missing Cautious Weight Decay ⚠️ CRITICAL

### What train_gpt.py has (lines 609-611):

```python
# "Cautious" weight decay (https://arxiv.org/abs/2510.12402)
mask = (v_chunk * param_chunk) >= 0
v_chunk.addcmul_(param_chunk, (eff_wd * mask).to(ref_param.dtype))
```

**Explanation**: Only apply weight decay when the update and parameter have the **same sign**. This prevents weight decay from interfering with the optimization direction.

### What we have (line 369-370):

```python
# Standard weight decay
if weight_decay > 0:
    p.mul_(1 - lr * weight_decay)
```

**Impact**: Standard weight decay can hurt finetuning performance by pulling parameters away from the optimization direction.

---

## Gap #3: Missing LR Scaling ⚠️ IMPORTANT

### What train_gpt.py has (lines 573-587):

```python
if "param_lr" not in group:
    group["param_lr"] = (
        max(1., param_shape[-2] / param_shape[-1]) ** 0.5
        * ref_param.new_tensor(
            [getattr(param, "lr_mul", 1.0) for param in params[module_idx:module_idx + num_params]]
        ).view(-1, 1, 1)
    )

    group["param_wd"] = ref_param.new_tensor(
        [getattr(param, "wd_mul", 1.0) for param in params[module_idx:module_idx + num_params]]
    ).view(-1, 1, 1)

# Determine LR and WD
eff_lr = group["lr"] * group["param_lr"]
eff_wd = group["lr"] * group["weight_decay"] * group["param_wd"]
```

**Features**:
1. Scales LR by `max(1, d_out/d_in)**0.5`
2. Supports per-parameter `lr_mul` multipliers
3. Supports per-parameter `wd_mul` multipliers

### What we have (line 346-380):

```python
# Fixed LR for all parameters
lr = group["lr"]
weight_decay = group["weight_decay"]

# NO per-parameter scaling
# NO lr_mul support
# NO wd_mul support
```

**Impact**: Without proper LR scaling, different layer shapes get inconsistent effective learning rates.

---

## Gap #4: Lerp vs Standard Momentum ⚠️ DESIGN CHOICE

### What train_gpt.py has (lines 555-556):

```python
momentum_buffer.lerp_(grad_chunk[:num_params], 1 - group["momentum"])
updated_grads = grad_chunk[:num_params].lerp_(momentum_buffer, group["momentum"])
```

**Effect**:
- `buf = buf * 0.95 + grad * 0.05`
- `update = grad * 0.05 + buf * 0.95`
- More conservative, gradient contribution is 20x smaller

### What we have (lines 363-366):

```python
# Standard momentum (from our "bug fix")
momentum_buffer.lerp_(grad, 1 - momentum)
update_grad = grad.lerp(momentum_buffer, momentum)
```

**Wait, we're using lerp too!** But our documentation said we "fixed" it to standard momentum. Let me check...

Actually looking at line 363-366, we ARE using lerp in our training script. So this is not a gap.

**However**, the muon_optimizer_complete.py was "fixed" to use standard momentum. This creates an inconsistency.

---

## Gap #5: Missing Beta2 Parameter

### What train_gpt.py has:

```python
# NorMuon needs beta2 for second momentum
group["beta2"]  # Used in line 598
```

### What we have:

No beta2 parameter defined in the optimizer.

**Impact**: Cannot implement NorMuon without beta2.

---

## Gap #6: Missing Attention Reshaping

### What train_gpt.py has (lines 559-563):

```python
if params[module_idx].label == 'attn':
    # Reshape attn params from [hdim, dim*4] to [4,hdim,dim]
    for p in params[module_idx:module_idx + num_params]:
        assert p.label == 'attn'
    updated_grads = updated_grads.view(4 * grad_shape[0], grad_shape[1], grad_shape[2] // 4)
```

**Purpose**: For attention projection matrices (QKV combined), reshape to orthogonalize each projection separately.

### What we have:

No special handling for attention parameters.

**Impact**: Attention matrices may not be orthogonalized properly.

---

## Gap #7: Batched Processing

### What train_gpt.py has:

All parameters in a group are processed together:
```python
# Stack all grads
stacked_grads = torch.stack([p.grad for p in params])

# Single batched polar_express call
v_chunk = polar_express(updated_grads)

# Batched parameter updates
```

### What we have:

Loop over parameters one by one:
```python
for p in group["params"]:
    # Process each parameter individually
    update = polar_express(update_grad)
```

**Impact**: Less efficient, more memory fragmentation.

---

## Summary Table

| Feature | train_gpt.py (NorMuon) | Our Implementation | Priority |
|---------|------------------------|-------------------|----------|
| Momentum | Lerp | Lerp ✅ | N/A |
| LR Scaling | `max(1, d_out/d_in)**0.5` | None ❌ | HIGH |
| Second Momentum (beta2) | Yes ✅ | No ❌ | **CRITICAL** |
| Adaptive Step Size | Yes ✅ | No ❌ | **CRITICAL** |
| Cautious Weight Decay | Yes ✅ | No ❌ | **CRITICAL** |
| Per-param lr_mul | Yes ✅ | No ❌ | MEDIUM |
| Per-param wd_mul | Yes ✅ | No ❌ | MEDIUM |
| Attention Reshaping | Yes ✅ | No ❌ | LOW |
| Batched Processing | Yes ✅ | No ❌ | LOW |

---

## Recommendation

The **three critical missing features** for NorMuon are:

1. **Second momentum buffer + adaptive step size** (NorMuon paper)
2. **Cautious weight decay** (prevents WD from hurting optimization)
3. **LR scaling** (ensures consistent effective LR across layers)

Without these, we're using **basic Muon**, not **NorMuon**. This explains why:
- AdamW finetuning works (it has adaptive step size via Adam's second moment)
- Muon finetuning fails (no adaptive step size, standard WD hurts optimization)

---

## Next Steps

### Option A: Implement Full NorMuon (Recommended)

Add the three critical features to our training script:
1. Second momentum buffer (beta2)
2. NorMuon adaptive step size calculation
3. Cautious weight decay
4. LR scaling

### Option B: Test Basic Muon First

Keep current implementation but:
1. Fix LR scaling (add `max(1, d_out/d_in)**0.5`)
2. Try cautious weight decay
3. See if this alone helps

### Option C: Use muon_optimizer_complete.py

But it was "fixed" to use standard momentum instead of lerp, which differs from train_gpt.py.

---

## Code References

- **train_gpt.py NorMuon**: [lines 490-646](train_gpt.py#L490-L646)
- **Our Muon class**: [lines 315-382](train_llama_muon_single_gpu.py#L315-L382)
- **NorMuon paper**: https://arxiv.org/pdf/2510.05491
- **Cautious WD paper**: https://arxiv.org/abs/2510.12402
