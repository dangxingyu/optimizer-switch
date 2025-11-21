# Moonlight Muon vs NorMuon Comparison

## Key Differences

### 1. Orthogonalization Method

**Moonlight (lines 49-76):**
```python
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    # Newton-Schulz iteration with quintic coefficients
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X
```
- Uses Newton-Schulz 5 (NS5) iteration
- Quintic polynomial with optimized coefficients
- `@torch.compile` decorator for optimization
- 5 iterations default

**NorMuon (train_gpt.py):**
```python
def polar_express(G):
    # Polar Express Sign Method
    # Uses 5 iterations with predefined coefficients
    # More complex with XXT kernel and ba_plus_cAA
```
- Uses Polar Express method
- Different mathematical approach
- More complex implementation

### 2. Momentum Formula

**Moonlight (lines 189-193):**
```python
buf.mul_(momentum).add_(g)        # Standard SGD momentum
if group["nesterov"]:
    g = g.add(buf, alpha=momentum)  # Nesterov
else:
    g = buf                         # Standard
```
- **Standard SGD momentum**: `buf = buf * momentum + grad`
- **Nesterov option**: `update = grad + buf * momentum`
- This is the **original Muon** approach

**NorMuon (train_gpt.py lines 555-556):**
```python
momentum_buffer.lerp_(grad, 1 - momentum)       # Lerp
updated_grads = grad.lerp(momentum_buffer, momentum)
```
- **Lerp momentum**: `buf = buf * 0.95 + grad * 0.05`
- More conservative (20x smaller gradient contribution)

### 3. LR Scaling

**Moonlight (lines 142-148):**
```python
def adjust_lr_for_muon(self, lr, param_shape):
    A, B = param_shape[:2]
    adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr
```
- Formula: `lr * 0.2 * sqrt(max(d_out, d_in))`
- For (512, 2048): `lr * 0.2 * sqrt(2048) = lr * 0.2 * 45.25 = lr * 9.05`

**NorMuon (train_gpt.py line 575):**
```python
scale = max(1., param_shape[-2] / param_shape[-1]) ** 0.5
```
- Formula: `lr * max(1, d_out/d_in)**0.5`
- For (512, 2048): `lr * max(1, 0.25)**0.5 = lr * 1.0`
- For (2048, 512): `lr * max(1, 4)**0.5 = lr * 2.0`

**Comparison:**
- Moonlight: More aggressive scaling, same for both directions
- NorMuon: Conservative, directional (depends on d_out/d_in ratio)

### 4. Adaptive Step Size

**Moonlight:**
- ❌ **NO adaptive step size**
- ❌ **NO second momentum**
- ❌ **NO variance tracking**
- Basic Muon only

**NorMuon:**
- ✅ Second momentum buffer for variance
- ✅ Adaptive step size: `1 / sqrt(variance)`
- ✅ Renormalization to preserve magnitude

### 5. Weight Decay

**Moonlight (line 200):**
```python
p.data.mul_(1 - lr * wd)  # Standard weight decay, always applied
```

**NorMuon (train_gpt.py lines 609-611):**
```python
mask = (v * p) >= 0  # Cautious: only when aligned
v.addcmul_(p, (eff_wd * mask).to(p.dtype))
```

### 6. Combined Optimizer

**Moonlight (lines 106-140):**
```python
def __init__(self, lr, wd, muon_params, adamw_params, ...):
    # Single optimizer handles both Muon and AdamW
    params = list(muon_params) + list(adamw_params)
    super().__init__(params, defaults)
```
- Single optimizer class
- Internal state tracks `use_muon` flag
- AdamW backup for non-2D params

**NorMuon:**
- Separate optimizers: `Muon()` and `AdamW()`
- Two optimizer instances
- Two `.step()` calls

## Summary Table

| Feature | Moonlight | NorMuon | Notes |
|---------|-----------|---------|-------|
| **Orthogonalization** | Newton-Schulz 5 | Polar Express | Different methods |
| **Momentum** | Standard SGD | Lerp | Moonlight 20x more aggressive |
| **Nesterov** | Optional ✅ | No ❌ | Moonlight has it |
| **LR Scaling** | `0.2 * sqrt(max(A,B))` | `max(1, A/B)**0.5` | Very different! |
| **Adaptive Step Size** | No ❌ | Yes ✅ | NorMuon only |
| **Second Momentum** | No ❌ | Yes ✅ | NorMuon only |
| **Cautious WD** | No ❌ | Yes ✅ | NorMuon only |
| **Combined Optimizer** | Yes ✅ | No ❌ | Moonlight cleaner |
| **torch.compile** | Yes ✅ | No ❌ | Moonlight faster |

## Which to Use?

### Use Moonlight if:
- ✅ Want simpler, more standard Muon
- ✅ Want Nesterov momentum option
- ✅ Want single unified optimizer
- ✅ Want faster orthogonalization (torch.compile)
- ✅ Training from scratch (pretraining)

### Use NorMuon if:
- ✅ Want adaptive step size (like Adam)
- ✅ Finetuning pretrained models
- ✅ Want cautious weight decay
- ✅ Following world-record train_gpt.py

### For Finetuning GSM8K:

**Previous results:**
- Basic Muon: ~2.3% (failed)
- AdamW: 13.5% (worked)

**Hypothesis:**
- Moonlight = Basic Muon + Nesterov + different scaling
- May still fail without adaptive step size
- NorMuon has adaptive step size like Adam → should work

**Recommendation:**
1. **Try NorMuon first** (we just implemented it)
2. If it works, great!
3. If it doesn't, **then try Moonlight**
4. Moonlight might work if the aggressive LR scaling compensates

## Implementation Plan

If switching to Moonlight:

1. Copy `zeropower_via_newtonschulz5` from moonlight_train.py
2. Copy `Muon` class from moonlight_train.py
3. Update parameter separation logic
4. Remove separate AdamW optimizer
5. Single optimizer handles both

Changes needed:
- Replace Polar Express with Newton-Schulz
- Change momentum formula (lerp → standard)
- Change LR scaling formula
- Remove NorMuon features (second momentum, cautious WD)
- Combine Muon + AdamW into single optimizer
