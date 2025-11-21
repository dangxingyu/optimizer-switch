# Muon实现对比分析

## 三个版本对比

### 1. Moonshot版本
### 2. Modded-NanoGPT原版 (2024-10-10_Muon)
### 3. 我们的实现 (muon_optimizer_complete.py)

---

## 关键差异对比

### A. Momentum更新

**Moonshot:**
```python
buf.mul_(momentum).add_(g)                    # buf = buf * momentum + grad
if nesterov:
    g = g.add(buf, alpha=momentum)            # g = grad + buf * momentum
else:
    g = buf                                    # g = buf
```

**Modded-NanoGPT:**
```python
buf.mul_(momentum).add_(g)                    # buf = buf * momentum + grad
if group['nesterov']:
    g = g.add(buf, alpha=momentum)            # g = grad + buf * momentum
```

**我们的实现 (修复后):**
```python
momentum_buffer.mul_(group["momentum"]).add_(grad)  # buf = buf * momentum + grad
# Nesterov
update_grad = grad + momentum_buffer * group["momentum"]
```

✅ **一致性**: 三个版本完全相同！

---

### B. Learning Rate Scaling

**Moonshot:**
```python
def adjust_lr_for_muon(self, lr, param_shape):
    A, B = param_shape[:2]
    adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr
```
即: `lr_eff = lr * 0.2 * sqrt(max(dim0, dim1))`

**Modded-NanoGPT:**
```python
scale = max(g.size(0), g.size(1))**0.5
p.data.add_(g, alpha=-lr * scale)
```
即: `lr_eff = lr * sqrt(max(dim0, dim1))`

**我们的实现 (修复后):**
```python
scale = max(p_example.size(-2), p_example.size(-1)) ** 0.5
eff_lr_val = group["lr"] * scale
param.add_(v_batch[i], alpha=-eff_lr_val)
```
即: `lr_eff = lr * sqrt(max(dim0, dim1))`

⚠️ **差异**: Moonshot多了一个**0.2的系数**！

---

### C. Weight Decay应用

**Moonshot:**
```python
# apply weight decay
p.data.mul_(1 - lr * wd)
# apply update
p.data.add_(u, alpha=-adjusted_lr)
```
即: weight decay **在update之前**应用

**Modded-NanoGPT:**
```python
# 没有显式的weight decay
# (可能在optimizer构造时通过其他方式处理)
```

**我们的实现:**
```python
# Apply weight decay
param.mul_(1 - eff_weight_decay_val)
# Apply zeropower update
param.add_(v_batch[i], alpha=-eff_lr_val)
```
其中: `eff_weight_decay_val = lr * weight_decay`

✅ **一致性**: Moonshot和我们的实现一致（先decay再update）

---

### D. Orthogonalization (Zeropower)

**Moonshot:**
```python
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X
```

**Modded-NanoGPT:**
```python
# 使用zeropower_backends字典，可能是NewtonSchulz或其他backend
g = zeropower_backend(g, steps=group['backend_steps'])
```

**我们的实现:**
```python
def polar_express(G: torch.Tensor):
    # Polar Express方法，不是Newton-Schulz
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
    # Perform iterations with polar_express_coeffs
    for a, b, c in polar_express_coeffs:
        XXT(X, out=A)
        ba_plus_cAA(A, alpha=c, beta=b, out=B)
        aX_plus_BX(X, B, X, beta=a, out=C)
        X, C = C, X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
```

⚠️ **差异**:
- Moonshot/Modded-NanoGPT: **Newton-Schulz 5** iteration
- 我们的实现: **Polar Express** method (更高级的算法)

---

## 总结差异

| 特性 | Moonshot | Modded-NanoGPT | 我们的实现 | 一致性 |
|------|----------|----------------|-----------|--------|
| Momentum公式 | `buf*m + g` | `buf*m + g` | `buf*m + g` | ✅ 完全一致 |
| Nesterov | `g + buf*m` | `g + buf*m` | `g + buf*m` | ✅ 完全一致 |
| LR Scaling | `lr * 0.2 * √max(d)` | `lr * √max(d)` | `lr * √max(d)` | ⚠️ Moonshot多0.2系数 |
| Weight Decay | `p*(1-lr*wd)` | 无 | `p*(1-lr*wd)` | ✅ 有WD的一致 |
| Orthogonalization | Newton-Schulz 5 | NewtonSchulz | Polar Express | ⚠️ 算法不同 |
| Batched Processing | 逐个参数 | 逐个参数 | Batched | ⚠️ 我们batched |

---

## 关键发现

### 1. ✅ Momentum和Nesterov完全正确
我们修复后的实现与Moonshot和原版完全一致！

### 2. ⚠️ LR Scaling差异 - **这可能是关键！**

让我计算实际的LR差异：

假设参数shape = (512, 2048):
- **Moonshot**: `lr * 0.2 * √2048 = lr * 0.2 * 45.25 = lr * 9.05`
- **我们/原版**: `lr * √2048 = lr * 45.25`

**我们的effective LR比Moonshot大5倍！**

这可能解释为什么我们的Muon finetuning学不动：
- 我们用的base LR可能太大了（相对于Moonshot的scaling）
- 或者说，Moonshot的0.2系数是专门为finetuning调整的

### 3. ⚠️ Polar Express vs Newton-Schulz

- **Polar Express**: 更新、更稳定的orthogonalization算法
- **Newton-Schulz**: 经典算法
- 两者都能work，但可能有微小的数值差异

### 4. ✅ Weight Decay处理正确

---

## 建议修改

### Option 1: 添加Moonshot的0.2系数（推荐用于finetuning）

```python
# 在step_single_gpu和distributed step中
scale = 0.2 * max(p_example.size(-2), p_example.size(-1)) ** 0.5
eff_lr_val = group["lr"] * scale
```

### Option 2: 保持原版scaling（推荐用于pretraining）

```python
# 保持现状
scale = max(p_example.size(-2), p_example.size(-1)) ** 0.5
eff_lr_val = group["lr"] * scale
```

**关键问题**: Moonshot的代码注释说"We believe it may not work well for finetuning pretrained models"，
但他们的实现却添加了0.2的保守系数，这可能正是为了让finetuning work！

---

## 测试建议

1. **先测试当前修复**: 看momentum和scaling bug修复后的效果
2. **如果还是不work**: 尝试添加0.2系数
3. **调整base LR**: Moonshot用`lr=1e-3`，我们用的LR可能需要相应调整

---

## 代码对齐建议

为了完全对齐Moonshot的实现，建议：

```python
def step_single_gpu(self):
    for group in self.param_groups:
        params = group["params"]
        if not params:
            continue

        # Use Moonshot's scaling for finetuning
        p_example = params[0]
        A, B = p_example.size(-2), p_example.size(-1)
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))  # Moonshot's formula
        eff_lr_val = group["lr"] * adjusted_ratio

        eff_weight_decay_val = group["lr"] * group["weight_decay"]

        for param in params:
            if param.grad is None:
                continue

            grad = param.grad
            state = self.state[param]

            if not state:
                state["momentum_buffer"] = torch.zeros_like(grad)

            momentum_buffer = state["momentum_buffer"]

            # Standard SGD momentum (same as Moonshot)
            momentum_buffer.mul_(group["momentum"]).add_(grad)

            # Nesterov (same as Moonshot)
            if group.get("nesterov", True):
                update_grad = grad.add(momentum_buffer, alpha=group["momentum"])
            else:
                update_grad = momentum_buffer

            # Orthogonalization
            # (use polar_express or switch to Newton-Schulz)

            # Apply weight decay (same as Moonshot)
            param.mul_(1 - eff_weight_decay_val)

            # Apply update (same as Moonshot)
            param.add_(orthogonalized_update, alpha=-eff_lr_val)
```
