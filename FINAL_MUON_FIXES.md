# Muon Optimizer 最终修复总结

## 发现并修复的3个Bug

### Bug #1: Momentum公式错误 ❌
**错误**: 使用`lerp`导致gradient被缩小20倍
```python
# 错误 (之前的实现)
momentum_buffer.lerp_(grad, 1 - group["momentum"])
# 等价于: buf = buf * 0.95 + grad * 0.05

# 正确 (修复后)
momentum_buffer.mul_(group["momentum"]).add_(grad)
# 等价于: buf = buf * 0.95 + grad * 1.0
```

### Bug #2: Learning Rate Scaling错误 - 版本混淆 ❌
**问题**: 使用了旧版本 (2024-10-10) 的scaling公式

```python
# 错误 (2024-10-10旧版)
scale = max(p.size(-2), p.size(-1)) ** 0.5
# 对于 (512, 2048): scale = 45.25

# 正确 (2025-06-15最新版)
scale = max(1, p.size(-2) / p.size(-1)) ** 0.5
# 对于 (512, 2048): scale = 1.0
```

**影响**: 旧版scaling使得LR **大了45倍**！导致finetuning不稳定。

### Bug #3: Nesterov实现不完整 ⚠️
**之前**: 我们只实现了Nesterov，但没有检查配置
```python
# 固定使用Nesterov
update_grad = grad + momentum_buffer * group["momentum"]
```

**现在**: 应该根据配置选择
```python
# 根据配置决定
if group.get("nesterov", True):
    update_grad = grad + momentum_buffer * group["momentum"]  # Nesterov
else:
    update_grad = momentum_buffer  # Standard momentum
```

---

## 版本演进历史

### Timeline

| 版本 | 日期 | Scaling公式 | 特点 |
|------|------|------------|------|
| v1 朴素版 | 2024-10-10 | `max(d_out, d_in)**0.5` | 用于pretraining from scratch |
| v2 KellerJordan | 2025-06-15 | `max(1, d_out/d_in)**0.5` | 最新版，更平衡 |
| v3 MuP | - | `(d_out/d_in)**0.5` | Maximal Update Parametrization |
| v4 Moonshot | - | `0.2 * max(d_out, d_in)**0.5` | 专为finetuning优化 |

### Scaling公式对比

假设参数shape = (512, 2048):

| 版本 | 公式 | Scale值 | 相对于v2 |
|------|------|---------|---------|
| v1 (旧版) | `max(512, 2048)**0.5` | 45.25 | **45x** ⚠️ |
| v2 (最新) | `max(1, 512/2048)**0.5` | 1.0 | 1x ✅ |
| v3 (MuP) | `(512/2048)**0.5` | 0.5 | 0.5x |
| v4 (Moonshot) | `0.2 * 45.25` | 9.05 | 9x |

---

## 为什么之前Muon Finetune完全失败？

### 组合效应分析

1. **Momentum Bug**: Gradient被缩小 **20倍**
2. **Scaling Bug**: LR被放大 **45倍**
3. **净效应**: `45 / 20 = 2.25x`

虽然净效应只是2.25倍，但：
- Gradient方向受momentum bug影响，可能不准确
- 过大的scaling导致不稳定
- 两个bug相互作用，产生unpredictable行为

**结果**: Accuracy停留在 ~2.3%，完全没学习！

---

## 最终正确的实现

### Single GPU版本
```python
def step_single_gpu(self):
    for group in self.param_groups:
        params = group["params"]
        if not params:
            continue

        # 1. Calculate scaling (最新版公式)
        p_example = params[0]
        scale = max(1, p_example.size(-2) / p_example.size(-1)) ** 0.5
        eff_lr_val = group["lr"] * scale
        eff_weight_decay_val = group["lr"] * group["weight_decay"]

        update_grads_list = []
        params_list = []

        for param in params:
            if param.grad is None:
                continue

            grad = param.grad
            state = self.state[param]

            if not state:
                state["momentum_buffer"] = torch.zeros_like(grad)

            momentum_buffer = state["momentum_buffer"]

            # 2. Momentum update (正确公式)
            momentum_buffer.mul_(group["momentum"]).add_(grad)

            # 3. Nesterov acceleration (if enabled)
            if group.get("nesterov", True):
                update_grad = grad + momentum_buffer * group["momentum"]
            else:
                update_grad = momentum_buffer

            update_grads_list.append(update_grad)
            params_list.append(param)

        if not update_grads_list:
            continue

        # 4. Batched orthogonalization
        batched_update_grads = torch.stack(update_grads_list)

        # Handle attn reshaping
        if getattr(params_list[0], 'label', None) == 'attn':
            original_shape = batched_update_grads.shape
            batch = 4 * original_shape[0]
            d1 = original_shape[1]
            d2 = original_shape[2] // 4
            batched = batched_update_grads.view(batch, d1, d2)
            v_batch = polar_express(batched)
            v_batch = v_batch.view(original_shape)
        else:
            v_batch = polar_express(batched_update_grads)

        # 5. Apply updates
        for i, param in enumerate(params_list):
            # Weight decay
            param.mul_(1 - eff_weight_decay_val)
            # Update
            param.add_(v_batch[i], alpha=-eff_lr_val)
```

---

## 与其他实现的对比

### vs Moonshot

**相同**:
- ✅ Momentum公式
- ✅ Nesterov实现
- ✅ Weight decay处理

**不同**:
- ⚠️ Moonshot用 `0.2 * max(d_out, d_in)**0.5`（更保守）
- ⚠️ 我们用 `max(1, d_out/d_in)**0.5`（KellerJordan最新版）

**权衡**:
- Moonshot的公式可能更适合finetuning（更保守）
- KellerJordan的公式更通用，但可能需要调整LR

### vs KellerJordan (2025-06-15)

**完全一致** ✅:
- ✅ Momentum: `buf * m + grad`
- ✅ Nesterov: `grad + buf * m`
- ✅ Scaling: `max(1, d_out/d_in)**0.5`
- ✅ Orthogonalization方法不同（但数学上等价）
  - KellerJordan: Newton-Schulz 5
  - 我们: Polar Express (更新的算法)

---

## 测试建议

### 1. 使用修复后的实现 + 相同的LR
先测试momentum和scaling bug修复后的效果，看是否有改善。

### 2. 如果还是不理想
考虑添加Moonshot的0.2系数：
```python
scale = 0.2 * max(p_example.size(-2), p_example.size(-1)) ** 0.5
```

或者直接降低base LR（等价）。

### 3. LR调整建议

**如果使用当前实现** (KellerJordan scaling):
- 对于finetuning，可能需要比pretraining更小的LR
- 建议范围: `1e-5` 到 `1e-4`

**如果添加0.2系数** (Moonshot scaling):
- 可以用稍大的LR
- 建议范围: `5e-5` 到 `5e-4`

---

## 修复清单 ✅

- [x] Bug #1: Momentum公式修复
- [x] Bug #2: Scaling公式更新到最新版
- [x] Bug #3: 添加Nesterov配置支持
- [x] Single GPU版本修复
- [x] Distributed版本修复
- [x] 语法检查通过
- [x] 创建详细文档
- [ ] 重新运行300M测试验证

---

## 下一步

建议重新运行300M的GSM8K finetuning测试：
- Base models: adamw_300m_8, muon_300m_8
- Finetune optimizers: adamw, muon
- Learning rates: 建议用相同的范围测试，因为我们现在的scaling更接近rational

预期：Muon finetune的accuracy应该从 ~2.3% 显著提升！

修复日期: 2025-11-19
