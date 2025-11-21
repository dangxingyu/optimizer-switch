# Muon Optimizer Bug Fixes

## 发现的两个严重Bug

### Bug #1: Momentum公式错误 ❌
**位置**: `muon_optimizer_complete.py` lines 426, 576

**错误的实现**:
```python
# 使用lerp，将gradient缩小了20倍！
momentum_buffer.lerp_(grad, 1 - group["momentum"])
# 等价于: buf = buf * 0.95 + grad * 0.05
```

**正确的实现**:
```python
# 原版Muon的标准momentum公式
momentum_buffer.mul_(group["momentum"]).add_(grad)
# 等价于: buf = buf * 0.95 + grad * 1.0
```

**影响**: Gradient被缩小了20倍 (1/0.05 = 20x)，导致学习极慢！

---

### Bug #2: Learning Rate Scaling公式错误 ❌
**位置**: `muon_optimizer_complete.py` lines 406, 539

**错误的实现**:
```python
# 错误的scaling公式
scale = max(1, p.size(-2) / p.size(-1)) ** 0.5
# 对于 (512, 2048) 的矩阵: scale = max(1, 0.25)**0.5 = 1.0
```

**正确的实现**:
```python
# 原版Muon的scaling公式
scale = max(p.size(-2), p.size(-1)) ** 0.5
# 对于 (512, 2048) 的矩阵: scale = 2048**0.5 = 45.25
```

**影响**: Learning rate被缩小了几十倍！对于常见的矩阵shape，差异可达45倍以上。

---

## 完整的正确算法

**原版Muon (2024-10-10_Muon/train_gpt2.py):**
```python
# 1. Momentum update
buf = buf * momentum + grad

# 2. Nesterov acceleration
if nesterov:
    g = grad + buf * momentum
else:
    g = buf

# 3. Orthogonalization (polar_express)
g = zeropower(g)

# 4. Learning rate scaling
scale = max(g.size(0), g.size(1)) ** 0.5

# 5. Parameter update
p = p - lr * scale * g
```

**修复后的实现 ✅:**
```python
# Single GPU版本和Distributed版本都已修复
# step_single_gpu() - lines 392-467
# step() (distributed) - lines 469-666

# 1. Momentum (line 434, 583)
momentum_buffer.mul_(group["momentum"]).add_(grad)

# 2. Nesterov (line 439, 588)
update_grad = grad + momentum_buffer * group["momentum"]

# 3. Orthogonalization (line 459, 614)
v_batch = polar_express(batched_update_grads)

# 4. Scaling (line 406, 539)
scale = max(p_example.size(-2), p_example.size(-1)) ** 0.5
eff_lr_val = group["lr"] * scale

# 5. Update (line 466, 625)
param.add_(v_batch[i], alpha=-eff_lr_val)
```

---

## 为什么之前的实现学不动？

1. **Momentum bug**: Gradient被缩小20倍
2. **Scaling bug**: Learning rate被再次缩小几十倍
3. **组合效果**: 实际学习率可能被缩小了 **上千倍**！

这解释了为什么：
- Muon finetune的accuracy停留在~2.3% (几乎没学习)
- AdamW finetune却能达到13.5% (正常学习)

---

## 验证测试

运行测试验证momentum公式：
```bash
python3 << 'EOF'
momentum = 0.95
grad = 2.0
buf_old = 10.0

# 正确的公式
buf_new_correct = buf_old * momentum + grad  # = 11.5

# 错误的lerp公式
buf_new_wrong = buf_old * momentum + grad * (1-momentum)  # = 9.6

print(f"正确: {buf_new_correct:.2f}")
print(f"错误: {buf_new_wrong:.2f}")
print(f"差异: {buf_new_correct / buf_new_wrong:.2f}x")
EOF
```

---

## 下一步

1. ✅ 两个bug都已修复
2. ⏳ 需要重新运行300M测试验证修复效果
3. ⏳ 预期Muon finetune的accuracy会显著提升

修复日期: 2025-11-19
