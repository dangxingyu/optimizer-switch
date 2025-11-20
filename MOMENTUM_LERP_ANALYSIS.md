# Momentum Lerp公式分析

## 发现：原版train_gpt.py也用lerp！

### 原版NorMuon代码 (line 555-556)
```python
momentum_buffer.lerp_(grad_chunk[:num_params], 1 - group["momentum"])
updated_grads = grad_chunk[:num_params].lerp_(momentum_buffer, group["momentum"])
```

### 数学推导

假设 `momentum = 0.95`, `buf_old = 10.0`, `grad = 2.0`

**Step 1: 更新momentum buffer**
```python
buf.lerp_(grad, 1 - momentum)
```
Lerp公式: `self = self * (1-weight) + other * weight`

所以: `buf = buf * momentum + grad * (1-momentum)`
     = `10.0 * 0.95 + 2.0 * 0.05 = 9.6`

**Step 2: 计算update**
```python
update = grad.lerp_(buf, momentum)
```
所以: `update = grad * (1-momentum) + buf * momentum`
     = `2.0 * 0.05 + 9.6 * 0.95 = 9.22`

### vs 标准Momentum公式

**标准SGD-Momentum:**
```python
buf = buf * momentum + grad * 1.0  # = 11.5
update = buf  # = 11.5
```

**标准Nesterov:**
```python
buf = buf * momentum + grad * 1.0  # = 11.5
update = grad + buf * momentum  # = 2.0 + 11.5 * 0.95 = 12.925
```

### 比较

| 方法 | Momentum Buffer | Update | 相对大小 |
|------|----------------|--------|---------|
| Lerp版本 | 9.6 | 9.22 | 1.0x |
| 标准Momentum | 11.5 | 11.5 | 1.25x |
| 标准Nesterov | 11.5 | 12.925 | 1.40x |

---

## 关键问题：Lerp是Bug还是Feature？

### 论据1：可能是Bug
1. 所有早期Muon论文和实现都用标准momentum
2. 2024-10-10版本用标准公式: `buf.mul_(momentum).add_(g)`
3. Moonshot实现用标准公式
4. Lerp会让gradient贡献缩小20倍

### 论据2：可能是Feature
1. 最新的train_gpt.py（world record代码）使用lerp
2. 可能是有意让optimizer更保守/稳定
3. 配合其他tricks (NorMuon, Cautious WD) 可能work得很好
4. 如果是bug，world record不应该能work

---

## 不同版本对比

### v1: 2024-10-10 (早期Muon)
```python
buf.mul_(momentum).add_(g)
if nesterov:
    g = g.add(buf, alpha=momentum)
scale = max(g.size(0), g.size(1))**0.5
```

### v2: 2025-06-15 (Optimization Leaderboard)
```python
# muon_update函数
momentum.lerp_(grad, 1 - beta)
update = grad.lerp_(momentum, beta) if nesterov else momentum
update *= max(1, grad.size(-2) / grad.size(-1))**0.5
```

### v3: train_gpt.py (NorMuon - World Record)
```python
momentum_buffer.lerp_(grad_chunk, 1 - group["momentum"])
updated_grads = grad_chunk.lerp_(momentum_buffer, group["momentum"])
# + NorMuon variance tracking
# + Cautious weight decay
scale = max(1., param_shape[-2] / param_shape[-1]) ** 0.5
```

---

## 结论

**最新的modded-nanogpt确实用lerp！**

这意味着：
1. v2 (2025-06-15) 开始引入lerp
2. 这可能是有意的设计改进
3. Lerp让optimizer更保守，可能更适合：
   - 大规模训练
   - 需要稳定性的场景
   - 配合其他advanced tricks

但是：
- 早期论文和Moonshot都用标准momentum
- 对于finetuning，可能需要实验两种方式

---

## 建议

### 选项A：跟随最新版 (使用Lerp)
```python
# 优点：和world record代码对齐
# 缺点：更保守，可能学得慢
momentum_buffer.lerp_(grad, 1 - group["momentum"])
update_grad = grad.lerp(momentum_buffer, group["momentum"])
```

### 选项B：使用标准Momentum
```python
# 优点：和论文、Moonshot对齐，更aggressive
# 缺点：可能不如最新版稳定
momentum_buffer.mul_(group["momentum"]).add_(grad)
update_grad = grad + momentum_buffer * group["momentum"]
```

### 选项C：可配置
```python
if group.get("use_lerp_momentum", False):
    momentum_buffer.lerp_(grad, 1 - group["momentum"])
    update_grad = grad.lerp(momentum_buffer, group["momentum"])
else:
    momentum_buffer.mul_(group["momentum"]).add_(grad)
    update_grad = grad + momentum_buffer * group["momentum"]
```

---

## 我们当前的情况

**我们现在用的是**: 标准momentum (mul + add)

**原因**:
1. 我看2024-10-10版本以为那是"正确的"
2. 但实际上最新版已经改成lerp了

**下一步**:
- 可以先用标准momentum测试
- 如果效果不好，再尝试lerp版本
- 或者同时测试两种版本
