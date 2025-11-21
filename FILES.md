# 文件说明

## 核心训练脚本

### train_llama_muon_single_gpu.py
- **主训练脚本** - 同时支持 AdamW 和 Moonlight Muon 优化器
- 通过 `--finetune_opt` 参数选择优化器：`adamw` 或 `moonlight_muon`
- 用于 SBATCH 批量实验

### 参考实现（仅供参考）

- **train_llama_adamw_single_gpu.py** - 原始 AdamW 实现
- **moonlight_train.py** - Moonlight Muon 参考实现

## SBATCH 脚本

### run_gsm8k_comprehensive_sweep.sbatch
- **主实验脚本** - 96个任务的完整 LR sweep
- 测试 6 个基础模型 × 2 个优化器 × 6 个学习率
- 基础模型：adamw_130m_{1,8}, muon_130m_{1,8}, adamw_300m_1, muon_300m_1
- 学习率：1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 2e-3

## 分析脚本

### analyze_comprehensive_sweep.py
- 从所有实验 logs 中提取结果
- 生成对比表格
- 保存为 CSV

### show_sweep_plan.py
- 显示实验计划
- 查看任务映射

## 工具模块

- **llama_model.py** - LLaMA 模型定义
- **eval_utils.py** - 评估工具
- **qa_data_utils.py** - GSM8K 数据处理

## 文档

- **COMPREHENSIVE_SWEEP.md** - 完整实验说明
- **README.md** - 项目总体说明

## Archive 目录

旧的脚本、分析和文档都在 `archive/` 目录下：
- `archive/old_scripts/` - 旧训练脚本
- `archive/old_sbatch/` - 旧 SBATCH 脚本
- `archive/old_analysis/` - 旧分析脚本
- `archive/old_docs/` - 旧文档
- `archive/analysis_results/` - 之前的 300m_8 分析结果

## 使用方法

### 提交实验
```bash
sbatch run_gsm8k_comprehensive_sweep.sbatch
```

### 查看计划
```bash
python3 show_sweep_plan.py
```

### 分析结果（实验完成后）
```bash
python3 analyze_comprehensive_sweep.py
```
