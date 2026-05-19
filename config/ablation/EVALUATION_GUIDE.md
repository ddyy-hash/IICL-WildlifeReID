# 消融实验评估说明

## ⚠️ 重要：使用统一的epoch模型进行公平对比

### 为什么要用epoch 100的模型？

**公平性原则：**
- ✅ 所有实验都训练130 epochs（15+15+100）
- ✅ 都在第100 epoch结束训练
- ✅ 使用相同epoch的模型确保公平对比
- ❌ 不能用各自的best模型（可能在不同epoch）

**模型路径：**
```
实验1: ./checkpoints/ablation/atrw_baseline/model_epoch100.pth
实验2: ./checkpoints/ablation/atrw_ipaid/model_epoch100.pth
实验3: ./checkpoints/ablation/atrw_fgid/model_epoch100.pth
实验4: ./checkpoints/ablation/atrw_iicl/model_epoch100.pth
实验5: ./checkpoints/atrw_v2/joint_phase3_epoch100.pth
```

---

## 📊 评估流程

### 1. 训练完成后的评估

每个实验训练完成后，立即运行评估：

```bash
# 评估实验1（Baseline）
bash evaluate_ablation.sh "Baseline" "./checkpoints/ablation/atrw_baseline/model_epoch100.pth"
```

### 2. 评估脚本会做什么？

`evaluate_ablation.sh` 会调用两个评估脚本：

**Closed-Set 评估：**
```bash
python tools/eval_atrw_closedset.py \
    --checkpoint <model_path> \
    --protocol animals_701 \
    --output submission_closedset.json
```

**Open-Set 评估：**
```bash
python tools/eval_atrw_openset.py \
    --checkpoint <model_path> \
    --output submission_openset.json
```

### 3. 评估结果保存

评估完成后，结果保存在：
```
checkpoints/ablation/<exp_name>/
├── model_epoch100.pth              # 训练的模型
├── submission_closedset.json       # Closed-Set提交文件
├── submission_openset.json         # Open-Set提交文件
└── results.json                    # 汇总的评估结果
```

---

## 🎯 关键指标

### 论文中使用的指标：

**主要指标（Open-Set Single-camera）：**
- Rank-1
- Rank-5
- **mAP** ⭐（最重要）

**次要指标（Closed-Set）：**
- Rank-1
- Rank-5
- mAP

**FGID贡献 = 实验3的mAP - 实验2的mAP**

---

## 📋 完整运行流程

### 方法1：自动运行（推荐）

```bash
# 运行所有实验（含训练+评估）
bash run_ablation_with_eval.sh
```

这个脚本会：
1. 训练实验1 → 评估实验1（epoch 100）
2. 训练实验2 → 评估实验2（epoch 100）
3. 训练实验3 → 评估实验3（epoch 100）
4. 训练实验4 → 评估实验4（epoch 100）
5. 评估实验5（使用V2的epoch 100模型）
6. 汇总所有结果

### 方法2：手动运行

```bash
# 1. 训练
python tools/train_joint.py --config config/ablation/atrw_ablation_1_baseline.yaml --data_dir ./data/processed/atrw/train --device cuda

# 2. 评估（使用epoch 100模型）
bash evaluate_ablation.sh "Baseline" "./checkpoints/ablation/atrw_baseline/model_epoch100.pth"

# 3. 重复上述步骤，完成所有实验
```

### 方法3：只评估（如果已经训练好）

```bash
# 评估所有实验（假设模型已训练）
bash evaluate_ablation.sh "Baseline" "./checkpoints/ablation/atrw_baseline/model_epoch100.pth"
bash evaluate_ablation.sh "+ IPAID" "./checkpoints/ablation/atrw_ipaid/model_epoch100.pth"
bash evaluate_ablation.sh "+ FGID" "./checkpoints/ablation/atrw_fgid/model_epoch100.pth"
bash evaluate_ablation.sh "+ IICL" "./checkpoints/ablation/atrw_iicl/model_epoch100.pth"
bash evaluate_ablation.sh "Full Model" "./checkpoints/atrw_v2/joint_phase3_epoch100.pth"

# 汇总结果
python tools/summarize_ablation_v2.py
```

---

## 📊 结果汇总

### 汇总脚本

```bash
# 使用更新版的汇总脚本（支持真实评估结果）
python tools/summarize_ablation_v2.py
```

### 预期输出

```
================================================================================
  消融实验结果汇总（基于真实评估）
================================================================================

✓ Baseline              | Rank-1:  82.30% | mAP:  65.20%
✓ + IPAID               | Rank-1:  85.70% | mAP:  68.50%
✓ + FGID                | Rank-1:  87.20% | mAP:  70.10%
✓ + IICL                | Rank-1:  88.10% | mAP:  71.20%
✓ Full Model            | Rank-1:  88.45% | mAP:  71.73%

--------------------------------------------------------------------------------
  增量分析
--------------------------------------------------------------------------------

+ IPAID vs Baseline: Δ mAP = +3.30%
+ FGID vs + IPAID: Δ mAP = +1.60%
  ⭐ FGID贡献: +1.60%
  ⚠️  可以投，但风险较高（录用概率30-35%）
+ IICL vs + FGID: Δ mAP = +1.10%
Full Model vs + IICL: Δ mAP = +0.53%
```

---

## ⚠️ 注意事项

### 1. 模型命名

确保训练脚本保存epoch 100的模型为：
- `model_epoch100.pth` 或
- `joint_phase3_epoch100.pth`（V2模型）

### 2. 评估协议

- **Closed-Set**: 使用 `animals_701` 协议（与Animals-2024论文对齐）
- **Open-Set**: 使用官方协议（Single-camera + Cross-camera）

### 3. 结果文件格式

`results.json` 应该包含：
```json
{
  "experiment": "Baseline",
  "checkpoint": "./checkpoints/ablation/atrw_baseline/model_epoch100.pth",
  "closedset": {
    "rank1": 98.15,
    "rank5": 99.57,
    "mAP": 81.38
  },
  "openset": {
    "single_camera": {
      "rank1": 88.45,
      "rank5": 96.86,
      "mAP": 71.73
    },
    "cross_camera": {
      "rank1": 77.33,
      "rank5": 90.22,
      "mAP": 42.78
    },
    "overall_mAP": 67.28
  }
}
```

---

## 🎯 决策标准

| FGID贡献 | 决策 | ACM MM录用概率 |
|---------|------|---------------|
| ≥ 2.0% | ✅ 投ACM MM | 40-45% |
| 1.5-2.0% | ⚠️ 可以投，但风险高 | 30-35% |
| < 1.5% | ❌ 不建议投ACM MM | <25% |

---

## 📝 论文表格

使用epoch 100的结果制作消融表格：

```latex
\begin{table}[t]
\centering
\caption{Ablation Study on ATRW Dataset (Open-Set Single-camera, Epoch 100)}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
Method & Rank-1 & Rank-5 & mAP \\
\midrule
Baseline (OSNet-AIN) & 82.3\% & 94.1\% & 65.2\% \\
+ IPAID & 85.7\% & 95.8\% & 68.5\% \\
+ FGID & 87.2\% & 96.3\% & 70.1\% \\
+ IICL & 88.1\% & 96.7\% & 71.2\% \\
Full Model & \textbf{88.45\%} & \textbf{96.86\%} & \textbf{71.73\%} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## ✅ 总结

1. ✅ **统一使用epoch 100模型**（公平对比）
2. ✅ **调用专门的评估脚本**（真实性能）
3. ✅ **保存完整的评估结果**（Closed-Set + Open-Set）
4. ✅ **关注Open-Set Single-camera的mAP**（论文主要指标）
5. ✅ **FGID贡献是决策关键**（是否投ACM MM）

**现在可以开始运行实验了！**

```bash
bash run_ablation_with_eval.sh
```

**3天后查看FGID的贡献，做最终决策！**
