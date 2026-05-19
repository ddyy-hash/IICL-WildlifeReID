# 配置文件对比：illumination_config_atrw_v2.yaml vs atrw_ablation_5_full.yaml

## 📊 对比结果

### ✅ 核心算法配置：完全一致

| 配置项 | V2配置 | Full配置 | 一致? |
|--------|--------|----------|-------|
| **光照模块启用** | true | true | ✅ |
| **IPAID: use_sensitivity** | true | true | ✅ |
| **IPAID: use_refinement** | true | true | ✅ |
| **FGID: use_feature_guided** | true | true | ✅ |
| **彩色光照** | true | true | ✅ |
| **IICL启用** | true | true | ✅ |
| **IICL权重** | 1.0 | 1.0 | ✅ |
| **Batch size** | 20 | 20 | ✅ |
| **Learning rate** | 0.00035 | 0.00035 | ✅ |
| **训练阶段** | 3阶段 | 3阶段 | ✅ |
| **损失函数权重** | 完全一致 | 完全一致 | ✅ |

### ⚠️ 非核心配置：有差异（不影响算法）

| 配置项 | V2配置 | Full配置 | 影响 |
|--------|--------|----------|------|
| **output_dir** | `./checkpoints/atrw_v2` | `./checkpoints/ablation/atrw_full` | ❌ 不影响算法 |
| **wandb.project** | `tiger-reid-v2-enhanced` | `tiger-reid-ablation-full` | ❌ 不影响算法 |
| **注释内容** | V2版本说明 | 消融实验说明 | ❌ 不影响算法 |

---

## 🎯 结论

### ✅ 两个配置在**算法层面完全一致**

**相同的部分（核心）：**
- ✅ 所有模块开关（IPAID, FGID, IICL）
- ✅ 所有超参数（学习率、batch size、权重）
- ✅ 训练策略（三阶段）
- ✅ 损失函数配置
- ✅ 数据增强策略
- ✅ 评估协议

**不同的部分（非核心）：**
- ⚠️ 输出目录路径（只是保存位置不同）
- ⚠️ Wandb项目名（只是日志标识不同）
- ⚠️ 注释内容（只是说明文字不同）

### 📝 这意味着什么？

1. **实验5确实可以使用V2的训练结果**
   - 因为算法配置完全一致
   - 只是保存路径不同

2. **但需要注意：**
   - V2模型保存在：`./checkpoints/atrw_v2/`
   - 如果运行实验5，会保存到：`./checkpoints/ablation/atrw_full/`
   - 这会重复训练，浪费时间

3. **建议做法：**
   - **不运行实验5**
   - 直接使用V2的结果作为实验5的结果
   - 在汇总时，从`./checkpoints/atrw_v2/`读取结果

---

## 🔧 修正建议

### 方案1：修改实验5配置，指向V2的输出目录（推荐）

```yaml
# 修改 atrw_ablation_5_full.yaml
output_dir: "./checkpoints/atrw_v2"  # 改为V2的目录
```

这样汇总脚本就能直接读取V2的结果。

### 方案2：在汇总脚本中特殊处理实验5

```python
# 在 summarize_ablation.py 中
if exp_name == 'Full Model':
    # 从V2目录读取结果
    results = load_results('./checkpoints/atrw_v2')
else:
    results = load_results(exp_dir)
```

### 方案3：创建符号链接（最简单）

```bash
# 创建符号链接
mkdir -p checkpoints/ablation
ln -s ../atrw_v2 checkpoints/ablation/atrw_full
```

这样实验5的路径会指向V2的实际目录。

---

## ✅ 验证

运行以下命令验证配置一致性：

```bash
# 对比关键配置
diff <(grep -v "^#" config/illumination_config_atrw_v2.yaml | grep -v "output_dir" | grep -v "project:") \
     <(grep -v "^#" config/ablation/atrw_ablation_5_full.yaml | grep -v "output_dir" | grep -v "project:")
```

如果输出为空或只有空白行差异，说明算法配置完全一致。

---

## 📋 总结

| 问题 | 答案 |
|------|------|
| **算法配置是否一致？** | ✅ 完全一致 |
| **可以用V2结果代替实验5吗？** | ✅ 可以 |
| **需要重新训练实验5吗？** | ❌ 不需要 |
| **差异会影响消融实验吗？** | ❌ 不影响 |

**建议：**
1. 不运行实验5
2. 修改汇总脚本，从`./checkpoints/atrw_v2/`读取实验5的结果
3. 或者创建符号链接：`ln -s ../atrw_v2 checkpoints/ablation/atrw_full`

这样可以节省8-10小时的训练时间！
