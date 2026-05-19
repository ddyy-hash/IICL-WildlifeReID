#!/bin/bash
# =============================================================================
# ATRW 消融实验运行脚本（含评估）
# =============================================================================
# 用法: bash run_ablation_with_eval.sh
#
# 这个脚本会：
# 1. 运行每个消融实验的训练
# 2. 训练完成后立即运行 Closed-Set 和 Open-Set 评估
# 3. 汇总所有结果

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ATRW 消融实验开始（含评估）${NC}"
echo -e "${GREEN}========================================${NC}"

# 数据目录
DATA_DIR="./data/processed/atrw/train"

# 检查数据目录
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}错误: 数据目录不存在: $DATA_DIR${NC}"
    exit 1
fi

# =============================================================================
# 实验 1: Baseline
# =============================================================================
echo -e "\n${YELLOW}[1/5] 实验 1: Baseline${NC}"

python tools/train_joint.py \
    --config config/ablation/atrw_ablation_1_baseline.yaml \
    --data_dir "$DATA_DIR" \
    --device cuda

echo -e "${YELLOW}评估实验 1（使用 phase1 epoch 140 模型）...${NC}"
bash evaluate_ablation.sh "Baseline" "./checkpoints/ablation/atrw_baseline/joint_phase1_epoch130.pth"

echo -e "${GREEN}✓ 实验 1 完成${NC}"

# =============================================================================
# 实验 2: + IPAID
# =============================================================================
echo -e "\n${YELLOW}[2/5] 实验 2: + IPAID${NC}"

python tools/train_joint.py \
    --config config/ablation/atrw_ablation_2_ipaid.yaml \
    --data_dir "$DATA_DIR" \
    --device cuda

echo -e "${YELLOW}评估实验 2（使用 phase3 epoch 80 模型）...${NC}"
bash evaluate_ablation.sh "+ IPAID" "./checkpoints/ablation/atrw_ipaid/joint_phase3_epoch100.pth"

echo -e "${GREEN}✓ 实验 2 完成${NC}"

# =============================================================================
# 实验 3: + FGID (核心创新)
# =============================================================================
echo -e "\n${YELLOW}[3/5] 实验 3: + FGID (核心创新)${NC}"

python tools/train_joint.py \
    --config config/ablation/atrw_ablation_3_fgid.yaml \
    --data_dir "$DATA_DIR" \
    --device cuda

echo -e "${YELLOW}评估实验 3（使用 phase3 epoch 80 模型）...${NC}"
bash evaluate_ablation.sh "+ FGID" "./checkpoints/ablation/atrw_fgid/joint_phase3_epoch100.pth"

echo -e "${GREEN}✓ 实验 3 完成${NC}"

# =============================================================================
# 实验 4: + IICL
# =============================================================================
echo -e "\n${YELLOW}[4/5] 实验 4: + IICL${NC}"

python tools/train_joint.py \
    --config config/ablation/atrw_ablation_4_iicl.yaml \
    --data_dir "$DATA_DIR" \
    --device cuda

echo -e "${YELLOW}评估实验 4（使用 phase3 epoch 80 模型）...${NC}"
bash evaluate_ablation.sh "+ IICL" "./checkpoints/ablation/atrw_iicl/joint_phase3_epoch100.pth"

echo -e "${GREEN}✓ 实验 4 完成${NC}"

# =============================================================================
# 实验 5: Full Model
# =============================================================================
echo -e "\n${YELLOW}[5/5] 实验 5: Full Model${NC}"

python tools/train_joint.py \
    --config config/ablation/atrw_ablation_5_full.yaml \
    --data_dir "$DATA_DIR" \
    --device cuda

echo -e "${YELLOW}评估实验 5（使用 phase3 epoch 80 模型）...${NC}"
bash evaluate_ablation.sh "Full Model" "./checkpoints/ablation/atrw_full/joint_phase3_epoch100.pth"

echo -e "${GREEN}✓ 实验 5 完成${NC}"

# =============================================================================
# 汇总所有结果
# =============================================================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  所有消融实验完成！${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}汇总结果...${NC}"
python tools/summarize_ablation.py

echo -e "\n${GREEN}完成！查看 ablation_results.csv 获取汇总结果${NC}"
