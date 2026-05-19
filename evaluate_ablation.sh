#!/bin/bash
# =============================================================================
# 消融实验评估脚本
# =============================================================================
# 用法: bash evaluate_ablation.sh <experiment_name> <checkpoint_path>
#
# 这个脚本会：
# 1. 运行 Closed-Set 评估（animals_701协议）
# 2. 运行 Open-Set 评估（官方协议）
# 3. 汇总结果到 JSON 文件

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 参数
EXP_NAME=$1
CHECKPOINT=$2

if [ -z "$EXP_NAME" ] || [ -z "$CHECKPOINT" ]; then
    echo -e "${RED}用法: bash evaluate_ablation.sh <experiment_name> <checkpoint_path>${NC}"
    echo "示例: bash evaluate_ablation.sh baseline ./checkpoints/ablation/atrw_baseline/model_best.pth"
    exit 1
fi

# 检查checkpoint是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}错误: Checkpoint不存在: $CHECKPOINT${NC}"
    exit 1
fi

OUTPUT_DIR=$(dirname "$CHECKPOINT")

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  评估实验: $EXP_NAME${NC}"
echo -e "${GREEN}  Checkpoint: $CHECKPOINT${NC}"
echo -e "${GREEN}========================================${NC}"

# =============================================================================
# 1. Closed-Set 评估
# =============================================================================
echo -e "\n${YELLOW}[1/2] 运行 Closed-Set 评估（animals_701协议）${NC}"

CLOSEDSET_OUTPUT="$OUTPUT_DIR/submission_closedset.json"

python tools/eval_atrw_closedset.py \
    --checkpoint "$CHECKPOINT" \
    --protocol animals_701 \
    --output "$CLOSEDSET_OUTPUT" \
    --batch_size 64

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Closed-Set 评估完成${NC}"
else
    echo -e "${RED}✗ Closed-Set 评估失败${NC}"
    exit 1
fi

# =============================================================================
# 2. Open-Set 评估
# =============================================================================
echo -e "\n${YELLOW}[2/2] 运行 Open-Set 评估（官方协议）${NC}"

OPENSET_OUTPUT="$OUTPUT_DIR/submission_openset.json"

python tools/eval_atrw_openset.py \
    --checkpoint "$CHECKPOINT" \
    --output "$OPENSET_OUTPUT" \
    --batch_size 64

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Open-Set 评估完成${NC}"
else
    echo -e "${RED}✗ Open-Set 评估失败${NC}"
    exit 1
fi

# =============================================================================
# 3. 汇总结果
# =============================================================================
echo -e "\n${YELLOW}[3/3] 汇总评估结果${NC}"

# 从评估脚本的输出中提取结果（假设评估脚本会打印结果）
# 这里需要根据实际的评估脚本输出格式来解析

# 创建结果JSON
RESULTS_FILE="$OUTPUT_DIR/results.json"

# 注意：这里需要根据实际的评估脚本输出来提取数据
# 暂时创建一个占位符
cat > "$RESULTS_FILE" << EOF
{
  "experiment": "$EXP_NAME",
  "checkpoint": "$CHECKPOINT",
  "closedset": {
    "submission": "$CLOSEDSET_OUTPUT",
    "note": "Run eval_atrw_closedset.py to get metrics"
  },
  "openset": {
    "submission": "$OPENSET_OUTPUT",
    "note": "Run eval_atrw_openset.py to get metrics"
  }
}
EOF

echo -e "${GREEN}✓ 结果已保存到: $RESULTS_FILE${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  评估完成！${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}结果文件:${NC}"
echo "  - Closed-Set: $CLOSEDSET_OUTPUT"
echo "  - Open-Set: $OPENSET_OUTPUT"
echo "  - 汇总: $RESULTS_FILE"

echo -e "\n${YELLOW}查看详细结果:${NC}"
echo "  cat $RESULTS_FILE"
