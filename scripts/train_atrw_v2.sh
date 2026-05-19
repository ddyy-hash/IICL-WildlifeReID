#!/bin/bash
# ATRW V2 增强版训练脚本
# 解决颜色偏移和细节丢失问题

set -e

echo "=========================================="
echo "ATRW V2 Enhanced Training Pipeline"
echo "=========================================="

# 配置
CONFIG="config/illumination_config_atrw_v2.yaml"
OUTPUT_DIR="checkpoints/atrw_v2_enhanced"
DEVICE="cuda"

# 1. 训练模型
echo ""
echo "[1/3] 开始训练..."
python tools/train_joint_reid.py \
  --config ${CONFIG} \
  --device ${DEVICE}

# 2. 可视化对比效果
echo ""
echo "[2/3] 生成可视化对比..."
python tools/visualize_color_correction_v2.py \
  --checkpoint ${OUTPUT_DIR}/joint_best.pth \
  --input_dir data/processed/atrw/query \
  --output_dir ${OUTPUT_DIR}/visualization \
  --num_examples 8 \
  --device ${DEVICE}

# 3. 完整分析
echo ""
echo "[3/3] 生成完整分析报告..."
python tools/visualize_joint_analysis.py \
  --checkpoint ${OUTPUT_DIR}/joint_best.pth \
  --query_dir data/processed/atrw/query \
  --gallery_dir data/processed/atrw/gallery \
  --output_dir ${OUTPUT_DIR}/analysis \
  --device ${DEVICE}

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "检查点: ${OUTPUT_DIR}/joint_best.pth"
echo "可视化: ${OUTPUT_DIR}/visualization/"
echo "分析报告: ${OUTPUT_DIR}/analysis/"
echo ""
echo "对比 V1 vs V2 效果："
echo "  V1: checkpoints/atrw_new_version_1/analysis_joint_best/"
echo "  V2: ${OUTPUT_DIR}/analysis/"
