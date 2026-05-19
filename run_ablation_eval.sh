#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${CONFIG_PATH:-config/illumination_config_atrw.yaml}"
DATA_DIR="${DATA_DIR:-data/processed/atrw/train}"
DATA_ROOT="${DATA_ROOT:-orignal_data/Amur Tiger Re-identification}"
EVAL_SCRIPT_DIR="${EVAL_SCRIPT_DIR:-ATRWEvalScript-main}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/ablation/atrw_main}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VARIANTS="${VARIANTS:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" tools/run_atrw_main_ablation.py \
  --config "$CONFIG_PATH" \
  --data_dir "$DATA_DIR" \
  --data_root "$DATA_ROOT" \
  --eval_script_dir "$EVAL_SCRIPT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --num_workers "$NUM_WORKERS" \
  --variants "$VARIANTS"
