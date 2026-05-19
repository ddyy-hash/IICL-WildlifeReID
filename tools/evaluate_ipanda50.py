#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            state_dict = checkpoint
        if isinstance(state_dict, dict):
            return state_dict
    raise ValueError("Checkpoint does not contain a valid state dict (model_state_dict/state_dict).")


def _build_model(checkpoint: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    from app.core.joint_model import JointReIDModel

    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}

    num_classes = int(checkpoint.get("num_classes", 50))
    backbone = model_cfg.get("backbone", "osnet_ain_x1_0")
    num_stripes = model_cfg.get("local_extractor", {}).get("num_parts", 6)
    dropout = model_cfg.get("local_extractor", {}).get("dropout", 0.0)
    use_ipaid = bool(model_cfg.get("illumination_module", {}).get("enabled", True))

    model = JointReIDModel(
        num_classes=num_classes,
        backbone_name=backbone,
        num_stripes=num_stripes,
        pretrained_backbone=False,
        soft_mask_temperature=10.0,
        soft_mask_type="sigmoid",
        use_ipaid=use_ipaid,
        dropout=dropout,
    ).to(device)

    state_dict = _extract_state_dict(checkpoint)
    load_result = model.load_state_dict(state_dict, strict=False)
    missing = getattr(load_result, "missing_keys", [])
    unexpected = getattr(load_result, "unexpected_keys", [])
    if missing:
        print(f"[WARN] Missing parameters: {len(missing)}")
    if unexpected:
        print(f"[WARN] Ignored parameters: {len(unexpected)}")

    return model


def main() -> None:
    from app.core.evaluation import ReIDEvaluator

    parser = argparse.ArgumentParser(description="iPanda50 official evaluation protocol")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--test_dir", type=str, default=None, help="iPanda50 test directory (official protocol)")
    parser.add_argument("--query_dir", type=str, default=None, help="Query directory (legacy compatibility)")
    parser.add_argument("--gallery_dir", type=str, default=None, help="Gallery directory (legacy compatibility)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--img_size", type=int, default=256, help="Image size")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cpu/cuda")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    device = _resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        checkpoint = {"state_dict": checkpoint}

    model = _build_model(checkpoint, device)

    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg, dict) else {}
    feature_cfg = eval_cfg.get("feature_extraction", {}) if isinstance(eval_cfg, dict) else {}
    num_workers = int(cfg.get("hardware", {}).get("num_workers", 0)) if isinstance(cfg, dict) else 0

    evaluator = ReIDEvaluator(
        model=model,
        device=device,
        img_height=args.img_size,
        img_width=args.img_size,
        batch_size=args.batch_size,
        flip_test=bool(feature_cfg.get("flip_test", eval_cfg.get("flip_test", True))),
        rerank=False,
        num_workers=num_workers,
    )

    results = None
    eval_target = None
    if args.test_dir:
        eval_target = args.test_dir
    elif args.query_dir and args.gallery_dir:
        inferred_test = os.path.join(os.path.dirname(args.query_dir), "test")
        if os.path.isdir(inferred_test):
            eval_target = inferred_test
        else:
            print("[WARN] No test directory found; falling back to query/gallery evaluation.")
            results = evaluator.evaluate(args.query_dir, args.gallery_dir)
    else:
        raise ValueError("Please specify --test_dir, or provide both --query_dir and --gallery_dir.")

    if eval_target is not None:
        results = evaluator.evaluate_ipanda50(eval_target)

    if not results:
        print("[ERROR] Evaluation failed")
        sys.exit(1)

    print("\n===== iPanda50 Evaluation Results =====")
    print(f"Rank-1  : {results.get('rank1', 0.0):.2f}%")
    print(f"Rank-5  : {results.get('rank5', 0.0):.2f}%")
    print(f"Rank-10 : {results.get('rank10', 0.0):.2f}%")
    print(f"mAP     : {results.get('mAP', 0.0):.2f}%")
    print("=======================================")

    output_file = os.path.join(os.path.dirname(args.checkpoint), "ipanda50_results.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("iPanda50 Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        if eval_target:
            f.write(f"Test Dir: {eval_target}\n")
        else:
            f.write(f"Query Dir: {args.query_dir}\n")
            f.write(f"Gallery Dir: {args.gallery_dir}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Rank-1  : {results.get('rank1', 0.0):.2f}%\n")
        f.write(f"Rank-5  : {results.get('rank5', 0.0):.2f}%\n")
        f.write(f"Rank-10 : {results.get('rank10', 0.0):.2f}%\n")
        f.write(f"mAP     : {results.get('mAP', 0.0):.2f}%\n")

    print(f"[INFO] Results saved to: {output_file}")


if __name__ == "__main__":
    main()
