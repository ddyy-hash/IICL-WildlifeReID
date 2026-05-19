#!/usr/bin/env python3
"""Thin wrapper for ReID retrieval evaluation using the packaged JointReIDModel."""

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


def _build_model(checkpoint: Dict[str, Any], baseline: bool, device: torch.device) -> torch.nn.Module:
    from app.core.joint_model import JointReIDModel
    from app.core.model_factory import extract_config_from_checkpoint, resolve_joint_model_init

    cfg = extract_config_from_checkpoint(checkpoint)
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}

    num_classes = int(checkpoint.get("num_classes", 100))
    backbone = model_cfg.get("backbone", "osnet_ain_x1_0")
    baseline_mode = str(
        checkpoint.get("baseline_mode")
        or (cfg.get("baseline", {}) if isinstance(cfg, dict) else {}).get("type", "")
    ).lower()

    # White-box baselines in this package are still reconstructed through
    # JointReIDModel, with the illumination branch disabled in configuration.
    use_joint_builder = (not baseline) or bool(baseline_mode)

    if not use_joint_builder:
        try:
            from tools.train_baselines import BaselineReIDModel
        except (ImportError, AttributeError):
            use_joint_builder = True
        else:
            num_stripes = model_cfg.get("local_extractor", {}).get("num_parts", 6)
            model = BaselineReIDModel(
                num_classes=num_classes,
                backbone_name=backbone,
                num_stripes=num_stripes,
                pretrained_backbone=False,
            ).to(device)

    if use_joint_builder:
        model = JointReIDModel(
            **resolve_joint_model_init(
                cfg,
                num_classes=num_classes,
                backbone_override=backbone,
                pretrained_backbone=False,
            )
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
    from app.core.model_factory import extract_config_from_checkpoint, resolve_eval_input_size

    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation with the packaged JointReIDModel."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to joint_best.pth")
    parser.add_argument("--query_dir", type=str, required=True, help="Query image root directory")
    parser.add_argument("--gallery_dir", type=str, required=True, help="Gallery image root directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument(
        "--img_height",
        type=int,
        default=None,
        help="Evaluation input height (defaults to checkpoint config)",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=None,
        help="Evaluation input width (defaults to checkpoint config)",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device: auto / cpu / cuda")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Instantiate the baseline variant without the illumination branch",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override DataLoader worker count (defaults to checkpoint config)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable k-reciprocal re-ranking",
    )
    parser.add_argument("--rerank_k1", type=int, default=20, help="Re-ranking parameter k1")
    parser.add_argument("--rerank_k2", type=int, default=6, help="Re-ranking parameter k2")
    parser.add_argument(
        "--rerank_lambda",
        type=float,
        default=0.3,
        help="Re-ranking lambda parameter",
    )
    parser.add_argument(
        "--use_local_distance",
        action="store_true",
        help="Use pattern-aware global-local distance fusion from model stripe descriptors",
    )
    parser.add_argument(
        "--local_weight",
        type=float,
        default=0.35,
        help="Weight for local stripe distance in global-local distance fusion",
    )
    parser.add_argument(
        "--local_metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Metric for local stripe descriptor distance",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = _resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        checkpoint = {"state_dict": checkpoint}

    model = _build_model(checkpoint, args.baseline, device)

    cfg = extract_config_from_checkpoint(checkpoint)
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg, dict) else {}
    feature_cfg = eval_cfg.get("feature_extraction", {}) if isinstance(eval_cfg, dict) else {}
    cfg_img_h, cfg_img_w = resolve_eval_input_size(cfg)
    img_h = int(args.img_height) if args.img_height is not None else cfg_img_h
    img_w = int(args.img_width) if args.img_width is not None else cfg_img_w
    cfg_num_workers = int(cfg.get("hardware", {}).get("num_workers", 0)) if isinstance(cfg, dict) else 0
    num_workers = int(args.num_workers) if args.num_workers is not None else cfg_num_workers
    print(f"[INFO] Evaluation input size: {img_h}x{img_w}")

    evaluator = ReIDEvaluator(
        model=model,
        device=device,
        img_height=img_h,
        img_width=img_w,
        batch_size=args.batch_size,
        flip_test=bool(feature_cfg.get("flip_test", eval_cfg.get("flip_test", True))),
        rerank=bool(args.rerank),
        rerank_params={
            "k1": args.rerank_k1,
            "k2": args.rerank_k2,
            "lambda_value": args.rerank_lambda,
        },
        use_local_distance=bool(args.use_local_distance),
        local_weight=float(args.local_weight),
        local_metric=str(args.local_metric),
        num_workers=num_workers,
        exclude_same_camera=bool(eval_cfg.get("exclude_same_camera", True)),
    )

    results = evaluator.evaluate(query_dir=args.query_dir, gallery_dir=args.gallery_dir)
    if not results:
        print("[ERROR] Evaluation failed.")
        sys.exit(1)

    print("\n===== ReID Evaluation Results =====")
    print(f"Rank-1  : {results.get('rank1', 0.0):.2f}%")
    print(f"Rank-5  : {results.get('rank5', 0.0):.2f}%")
    print(f"Rank-10 : {results.get('rank10', 0.0):.2f}%")
    print(f"mAP     : {results.get('mAP', 0.0):.2f}%")
    if "rank1_seen" in results or "rank1_unseen" in results:
        print(f"Rank-1 Seen   : {results.get('rank1_seen', 0.0):.2f}%")
        print(f"Rank-1 Unseen : {results.get('rank1_unseen', 0.0):.2f}%")
    if "mAP_seen" in results or "mAP_unseen" in results:
        print(f"mAP Seen      : {results.get('mAP_seen', 0.0):.2f}%")
        print(f"mAP Unseen    : {results.get('mAP_unseen', 0.0):.2f}%")
    print("===================================")


if __name__ == "__main__":
    main()
