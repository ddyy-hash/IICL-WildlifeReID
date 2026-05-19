#!/usr/bin/env python3
"""Thin wrapper for ATRW open-set evaluation under the official protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


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


def get_test_images(test_dir: str, valid_imgids: Optional[Set[int]] = None) -> List[Tuple[str, int]]:
    """Read ATRW test images and optionally filter them by valid image id."""
    samples: List[Tuple[str, int]] = []
    for img_name in sorted(os.listdir(test_dir)):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(img_name)[0]
        if not stem.isdigit():
            continue
        imgid = int(stem)
        if valid_imgids is not None and imgid not in valid_imgids:
            continue
        samples.append((os.path.join(test_dir, img_name), imgid))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="ATRW open-set evaluation under the official protocol")
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="Path to the test-image directory (auto-detected if omitted)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="orignal_data/Amur Tiger Re-identification",
        help="ATRW data root directory",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--backbone", type=str, default="osnet_ain_x1_0", help="Backbone type")
    parser.add_argument(
        "--num_classes",
        type=int,
        default=107,
        help="Fallback number of training classes used during model construction",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission_openset.json",
        help="Path to the submission JSON file",
    )
    parser.add_argument(
        "--eval_script_dir",
        type=str,
        default="ATRWEvalScript-main",
        help="Path to the ATRW evaluation-script directory",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_height", type=int, default=256, help="Input image height")
    parser.add_argument("--img_width", type=int, default=512, help="Input image width")
    parser.add_argument("--skip_eval", action="store_true", help="Skip the final official evaluation call")
    parser.add_argument("--flip-test", dest="flip_test", action="store_true", help="Enable flip test")
    parser.add_argument("--no-flip-test", dest="flip_test", action="store_false", help="Disable flip test")
    parser.add_argument("--rerank", action="store_true", help="Apply k-reciprocal reranking before official scoring")
    parser.add_argument("--rerank_k1", type=int, default=None, help="Reranking parameter k1")
    parser.add_argument("--rerank_k2", type=int, default=None, help="Reranking parameter k2")
    parser.add_argument("--rerank_lambda", type=float, default=None, help="Reranking lambda parameter")
    parser.set_defaults(flip_test=None)
    args = parser.parse_args()

    from app.core.evaluation import (
        ReIDDataset,
        build_submission_from_distance,
        build_submission_from_features,
        evaluate_atrw_official,
        extract_features,
        load_atrw_gt,
    )
    from app.core.joint_model import JointReIDModel
    from app.core.model_factory import extract_config_from_checkpoint, resolve_eval_input_size, resolve_joint_model_init
    from tools.reranking import re_ranking

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.test_dir is None:
        test_dir = os.path.join(args.data_root, "test")
        if not os.path.exists(test_dir):
            test_dir = os.path.join(args.data_root, "atrw_reid_test", "test")
        args.test_dir = test_dir

    if not os.path.exists(args.test_dir):
        raise FileNotFoundError(f"Test directory not found: {args.test_dir}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    gt_file = os.path.join(args.eval_script_dir, "annotations", "gt_test_plain.json")
    if not os.path.exists(gt_file):
        print(f"[WARN] Ground truth file not found: {gt_file}")
        gt_file = None

    gt_data = load_atrw_gt(gt_file) if gt_file else None
    valid_imgids = set(gt_data["imgids"]) if gt_data else None
    samples = get_test_images(args.test_dir, valid_imgids)
    if not samples:
        raise RuntimeError("No valid test images were found.")
    print(f"Loaded {len(samples)} test images")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_num_classes = checkpoint.get("num_classes") if isinstance(checkpoint, dict) else None
    num_classes = int(ckpt_num_classes or args.num_classes)
    config = extract_config_from_checkpoint(checkpoint)
    cfg_img_h, cfg_img_w = resolve_eval_input_size(config)
    if args.img_height == parser.get_default("img_height"):
        args.img_height = cfg_img_h
    if args.img_width == parser.get_default("img_width"):
        args.img_width = cfg_img_w

    eval_cfg = config.get("evaluation", {}) if isinstance(config, dict) else {}
    feature_extraction_cfg = eval_cfg.get("feature_extraction", {}) if isinstance(eval_cfg, dict) else {}
    if args.flip_test is None:
        args.flip_test = bool(feature_extraction_cfg.get("flip_test", eval_cfg.get("flip_test", False)))

    rerank_params = dict(eval_cfg.get("rerank_params", {})) if isinstance(eval_cfg, dict) else {}
    if args.rerank_k1 is None:
        args.rerank_k1 = int(rerank_params.get("k1", 25))
    if args.rerank_k2 is None:
        args.rerank_k2 = int(rerank_params.get("k2", 6))
    if args.rerank_lambda is None:
        args.rerank_lambda = float(rerank_params.get("lambda_value", 0.2))
    args.rerank = bool(args.rerank or eval_cfg.get("rerank", False))

    print(f"Input size: {args.img_height}x{args.img_width}")
    print(f"Flip test: {args.flip_test}")
    if args.rerank:
        print(
            f"Reranking: enabled=True "
            f"(k1={args.rerank_k1}, k2={args.rerank_k2}, lambda={args.rerank_lambda})"
        )
    else:
        print("Reranking: enabled=False")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
    ])
    dataset = ReIDDataset(samples=samples, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if os.name != "nt" else 0,
        pin_memory=device.type == "cuda",
    )

    model = JointReIDModel(
        **resolve_joint_model_init(
            config,
            num_classes=num_classes,
            backbone_override=args.backbone,
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

    features, imgids, _, _ = extract_features(
        model,
        dataloader,
        device,
        flip_test=args.flip_test,
    )
    if args.rerank:
        distmat = re_ranking(
            features,
            features,
            k1=args.rerank_k1,
            k2=args.rerank_k2,
            lambda_value=args.rerank_lambda,
        )
        submission = build_submission_from_distance(imgids, distmat)
    else:
        submission = build_submission_from_features(imgids, features)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False)
    print(f"Submission saved to {args.output}")

    if not args.skip_eval and gt_data:
        results = evaluate_atrw_official(gt_data, submission)
        print("\n" + "=" * 60)
        print("Open-set evaluation results (official ATRW protocol)")
        print("=" * 60)
        print(f"{'Scenario':<20} {'Rank-1':<12} {'Rank-5':<12} {'mAP':<12}")
        print("-" * 60)
        print(
            f"{'Single-camera':<20} "
            f"{results['rank1_single']:>6.2f}%      "
            f"{results['rank5_single']:>6.2f}%      "
            f"{results['mAP_single']:>6.2f}%"
        )
        print(
            f"{'Cross-camera':<20} "
            f"{results['rank1_cross']:>6.2f}%      "
            f"{results['rank5_cross']:>6.2f}%      "
            f"{results['mAP_cross']:>6.2f}%"
        )
        print(f"{'mmAP':<20} {'':<12} {'':<12} {results['mmAP']:>6.2f}%")
        print("-" * 60)


if __name__ == "__main__":
    main()
