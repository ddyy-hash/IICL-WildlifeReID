#!/usr/bin/env python3
"""ATRW closed-set evaluation helper.

Supported protocols:
1) ``train70_val30``: identity-level 70:30 split inside the training set
2) ``animals_701``: Animals-2024-style evaluation on the 701 ``query=="sing"`` images
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_atrw_train_data(data_root: str) -> Tuple[List[Tuple[str, int]], Dict[int, List[int]]]:
    """Load ATRW training annotations and return image samples plus id indices."""
    train_dir = os.path.join(data_root, "train")
    anno_file = os.path.join(data_root, "reid_list_train.csv")

    if not os.path.exists(train_dir):
        train_dir = os.path.join(data_root, "atrw_reid_train", "train")
    if not os.path.exists(anno_file):
        anno_file = os.path.join(data_root, "atrw_anno_reid_train", "reid_list_train.csv")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(anno_file):
        raise FileNotFoundError(f"Annotation file not found: {anno_file}")

    filename_to_id: Dict[str, int] = {}
    with open(anno_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("entityid"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                filename_to_id[parts[1]] = int(parts[0])

    samples: List[Tuple[str, int]] = []
    id_to_samples: Dict[int, List[int]] = defaultdict(list)

    for img_name in sorted(os.listdir(train_dir)):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        if img_name not in filename_to_id:
            continue

        entity_id = filename_to_id[img_name]
        img_path = os.path.join(train_dir, img_name)
        sample_idx = len(samples)
        samples.append((img_path, entity_id))
        id_to_samples[entity_id].append(sample_idx)

    return samples, id_to_samples


def split_train_val(
    samples: Sequence[Tuple[str, int]],
    id_to_samples: Dict[int, List[int]],
    train_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], int]:
    """Split each identity into train/val partitions and remap labels to contiguous ids."""
    random.seed(seed)
    np.random.seed(seed)

    train_samples: List[Tuple[str, int]] = []
    val_samples: List[Tuple[str, int]] = []

    unique_ids = sorted(id_to_samples.keys())
    id_to_label = {entity_id: i for i, entity_id in enumerate(unique_ids)}

    for entity_id, indices in id_to_samples.items():
        local_indices = list(indices)
        random.shuffle(local_indices)
        label = id_to_label[entity_id]

        n_train = max(1, int(len(local_indices) * train_ratio))
        for idx_pos, sample_idx in enumerate(local_indices):
            img_path, _ = samples[sample_idx]
            if idx_pos < n_train:
                train_samples.append((img_path, label))
            else:
                val_samples.append((img_path, label))

    return train_samples, val_samples, len(unique_ids)


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


def _resolve_test_dir(data_root: str) -> str:
    candidate_1 = os.path.join(data_root, "test")
    candidate_2 = os.path.join(data_root, "atrw_reid_test", "test")
    if os.path.exists(candidate_1):
        return candidate_1
    if os.path.exists(candidate_2):
        return candidate_2
    raise FileNotFoundError(
        "ATRW test directory not found. Tried: "
        f"{candidate_1} | {candidate_2}"
    )


def prepare_train70_val30_samples(
    data_root: str,
    train_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    """Prepare samples for the train70_val30 protocol."""
    samples, id_to_samples = load_atrw_train_data(data_root)
    _, val_samples, _ = split_train_val(samples, id_to_samples, train_ratio, seed)
    entities = len(set(int(label) for _, label in val_samples))
    info = {
        "total_train_images": len(samples),
        "total_train_entities": len(id_to_samples),
        "eval_images": len(val_samples),
        "eval_entities": entities,
    }
    return val_samples, info


def prepare_animals701_samples(
    data_root: str,
    eval_script_dir: str,
) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    """Prepare samples for the animals_701 protocol (``query == 'sing'``)."""
    gt_file = os.path.join(eval_script_dir, "annotations", "gt_test_plain.json")
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Official GT file not found: {gt_file}")

    test_dir = _resolve_test_dir(data_root)
    with open(gt_file, "r", encoding="utf-8") as f:
        gt_annotations = json.load(f)

    if not isinstance(gt_annotations, list):
        raise ValueError(f"GT file has an unexpected format (expected a list): {gt_file}")

    query_sing = [x for x in gt_annotations if str(x.get("query", "")).strip().lower() == "sing"]
    if not query_sing:
        raise RuntimeError("No samples with query=='sing' were found in the GT file.")

    imgid_to_path: Dict[int, str] = {}
    for img_name in sorted(os.listdir(test_dir)):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem, _ = os.path.splitext(img_name)
        if not stem.isdigit():
            continue
        imgid_to_path[int(stem)] = os.path.join(test_dir, img_name)

    samples: List[Tuple[str, int]] = []
    missing_count = 0
    for item in sorted(query_sing, key=lambda x: int(x["imgid"])):
        imgid = int(item["imgid"])
        entity_id = int(item["entityid"])
        img_path = imgid_to_path.get(imgid)
        if img_path is None:
            missing_count += 1
            continue
        samples.append((img_path, entity_id))

    if not samples:
        raise RuntimeError(
            "Failed to build the animals_701 protocol: no valid images were matched."
        )

    info = {
        "expected_images": len(query_sing),
        "expected_entities": len(set(int(x["entityid"]) for x in query_sing)),
        "eval_images": len(samples),
        "eval_entities": len(set(int(x[1]) for x in samples)),
        "missing_images": missing_count,
    }
    return samples, info


def build_model(
    checkpoint: Dict[str, Any],
    device: torch.device,
    backbone_name: str,
    fallback_num_classes: int,
) -> torch.nn.Module:
    """Build the model and load checkpoint weights."""
    from app.core.joint_model import JointReIDModel
    from app.core.model_factory import extract_config_from_checkpoint, resolve_joint_model_init

    ckpt_num_classes = checkpoint.get("num_classes") if isinstance(checkpoint, dict) else None
    num_classes = int(ckpt_num_classes or fallback_num_classes)

    config = extract_config_from_checkpoint(checkpoint)
    model = JointReIDModel(
        **resolve_joint_model_init(
            config,
            num_classes=num_classes,
            backbone_override=backbone_name,
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


def evaluate_samples(
    model: torch.nn.Module,
    samples: Sequence[Tuple[str, int]],
    device: torch.device,
    batch_size: int,
    distance_metric: str,
    img_height: int,
    img_width: int,
    flip_test: bool = False,
) -> Dict[str, float]:
    """Run all-vs-all closed-set evaluation on the provided sample pool."""
    from app.core.evaluation import ReIDDataset, compute_cmc_map, compute_distance_matrix, extract_features

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])
    dataset = ReIDDataset(samples=list(samples), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if os.name == "nt" else 4,
        pin_memory=device.type == "cuda",
    )

    features, labels, _, paths = extract_features(model, loader, device, flip_test=flip_test)
    print(f"Feature shape: {features.shape}")

    distmat = compute_distance_matrix(features, features, metric=distance_metric)
    np.fill_diagonal(distmat, np.inf)

    cmc, m_ap = compute_cmc_map(
        distmat,
        labels,
        labels,
        query_cams=None,
        gallery_cams=None,
        query_paths=paths,
        gallery_paths=paths,
        max_rank=10,
        exclude_same_camera=False,
    )

    return {
        "rank1": float(cmc[0] * 100),
        "rank5": float(cmc[min(4, len(cmc) - 1)] * 100),
        "rank10": float(cmc[min(9, len(cmc) - 1)] * 100),
        "mAP": float(m_ap * 100),
    }


def main() -> Dict[str, float]:
    parser = argparse.ArgumentParser(description="ATRW Closed-Set Evaluation")
    parser.add_argument("--data_root", type=str, default="orignal_data/Amur Tiger Re-identification", help="ATRW data root directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--backbone", type=str, default="osnet_ain_x1_0", help="Backbone type")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_height", type=int, default=256, help="Input image height")
    parser.add_argument("--img_width", type=int, default=512, help="Input image width")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (affects the 70:30 split)")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training-set ratio")
    parser.add_argument(
        "--protocol",
        type=str,
        default="train70_val30",
        choices=["train70_val30", "animals_701"],
        help="Evaluation protocol: train70_val30 or animals_701",
    )
    parser.add_argument(
        "--eval_script_dir",
        type=str,
        default="ATRWEvalScript-main",
        help="Directory containing the ATRW official evaluation script (required for animals_701)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=107,
        help="Fallback number of classes when the checkpoint does not store num_classes",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON path for saving evaluation results",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Protocol: {args.protocol}")
    print(f"Input size: {args.img_height}x{args.img_width}")

    print("\n" + "=" * 60)
    print("Step 1: Prepare evaluation samples")
    print("=" * 60)
    if args.protocol == "animals_701":
        eval_samples, data_info = prepare_animals701_samples(args.data_root, args.eval_script_dir)
        print(f"Expected: ~{data_info['expected_images']} images / {data_info['expected_entities']} entities")
        print(f"Matched samples: {data_info['eval_images']} images / {data_info['eval_entities']} entities")
        if data_info["missing_images"] > 0:
            print(f"[WARN] GT images missing from the test directory: {data_info['missing_images']}")
        distance_metric = "euclidean"
    else:
        eval_samples, data_info = prepare_train70_val30_samples(args.data_root, args.train_ratio, args.seed)
        print(
            f"Training-set total: {data_info['total_train_images']} images / "
            f"{data_info['total_train_entities']} entities"
        )
        print(f"Validation samples: {data_info['eval_images']} images / {data_info['eval_entities']} entities")
        print(f"Split ratio: {int(args.train_ratio * 100)}:{int((1 - args.train_ratio) * 100)}, seed={args.seed}")
        distance_metric = "cosine"

    print("\n" + "=" * 60)
    print("Step 2: Load the model")
    print("=" * 60)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        checkpoint = {"state_dict": checkpoint}
    from app.core.model_factory import extract_config_from_checkpoint, resolve_eval_input_size
    cfg_img_h, cfg_img_w = resolve_eval_input_size(extract_config_from_checkpoint(checkpoint))
    if args.img_height == parser.get_default("img_height"):
        args.img_height = cfg_img_h
    if args.img_width == parser.get_default("img_width"):
        args.img_width = cfg_img_w
    print(f"Resolved input size: {args.img_height}x{args.img_width}")
    model = build_model(
        checkpoint=checkpoint,
        device=device,
        backbone_name=args.backbone,
        fallback_num_classes=args.num_classes,
    )

    print("\n" + "=" * 60)
    print("Step 3: Extract features and evaluate")
    print("=" * 60)
    results = evaluate_samples(
        model=model,
        samples=eval_samples,
        device=device,
        batch_size=args.batch_size,
        distance_metric=distance_metric,
        img_height=args.img_height,
        img_width=args.img_width,
    )

    print("\n" + "=" * 60)
    if args.protocol == "animals_701":
        print("Closed-set evaluation results (animals_701, aligned with the Animals-2024 protocol)")
    else:
        print("Closed-set evaluation results (train70_val30, legacy-compatible)")
    print("=" * 60)
    print(f"  Rank-1:  {results['rank1']:.2f}%")
    print(f"  Rank-5:  {results['rank5']:.2f}%")
    print(f"  Rank-10: {results['rank10']:.2f}%")
    print(f"  mAP:     {results['mAP']:.2f}%")

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        payload: Dict[str, Any] = {
            "protocol": args.protocol,
            "checkpoint": args.checkpoint,
            "backbone": args.backbone,
            "data_root": args.data_root,
            "eval_script_dir": args.eval_script_dir,
            "batch_size": args.batch_size,
            "img_height": args.img_height,
            "img_width": args.img_width,
            "num_classes_fallback": args.num_classes,
            "dataset_info": data_info,
            "results": results,
            "paper_reference": {
                "source": "animals-14-01106 (Animals 2024)",
                "target_metrics": {
                    "rank1": 96.3,
                    "rank5": 98.9,
                    "mAP": 78.7,
                },
            },
        }
        if args.protocol == "animals_701":
            payload["paper_delta"] = {
                "rank1": results["rank1"] - 96.3,
                "rank5": results["rank5"] - 98.9,
                "mAP": results["mAP"] - 78.7,
            }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
