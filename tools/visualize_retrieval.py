#!/usr/bin/env python3
"""Visualize ReID retrieval results for query/gallery folders."""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.core.evaluation import ReIDDataset, extract_features


def _import_joint_model_class():
    try:
        from app.core.joint_model import JointReIDModel
    except ModuleNotFoundError as exc:
        if exc.name == "torchreid":
            raise ModuleNotFoundError(
                "Missing dependency 'torchreid'. Activate the same environment where "
                "tools/visualize_color_correction_v2.py works (for example conda env 'dog_train'), "
                "or install it via: pip install torchreid"
            ) from exc
        raise
    return JointReIDModel


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            state_dict = checkpoint
        if isinstance(state_dict, dict):
            return state_dict
    raise ValueError("checkpoint does not contain model_state_dict/state_dict")


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    device: torch.device,
    fallback_backbone: str,
    fallback_num_classes: int,
):
    JointReIDModel = _import_joint_model_class()

    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    illum_cfg_model = model_cfg.get("illumination_module", {}) if isinstance(model_cfg, dict) else {}
    illum_cfg_top = cfg.get("illumination_module", {}) if isinstance(cfg, dict) else {}
    local_cfg = model_cfg.get("local_extractor", {}) if isinstance(model_cfg, dict) else {}

    num_classes = int(checkpoint.get("num_classes", fallback_num_classes))
    backbone = model_cfg.get("backbone", fallback_backbone)
    num_stripes = int(local_cfg.get("num_parts", 6))
    dropout = float(local_cfg.get("dropout", 0.0))

    if "enabled" in illum_cfg_model:
        use_ipaid = bool(illum_cfg_model.get("enabled", True))
    else:
        module_type = str(illum_cfg_top.get("module_type", "IPAIDModule")).lower()
        use_ipaid = module_type not in {"none", "disabled", "null"}

    ipaid_params = illum_cfg_model.get("module_params")
    if not ipaid_params:
        ipaid_params = illum_cfg_top.get("module_params", {})

    model = JointReIDModel(
        num_classes=num_classes,
        backbone_name=backbone,
        num_stripes=num_stripes,
        pretrained_backbone=False,
        soft_mask_temperature=10.0,
        soft_mask_type="sigmoid",
        use_ipaid=use_ipaid,
        dropout=dropout,
        ipaid_params=ipaid_params,
    ).to(device)

    state_dict = _extract_state_dict(checkpoint)
    load_ret = model.load_state_dict(state_dict, strict=False)
    missing = getattr(load_ret, "missing_keys", [])
    unexpected = getattr(load_ret, "unexpected_keys", [])
    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")

    model.eval()
    return model, cfg


def _build_loader(root: str, img_height: int, img_width: int, batch_size: int, device: torch.device):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
        ]
    )
    dataset = ReIDDataset(root=root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    return dataset, loader


def compute_distance_matrix(query_features: np.ndarray, gallery_features: np.ndarray) -> np.ndarray:
    query_features = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-12)
    gallery_features = gallery_features / (np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-12)
    return 1.0 - np.dot(query_features, gallery_features.T)


def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _sample_query_indices(
    success_indices: Sequence[int],
    failure_indices: Sequence[int],
    num_queries: int,
    rng: np.random.Generator,
) -> Tuple[List[int], int, int]:
    total_available = len(success_indices) + len(failure_indices)
    num_queries = min(num_queries, total_available)

    num_success = min(num_queries // 2, len(success_indices))
    num_failure = min(num_queries - num_success, len(failure_indices))

    remaining = num_queries - num_success - num_failure
    if remaining > 0:
        add_success = min(remaining, len(success_indices) - num_success)
        num_success += add_success
        remaining -= add_success
    if remaining > 0:
        add_failure = min(remaining, len(failure_indices) - num_failure)
        num_failure += add_failure

    selected: List[int] = []
    if num_success > 0:
        selected.extend(rng.choice(success_indices, size=num_success, replace=False).tolist())
    if num_failure > 0:
        selected.extend(rng.choice(failure_indices, size=num_failure, replace=False).tolist())

    random.shuffle(selected)
    return selected, num_success, num_failure


def visualize_retrieval_results(
    query_paths: List[str],
    gallery_paths: List[str],
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    query_pids: List[Any],
    gallery_pids: List[Any],
    output_dir: str,
    top_k: int = 10,
    num_queries: int = 20,
    exclude_same_image: bool = True,
    seed: int = 42,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if len(query_paths) == 0:
        raise ValueError("query set is empty")
    if len(gallery_paths) == 0:
        raise ValueError("gallery set is empty")

    print("[INFO] Computing distance matrix...")
    distmat = compute_distance_matrix(query_features, gallery_features)

    gallery_paths_norm = [_normalize_path(p) for p in gallery_paths]

    def ranked_indices_for_query(q_idx: int) -> np.ndarray:
        ranked = np.argsort(distmat[q_idx])
        if not exclude_same_image:
            return ranked

        q_path = _normalize_path(query_paths[q_idx])
        filtered = [g_idx for g_idx in ranked if gallery_paths_norm[g_idx] != q_path]
        return np.asarray(filtered, dtype=np.int64)

    print("[INFO] Computing retrieval statistics...")
    rank1_correct = 0
    rank5_correct = 0
    rank10_correct = 0
    total_queries = len(query_paths)

    success_indices: List[int] = []
    failure_indices: List[int] = []

    for q_idx in range(total_queries):
        query_pid = query_pids[q_idx]
        ranked = ranked_indices_for_query(q_idx)

        first_correct_rank: int | None = None
        for rank, g_idx in enumerate(ranked[:10], start=1):
            if gallery_pids[g_idx] == query_pid:
                first_correct_rank = rank
                break

        if first_correct_rank == 1:
            rank1_correct += 1
        if first_correct_rank is not None and first_correct_rank <= 5:
            rank5_correct += 1
        if first_correct_rank is not None and first_correct_rank <= 10:
            rank10_correct += 1

        if first_correct_rank == 1:
            success_indices.append(q_idx)
        else:
            failure_indices.append(q_idx)

    rank1_acc = 100.0 * rank1_correct / total_queries
    rank5_acc = 100.0 * rank5_correct / total_queries
    rank10_acc = 100.0 * rank10_correct / total_queries

    print("[INFO] Overall Statistics:")
    print(f"  Rank-1:  {rank1_acc:.2f}% ({rank1_correct}/{total_queries})")
    print(f"  Rank-5:  {rank5_acc:.2f}% ({rank5_correct}/{total_queries})")
    print(f"  Rank-10: {rank10_acc:.2f}% ({rank10_correct}/{total_queries})")

    rng = np.random.default_rng(seed)
    query_indices, num_success, num_failure = _sample_query_indices(
        success_indices=success_indices,
        failure_indices=failure_indices,
        num_queries=num_queries,
        rng=rng,
    )

    print(
        f"[INFO] Visualizing {len(query_indices)} queries "
        f"({num_success} success, {num_failure} failure) with top-{top_k} results..."
    )

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("ReID Retrieval Visualization Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Queries: {total_queries}\n")
        f.write(f"Total Gallery: {len(gallery_paths)}\n")
        f.write(f"Unique Identities: {len(set(query_pids))}\n")
        f.write(f"Exclude same image: {exclude_same_image}\n\n")
        f.write(f"Rank-1 Accuracy:  {rank1_acc:.2f}% ({rank1_correct}/{total_queries})\n")
        f.write(f"Rank-5 Accuracy:  {rank5_acc:.2f}% ({rank5_correct}/{total_queries})\n")
        f.write(f"Rank-10 Accuracy: {rank10_acc:.2f}% ({rank10_correct}/{total_queries})\n\n")
        f.write(f"Visualized Queries: {len(query_indices)}\n")
        f.write(f"  - Success cases: {num_success}\n")
        f.write(f"  - Failure cases: {num_failure}\n")
        f.write("=" * 60 + "\n")

    for idx, q_idx in enumerate(query_indices):
        query_path = query_paths[q_idx]
        query_pid = query_pids[q_idx]

        ranked = ranked_indices_for_query(q_idx)
        indices = ranked[:top_k]
        if len(indices) == 0:
            continue

        query_rank = None
        for rank, g_idx in enumerate(indices, start=1):
            if gallery_pids[g_idx] == query_pid:
                query_rank = rank
                break

        cols = 1 + len(indices)
        fig = plt.figure(figsize=(2.2 * cols, 4.5))

        status = (
            "SUCCESS"
            if query_rank == 1
            else f"FAILURE (Correct at Rank-{query_rank})"
            if query_rank
            else f"FAILURE (Not in Top-{len(indices)})"
        )
        status_color = "green" if query_rank == 1 else "orange" if query_rank else "red"
        fig.suptitle(
            f"Query {idx + 1}/{len(query_indices)} - {status}",
            fontsize=14,
            fontweight="bold",
            color=status_color,
        )

        ax = plt.subplot(1, cols, 1)
        query_img = cv2.imread(query_path)
        if query_img is not None:
            query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
            ax.imshow(query_img)

        query_filename = os.path.basename(query_path)
        ax.set_title(f"QUERY\\nID: {query_pid}\\n{query_filename}", fontsize=10, fontweight="bold", color="blue")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("blue")
            spine.set_linewidth(4)

        for rank, g_idx in enumerate(indices, start=1):
            ax = plt.subplot(1, cols, rank + 1)
            gallery_path = gallery_paths[g_idx]
            gallery_pid = gallery_pids[g_idx]
            distance = float(distmat[q_idx, g_idx])

            gallery_img = cv2.imread(gallery_path)
            if gallery_img is not None:
                gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
                ax.imshow(gallery_img)

            is_correct = gallery_pid == query_pid
            color = "green" if is_correct else "red"
            gallery_filename = os.path.basename(gallery_path)
            title = f"Rank-{rank}\\nID: {gallery_pid}\\nDist: {distance:.3f}\\n{gallery_filename}"

            ax.set_title(title, fontsize=8, color=color, fontweight="bold" if is_correct else "normal")
            ax.axis("off")

            border_color = "green" if is_correct else "lightgray"
            border_width = 4 if is_correct else 1
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        category = "success" if query_rank == 1 else "failure"
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        safe_pid = str(query_pid).replace("/", "_").replace("\\", "_")
        output_path = os.path.join(category_dir, f"retrieval_{idx + 1:03d}_query_{safe_pid}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"[{idx + 1}/{len(query_indices)}] Saved: {output_path}")

    print("\n[INFO] Visualization completed!")
    print(f"  Results saved to: {output_dir}")
    print(f"  Success cases: {os.path.join(output_dir, 'success')}")
    print(f"  Failure cases: {os.path.join(output_dir, 'failure')}")
    print(f"  Summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ReID retrieval results")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint path")
    parser.add_argument("--query_dir", type=str, required=True, help="query image root")
    parser.add_argument("--gallery_dir", type=str, required=True, help="gallery image root")
    parser.add_argument("--output_dir", type=str, default="./visualizations/retrieval", help="output directory")
    parser.add_argument("--top_k", type=int, default=10, help="top-k retrieval results to show")
    parser.add_argument("--num_queries", type=int, default=20, help="number of queries to visualize")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument("--backbone", type=str, default="osnet_ain_x1_0", help="fallback backbone")
    parser.add_argument("--num_classes", type=int, default=107, help="fallback number of classes")
    parser.add_argument("--img_height", type=int, default=256, help="fallback image height")
    parser.add_argument("--img_width", type=int, default=256, help="fallback image width")
    parser.add_argument("--no_flip_test", action="store_true", help="disable flip test")
    parser.add_argument("--keep_same_image", action="store_true", help="keep exact same image in gallery ranking")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.isdir(args.query_dir):
        raise FileNotFoundError(f"Query directory not found: {args.query_dir}")
    if not os.path.isdir(args.gallery_dir):
        raise FileNotFoundError(f"Gallery directory not found: {args.gallery_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    print("[1/4] Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        checkpoint = {"state_dict": checkpoint}

    model, cfg = build_model_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
        fallback_backbone=args.backbone,
        fallback_num_classes=args.num_classes,
    )

    training_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    img_h = int(training_cfg.get("image_height", args.img_height))
    img_w = int(training_cfg.get("image_width", args.img_width))

    print("[2/4] Loading datasets...")
    query_dataset, query_loader = _build_loader(args.query_dir, img_h, img_w, args.batch_size, device)
    gallery_dataset, gallery_loader = _build_loader(args.gallery_dir, img_h, img_w, args.batch_size, device)
    print(f"  Query images: {len(query_dataset)}")
    print(f"  Gallery images: {len(gallery_dataset)}")

    print("[3/4] Extracting features...")
    flip_test = not args.no_flip_test
    query_features, query_pids, _, query_paths = extract_features(model, query_loader, device, flip_test=flip_test)
    gallery_features, gallery_pids, _, gallery_paths = extract_features(model, gallery_loader, device, flip_test=flip_test)

    if query_features.size == 0:
        raise RuntimeError("No query features extracted")
    if gallery_features.size == 0:
        raise RuntimeError("No gallery features extracted")

    print(f"  Query: {len(query_features)} images, {len(set(query_pids))} identities")
    print(f"  Gallery: {len(gallery_features)} images, {len(set(gallery_pids))} identities")

    print("[4/4] Generating retrieval visualization...")
    visualize_retrieval_results(
        query_paths=query_paths,
        gallery_paths=gallery_paths,
        query_features=query_features,
        gallery_features=gallery_features,
        query_pids=query_pids,
        gallery_pids=gallery_pids,
        output_dir=args.output_dir,
        top_k=args.top_k,
        num_queries=args.num_queries,
        exclude_same_image=not args.keep_same_image,
        seed=args.seed,
    )

    print("\nDone!")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
