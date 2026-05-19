#!/usr/bin/env python3
"""Generate qualitative and quantitative analysis panels for the joint ReID model."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except Exception:
    TSNE = None
    SKLEARN_AVAILABLE = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.core.evaluation import ReIDDataset, compute_cmc_map, compute_distance_matrix
from app.core.joint_model import JointReIDModel


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            state_dict = checkpoint
        if isinstance(state_dict, dict):
            return state_dict
    raise ValueError("Checkpoint does not contain model_state_dict/state_dict.")


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _to_rgb_u8(img_chw: torch.Tensor) -> np.ndarray:
    arr = img_chw.detach().cpu().clamp(0.0, 1.0).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _to_gray_heatmap(gray_2d: np.ndarray) -> np.ndarray:
    x = gray_2d.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    x_u8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    hm = cv2.applyColorMap(x_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < 1e-12 or bn < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def _simulate_lighting(img: torch.Tensor) -> Dict[str, torch.Tensor]:
    out = {"original": img}
    out["dark"] = torch.clamp(img * 0.45, 0.0, 1.0)
    out["bright"] = torch.clamp(img * 1.55, 0.0, 1.0)
    out["low_contrast"] = torch.clamp((img - 0.5) * 0.5 + 0.5, 0.0, 1.0)
    warm = img.clone()
    warm[0] = torch.clamp(warm[0] * 1.15, 0.0, 1.0)
    warm[2] = torch.clamp(warm[2] * 0.85, 0.0, 1.0)
    out["warm_color"] = warm
    return out


@dataclass
class FeaturePack:
    feats: np.ndarray
    ids: List[Any]
    cams: List[int]
    paths: List[str]


def save_distance_histogram(
    q_with: FeaturePack,
    g_with: FeaturePack,
    q_without: FeaturePack,
    g_without: FeaturePack,
    output_path: str,
    metric: str = "cosine",
) -> None:
    d_with = compute_distance_matrix(q_with.feats, g_with.feats, metric=metric)
    d_without = compute_distance_matrix(q_without.feats, g_without.feats, metric=metric)

    q_ids = np.asarray(q_with.ids)
    g_ids = np.asarray(g_with.ids)
    pos_mask = q_ids[:, None] == g_ids[None, :]
    neg_mask = ~pos_mask

    pos_with = d_with[pos_mask]
    neg_with = d_with[neg_mask]
    pos_without = d_without[pos_mask]
    neg_without = d_without[neg_mask]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(pos_with, bins=60, alpha=0.6, label="Positive", color="#2E8B57", density=True)
    axes[0].hist(neg_with, bins=60, alpha=0.6, label="Negative", color="#CD5C5C", density=True)
    axes[0].set_title("With IPAID")
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].hist(pos_without, bins=60, alpha=0.6, label="Positive", color="#2E8B57", density=True)
    axes[1].hist(neg_without, bins=60, alpha=0.6, label="Negative", color="#CD5C5C", density=True)
    axes[1].set_title("Bypass IPAID")
    axes[1].set_xlabel("Distance")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    fig.suptitle("Positive/Negative Distance Distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    device: torch.device,
    fallback_backbone: str,
    fallback_num_classes: int,
) -> JointReIDModel:
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
        print(f"[WARN] missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}")
    return model


@torch.no_grad()
def forward_features(
    model: JointReIDModel,
    loader: DataLoader,
    device: torch.device,
    bypass_ipaid: bool = False,
) -> FeaturePack:
    model.eval()
    feats_list: List[torch.Tensor] = []
    ids: List[Any] = []
    cams: List[int] = []
    paths: List[str] = []

    prev_flag = getattr(model, "use_ipaid", None)
    if bypass_ipaid and prev_flag is not None:
        model.use_ipaid = False

    for batch in loader:
        imgs = batch[0].to(device)
        output = model(imgs, boxes_list=None, return_illuminated=False)
        feat = output["features"] if isinstance(output, dict) else output[0]
        feat = F.normalize(feat, p=2, dim=1)
        feats_list.append(feat.cpu())

        batch_ids = batch[1]
        batch_cams = batch[2]
        batch_paths = batch[3]

        if isinstance(batch_ids, torch.Tensor):
            ids.extend(batch_ids.tolist())
        else:
            ids.extend(list(batch_ids))

        if isinstance(batch_cams, torch.Tensor):
            cams.extend([int(x) for x in batch_cams.tolist()])
        else:
            cams.extend([int(x) for x in batch_cams])

        paths.extend([str(p) for p in batch_paths])

    if bypass_ipaid and prev_flag is not None:
        model.use_ipaid = prev_flag

    feats = torch.cat(feats_list, dim=0).numpy() if feats_list else np.empty((0, 0), dtype=np.float32)
    return FeaturePack(feats=feats, ids=ids, cams=cams, paths=paths)


def make_dataloaders(
    query_dir: str,
    gallery_dir: str,
    img_height: int,
    img_width: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[ReIDDataset, ReIDDataset, DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])
    q_ds = ReIDDataset(root=query_dir, transform=transform)
    g_ds = ReIDDataset(root=gallery_dir, transform=transform)

    q_loader = DataLoader(
        q_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if os.name != "nt" else 0,
        pin_memory=device.type == "cuda",
    )
    g_loader = DataLoader(
        g_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if os.name != "nt" else 0,
        pin_memory=device.type == "cuda",
    )
    return q_ds, g_ds, q_loader, g_loader


def evaluate_reid(
    q: FeaturePack,
    g: FeaturePack,
    metric: str = "cosine",
) -> Dict[str, float]:
    dist = compute_distance_matrix(q.feats, g.feats, metric=metric)
    cmc, m_ap = compute_cmc_map(
        distmat=dist,
        query_ids=q.ids,
        gallery_ids=g.ids,
        query_cams=q.cams,
        gallery_cams=g.cams,
        query_paths=q.paths,
        gallery_paths=g.paths,
        max_rank=10,
        exclude_same_camera=True,
    )
    return {
        "rank1": float(cmc[0] * 100.0),
        "rank5": float(cmc[min(4, len(cmc) - 1)] * 100.0),
        "rank10": float(cmc[min(9, len(cmc) - 1)] * 100.0),
        "mAP": float(m_ap * 100.0),
    }


def save_tsne_plot(
    feats_with: np.ndarray,
    ids_with: Sequence[Any],
    feats_without: np.ndarray,
    ids_without: Sequence[Any],
    output_path: str,
    max_points: int,
    seed: int,
) -> None:
    if not SKLEARN_AVAILABLE:
        print("[WARN] sklearn is unavailable; skipping the t-SNE figure.")
        return

    n = min(len(ids_with), len(ids_without), max_points)
    if n < 20:
        print("[WARN] Too few samples for t-SNE; skipping.")
        return

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ids_with), size=n, replace=False)
    f1 = feats_with[idx]
    y1 = [ids_with[i] for i in idx]
    f2 = feats_without[idx]
    y2 = [ids_without[i] for i in idx]

    tsne1 = TSNE(n_components=2, random_state=seed, perplexity=min(30, max(5, n // 12)))
    tsne2 = TSNE(n_components=2, random_state=seed, perplexity=min(30, max(5, n // 12)))
    c1 = tsne1.fit_transform(f1)
    c2 = tsne2.fit_transform(f2)

    unique_ids = sorted({str(x) for x in y1})
    id_to_idx = {k: i for i, k in enumerate(unique_ids)}
    color1 = [id_to_idx[str(v)] for v in y1]
    color2 = [id_to_idx[str(v)] for v in y2]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(c1[:, 0], c1[:, 1], c=color1, cmap="tab20", s=14, alpha=0.75)
    axes[0].set_title("With IPAID")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    axes[0].grid(alpha=0.2)

    axes[1].scatter(c2[:, 0], c2[:, 1], c=color2, cmap="tab20", s=14, alpha=0.75)
    axes[1].set_title("Bypass IPAID")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].grid(alpha=0.2)

    fig.suptitle("Feature Space Distribution (Same ID Colors)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_retrieval_grid(
    q_ds: ReIDDataset,
    g_ds: ReIDDataset,
    dist_with: np.ndarray,
    dist_without: np.ndarray,
    output_path: str,
    num_queries: int,
    topk: int,
    seed: int,
) -> None:
    if len(q_ds) == 0 or len(g_ds) == 0:
        print("[WARN] Empty data; skipping the retrieval grid.")
        return

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(q_ds), size=min(num_queries, len(q_ds)), replace=False)

    rows = len(chosen) * 2
    cols = topk + 1
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 1.9 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    def _read_rgb(path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            return np.zeros((128, 256, 3), dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for r_idx, q_idx in enumerate(chosen):
        q_path, q_id, _ = q_ds.samples[q_idx]
        q_img = _read_rgb(q_path)

        order_with = np.argsort(dist_with[q_idx])[:topk]
        order_without = np.argsort(dist_without[q_idx])[:topk]

        row_a = r_idx * 2
        axes[row_a, 0].imshow(q_img)
        axes[row_a, 0].set_title(f"Q:{q_id}")
        axes[row_a, 0].axis("off")
        for j, g_idx in enumerate(order_with, start=1):
            g_path, g_id, _ = g_ds.samples[g_idx]
            g_img = _read_rgb(g_path)
            axes[row_a, j].imshow(g_img)
            hit = (str(g_id) == str(q_id))
            axes[row_a, j].set_title(f"W-{j} {'OK' if hit else 'X'}", fontsize=8)
            for spine in axes[row_a, j].spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_edgecolor("green" if hit else "red")
            axes[row_a, j].axis("off")

        row_b = row_a + 1
        axes[row_b, 0].imshow(q_img)
        axes[row_b, 0].set_title(f"Q:{q_id} (Bypass)")
        axes[row_b, 0].axis("off")
        for j, g_idx in enumerate(order_without, start=1):
            g_path, g_id, _ = g_ds.samples[g_idx]
            g_img = _read_rgb(g_path)
            axes[row_b, j].imshow(g_img)
            hit = (str(g_id) == str(q_id))
            axes[row_b, j].set_title(f"B-{j} {'OK' if hit else 'X'}", fontsize=8)
            for spine in axes[row_b, j].spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_edgecolor("green" if hit else "red")
            axes[row_b, j].axis("off")

    fig.suptitle("Retrieval Comparison: With IPAID vs Bypass IPAID", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_illumination_processing_grid(
    model: JointReIDModel,
    dataset: ReIDDataset,
    device: torch.device,
    output_path: str,
    num_examples: int,
) -> None:
    if len(dataset) == 0:
        print("[WARN] Empty data; skipping the illumination-processing figure.")
        return

    n = min(num_examples, len(dataset))
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    prev_flag = getattr(model, "use_ipaid", None)
    if prev_flag is not None:
        model.use_ipaid = True
    model.eval()

    for i in range(n):
        img_t, pid, _, path = dataset[i]
        inp = img_t.unsqueeze(0).to(device)
        out = model(inp, boxes_list=None, return_illuminated=True)

        original = _to_rgb_u8(img_t)
        illuminated = out.get("illuminated", inp).squeeze(0)
        illuminated_u8 = _to_rgb_u8(illuminated)

        illum_map = out.get("illumination_map")
        if illum_map is not None:
            illum_map_np = illum_map.squeeze().detach().cpu().numpy()
            if illum_map_np.ndim == 3:
                illum_map_np = np.mean(illum_map_np, axis=0)
            illum_hm = _to_gray_heatmap(illum_map_np)
        else:
            gray = cv2.cvtColor(illuminated_u8, cv2.COLOR_RGB2GRAY)
            illum_hm = _to_gray_heatmap(gray.astype(np.float32))

        diff = np.abs(original.astype(np.float32) - illuminated_u8.astype(np.float32)).mean(axis=2)
        diff_hm = _to_gray_heatmap(diff)

        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f"Original\nID:{pid}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(illuminated_u8)
        axes[i, 1].set_title("After IPAID")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(illum_hm)
        axes[i, 2].set_title("Illumination Map")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(diff_hm)
        axes[i, 3].set_title("|Original - IPAID|")
        axes[i, 3].axis("off")

    if prev_flag is not None:
        model.use_ipaid = prev_flag

    fig.suptitle("Illumination Processing Visualization", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _colorfulness(image: np.ndarray) -> float:
    rg = image[:, :, 0].astype(np.float32) - image[:, :, 1].astype(np.float32)
    yb = 0.5 * (image[:, :, 0].astype(np.float32) + image[:, :, 1].astype(np.float32)) - image[:, :, 2].astype(np.float32)
    std_root = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
    mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    return std_root + 0.3 * mean_root


def compute_image_stats(image: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    colorfulness = float(_colorfulness(image) / 255.0)
    return {
        "brightness": brightness,
        "contrast": contrast,
        "colorfulness": colorfulness,
    }


def analyze_visual_statistics(
    model: JointReIDModel,
    dataset: ReIDDataset,
    device: torch.device,
    num_examples: int,
    seed: int,
) -> Dict[str, Any]:
    if len(dataset) == 0:
        return {}

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(num_examples, len(dataset)), replace=False)
    agg: Dict[str, List[float]] = {
        "brightness_original": [],
        "contrast_original": [],
        "colorfulness_original": [],
        "brightness_illuminated": [],
        "contrast_illuminated": [],
        "colorfulness_illuminated": [],
    }
    per_sample: List[Dict[str, Dict[str, float]]] = []

    prev_flag = getattr(model, "use_ipaid", None)
    if prev_flag is not None:
        model.use_ipaid = True
    model.eval()

    for idx in indices:
        img_t, _, _, _ = dataset[idx]
        inp = img_t.unsqueeze(0).to(device)
        out = model(inp, boxes_list=None, return_illuminated=True)
        illuminated = out.get("illuminated", inp).squeeze(0)

        original = _to_rgb_u8(img_t)
        illuminated_u8 = _to_rgb_u8(illuminated)

        orig_stats = compute_image_stats(original)
        ill_stats = compute_image_stats(illuminated_u8)

        agg["brightness_original"].append(orig_stats["brightness"])
        agg["contrast_original"].append(orig_stats["contrast"])
        agg["colorfulness_original"].append(orig_stats["colorfulness"])
        agg["brightness_illuminated"].append(ill_stats["brightness"])
        agg["contrast_illuminated"].append(ill_stats["contrast"])
        agg["colorfulness_illuminated"].append(ill_stats["colorfulness"])

        per_sample.append({"original": orig_stats, "illuminated": ill_stats})

    if prev_flag is not None:
        model.use_ipaid = prev_flag

    def mean(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    summary: Dict[str, Dict[str, float]] = {
        "original": {
            "brightness_mean": mean(agg["brightness_original"]),
            "contrast_mean": mean(agg["contrast_original"]),
            "colorfulness_mean": mean(agg["colorfulness_original"]),
        },
        "illuminated": {
            "brightness_mean": mean(agg["brightness_illuminated"]),
            "contrast_mean": mean(agg["contrast_illuminated"]),
            "colorfulness_mean": mean(agg["colorfulness_illuminated"]),
        },
    }

    colorfulness_vals = [s["original"]["colorfulness"] for s in per_sample]
    if colorfulness_vals:
        threshold = float(np.percentile(colorfulness_vals, 25))
        low_texture = [s for s in per_sample if s["original"]["colorfulness"] <= threshold]
    else:
        threshold = 0.0
        low_texture = []

    def avg(samples: List[Dict[str, Dict[str, float]]], key: str) -> Dict[str, float]:
        if not samples:
            return {"brightness_mean": 0.0, "contrast_mean": 0.0, "colorfulness_mean": 0.0}
        return {
            "brightness_mean": float(np.mean([s[key]["brightness"] for s in samples])),
            "contrast_mean": float(np.mean([s[key]["contrast"] for s in samples])),
            "colorfulness_mean": float(np.mean([s[key]["colorfulness"] for s in samples])),
        }

    summary["low_texture"] = {
        "count": len(low_texture),
        "threshold": threshold,
        "original": avg(low_texture, "original"),
        "illuminated": avg(low_texture, "illuminated"),
    }

    return summary


def save_visual_stats_plot(
    stats: Dict[str, Dict[str, float]],
    output_path: str,
) -> None:
    if not stats:
        print("[WARN] No visual statistics are available; skipping the pie chart.")
        return

    categories = ["brightness", "contrast", "colorfulness"]
    original = [stats.get("original", {}).get(f"{c}_mean", 0.0) for c in categories]
    illuminated = [stats.get("illuminated", {}).get(f"{c}_mean", 0.0) for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, original, width, label="Original", color="#5DA5E7")
    ax.bar(x + width / 2, illuminated, width, label="IPAID", color="#F28E2B")

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.set_ylabel("Normalized Value")
    ax.set_title("Visual Statistics: Original vs IPAID")
    ax.legend()
    ax.grid(alpha=0.2)

    for i in range(len(categories)):
        ax.text(x[i] - width / 2, original[i] + 0.01, f"{original[i]:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + width / 2, illuminated[i] + 0.01, f"{illuminated[i]:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_illumination_robustness_boxplot(
    model: JointReIDModel,
    dataset: ReIDDataset,
    device: torch.device,
    output_path: str,
    num_examples: int,
) -> Dict[str, Dict[str, float]]:
    if len(dataset) == 0:
        print("[WARN] Empty data; skipping the illumination-robustness figure.")
        return {}

    n = min(num_examples, len(dataset))
    conditions = ["dark", "bright", "low_contrast", "warm_color"]
    sims_with: Dict[str, List[float]] = {k: [] for k in conditions}
    sims_without: Dict[str, List[float]] = {k: [] for k in conditions}

    model.eval()
    prev_flag = getattr(model, "use_ipaid", None)

    for i in range(n):
        img_t, _, _, _ = dataset[i]
        variants = _simulate_lighting(img_t)

        if prev_flag is not None:
            model.use_ipaid = True
        base_with = model(variants["original"].unsqueeze(0).to(device), boxes_list=None, return_illuminated=False)["features"]
        base_with = F.normalize(base_with, p=2, dim=1).squeeze(0).detach().cpu().numpy()

        if prev_flag is not None:
            model.use_ipaid = False
        base_without = model(variants["original"].unsqueeze(0).to(device), boxes_list=None, return_illuminated=False)["features"]
        base_without = F.normalize(base_without, p=2, dim=1).squeeze(0).detach().cpu().numpy()

        for cond in conditions:
            v = variants[cond].unsqueeze(0).to(device)

            if prev_flag is not None:
                model.use_ipaid = True
            f_with = model(v, boxes_list=None, return_illuminated=False)["features"]
            f_with = F.normalize(f_with, p=2, dim=1).squeeze(0).detach().cpu().numpy()
            sims_with[cond].append(_cosine_sim(base_with, f_with))

            if prev_flag is not None:
                model.use_ipaid = False
            f_without = model(v, boxes_list=None, return_illuminated=False)["features"]
            f_without = F.normalize(f_without, p=2, dim=1).squeeze(0).detach().cpu().numpy()
            sims_without[cond].append(_cosine_sim(base_without, f_without))

    if prev_flag is not None:
        model.use_ipaid = prev_flag

    positions = np.arange(len(conditions))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    bp1 = ax.boxplot(
        [sims_with[c] for c in conditions],
        positions=positions - width / 2,
        widths=0.30,
        patch_artist=True,
        showfliers=False,
    )
    bp2 = ax.boxplot(
        [sims_without[c] for c in conditions],
        positions=positions + width / 2,
        widths=0.30,
        patch_artist=True,
        showfliers=False,
    )

    for b in bp1["boxes"]:
        b.set_facecolor("#2E8B57")
        b.set_alpha(0.7)
    for b in bp2["boxes"]:
        b.set_facecolor("#CD5C5C")
        b.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Cosine Similarity to Original Feature")
    ax.set_title("Illumination Robustness: With IPAID vs Bypass IPAID")
    ax.grid(alpha=0.2)
    ax.legend(
        [bp1["boxes"][0], bp2["boxes"][0]],
        ["With IPAID", "Bypass IPAID"],
        loc="lower left",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary: Dict[str, Dict[str, float]] = {}
    for c in conditions:
        summary[c] = {
            "with_ipaid_mean": float(np.mean(sims_with[c])) if sims_with[c] else 0.0,
            "with_ipaid_std": float(np.std(sims_with[c])) if sims_with[c] else 0.0,
            "bypass_mean": float(np.mean(sims_without[c])) if sims_without[c] else 0.0,
            "bypass_std": float(np.std(sims_without[c])) if sims_without[c] else 0.0,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint ReID visualization analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint path")
    parser.add_argument("--query_dir", type=str, required=True, help="query directory")
    parser.add_argument("--gallery_dir", type=str, required=True, help="gallery directory")
    parser.add_argument("--output_dir", type=str, default="outputs/joint_analysis", help="output directory")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument("--backbone", type=str, default="osnet_ain_x1_0", help="fallback backbone")
    parser.add_argument("--num_classes", type=int, default=107, help="fallback num classes")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_width", type=int, default=512)
    parser.add_argument("--tsne_max_points", type=int, default=1200)
    parser.add_argument("--num_retrieval_queries", type=int, default=6)
    parser.add_argument("--retrieval_topk", type=int, default=5)
    parser.add_argument("--num_illum_examples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not os.path.isdir(args.query_dir):
        raise FileNotFoundError(f"query_dir not found: {args.query_dir}")
    if not os.path.isdir(args.gallery_dir):
        raise FileNotFoundError(f"gallery_dir not found: {args.gallery_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = _resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Input size: {args.img_height}x{args.img_width}")

    print("\n[1/6] Loading model...")
    checkpoint_data = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not isinstance(checkpoint_data, dict):
        checkpoint_data = {"state_dict": checkpoint_data}
    model = build_model_from_checkpoint(
        checkpoint=checkpoint_data,
        device=device,
        fallback_backbone=args.backbone,
        fallback_num_classes=args.num_classes,
    )

    print("[2/6] Building dataloaders...")
    q_ds, g_ds, q_loader, g_loader = make_dataloaders(
        query_dir=args.query_dir,
        gallery_dir=args.gallery_dir,
        img_height=args.img_height,
        img_width=args.img_width,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    print(f"  query: {len(q_ds)} images, gallery: {len(g_ds)} images")

    print("[3/6] Extracting features (with IPAID / bypass IPAID)...")
    q_with = forward_features(model, q_loader, device, bypass_ipaid=False)
    g_with = forward_features(model, g_loader, device, bypass_ipaid=False)
    q_without = forward_features(model, q_loader, device, bypass_ipaid=True)
    g_without = forward_features(model, g_loader, device, bypass_ipaid=True)

    metrics_with = evaluate_reid(q_with, g_with, metric="cosine")
    metrics_without = evaluate_reid(q_without, g_without, metric="cosine")

    print("[4/6] Saving feature-space and distance plots...")
    save_tsne_plot(
        feats_with=np.concatenate([q_with.feats, g_with.feats], axis=0),
        ids_with=q_with.ids + g_with.ids,
        feats_without=np.concatenate([q_without.feats, g_without.feats], axis=0),
        ids_without=q_without.ids + g_without.ids,
        output_path=os.path.join(args.output_dir, "feature_tsne_with_vs_bypass.png"),
        max_points=args.tsne_max_points,
        seed=args.seed,
    )
    save_distance_histogram(
        q_with=q_with,
        g_with=g_with,
        q_without=q_without,
        g_without=g_without,
        output_path=os.path.join(args.output_dir, "distance_hist_with_vs_bypass.png"),
        metric="cosine",
    )

    print("[5/6] Saving retrieval and illumination plots...")
    dist_with = compute_distance_matrix(q_with.feats, g_with.feats, metric="cosine")
    dist_without = compute_distance_matrix(q_without.feats, g_without.feats, metric="cosine")
    save_retrieval_grid(
        q_ds=q_ds,
        g_ds=g_ds,
        dist_with=dist_with,
        dist_without=dist_without,
        output_path=os.path.join(args.output_dir, "retrieval_with_vs_bypass.png"),
        num_queries=args.num_retrieval_queries,
        topk=args.retrieval_topk,
        seed=args.seed,
    )
    save_illumination_processing_grid(
        model=model,
        dataset=q_ds,
        device=device,
        output_path=os.path.join(args.output_dir, "illumination_processing_examples.png"),
        num_examples=args.num_illum_examples,
    )
    robustness_summary = save_illumination_robustness_boxplot(
        model=model,
        dataset=q_ds,
        device=device,
        output_path=os.path.join(args.output_dir, "illumination_robustness_boxplot.png"),
        num_examples=args.num_illum_examples,
    )
    visual_stats = analyze_visual_statistics(
        model=model,
        dataset=q_ds,
        device=device,
        num_examples=args.num_illum_examples,
        seed=args.seed,
    )
    visual_stats_plot_path = os.path.join(args.output_dir, "visual_stats_comparison.png")
    save_visual_stats_plot(visual_stats, visual_stats_plot_path)

    print("[6/6] Writing summary JSON...")
    summary = {
        "checkpoint": args.checkpoint,
        "query_dir": args.query_dir,
        "gallery_dir": args.gallery_dir,
        "img_height": args.img_height,
        "img_width": args.img_width,
        "metrics_with_ipaid": metrics_with,
        "metrics_bypass_ipaid": metrics_without,
        "metric_deltas": {
            "rank1_delta": metrics_with["rank1"] - metrics_without["rank1"],
            "rank5_delta": metrics_with["rank5"] - metrics_without["rank5"],
            "rank10_delta": metrics_with["rank10"] - metrics_without["rank10"],
            "mAP_delta": metrics_with["mAP"] - metrics_without["mAP"],
        },
        "illumination_robustness": robustness_summary,
        "visual_statistics": visual_stats,
        "outputs": {
            "tsne": "feature_tsne_with_vs_bypass.png",
            "distance_hist": "distance_hist_with_vs_bypass.png",
            "retrieval": "retrieval_with_vs_bypass.png",
            "illum_processing": "illumination_processing_examples.png",
            "illum_robustness": "illumination_robustness_boxplot.png",
            "visual_stats": "visual_stats_comparison.png",
        },
    }
    out_json = os.path.join(args.output_dir, "analysis_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Saved to: {args.output_dir}")
    print(json.dumps(summary["metric_deltas"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
