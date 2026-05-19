#!/usr/bin/env python3
"""Mine stronger RIIC-ReID main-paper figure cases from ATRW assets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import load_config
from app.core.joint_model import JointReIDModel


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Sample:
    split: str
    label: str
    path: str
    relpath: str
    stem: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine RIIC-ReID main-paper figure cases")
    parser.add_argument("--config", default="config/illumination_config_atrw.yaml")
    parser.add_argument("--query_dir", default="data/processed/atrw/query")
    parser.add_argument("--gallery_dir", default="data/processed/atrw/gallery")
    parser.add_argument("--riic_ckpt", default="checkpoints/atrw_routeb_theoryB/joint_best.pth")
    parser.add_argument(
        "--retinexnet_dir",
        default=(
            "downloads/westc_perceptual_assets_20260325/root/autodl-tmp/v2_2/"
            "dog_reid_web/data/perceptual_baselines/atrw/retinexnet/test"
        ),
    )
    parser.add_argument(
        "--zerodcepp_dir",
        default=(
            "downloads/westc_perceptual_assets_20260325/root/autodl-tmp/v2_2/"
            "dog_reid_web/data/perceptual_baselines/atrw/zerodcepp/test"
        ),
    )
    parser.add_argument(
        "--output_root",
        default="docs/figures/riic_reid_main_paper_20260327/assets",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--claim_topk", type=int, default=5)
    parser.add_argument("--claim_candidates", type=int, default=25)
    parser.add_argument("--trust_scan_topn", type=int, default=80)
    parser.add_argument("--force_claim_query", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)

    def _default(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value))

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_default)


def load_split(root_dir: str, split: str) -> List[Sample]:
    root = Path(root_dir)
    samples: List[Sample] = []
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        for img_path in sorted(label_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            samples.append(
                Sample(
                    split=split,
                    label=label_dir.name,
                    path=str(img_path),
                    relpath=f"{split}/{label_dir.name}/{img_path.name}",
                    stem=img_path.stem,
                )
            )
    return samples


def build_transform(config_path: str):
    cfg = load_config(config_path)
    train_cfg = cfg.get("training", {})
    height = int(train_cfg.get("image_height", 256))
    width = int(train_cfg.get("image_width", 384))
    return transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])


def load_rgb(path: str, tfm) -> torch.Tensor:
    return tfm(Image.open(path).convert("RGB"))


def save_tensor(path: Path, x: torch.Tensor) -> None:
    ensure_dir(path.parent)
    arr = x.detach().cpu().clamp(0.0, 1.0).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_heatmap(path: Path, x: torch.Tensor | None, title: str) -> None:
    if x is None:
        return
    arr = x.detach().cpu().float().squeeze().numpy()
    if arr.ndim != 2:
        return
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(arr, cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(4.0, 3.1))
    ax.imshow(heatmap)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def infer_num_classes(state_dict: dict) -> int:
    for key, value in state_dict.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim == 2
            and key.endswith(("classifier.weight", "global_classifier.weight", "arcface.weight"))
        ):
            return int(value.shape[0])
    return 107


def load_model(ckpt_path: str, config_path: str, device: torch.device) -> JointReIDModel:
    cfg = load_config(config_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    model_cfg = cfg.get("model", {})
    illum_cfg = cfg.get("illumination_module", {})
    local_cfg = model_cfg.get("local_extractor", {})
    ipaid_params = dict(
        model_cfg.get("illumination_module", {}).get("module_params")
        or illum_cfg.get("module_params", {})
    )
    for key in (
        "feature_fusion",
        "branch_attention_fusion",
        "nuisance_head",
        "reid_head",
        "backbone_random_erasing",
    ):
        sub_cfg = model_cfg.get(key, {})
        if sub_cfg:
            ipaid_params[f"_{key}"] = sub_cfg

    model = JointReIDModel(
        num_classes=infer_num_classes(state),
        backbone_name=model_cfg.get("backbone", "osnet_ain_x1_0"),
        num_stripes=int(local_cfg.get("num_parts", 6)),
        pretrained_backbone=False,
        use_ipaid=True,
        dropout=float(local_cfg.get("dropout", 0.0)),
        ipaid_params=ipaid_params,
    )
    model_state = model.state_dict()
    keep = {
        key: value
        for key, value in state.items()
        if key in model_state
        and isinstance(value, torch.Tensor)
        and value.shape == model_state[key].shape
    }
    model.load_state_dict(keep, strict=False)
    return model.to(device).eval()


def luminance(x: torch.Tensor) -> float:
    return float((0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).mean().item())


def batched(seq: List[str], batch_size: int) -> Iterable[List[str]]:
    for start in range(0, len(seq), batch_size):
        yield seq[start : start + batch_size]


@torch.no_grad()
def extract_raw_features(model: JointReIDModel, batch: torch.Tensor) -> torch.Tensor:
    out = model.forward_raw_reference(batch, detach=True)
    return F.normalize(out["features"], p=2, dim=1)


@torch.no_grad()
def extract_riic_features(model: JointReIDModel, batch: torch.Tensor) -> torch.Tensor:
    out = model(batch, return_illuminated=False)
    return F.normalize(out["features"], p=2, dim=1)


def extract_features_from_paths(
    paths: List[str],
    tfm,
    device: torch.device,
    model: JointReIDModel,
    batch_size: int,
    mode: str,
) -> np.ndarray:
    chunks: List[torch.Tensor] = []
    for chunk in batched(paths, batch_size):
        batch = torch.stack([load_rgb(path, tfm) for path in chunk], 0).to(device)
        if mode == "riic":
            feat = extract_riic_features(model, batch)
        else:
            feat = extract_raw_features(model, batch)
        chunks.append(feat.cpu())
    return torch.cat(chunks, 0).numpy()


def compute_ap(dists: np.ndarray, gallery_labels: np.ndarray, query_label: str) -> float:
    order = np.argsort(dists)
    matches = (gallery_labels[order] == query_label).astype(np.float32)
    if matches.sum() <= 0:
        return 0.0
    precision = np.cumsum(matches) / (np.arange(matches.size, dtype=np.float32) + 1.0)
    return float((precision * matches).sum() / matches.sum())


def correct_at_k(dists: np.ndarray, gallery_labels: np.ndarray, query_label: str, topk: int) -> int:
    order = np.argsort(dists)[:topk]
    return int(np.sum(gallery_labels[order] == query_label))


def pca2(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    y = x @ vt[:2].T
    if y.shape[1] < 2:
        y = np.pad(y, ((0, 0), (0, 2 - y.shape[1])))
    return y


def resolve_enhanced_path(sample: Sample, root: Path) -> str:
    preferred = root / f"{sample.stem}.jpg"
    if preferred.exists():
        return str(preferred)
    matches = sorted(root.glob(f"{sample.stem}.*"))
    if matches:
        return str(matches[0])
    return sample.path


def score_claim_case(record: dict, lum: float) -> float:
    best_perceptual_ap = max(record["ap_retinexnet"], record["ap_zerodcepp"])
    best_perceptual_c5 = max(record["c5_retinexnet"], record["c5_zerodcepp"])
    riic_gain = record["ap_riic"] - best_perceptual_ap
    c5_gain = record["c5_riic"] - best_perceptual_c5
    darkness_bonus = max(0.0, 0.52 - lum)
    return 1.8 * riic_gain + 0.12 * c5_gain + 0.10 * darkness_bonus


def select_claim_case(
    query_samples: List[Sample],
    query_lums: np.ndarray,
    query_feats: Dict[str, np.ndarray],
    gallery_feats: Dict[str, np.ndarray],
    gallery_labels: np.ndarray,
    topk: int,
    topn: int,
) -> List[dict]:
    rows: List[dict] = []
    for idx, query in enumerate(query_samples):
        if query_lums[idx] > 0.55:
            continue
        metrics = {}
        for key in ("raw", "retinexnet", "zerodcepp", "riic"):
            dists = 1.0 - gallery_feats[key] @ query_feats[key][idx]
            metrics[f"ap_{key}"] = compute_ap(dists, gallery_labels, query.label)
            metrics[f"c5_{key}"] = correct_at_k(dists, gallery_labels, query.label, topk=topk)
        row = {
            "query_index": idx,
            "query_relpath": query.relpath,
            "label": query.label,
            "luminance": float(query_lums[idx]),
            **metrics,
        }
        row["claim_score"] = score_claim_case(row, query_lums[idx])
        rows.append(row)
    rows.sort(key=lambda item: item["claim_score"], reverse=True)
    return rows


def _display_order(
    dists: np.ndarray,
    gallery_samples: List[Sample],
    query_stem: str,
    topk: int,
) -> List[int]:
    order = np.argsort(dists)
    display = [int(idx) for idx in order if gallery_samples[int(idx)].stem != query_stem]
    return display[:topk]


def export_claim_assets(
    out_dir: Path,
    selected: dict,
    query_samples: List[Sample],
    gallery_samples: List[Sample],
    gallery_labels: np.ndarray,
    query_paths: Dict[str, List[str]],
    gallery_feats: Dict[str, np.ndarray],
    query_feats: Dict[str, np.ndarray],
    tfm,
    model: JointReIDModel,
    device: torch.device,
    topk: int,
) -> dict:
    ensure_dir(out_dir)
    idx = int(selected["query_index"])
    query = query_samples[idx]

    raw_tensor = load_rgb(query.path, tfm)
    retinex_tensor = load_rgb(query_paths["retinexnet"][idx], tfm)
    zerodce_tensor = load_rgb(query_paths["zerodcepp"][idx], tfm)
    with torch.no_grad():
        riic_out = model(raw_tensor.unsqueeze(0).to(device), return_illuminated=True)
    riic_tensor = riic_out["illuminated"][0].detach().cpu().clamp(0.0, 1.0)

    save_tensor(out_dir / "query_raw.png", raw_tensor)
    save_tensor(out_dir / "query_retinexnet.png", retinex_tensor)
    save_tensor(out_dir / "query_zerodcepp.png", zerodce_tensor)
    save_tensor(out_dir / "query_riic.png", riic_tensor)

    method_meta: Dict[str, dict] = {}
    for method_key, label in (
        ("raw", "Matched ReID baseline"),
        ("retinexnet", "RetinexNet"),
        ("zerodcepp", "Zero-DCE++"),
        ("riic", "RIIC-ReID"),
    ):
        dists = 1.0 - gallery_feats[method_key] @ query_feats[method_key][idx]
        display_order = _display_order(dists, gallery_samples, query.stem, topk=topk)
        ranked = []
        for rank, gi in enumerate(display_order, start=1):
            sample = gallery_samples[gi]
            match = bool(sample.label == query.label)
            tag = "pos" if match else "neg"
            name = f"rank{rank:02d}_{tag}_{sample.label}_{sample.stem}.png"
            save_tensor(out_dir / f"gallery_{method_key}" / name, load_rgb(sample.path, tfm))
            ranked.append(
                {
                    "rank": rank,
                    "gallery_relpath": sample.relpath,
                    "gallery_label": sample.label,
                    "match": match,
                    "saved_path": str((out_dir / f"gallery_{method_key}" / name).as_posix()),
                }
            )
        method_meta[method_key] = {
            "label": label,
            "ap": compute_ap(dists, gallery_labels, query.label),
            "correct_at_5": int(sum(1 for row in ranked if row["match"])),
            "ranked": ranked,
        }

    claim_meta = {
        "query_relpath": query.relpath,
        "query_label": query.label,
        "query_luminance": float(selected["luminance"]),
        "topk": topk,
        "methods": method_meta,
    }
    save_json(out_dir / "metrics.json", claim_meta)
    return claim_meta


def score_trust_case(metrics: dict) -> float:
    return (
        0.9 * metrics["rollback_mean"]
        + 1.8 * metrics["rollback_std"]
        + 0.7 * metrics["identity_mean"]
        + 6.0 * metrics["correction_gap_mean"]
        + 0.6 * metrics["color_risk_mean"]
        + 0.3 * metrics["correction_gap_std"]
    )


def save_branch_attention(case_dir: Path, branch_weights: torch.Tensor | None) -> dict | None:
    if branch_weights is None or not isinstance(branch_weights, torch.Tensor):
        return None
    weights = branch_weights.detach().cpu().float()
    if weights.ndim != 3 or weights.size(-1) != 3:
        return None
    arr = weights[0].numpy()

    csv_path = case_dir / "branch_attention.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stripe", "raw", "base", "adapted"])
        for idx, row in enumerate(arr, start=1):
            writer.writerow([idx, float(row[0]), float(row[1]), float(row[2])])

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    y = np.arange(arr.shape[0])
    left = np.zeros(arr.shape[0], dtype=np.float32)
    colors = ["#355FDB", "#22A54A", "#D97706"]
    labels = ["raw", "base", "adapted"]
    for branch_idx in range(arr.shape[1]):
        ax.barh(y, arr[:, branch_idx], left=left, color=colors[branch_idx], label=labels[branch_idx], height=0.72)
        left += arr[:, branch_idx]
    ax.set_xlim(0.0, 1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"s{i+1}" for i in range(arr.shape[0])])
    ax.invert_yaxis()
    ax.set_title("Stripe-wise branch attention", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(case_dir / "branch_attention.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    return {
        "csv": str(csv_path.as_posix()),
        "png": str((case_dir / "branch_attention.png").as_posix()),
    }


def export_trust_case(
    out_dir: Path,
    query: Sample,
    tfm,
    model: JointReIDModel,
    device: torch.device,
    luminance_value: float,
) -> dict:
    ensure_dir(out_dir)
    raw_x = load_rgb(query.path, tfm)
    with torch.no_grad():
        out = model(raw_x.unsqueeze(0).to(device), return_illuminated=True)

    corrected = out.get("illuminated", raw_x.unsqueeze(0).to(device))[0].detach().cpu().clamp(0.0, 1.0)
    ipaid = out.get("ipaid_details") or {}

    save_tensor(out_dir / "raw.png", raw_x)
    save_tensor(out_dir / "corrected.png", corrected)
    if isinstance(ipaid.get("reflectance_base"), torch.Tensor):
        save_tensor(out_dir / "reflectance_base.png", ipaid["reflectance_base"][0].detach().cpu().clamp(0.0, 1.0))
    if isinstance(ipaid.get("reflectance_att"), torch.Tensor):
        save_tensor(out_dir / "reflectance_att.png", ipaid["reflectance_att"][0].detach().cpu().clamp(0.0, 1.0))

    save_heatmap(out_dir / "identity_protection_map.png", ipaid.get("identity_protection_map"), "identity protection")
    save_heatmap(out_dir / "rollback_alpha_map.png", ipaid.get("rollback_alpha_map"), "rollback alpha")
    save_heatmap(out_dir / "correction_gap.png", ipaid.get("correction_gap"), "correction gap")
    save_heatmap(out_dir / "illumination_map.png", ipaid.get("illumination"), "illumination")
    save_heatmap(out_dir / "color_risk.png", ipaid.get("color_risk"), "color risk")
    branch_info = save_branch_attention(out_dir, ipaid.get("branch_attention_weights"))

    rollback = ipaid.get("rollback_alpha")
    correction_gap = ipaid.get("correction_gap")
    identity_map = ipaid.get("identity_protection_map")
    color_risk = ipaid.get("color_risk")

    metrics = {
        "sample_relpath": query.relpath,
        "label": query.label,
        "luminance": float(luminance_value),
        "rollback_mean": float(rollback.mean().item()) if isinstance(rollback, torch.Tensor) else 0.0,
        "rollback_std": float(rollback.std(unbiased=False).item()) if isinstance(rollback, torch.Tensor) else 0.0,
        "identity_mean": float(identity_map.mean().item()) if isinstance(identity_map, torch.Tensor) else 0.0,
        "identity_std": float(identity_map.std(unbiased=False).item()) if isinstance(identity_map, torch.Tensor) else 0.0,
        "correction_gap_mean": float(correction_gap.mean().item()) if isinstance(correction_gap, torch.Tensor) else 0.0,
        "correction_gap_std": float(correction_gap.std(unbiased=False).item()) if isinstance(correction_gap, torch.Tensor) else 0.0,
        "color_risk_mean": float(color_risk.mean().item()) if isinstance(color_risk, torch.Tensor) else 0.0,
        "branch_attention": branch_info,
    }
    metrics["trust_score"] = score_trust_case(metrics)
    save_json(out_dir / "metrics.json", metrics)
    return metrics


def export_geometry_case(
    out_dir: Path,
    query: Sample,
    tfm,
    model: JointReIDModel,
    device: torch.device,
    gallery_samples: List[Sample],
    batch_size: int,
    perceptual_query_path: str,
) -> dict:
    ensure_dir(out_dir)
    pos_gallery = [sample for sample in gallery_samples if sample.label == query.label]
    neg_gallery = [sample for sample in gallery_samples if sample.label != query.label]

    teacher_feats = extract_features_from_paths([sample.path for sample in pos_gallery], tfm, device, model, batch_size, "raw")
    neg_feats = extract_features_from_paths([sample.path for sample in neg_gallery], tfm, device, model, batch_size, "raw")
    query_raw_feat = extract_features_from_paths([query.path], tfm, device, model, 1, "raw")[0]
    query_perc_feat = extract_features_from_paths([perceptual_query_path], tfm, device, model, 1, "raw")[0]
    query_riic_feat = extract_features_from_paths([query.path], tfm, device, model, 1, "riic")[0]

    teacher_center = teacher_feats.mean(axis=0, keepdims=True)
    neg_d = np.linalg.norm(neg_feats - teacher_center, axis=1)
    nn_idx = np.argsort(neg_d)[:6]
    hard_neg_feats = neg_feats[nn_idx]
    hard_neg_samples = [neg_gallery[int(i)] for i in nn_idx]

    proj = pca2(
        np.concatenate(
            [
                teacher_feats,
                hard_neg_feats,
                query_raw_feat[None, :],
                query_perc_feat[None, :],
                query_riic_feat[None, :],
            ],
            axis=0,
        )
    )
    teacher_proj = proj[: teacher_feats.shape[0]]
    neg_proj = proj[teacher_feats.shape[0] : teacher_feats.shape[0] + hard_neg_feats.shape[0]]
    query_proj = proj[-3:]

    csv_path = out_dir / "teacher_student_projection.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "role", "label", "x", "y"])
        for idx, sample in enumerate(pos_gallery):
            writer.writerow([f"teacher_pos_{idx+1}", "teacher_positive", sample.label, teacher_proj[idx, 0], teacher_proj[idx, 1]])
        for idx, sample in enumerate(hard_neg_samples):
            writer.writerow([f"hard_neg_{idx+1}", "hard_negative", sample.label, neg_proj[idx, 0], neg_proj[idx, 1]])
        writer.writerow(["query_raw", "query", query.label, query_proj[0, 0], query_proj[0, 1]])
        writer.writerow(["query_perceptual", "query", query.label, query_proj[1, 0], query_proj[1, 1]])
        writer.writerow(["query_riic", "query", query.label, query_proj[2, 0], query_proj[2, 1]])

    stats = {
        "query_relpath": query.relpath,
        "query_label": query.label,
        "num_teacher_positives": len(pos_gallery),
        "teacher_radius_l2": float(np.linalg.norm(teacher_feats - teacher_center, axis=1).max()),
        "query_raw_dist_to_center": float(np.linalg.norm(query_raw_feat - teacher_center.squeeze(0))),
        "query_perceptual_dist_to_center": float(np.linalg.norm(query_perc_feat - teacher_center.squeeze(0))),
        "query_riic_dist_to_center": float(np.linalg.norm(query_riic_feat - teacher_center.squeeze(0))),
        "nearest_negative_dist_to_center": float(neg_d[nn_idx[0]]),
        "geometry_gain": float(
            np.linalg.norm(query_perc_feat - teacher_center.squeeze(0))
            - np.linalg.norm(query_riic_feat - teacher_center.squeeze(0))
        ),
        "hard_negative_relpaths": [sample.relpath for sample in hard_neg_samples],
    }
    save_json(out_dir / "metrics.json", stats)
    return stats


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device_name = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    output_root = Path(args.output_root)
    claim_dir = output_root / "claim_case"
    trust_dir = output_root / "trust_case"
    geometry_dir = output_root / "geometry_case"
    ensure_dir(output_root)

    tfm = build_transform(args.config)
    model = load_model(args.riic_ckpt, args.config, device)

    query_samples = load_split(args.query_dir, "query")
    gallery_samples = load_split(args.gallery_dir, "gallery")
    gallery_labels = np.array([sample.label for sample in gallery_samples], dtype=object)

    retinex_root = Path(args.retinexnet_dir)
    zerodce_root = Path(args.zerodcepp_dir)
    query_paths = {
        "raw": [sample.path for sample in query_samples],
        "retinexnet": [resolve_enhanced_path(sample, retinex_root) for sample in query_samples],
        "zerodcepp": [resolve_enhanced_path(sample, zerodce_root) for sample in query_samples],
    }
    gallery_paths = {
        "raw": [sample.path for sample in gallery_samples],
        "retinexnet": [resolve_enhanced_path(sample, retinex_root) for sample in gallery_samples],
        "zerodcepp": [resolve_enhanced_path(sample, zerodce_root) for sample in gallery_samples],
    }

    print(f"Query images: {len(query_samples)} | Gallery images: {len(gallery_samples)}")
    print("Extracting gallery features...")
    gallery_feats = {
        "raw": extract_features_from_paths(gallery_paths["raw"], tfm, device, model, args.batch_size, "raw"),
        "retinexnet": extract_features_from_paths(gallery_paths["retinexnet"], tfm, device, model, args.batch_size, "raw"),
        "zerodcepp": extract_features_from_paths(gallery_paths["zerodcepp"], tfm, device, model, args.batch_size, "raw"),
        "riic": extract_features_from_paths(gallery_paths["raw"], tfm, device, model, args.batch_size, "riic"),
    }

    print("Extracting query features...")
    query_feats = {
        "raw": extract_features_from_paths(query_paths["raw"], tfm, device, model, args.batch_size, "raw"),
        "retinexnet": extract_features_from_paths(query_paths["retinexnet"], tfm, device, model, args.batch_size, "raw"),
        "zerodcepp": extract_features_from_paths(query_paths["zerodcepp"], tfm, device, model, args.batch_size, "raw"),
        "riic": extract_features_from_paths(query_paths["raw"], tfm, device, model, args.batch_size, "riic"),
    }

    print("Computing query luminance...")
    query_lums = np.array([luminance(load_rgb(sample.path, tfm)) for sample in query_samples], dtype=np.float32)

    print("Selecting claim candidates...")
    claim_rows = select_claim_case(
        query_samples=query_samples,
        query_lums=query_lums,
        query_feats=query_feats,
        gallery_feats=gallery_feats,
        gallery_labels=gallery_labels,
        topk=args.claim_topk,
        topn=args.claim_candidates,
    )
    if not claim_rows:
        raise RuntimeError("No claim candidates were found.")

    claim_candidates = claim_rows[: args.claim_candidates]
    selected_claim = claim_candidates[0]
    if args.force_claim_query:
        matched = [row for row in claim_rows if row["query_relpath"] == args.force_claim_query]
        if not matched:
            matched = [row for row in claim_rows if args.force_claim_query in row["query_relpath"]]
        if not matched:
            raise ValueError(f"Could not find forced claim query: {args.force_claim_query}")
        selected_claim = matched[0]

    claim_meta = export_claim_assets(
        out_dir=claim_dir,
        selected=selected_claim,
        query_samples=query_samples,
        gallery_samples=gallery_samples,
        gallery_labels=gallery_labels,
        query_paths=query_paths,
        gallery_feats=gallery_feats,
        query_feats=query_feats,
        tfm=tfm,
        model=model,
        device=device,
        topk=args.claim_topk,
    )

    print("Selecting trust candidate from top claim candidates...")
    trust_candidates: List[dict] = []
    for candidate in claim_candidates[: args.trust_scan_topn]:
        query = query_samples[int(candidate["query_index"])]
        temp_dir = trust_dir / "_tmp" / query.stem
        metrics = export_trust_case(
            out_dir=temp_dir,
            query=query,
            tfm=tfm,
            model=model,
            device=device,
            luminance_value=float(candidate["luminance"]),
        )
        metrics["query_index"] = int(candidate["query_index"])
        trust_candidates.append(metrics)

    trust_candidates.sort(key=lambda item: item["trust_score"], reverse=True)
    if not trust_candidates:
        raise RuntimeError("No trust candidate was found.")
    trust_best = trust_candidates[0]
    best_query = query_samples[int(trust_best["query_index"])]

    # Re-export the best trust case into the canonical directory.
    if trust_dir.exists():
        ensure_dir(trust_dir)
    trust_meta = export_trust_case(
        out_dir=trust_dir,
        query=best_query,
        tfm=tfm,
        model=model,
        device=device,
        luminance_value=float(query_lums[int(trust_best["query_index"])]),
    )

    print("Exporting geometry case from the selected claim case...")
    claim_query = query_samples[int(selected_claim["query_index"])]
    geometry_meta = export_geometry_case(
        out_dir=geometry_dir,
        query=claim_query,
        tfm=tfm,
        model=model,
        device=device,
        gallery_samples=gallery_samples,
        batch_size=args.batch_size,
        perceptual_query_path=query_paths["zerodcepp"][int(selected_claim["query_index"])],
    )

    summary = {
        "device": str(device),
        "config": args.config,
        "riic_ckpt": args.riic_ckpt,
        "claim_candidates": claim_candidates,
        "selected_claim_case": claim_meta,
        "selected_trust_case": trust_meta,
        "selected_geometry_case": geometry_meta,
    }
    save_json(output_root / "case_summary.json", summary)
    print(f"Saved RIIC-ReID main-paper assets to {output_root}")


if __name__ == "__main__":
    main()
