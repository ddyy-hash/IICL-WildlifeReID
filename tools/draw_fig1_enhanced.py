#!/usr/bin/env python3
"""Generate enhanced Fig 1 visualizations for RIIC-ReID paper.

Two sub-figures:
  (a) Per-stripe cosine similarity to positive gallery centre
  (b) PCA scatter of query embeddings vs positive/negative gallery

Requires: conda activate dog_train
Usage:    python tools/draw_fig1_enhanced.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


# ---------------------------------------------------------------------------
# Paths  (same defaults as mine_riic_main_paper_cases.py)
# ---------------------------------------------------------------------------
CONFIG          = "config/illumination_config_atrw.yaml"
CKPT            = "checkpoints/atrw_routeb_theoryB/joint_best.pth"
QUERY_DIR       = "data/processed/atrw/query"
GALLERY_DIR     = "data/processed/atrw/gallery"
RETINEX_DIR     = (
    "downloads/westc_perceptual_assets_20260325/root/autodl-tmp/v2_2/"
    "dog_reid_web/data/perceptual_baselines/atrw/retinexnet/test"
)
ZERODCE_DIR     = (
    "downloads/westc_perceptual_assets_20260325/root/autodl-tmp/v2_2/"
    "dog_reid_web/data/perceptual_baselines/atrw/zerodcepp/test"
)
OUTPUT_DIR      = "docs/figures/riic_reid_main_paper_20260327/final"

# The claim case selected by mine_riic_main_paper_cases.py
CLAIM_META_PATH = "docs/figures/riic_reid_main_paper_20260327/assets/claim_case/metrics.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ---------------------------------------------------------------------------
# Model loading  (reuse logic from mine script)
# ---------------------------------------------------------------------------

def infer_num_classes(state_dict: dict) -> int:
    for key, value in state_dict.items():
        if (isinstance(value, torch.Tensor) and value.ndim == 2
                and key.endswith(("classifier.weight", "global_classifier.weight", "arcface.weight"))):
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
    for key in ("feature_fusion", "branch_attention_fusion", "nuisance_head",
                "reid_head", "backbone_random_erasing"):
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
    keep = {k: v for k, v in state.items()
            if k in model_state and isinstance(v, torch.Tensor) and v.shape == model_state[k].shape}
    model.load_state_dict(keep, strict=False)
    return model.to(device).eval()


def build_transform(config_path: str):
    cfg = load_config(config_path)
    train_cfg = cfg.get("training", {})
    h = int(train_cfg.get("image_height", 256))
    w = int(train_cfg.get("image_width", 384))
    return transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor()])


def load_rgb(path: str, tfm) -> torch.Tensor:
    return tfm(Image.open(path).convert("RGB"))


# ---------------------------------------------------------------------------
# Per-stripe feature extraction (hooks into LocalFeatureExtractor internals)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_stripe_features(
    model: JointReIDModel,
    image_tensor: torch.Tensor,
    mode: str = "riic",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (stripe_features [S, D], global_feature [D]).

    mode: 'riic' = with illumination correction
          'raw'  = bypass illumination, raw backbone only
    """
    device = next(model.parameters()).device
    x = image_tensor.unsqueeze(0).to(device)

    if mode == "riic":
        out = model(x, return_illuminated=True)
    else:
        # Raw path: skip illumination, go straight through backbone
        normalized = model._prepare_backbone_input(x)
        feature_map = model.extract_backbone_features(normalized)
        # We need to manually walk through the local extractor
        # to get per-stripe features
        pass

    # For both modes, we need the feature map.  Let's extract it cleanly.
    if mode == "riic":
        # Re-run to get feature_map (the model doesn't return it directly)
        pass

    # Simpler approach: hook into the local extractor
    le = model.local_extractor
    if not hasattr(le, 'stripe_convs'):
        raise RuntimeError("Model uses PlainGlobalExtractor, no stripe features available")

    # Get the feature map
    if mode == "riic":
        # Full RIIC pipeline
        with torch.amp.autocast(device_type=device.type, enabled=False):
            illum_images = x.float()
            coarse_out, feat_mid = model._compute_illumination_guidance(illum_images)
            identity_protection_map = model._build_identity_protection_map(
                feat_mid, image_size=(x.shape[2], x.shape[3]))
            ipaid_details = model.illumination.forward_refine(
                illum_images, coarse_out, feat_mid,
                identity_protection_map=identity_protection_map)
        illuminated = ipaid_details['reflectance']

        # Branch attention fusion path
        if model.branch_attention_fusion_enabled:
            fusion_stats = model._build_feature_fusion_stats(ipaid_details)
            raw_fm = model._extract_raw_feature_map_for_fusion(x)
            base_fm = model.extract_backbone_features(
                model._prepare_backbone_input(ipaid_details.get("reflectance_base", illuminated)))
            adapted_fm = model.extract_backbone_features(
                model._prepare_backbone_input(illuminated))
            feature_map, _ = model._maybe_fuse_branch_feature_maps(
                [raw_fm, base_fm, adapted_fm], aux_stats=fusion_stats)
        else:
            feature_map = model.extract_backbone_features(
                model._prepare_backbone_input(illuminated))
            raw_fm = model._extract_raw_feature_map_for_fusion(x)
            fusion_stats = model._build_feature_fusion_stats(ipaid_details)
            feature_map = model._maybe_fuse_feature_maps(raw_fm, feature_map, aux_stats=fusion_stats)
    else:
        normalized = model._prepare_backbone_input(x)
        feature_map = model.extract_backbone_features(normalized)

    # Now extract per-stripe features from feature_map
    B, C, H, W = feature_map.shape
    num_stripes = le.num_stripes
    stripe_feats = []
    for i in range(num_stripes):
        stripe_h = H // num_stripes
        if le.use_deformable_stripes:
            base_center = (i + 0.5) * stripe_h
            offset = torch.tanh(le.stripe_offsets[i]) * le.max_offset_ratio * H
            center = base_center + offset
            start_h = int(torch.clamp(center - stripe_h / 2.0, 0, max(H - 1, 0)).item())
            end_h = int(torch.clamp(center + stripe_h / 2.0, start_h + 1, H).item())
        else:
            start_h = i * stripe_h
            end_h = (i + 1) * stripe_h if i < num_stripes - 1 else H
        stripe = feature_map[:, :, start_h:end_h, :]
        stripe_feat = le.stripe_convs[i](stripe)
        stripe_feat = F.adaptive_avg_pool2d(stripe_feat, 1).flatten(1)
        stripe_feats.append(stripe_feat.squeeze(0))  # [D]

    global_feat = le.global_conv(feature_map)
    global_feat = F.adaptive_avg_pool2d(global_feat, 1).flatten(1).squeeze(0)  # [D]

    stripe_feats = torch.stack(stripe_feats, dim=0)  # [S, D]
    return stripe_feats.cpu(), global_feat.cpu()


@torch.no_grad()
def extract_fused_feature(
    model: JointReIDModel,
    image_tensor: torch.Tensor,
    mode: str = "riic",
) -> torch.Tensor:
    """Return the final fused ReID feature vector [D]."""
    device = next(model.parameters()).device
    x = image_tensor.unsqueeze(0).to(device)
    if mode == "riic":
        out = model(x, return_illuminated=False)
    else:
        out = model.forward_raw_reference(x, detach=True)
    return F.normalize(out["features"], p=2, dim=1).cpu().squeeze(0)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_split_paths(root_dir: str) -> List[Tuple[str, str]]:
    """Return list of (label, path) tuples."""
    root = Path(root_dir)
    samples = []
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        for img_path in sorted(label_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTS:
                samples.append((label_dir.name, str(img_path)))
    return samples


def resolve_enhanced(stem: str, root: Path) -> str:
    preferred = root / f"{stem}.jpg"
    if preferred.exists():
        return str(preferred)
    matches = sorted(root.glob(f"{stem}.*"))
    return str(matches[0]) if matches else ""


def pca_2d(features: np.ndarray) -> np.ndarray:
    """Project to 2D via PCA."""
    x = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load claim case metadata to know which query to use
    claim_meta = json.loads(Path(CLAIM_META_PATH).read_text(encoding="utf-8"))
    query_label = claim_meta["query_label"]
    query_relpath = claim_meta["query_relpath"]
    # Extract the query stem (e.g. "001132" from "query/21/001132.jpg")
    query_stem = Path(query_relpath).stem

    print(f"Claim case: label={query_label}, relpath={query_relpath}")

    # Load model & transform
    tfm = build_transform(CONFIG)
    model = load_model(CKPT, CONFIG, device)

    # Load the specific query image
    query_samples = load_split_paths(QUERY_DIR)
    query_path = None
    for lbl, pth in query_samples:
        if lbl == query_label and Path(pth).stem == query_stem:
            query_path = pth
            break
    if query_path is None:
        raise RuntimeError(f"Could not find query {query_relpath}")

    print(f"Query path: {query_path}")

    # Resolve enhanced paths for this query
    retinex_path = resolve_enhanced(query_stem, Path(RETINEX_DIR))
    zerodce_path = resolve_enhanced(query_stem, Path(ZERODCE_DIR))
    print(f"RetinexNet path: {retinex_path}")
    print(f"ZeroDCE++ path: {zerodce_path}")

    # Load gallery
    gallery_samples = load_split_paths(GALLERY_DIR)
    pos_gallery = [(lbl, pth) for lbl, pth in gallery_samples if lbl == query_label]
    neg_gallery = [(lbl, pth) for lbl, pth in gallery_samples if lbl != query_label]
    print(f"Positive gallery: {len(pos_gallery)}, Negative gallery: {len(neg_gallery)}")

    # ========================================================================
    # (a) Per-stripe cosine similarity to positive gallery centre
    # ========================================================================
    print("\n=== Extracting per-stripe features ===")

    # Query features for each method
    query_raw_tensor = load_rgb(query_path, tfm)
    query_retinex_tensor = load_rgb(retinex_path, tfm) if retinex_path else query_raw_tensor
    query_zerodce_tensor = load_rgb(zerodce_path, tfm) if zerodce_path else query_raw_tensor

    stripe_raw, global_raw = extract_stripe_features(model, query_raw_tensor, mode="raw")
    stripe_riic, global_riic = extract_stripe_features(model, query_raw_tensor, mode="riic")
    stripe_retinex, global_retinex = extract_stripe_features(model, query_retinex_tensor, mode="raw")
    stripe_zerodce, global_zerodce = extract_stripe_features(model, query_zerodce_tensor, mode="raw")

    print(f"Stripe features shape: {stripe_raw.shape}")

    # Gallery positive centre (per-stripe) — use raw backbone features
    print("Extracting positive gallery stripe features...")
    gallery_stripe_sum = None
    gallery_global_sum = None
    n_pos = 0
    for lbl, pth in pos_gallery:
        g_tensor = load_rgb(pth, tfm)
        g_stripe, g_global = extract_stripe_features(model, g_tensor, mode="raw")
        if gallery_stripe_sum is None:
            gallery_stripe_sum = g_stripe
            gallery_global_sum = g_global
        else:
            gallery_stripe_sum = gallery_stripe_sum + g_stripe
            gallery_global_sum = gallery_global_sum + g_global
        n_pos += 1
    gallery_stripe_centre = F.normalize(gallery_stripe_sum / n_pos, p=2, dim=1)  # [S, D]
    gallery_global_centre = F.normalize((gallery_global_sum / n_pos).unsqueeze(0), p=2, dim=1).squeeze(0)

    # Compute per-stripe cosine similarity
    num_stripes = stripe_raw.shape[0]
    methods = {
        "Raw": stripe_raw,
        "RetinexNet": stripe_retinex,
        "Zero-DCE++": stripe_zerodce,
        "RIIC-ReID": stripe_riic,
    }
    stripe_sims = {}
    for name, stripe_feat in methods.items():
        stripe_feat_norm = F.normalize(stripe_feat, p=2, dim=1)
        sims = (stripe_feat_norm * gallery_stripe_centre).sum(dim=1).numpy()
        stripe_sims[name] = sims
        print(f"  {name}: stripe sims = {sims}")

    # Also compute global cosine similarity
    global_sims = {}
    global_methods = {
        "Raw": global_raw,
        "RetinexNet": global_retinex,
        "Zero-DCE++": global_zerodce,
        "RIIC-ReID": global_riic,
    }
    for name, g_feat in global_methods.items():
        g_feat_norm = F.normalize(g_feat.unsqueeze(0), p=2, dim=1).squeeze(0)
        sim = (g_feat_norm * gallery_global_centre).sum().item()
        global_sims[name] = sim
        print(f"  {name}: global sim = {sim:.4f}")

    # ========================================================================
    # (b) Feature space PCA scatter
    # ========================================================================
    print("\n=== Extracting embeddings for PCA scatter ===")

    # Positive gallery embeddings
    pos_feats = []
    for lbl, pth in pos_gallery:
        feat = extract_fused_feature(model, load_rgb(pth, tfm), mode="raw")
        pos_feats.append(feat.numpy())
    pos_feats = np.stack(pos_feats, axis=0)

    # Hard negative gallery embeddings (nearest to positive centre)
    pos_centre = pos_feats.mean(axis=0, keepdims=True)
    neg_feats_all = []
    neg_labels_all = []
    for lbl, pth in neg_gallery:
        feat = extract_fused_feature(model, load_rgb(pth, tfm), mode="raw")
        neg_feats_all.append(feat.numpy())
        neg_labels_all.append(lbl)
    neg_feats_all = np.stack(neg_feats_all, axis=0)
    neg_dists = np.linalg.norm(neg_feats_all - pos_centre, axis=1)
    hard_neg_idx = np.argsort(neg_dists)[:8]
    hard_neg_feats = neg_feats_all[hard_neg_idx]

    # Query embeddings
    query_feat_raw = extract_fused_feature(model, query_raw_tensor, mode="raw").numpy()
    query_feat_riic = extract_fused_feature(model, query_raw_tensor, mode="riic").numpy()
    query_feat_retinex = extract_fused_feature(model, query_retinex_tensor, mode="raw").numpy()
    query_feat_zerodce = extract_fused_feature(model, query_zerodce_tensor, mode="raw").numpy()

    # PCA projection
    all_feats = np.concatenate([
        pos_feats,
        hard_neg_feats,
        query_feat_raw[None],
        query_feat_retinex[None],
        query_feat_zerodce[None],
        query_feat_riic[None],
    ], axis=0)
    proj = pca_2d(all_feats)
    n_pos_pts = len(pos_feats)
    n_neg_pts = len(hard_neg_feats)
    pos_proj = proj[:n_pos_pts]
    neg_proj = proj[n_pos_pts:n_pos_pts + n_neg_pts]
    query_projs = proj[n_pos_pts + n_neg_pts:]  # raw, retinex, zerodce, riic

    # ========================================================================
    # Draw the combined figure
    # ========================================================================
    print("\n=== Drawing figure ===")
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 5.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1], wspace=0.30)

    # --- (a) Per-stripe bar chart ---
    ax1 = fig.add_subplot(gs[0])
    x_positions = np.arange(num_stripes + 1)  # +1 for global
    bar_width = 0.19
    colors = {"Raw": "#8B8B8B", "RetinexNet": "#E8833A", "Zero-DCE++": "#4C9BD6", "RIIC-ReID": "#2CA02C"}
    stripe_labels = [f"S{i+1}" for i in range(num_stripes)] + ["Global"]

    for idx, (name, sims) in enumerate(stripe_sims.items()):
        values = np.append(sims, global_sims[name])
        offset = (idx - 1.5) * bar_width
        bars = ax1.bar(x_positions + offset, values, bar_width,
                       label=name, color=colors[name], edgecolor="white", linewidth=0.5)
        # Add value labels on RIIC bars
        if name == "RIIC-ReID":
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold",
                         color=colors[name])

    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(stripe_labels, fontsize=9)
    ax1.set_ylabel("Cosine similarity to positive gallery centre", fontsize=9.5)
    ax1.set_title("(a) Per-stripe feature similarity", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8.5, loc="lower right", framealpha=0.85)
    ax1.set_ylim(0, min(1.0, max(max(v) for v in stripe_sims.values()) + 0.08))
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- (b) PCA scatter ---
    ax2 = fig.add_subplot(gs[1])

    # Draw a convex hull / ellipse around positive gallery
    from matplotlib.patches import Ellipse
    pos_mean = pos_proj.mean(axis=0)
    pos_std = pos_proj.std(axis=0)
    ellipse = Ellipse(xy=pos_mean, width=4 * pos_std[0], height=4 * pos_std[1],
                      angle=0, facecolor="#2CA02C", alpha=0.08, edgecolor="#2CA02C",
                      linestyle="--", linewidth=1.2, label="Positive manifold")
    ax2.add_patch(ellipse)

    ax2.scatter(pos_proj[:, 0], pos_proj[:, 1], c="#2CA02C", s=40, marker="o",
                alpha=0.6, label=f"Gallery (ID={query_label})", zorder=3, edgecolors="white", linewidth=0.5)
    ax2.scatter(neg_proj[:, 0], neg_proj[:, 1], c="#D62728", s=40, marker="x",
                alpha=0.7, label="Hard negatives", zorder=3)

    # Query points with distinct markers
    query_markers = [
        (query_projs[0], "Raw", "^", "#8B8B8B", 110),
        (query_projs[1], "RetinexNet", "s", "#E8833A", 100),
        (query_projs[2], "Zero-DCE++", "D", "#4C9BD6", 90),
        (query_projs[3], "RIIC-ReID", "*", "#2CA02C", 200),
    ]
    for pt, label, marker, color, size in query_markers:
        ax2.scatter(pt[0], pt[1], c=color, s=size, marker=marker, label=f"Query ({label})",
                    zorder=5, edgecolors="black", linewidth=0.8)

    # Draw arrows from raw -> each method
    for i, (pt, label, _, color, _) in enumerate(query_markers[1:], start=1):
        ax2.annotate("", xy=(pt[0], pt[1]),
                     xytext=(query_projs[0][0], query_projs[0][1]),
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.7))

    ax2.set_xlabel("PC 1", fontsize=9.5)
    ax2.set_ylabel("PC 2", fontsize=9.5)
    ax2.set_title("(b) Feature space embedding (PCA)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=7.5, loc="best", framealpha=0.85, ncol=1)
    ax2.grid(alpha=0.2, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        f"RIIC-ReID feature-level analysis  —  Query ID {query_label} (low-light)",
        fontsize=12, fontweight="bold", y=0.98,
    )

    for fmt in ("png", "pdf"):
        save_path = out_dir / f"fig1_enhanced_stripe_pca.{fmt}"
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.close(fig)

    # Save metrics JSON
    metrics = {
        "query_label": query_label,
        "query_relpath": query_relpath,
        "stripe_cosine_similarities": {k: v.tolist() for k, v in stripe_sims.items()},
        "global_cosine_similarities": global_sims,
        "pca_projections": {
            "positive_gallery": pos_proj.tolist(),
            "hard_negatives": neg_proj.tolist(),
            "query_raw": query_projs[0].tolist(),
            "query_retinexnet": query_projs[1].tolist(),
            "query_zerodcepp": query_projs[2].tolist(),
            "query_riic": query_projs[3].tolist(),
        },
    }
    metrics_path = out_dir / "fig1_enhanced_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved metrics: {metrics_path}")
    print("Done!")


if __name__ == "__main__":
    main()
