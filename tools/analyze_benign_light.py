#!/usr/bin/env python3
"""Analyze benign-light behavior for the current RIIC full-model checkpoint.

This script builds a bright benign-light proxy subset from real ATRW query
images using only original-image brightness statistics, then compares the
same full checkpoint under:

1. normal inference (IPAID/RIIC enabled), and
2. bypassed illumination front-end (raw-branch proxy).

It reports retrieval deltas on the subset and summarizes no-harm signals such
as small image perturbation, rollback-to-raw strength, and raw-branch attention.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.visualize_joint_analysis import (  # noqa: E402
    FeaturePack,
    _resolve_device,
    build_model_from_checkpoint,
    compute_image_stats,
    evaluate_reid,
    forward_features,
    make_dataloaders,
)


def _mean_gray_brightness(imgs: torch.Tensor) -> torch.Tensor:
    """Compute grayscale brightness in [0, 1] for a BCHW image batch."""
    weights = imgs.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    gray = (imgs * weights).sum(dim=1)
    return gray.mean(dim=(1, 2))


def _std_gray_contrast(imgs: torch.Tensor) -> torch.Tensor:
    """Compute grayscale contrast as std(gray) for a BCHW image batch."""
    weights = imgs.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    gray = (imgs * weights).sum(dim=1)
    return gray.flatten(1).std(dim=1, unbiased=False)


def _slice_feature_pack(pack: FeaturePack, indices: Sequence[int]) -> FeaturePack:
    idx = np.asarray(indices, dtype=np.int64)
    return FeaturePack(
        feats=pack.feats[idx],
        ids=[pack.ids[i] for i in idx],
        cams=[pack.cams[i] for i in idx],
        paths=[pack.paths[i] for i in idx],
    )


def _safe_mean_tensor(x: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
    if x is None:
        return torch.full((batch_size,), np.nan, device=device)
    return x.view(batch_size, -1).mean(dim=1)


@torch.no_grad()
def collect_full_model_diagnostics(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Collect per-query diagnostics for the full model only."""
    model.eval()

    rows: List[Dict[str, Any]] = []
    prev_flag = getattr(model, "use_ipaid", None)
    if prev_flag is not None:
        model.use_ipaid = True

    for batch in loader:
        imgs = batch[0]
        ids = batch[1]
        cams = batch[2]
        paths = batch[3]

        imgs_device = imgs.to(device)
        output = model(imgs_device, boxes_list=None, return_illuminated=True)
        features = F.normalize(output["features"], p=2, dim=1).cpu().numpy()
        illuminated = output.get("illuminated", imgs_device)
        ipaid_details = output.get("ipaid_details") or {}

        batch_size = imgs_device.size(0)
        pixel_diff = (illuminated - imgs_device).abs().mean(dim=(1, 2, 3)).detach()
        brightness = _mean_gray_brightness(imgs).detach()
        contrast = _std_gray_contrast(imgs).detach()

        rollback_alpha = _safe_mean_tensor(ipaid_details.get("rollback_alpha"), batch_size, device)
        correction_gap = _safe_mean_tensor(ipaid_details.get("correction_gap"), batch_size, device)
        color_risk = _safe_mean_tensor(ipaid_details.get("color_risk"), batch_size, device)

        branch_attention = ipaid_details.get("branch_attention_weights")
        if branch_attention is None:
            raw_branch_weight = torch.full((batch_size,), np.nan, device=device)
        else:
            raw_branch_weight = branch_attention[:, :, 0].mean(dim=1)

        if isinstance(ids, torch.Tensor):
            batch_ids = ids.tolist()
        else:
            batch_ids = list(ids)
        if isinstance(cams, torch.Tensor):
            batch_cams = [int(x) for x in cams.tolist()]
        else:
            batch_cams = [int(x) for x in cams]
        batch_paths = [str(p) for p in paths]

        # Keep a simple colorfulness statistic on the resized original image.
        colorfulness_list: List[float] = []
        for img_t in imgs:
            arr = np.clip(img_t.numpy().transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
            colorfulness_list.append(float(compute_image_stats(arr)["colorfulness"]))

        for i in range(batch_size):
            rows.append(
                {
                    "index": len(rows),
                    "path": batch_paths[i],
                    "id": batch_ids[i],
                    "cam": batch_cams[i],
                    "brightness": float(brightness[i].item()),
                    "contrast": float(contrast[i].item()),
                    "colorfulness": float(colorfulness_list[i]),
                    "pixel_diff_l1": float(pixel_diff[i].item()),
                    "rollback_alpha_mean": float(rollback_alpha[i].item()) if not torch.isnan(rollback_alpha[i]) else None,
                    "rollback_to_raw_share": (
                        float(1.0 - rollback_alpha[i].item()) if not torch.isnan(rollback_alpha[i]) else None
                    ),
                    "raw_branch_weight_mean": (
                        float(raw_branch_weight[i].item()) if not torch.isnan(raw_branch_weight[i]) else None
                    ),
                    "correction_gap_mean": (
                        float(correction_gap[i].item()) if not torch.isnan(correction_gap[i]) else None
                    ),
                    "color_risk_mean": float(color_risk[i].item()) if not torch.isnan(color_risk[i]) else None,
                    "feature_norm_enabled": float(np.linalg.norm(features[i])),
                }
            )

    if prev_flag is not None:
        model.use_ipaid = prev_flag

    return rows


def _feature_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.clip(denom, 1e-12, None)
    return np.sum(a * b, axis=1) / denom


def _summarize_numeric(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Dict[str, float | None]]:
    summary: Dict[str, Dict[str, float | None]] = {}
    for key in keys:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        if not vals:
            summary[key] = {"mean": None, "std": None, "median": None}
            continue
        arr = np.asarray(vals, dtype=np.float64)
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
        }
    return summary


def _metric_delta(enabled: Dict[str, float], bypass: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in enabled:
        out[f"{key}_delta"] = float(enabled[key] - bypass[key])
    return out


def save_brightness_histogram(
    brightness: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(brightness, bins=30, color="#4E79A7", alpha=0.8, edgecolor="white")
    ax.axvline(threshold, color="#E15759", linestyle="--", linewidth=2, label=f"q75={threshold:.3f}")
    ax.set_title("ATRW Query Brightness Distribution")
    ax.set_xlabel("Mean Grayscale Brightness")
    ax.set_ylabel("Query Count")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_no_harm_boxplots(
    rows_all: Sequence[Dict[str, Any]],
    rows_subset: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    metrics = [
        ("pixel_diff_l1", "Mean |Illuminated - Original|"),
        ("feature_cosine_enabled_vs_bypass", "Cosine(Enabled, Bypass)"),
        ("raw_branch_weight_mean", "Raw Branch Weight"),
        ("rollback_to_raw_share", "Rollback-to-Raw Share"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, metrics):
        all_vals = [float(r[key]) for r in rows_all if r.get(key) is not None]
        subset_vals = [float(r[key]) for r in rows_subset if r.get(key) is not None]
        if all_vals and subset_vals:
            bp = ax.boxplot([all_vals, subset_vals], patch_artist=True, showfliers=False)
            colors = ["#A0CBE8", "#F28E2B"]
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            ax.set_xticklabels(["All queries", "Bright benign-light proxy"])
        ax.set_title(title)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary_markdown(summary: Dict[str, Any], output_path: Path) -> None:
    subset = summary["subset_definition"]
    retrieval = summary["retrieval"]
    no_harm = summary["no_harm"]

    lines = [
        "# Benign-Light Proxy Analysis",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Query set size: `{summary['query_count']}`",
        f"- Gallery size: `{summary['gallery_count']}`",
        (
            f"- Bright benign-light proxy: top `{int((1.0 - subset['quantile']) * 100)}`% by original-query "
            f"brightness (`threshold={subset['brightness_threshold']:.4f}`, `count={subset['count']}`)"
        ),
        "",
        "## Retrieval",
        "",
        "### All queries",
        "",
        f"- Enabled mAP: `{retrieval['all_queries']['enabled']['mAP']:.2f}`",
        f"- Bypass mAP: `{retrieval['all_queries']['bypass']['mAP']:.2f}`",
        f"- Enabled Rank-1: `{retrieval['all_queries']['enabled']['rank1']:.2f}`",
        f"- Bypass Rank-1: `{retrieval['all_queries']['bypass']['rank1']:.2f}`",
        "",
        "### Bright benign-light proxy subset",
        "",
        f"- Enabled mAP: `{retrieval['subset']['enabled']['mAP']:.2f}`",
        f"- Bypass mAP: `{retrieval['subset']['bypass']['mAP']:.2f}`",
        f"- Enabled Rank-1: `{retrieval['subset']['enabled']['rank1']:.2f}`",
        f"- Bypass Rank-1: `{retrieval['subset']['bypass']['rank1']:.2f}`",
        "",
        "## No-Harm Signals on the Bright Proxy Subset",
        "",
        f"- Mean pixel perturbation: `{no_harm['subset_stats']['pixel_diff_l1']['mean']:.6f}`",
        f"- Mean feature cosine between enabled and bypass descriptors: `{no_harm['subset_stats']['feature_cosine_enabled_vs_bypass']['mean']:.6f}`",
    ]

    raw_branch = no_harm["subset_stats"]["raw_branch_weight_mean"]["mean"]
    rollback = no_harm["subset_stats"]["rollback_to_raw_share"]["mean"]
    if raw_branch is not None:
        lines.append(f"- Mean raw-branch attention weight: `{raw_branch:.6f}`")
    if rollback is not None:
        lines.append(f"- Mean rollback-to-raw share: `{rollback:.6f}`")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is a bright benign-light proxy based only on original-image brightness, not a curated controlled-light benchmark.",
            "- The analysis compares the same full checkpoint with the illumination front-end enabled vs bypassed.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze benign-light behavior for a RIIC full checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/atrw_new_version_1/joint_best_reid_best.pth",
        help="full-model checkpoint path",
    )
    parser.add_argument("--query_dir", type=str, default="data/processed/atrw/query", help="query directory")
    parser.add_argument("--gallery_dir", type=str, default="data/processed/atrw/gallery", help="gallery directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/benign_light_20260402",
        help="output directory",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument("--backbone", type=str, default="osnet_ain_x1_0", help="fallback backbone")
    parser.add_argument("--num_classes", type=int, default=107, help="fallback num classes")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_width", type=int, default=384)
    parser.add_argument(
        "--brightness_quantile",
        type=float,
        default=0.75,
        help="top brightness quantile used to define the bright benign-light proxy subset",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    print(f"[1/6] Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        checkpoint = {"state_dict": checkpoint}
    model = build_model_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
        fallback_backbone=args.backbone,
        fallback_num_classes=args.num_classes,
    )

    print("[2/6] Building ATRW query/gallery dataloaders")
    q_ds, g_ds, q_loader, g_loader = make_dataloaders(
        query_dir=args.query_dir,
        gallery_dir=args.gallery_dir,
        img_height=args.img_height,
        img_width=args.img_width,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    print(f"  query={len(q_ds)} gallery={len(g_ds)}")

    print("[3/6] Extracting retrieval features (enabled / bypass)")
    q_enabled = forward_features(model, q_loader, device, bypass_ipaid=False)
    g_enabled = forward_features(model, g_loader, device, bypass_ipaid=False)
    q_bypass = forward_features(model, q_loader, device, bypass_ipaid=True)
    g_bypass = forward_features(model, g_loader, device, bypass_ipaid=True)

    print("[4/6] Collecting per-query benign-light diagnostics")
    diagnostics = collect_full_model_diagnostics(model, q_loader, device)
    if len(diagnostics) != len(q_enabled.paths):
        raise RuntimeError("Diagnostic row count does not match query feature count.")

    feature_cos = _feature_cosine_similarity(q_enabled.feats, q_bypass.feats)
    for row, cos_val in zip(diagnostics, feature_cos):
        row["feature_cosine_enabled_vs_bypass"] = float(cos_val)

    brightness = np.asarray([row["brightness"] for row in diagnostics], dtype=np.float64)
    threshold = float(np.quantile(brightness, args.brightness_quantile))
    subset_indices = [i for i, row in enumerate(diagnostics) if float(row["brightness"]) >= threshold]
    subset_rows = [diagnostics[i] for i in subset_indices]
    print(f"  brightness threshold={threshold:.4f}, subset={len(subset_indices)}/{len(diagnostics)}")

    print("[5/6] Evaluating retrieval on all queries and the bright benign-light proxy subset")
    q_enabled_subset = _slice_feature_pack(q_enabled, subset_indices)
    q_bypass_subset = _slice_feature_pack(q_bypass, subset_indices)

    retrieval_all_enabled = evaluate_reid(q_enabled, g_enabled)
    retrieval_all_bypass = evaluate_reid(q_bypass, g_bypass)
    retrieval_subset_enabled = evaluate_reid(q_enabled_subset, g_enabled)
    retrieval_subset_bypass = evaluate_reid(q_bypass_subset, g_bypass)

    print("[6/6] Writing summaries and plots")
    numeric_keys = [
        "brightness",
        "contrast",
        "colorfulness",
        "pixel_diff_l1",
        "feature_cosine_enabled_vs_bypass",
        "rollback_alpha_mean",
        "rollback_to_raw_share",
        "raw_branch_weight_mean",
        "correction_gap_mean",
        "color_risk_mean",
    ]

    summary: Dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "query_dir": str(Path(args.query_dir).resolve()),
        "gallery_dir": str(Path(args.gallery_dir).resolve()),
        "query_count": len(q_ds),
        "gallery_count": len(g_ds),
        "subset_definition": {
            "name": "bright_benign_light_proxy",
            "brightness_metric": "mean grayscale brightness on resized original query image",
            "quantile": args.brightness_quantile,
            "brightness_threshold": threshold,
            "count": len(subset_indices),
            "fraction": float(len(subset_indices) / max(len(diagnostics), 1)),
        },
        "retrieval": {
            "all_queries": {
                "enabled": retrieval_all_enabled,
                "bypass": retrieval_all_bypass,
                "delta": _metric_delta(retrieval_all_enabled, retrieval_all_bypass),
            },
            "subset": {
                "enabled": retrieval_subset_enabled,
                "bypass": retrieval_subset_bypass,
                "delta": _metric_delta(retrieval_subset_enabled, retrieval_subset_bypass),
            },
        },
        "no_harm": {
            "all_query_stats": _summarize_numeric(diagnostics, numeric_keys),
            "subset_stats": _summarize_numeric(subset_rows, numeric_keys),
        },
        "artifacts": {
            "brightness_histogram": "brightness_histogram.png",
            "no_harm_boxplots": "no_harm_boxplots.png",
            "diagnostics_csv": "per_query_diagnostics.csv",
            "summary_json": "summary.json",
            "summary_md": "summary.md",
        },
    }

    save_brightness_histogram(brightness, threshold, output_dir / "brightness_histogram.png")
    save_no_harm_boxplots(diagnostics, subset_rows, output_dir / "no_harm_boxplots.png")

    csv_path = output_dir / "per_query_diagnostics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(diagnostics[0].keys()))
        writer.writeheader()
        writer.writerows(diagnostics)

    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_markdown(summary, output_dir / "summary.md")

    print(json.dumps(summary["retrieval"]["subset"]["delta"], ensure_ascii=False, indent=2))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
