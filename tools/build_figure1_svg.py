#!/usr/bin/env python3
"""Build the paper-quality Figure 1 teaser as editable SVG/PNG."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from tools.evaluate_reid import _build_model
from tools.prepare_rift_paper_figure_assets import (
    Sample,
    build_transform,
    compute_ap,
    filter_display_order,
    first_idx,
    load_model as load_rift_model,
    load_rgb,
    pca2,
    to_u8,
)

RAW_COLOR = "#6B7280"
GENERIC_COLOR = "#2563EB"
RIFT_COLOR = "#D97706"
POS_COLOR = "#16A34A"
NEG_COLOR = "#DC2626"
TEXT_COLOR = "#111827"
SUBTLE_TEXT = "#4B5563"
BOX_EDGE = "#D1D5DB"
BOX_FILL = "#F8FAFC"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Figure 1 teaser SVG")
    parser.add_argument("--query_relpath", default="query/112/003457.jpg")
    parser.add_argument(
        "--rift_ckpt",
        default=str(PROJECT_ROOT / "checkpoints" / "atrw_routeb_theoryB" / "joint_best.pth"),
    )
    parser.add_argument(
        "--baseline_ckpt",
        default=str(
            PROJECT_ROOT
            / "downloads"
            / "westc_perceptual_assets_20260325"
            / "root"
            / "dog_reid_storage"
            / "checkpoints"
            / "perceptual_baselines"
            / "atrw"
            / "zerodcepp"
            / "osnet_ain_x1_0"
            / "baseline_best.pth"
        ),
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "illumination_config_atrw.yaml"),
    )
    parser.add_argument(
        "--raw_query_root",
        default=str(PROJECT_ROOT / "data" / "processed" / "atrw" / "query"),
    )
    parser.add_argument(
        "--raw_gallery_root",
        default=str(PROJECT_ROOT / "data" / "processed" / "atrw" / "gallery"),
    )
    parser.add_argument(
        "--enhanced_root",
        default=str(
            PROJECT_ROOT
            / "downloads"
            / "westc_perceptual_assets_20260325"
            / "root"
            / "autodl-tmp"
            / "v2_2"
            / "dog_reid_web"
            / "data"
            / "perceptual_baselines"
            / "atrw"
            / "zerodcepp"
            / "test"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "docs" / "figures" / "rift_paper_20260325" / "fig1_svg"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=4)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_status(path: Path, phase: str, extra: dict | None = None) -> None:
    payload = {"phase": phase}
    if extra:
        payload.update(extra)
    save_json(path, payload)
    print(f"[STATUS] {phase}", flush=True)


def load_folder_samples(root: Path, split: str) -> List[Sample]:
    samples: List[Sample] = []
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        for img_path in sorted(label_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            samples.append(
                Sample(
                    split=split,
                    label=label_dir.name,
                    path=str(img_path),
                    relpath=f"{split}/{label_dir.name}/{img_path.name}",
                )
            )
    return samples


def resolve_query(raw_query_samples: List[Sample], query_relpath: str) -> Sample:
    for sample in raw_query_samples:
        if sample.relpath == query_relpath:
            return sample
    raise FileNotFoundError(f"Query sample not found: {query_relpath}")


def build_enhanced_gallery_samples(raw_gallery_samples: List[Sample], enhanced_root: Path) -> List[Sample]:
    enhanced_samples: List[Sample] = []
    missing: List[str] = []
    for sample in raw_gallery_samples:
        enhanced_path = enhanced_root / Path(sample.path).name
        if not enhanced_path.exists():
            missing.append(str(enhanced_path))
            continue
        enhanced_samples.append(
            Sample(
                split=sample.split,
                label=sample.label,
                path=str(enhanced_path),
                relpath=sample.relpath,
            )
        )
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} enhanced gallery images, first: {missing[0]}")
    return enhanced_samples


def extract_descriptor_features(
    model: torch.nn.Module,
    samples: Iterable[Sample],
    tfm,
    device: torch.device,
    batch_size: int,
    use_corrected_forward: bool,
) -> np.ndarray:
    sample_list = list(samples)
    if not sample_list:
        return np.zeros((0, 256), dtype=np.float32)
    model.eval()
    features: List[np.ndarray] = []
    start = 0
    current_bs = max(1, batch_size)
    with torch.no_grad():
        while start < len(sample_list):
            chunk = sample_list[start : start + current_bs]
            try:
                batch = torch.stack([load_rgb(s, tfm) for s in chunk], dim=0).to(device)
                output = model(batch, boxes_list=None, return_illuminated=False)
                feat = F.normalize(output["features"], p=2, dim=1)
                features.append(feat.cpu().numpy().astype(np.float32))
                start += len(chunk)
            except torch.OutOfMemoryError:
                release_cuda()
                if current_bs == 1:
                    raise
                current_bs = max(1, current_bs // 2)
                continue
            finally:
                release_cuda()
    return np.concatenate(features, axis=0)


def extract_raw_branch_features(
    model: torch.nn.Module,
    samples: Iterable[Sample],
    tfm,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    sample_list = list(samples)
    if not sample_list:
        return np.zeros((0, 256), dtype=np.float32)
    model.eval()
    features: List[np.ndarray] = []
    start = 0
    current_bs = max(1, batch_size)
    with torch.no_grad():
        while start < len(sample_list):
            chunk = sample_list[start : start + current_bs]
            try:
                batch = torch.stack([load_rgb(s, tfm) for s in chunk], dim=0).to(device)
                output = model.forward_raw_reference(batch, detach=True)
                feat = F.normalize(output["features"], p=2, dim=1)
                features.append(feat.cpu().numpy().astype(np.float32))
                start += len(chunk)
            except torch.OutOfMemoryError:
                release_cuda()
                if current_bs == 1:
                    raise
                current_bs = max(1, current_bs // 2)
                continue
            finally:
                release_cuda()
    return np.concatenate(features, axis=0)


def load_or_compute(cache_path: Path, fn):
    if cache_path.exists():
        return np.load(cache_path)
    arr = fn()
    ensure_dir(cache_path.parent)
    np.save(cache_path, arr)
    return arr


def release_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def add_card_frame(ax, edgecolor: str, linewidth: float = 1.6) -> None:
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(edgecolor)
        sp.set_linewidth(linewidth)
    ax.set_xticks([])
    ax.set_yticks([])


def add_panel_title(ax, title: str, subtitle: str, color: str) -> None:
    ax.text(
        0.0,
        1.06,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    ax.text(
        0.0,
        1.01,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=color,
    )


def add_badge(ax, text: str, color: str, x: float = 0.03, y: float = 0.96) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="white",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=color, edgecolor=color, linewidth=0.0),
    )


def draw_header(ax) -> None:
    ax.axis("off")
    ax.text(
        0.0,
        0.82,
        "Figure 1. Why model-centric illumination correction?",
        fontsize=18,
        fontweight="bold",
        color=TEXT_COLOR,
        ha="left",
        va="center",
    )
    ax.text(
        0.0,
        0.40,
        "Should correction be defined by human-perceptual brightness, or by the feature space preferred by the model?",
        fontsize=11.5,
        color=SUBTLE_TEXT,
        ha="left",
        va="center",
    )
    ax.text(
        0.0,
        0.08,
        "A real ATRW example. Zero-DCE++ brightens the query more aggressively, while RIFT produces a stronger retrieval outcome and a more favorable move in the RIFT descriptor space.",
        fontsize=10.2,
        color=SUBTLE_TEXT,
        ha="left",
        va="center",
    )


def draw_image_panel(ax, image: np.ndarray, title: str, subtitle: str, color: str, badge: str) -> None:
    ax.imshow(image)
    add_card_frame(ax, BOX_EDGE, linewidth=1.6)
    add_panel_title(ax, title, subtitle, color)
    add_badge(ax, badge, color)


def draw_retrieval_row(fig, subgrid, row_idx: int, row_title: str, corrected_img: np.ndarray, order: np.ndarray, ap_value: float, gallery_samples: List[Sample], gallery_labels: np.ndarray, query_label: str, tfm, topk: int, color: str, query_raw_img: np.ndarray) -> List[plt.Axes]:
    row_axes = [fig.add_subplot(subgrid[row_idx, col]) for col in range(topk + 2)]
    row_axes[0].imshow(query_raw_img)
    add_card_frame(row_axes[0], BOX_EDGE, linewidth=1.2)
    row_axes[0].set_title("Query", fontsize=8.5, color=TEXT_COLOR, pad=5)

    row_axes[1].imshow(corrected_img)
    add_card_frame(row_axes[1], color, linewidth=1.8)
    row_axes[1].set_title(row_title, fontsize=8.5, color=color, pad=5)

    for k in range(topk):
        gi = int(order[k])
        gx = load_rgb(gallery_samples[gi], tfm)
        match = bool(gallery_labels[gi] == query_label)
        border = POS_COLOR if match else NEG_COLOR
        row_axes[k + 2].imshow(to_u8(gx))
        add_card_frame(row_axes[k + 2], border, linewidth=2.8)
        row_axes[k + 2].set_title(
            f"#{k + 1} {'match' if match else 'distractor'}",
            fontsize=7.5,
            color=border,
            pad=5,
        )

    row_axes[1].text(
        0.97,
        0.05,
        f"AP {ap_value:.3f}",
        transform=row_axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=color,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=color, linewidth=0.8, alpha=0.92),
    )
    return row_axes


def draw_feature_bridge(ax, proj: np.ndarray) -> None:
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(BOX_EDGE)
        sp.set_linewidth(1.2)

    names = ["raw query", "Zero-DCE++", "RIFT", "correct match", "distractor"]
    colors = [RAW_COLOR, GENERIC_COLOR, RIFT_COLOR, POS_COLOR, NEG_COLOR]
    markers = ["o", "o", "o", "^", "X"]
    sizes = [90, 96, 106, 110, 112]

    offsets = {
        "raw query": (0.030, 0.012),
        "Zero-DCE++": (0.030, 0.022),
        "RIFT": (-0.055, -0.045),
        "correct match": (0.018, 0.020),
        "distractor": (0.018, 0.020),
    }

    for name, color, marker, size, pt in zip(names, colors, markers, sizes, proj):
        ax.scatter(pt[0], pt[1], s=size, c=color, marker=marker, edgecolors="black", linewidths=0.75, zorder=3)
        dx, dy = offsets.get(name, (0.025, 0.025))
        ax.text(
            pt[0] + dx,
            pt[1] + dy,
            name,
            fontsize=8,
            color=TEXT_COLOR,
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.88),
        )

    ax.add_patch(
        FancyArrowPatch(
            posA=proj[0],
            posB=proj[1],
            arrowstyle="-|>",
            mutation_scale=12,
            lw=1.8,
            color=GENERIC_COLOR,
            alpha=0.9,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            posA=proj[0],
            posB=proj[2],
            arrowstyle="-|>",
            mutation_scale=12,
            lw=2.0,
            color=RIFT_COLOR,
            alpha=0.95,
        )
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="query variants", markerfacecolor=RAW_COLOR, markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="^", color="w", label="correct match", markerfacecolor=POS_COLOR, markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="X", color="w", label="hard distractor", markerfacecolor=NEG_COLOR, markeredgecolor="black", markersize=8),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right", frameon=False)


def add_background_box(fig, axes: List[plt.Axes]) -> tuple[float, float, float, float]:
    boxes = [ax.get_position() for ax in axes]
    x0 = min(b.x0 for b in boxes) - 0.006
    y0 = min(b.y0 for b in boxes) - 0.012
    x1 = max(b.x1 for b in boxes) + 0.006
    y1 = max(b.y1 for b in boxes) + 0.012
    patch = FancyBboxPatch(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        transform=fig.transFigure,
        boxstyle="round,pad=0.006,rounding_size=0.010",
        linewidth=1.0,
        edgecolor=BOX_EDGE,
        facecolor=BOX_FILL,
        zorder=-10,
    )
    fig.patches.append(patch)
    return x0, y0, x1, y1


def add_section_heading(fig, box: tuple[float, float, float, float], title: str, subtitle: str) -> None:
    x0, _, _, y1 = box
    fig.text(x0 + 0.010, y1 + 0.010, title, fontsize=11.5, fontweight="bold", color=TEXT_COLOR, ha="left", va="bottom")
    if subtitle:
        fig.text(x0 + 0.010, y1 - 0.002, subtitle, fontsize=8.8, color=SUBTLE_TEXT, ha="left", va="bottom")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"
    ensure_dir(output_dir)
    ensure_dir(cache_dir)
    status_path = output_dir / "status.json"
    write_status(status_path, "start")

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tfm = build_transform(args.config)

    raw_query_samples = load_folder_samples(Path(args.raw_query_root), "query")
    raw_gallery_samples = load_folder_samples(Path(args.raw_gallery_root), "gallery")
    query_sample = resolve_query(raw_query_samples, args.query_relpath)
    gallery_labels = np.array([s.label for s in raw_gallery_samples], dtype=object)
    write_status(status_path, "loaded_raw_samples", {"query_count": len(raw_query_samples), "gallery_count": len(raw_gallery_samples)})

    enhanced_root = Path(args.enhanced_root)
    enhanced_query_sample = Sample(
        split="query",
        label=query_sample.label,
        path=str(enhanced_root / Path(query_sample.path).name),
        relpath=query_sample.relpath,
    )
    if not Path(enhanced_query_sample.path).exists():
        raise FileNotFoundError(f"Enhanced query image missing: {enhanced_query_sample.path}")
    enhanced_gallery_samples = build_enhanced_gallery_samples(raw_gallery_samples, enhanced_root)
    write_status(status_path, "resolved_enhanced_samples")

    baseline_ckpt = torch.load(args.baseline_ckpt, map_location="cpu", weights_only=False)
    baseline_model = _build_model(baseline_ckpt, baseline=True, device=device)
    baseline_model.eval()
    write_status(status_path, "loaded_baseline_model")

    query_key = f"{query_sample.label}_{Path(query_sample.path).stem}"
    base_gallery_cache = cache_dir / "atrw_zerodcepp_gallery_feats.npy"
    rift_gallery_cache = cache_dir / "atrw_rift_gallery_feats.npy"

    baseline_gallery_feats = load_or_compute(
        base_gallery_cache,
        lambda: extract_descriptor_features(
            baseline_model,
            enhanced_gallery_samples,
            tfm,
            device,
            args.batch_size,
            use_corrected_forward=False,
        ),
    )
    write_status(status_path, "ready_baseline_gallery_cache", {"cache": str(base_gallery_cache)})
    baseline_query_feat = extract_descriptor_features(
        baseline_model,
        [enhanced_query_sample],
        tfm,
        device,
        batch_size=1,
        use_corrected_forward=False,
    )[0]

    del baseline_model
    del baseline_ckpt
    release_cuda()
    write_status(status_path, "released_baseline_model")

    rift_model = load_rift_model(args.rift_ckpt, args.config, device)
    rift_model.eval()
    write_status(status_path, "loaded_rift_model")

    rift_gallery_feats = load_or_compute(
        rift_gallery_cache,
        lambda: extract_descriptor_features(
            rift_model,
            raw_gallery_samples,
            tfm,
            device,
            args.batch_size,
            use_corrected_forward=True,
        ),
    )
    write_status(status_path, "ready_rift_gallery_cache", {"cache": str(rift_gallery_cache)})

    rift_query_feat = extract_descriptor_features(
        rift_model,
        [query_sample],
        tfm,
        device,
        batch_size=1,
        use_corrected_forward=True,
    )[0]

    d_base = 1.0 - baseline_gallery_feats @ baseline_query_feat
    d_rift = 1.0 - rift_gallery_feats @ rift_query_feat
    order_base = np.argsort(d_base)
    order_rift = np.argsort(d_rift)
    display_order_base = filter_display_order(order_base, raw_gallery_samples, query_sample)
    display_order_rift = filter_display_order(order_rift, raw_gallery_samples, query_sample)
    ap_base = compute_ap(d_base, gallery_labels, query_sample.label)
    ap_rift = compute_ap(d_rift, gallery_labels, query_sample.label)
    write_status(status_path, "computed_retrieval_metrics", {"ap_base": float(ap_base), "ap_rift": float(ap_rift)})

    query_raw_img = to_u8(load_rgb(query_sample, tfm))
    query_generic_img = to_u8(load_rgb(enhanced_query_sample, tfm))
    with torch.no_grad():
        rift_out = rift_model(load_rgb(query_sample, tfm).unsqueeze(0).to(device), return_illuminated=True)
    query_rift_img = to_u8(rift_out.get("illuminated")[0].detach().cpu().clamp(0.0, 1.0))

    rift_raw_query_feat = extract_raw_branch_features(rift_model, [query_sample], tfm, device, batch_size=1)[0]
    rift_generic_query_feat = extract_raw_branch_features(rift_model, [enhanced_query_sample], tfm, device, batch_size=1)[0]
    pos_idx = first_idx(display_order_rift, gallery_labels, query_sample.label, True)
    neg_idx = first_idx(display_order_base, gallery_labels, query_sample.label, False)
    pos_feat = extract_raw_branch_features(rift_model, [raw_gallery_samples[pos_idx]], tfm, device, batch_size=1)[0]
    neg_feat = extract_raw_branch_features(rift_model, [raw_gallery_samples[neg_idx]], tfm, device, batch_size=1)[0]
    bridge_proj = pca2(
        np.stack(
            [rift_raw_query_feat, rift_generic_query_feat, rift_query_feat, pos_feat, neg_feat],
            axis=0,
        )
    )
    write_status(status_path, "computed_feature_bridge")

    fig = plt.figure(figsize=(13.8, 8.0), facecolor="white")
    outer = fig.add_gridspec(
        3,
        12,
        height_ratios=[0.95, 2.8, 3.5],
        hspace=0.40,
        wspace=0.40,
    )

    header_ax = fig.add_subplot(outer[0, :])
    draw_header(header_ax)

    top_grid = outer[1, :].subgridspec(1, 3, wspace=0.18)
    ax_raw = fig.add_subplot(top_grid[0, 0])
    ax_generic = fig.add_subplot(top_grid[0, 1])
    ax_rift = fig.add_subplot(top_grid[0, 2])
    draw_image_panel(ax_raw, query_raw_img, "Original query", "low-light input", RAW_COLOR, "A")
    draw_image_panel(ax_generic, query_generic_img, "Generic enhancement", "Zero-DCE++ reference", GENERIC_COLOR, "B")
    draw_image_panel(ax_rift, query_rift_img, "RIFT correction", "model-centric adaptation", RIFT_COLOR, "C")

    retrieval_grid = outer[2, :8].subgridspec(2, args.topk + 2, wspace=0.10, hspace=0.22)
    retrieval_axes_1 = draw_retrieval_row(
        fig,
        retrieval_grid,
        row_idx=0,
        row_title="Zero-DCE++ retrieval",
        corrected_img=query_generic_img,
        order=display_order_base,
        ap_value=ap_base,
        gallery_samples=raw_gallery_samples,
        gallery_labels=gallery_labels,
        query_label=query_sample.label,
        tfm=tfm,
        topk=args.topk,
        color=GENERIC_COLOR,
        query_raw_img=query_raw_img,
    )
    retrieval_axes_2 = draw_retrieval_row(
        fig,
        retrieval_grid,
        row_idx=1,
        row_title="RIFT retrieval",
        corrected_img=query_rift_img,
        order=display_order_rift,
        ap_value=ap_rift,
        gallery_samples=raw_gallery_samples,
        gallery_labels=gallery_labels,
        query_label=query_sample.label,
        tfm=tfm,
        topk=args.topk,
        color=RIFT_COLOR,
        query_raw_img=query_raw_img,
    )

    bridge_ax = fig.add_subplot(outer[2, 8:])
    draw_feature_bridge(bridge_ax, bridge_proj)

    retrieval_box = add_background_box(fig, retrieval_axes_1 + retrieval_axes_2)
    bridge_box = add_background_box(fig, [bridge_ax])
    add_section_heading(
        fig,
        retrieval_box,
        "Retrieval evidence",
        "",
    )
    add_section_heading(
        fig,
        bridge_box,
        "Feature-space bridge",
        "",
    )

    stem = output_dir / f"figure1_{query_key}"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)
    write_status(status_path, "saved_figure", {"svg": str(stem.with_suffix('.svg')), "png": str(stem.with_suffix('.png'))})

    metrics = {
        "query_relpath": query_sample.relpath,
        "query_label": query_sample.label,
        "query_name": Path(query_sample.path).name,
        "zerodcepp_ap": float(ap_base),
        "rift_ap": float(ap_rift),
        "delta_ap": float(ap_rift - ap_base),
        "topk_zerodcepp": [
            {
                "rank": k + 1,
                "gallery_relpath": raw_gallery_samples[int(display_order_base[k])].relpath,
                "match": bool(gallery_labels[int(display_order_base[k])] == query_sample.label),
            }
            for k in range(args.topk)
        ],
        "topk_rift": [
            {
                "rank": k + 1,
                "gallery_relpath": raw_gallery_samples[int(display_order_rift[k])].relpath,
                "match": bool(gallery_labels[int(display_order_rift[k])] == query_sample.label),
            }
            for k in range(args.topk)
        ],
        "positive_gallery_relpath": raw_gallery_samples[pos_idx].relpath,
        "negative_gallery_relpath": raw_gallery_samples[neg_idx].relpath,
        "outputs": {
            "svg": str(stem.with_suffix(".svg")),
            "png": str(stem.with_suffix(".png")),
        },
    }
    save_json(stem.with_suffix(".json"), metrics)
    write_status(status_path, "done")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
