#!/usr/bin/env python3
"""Draw the RIIC-ReID main-paper figure set from regenerated assets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
from PIL import Image


PALETTE = {
    "ink": "#18324A",
    "slate": "#6C7C90",
    "muted": "#93A1B2",
    "green": "#1E9E59",
    "red": "#D6453D",
    "blue": "#2E6BD8",
    "purple": "#7C3E9D",
    "orange": "#D97706",
    "amber": "#E7A34B",
    "teal": "#2B9B90",
    "cloud": "#F5F8FB",
    "mist": "#EEF3F8",
    "warm": "#FFF4E8",
    "salmon": "#FFF0EC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw RIIC-ReID main-paper figures")
    parser.add_argument(
        "--assets_root",
        default="docs/figures/riic_reid_main_paper_20260327/assets",
    )
    parser.add_argument(
        "--output_root",
        default="docs/figures/riic_reid_main_paper_20260327/final",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_all(fig: plt.Figure, stem: Path) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def image_axes(ax: plt.Axes, image: np.ndarray, edge: str | None = None, lw: float = 2.2) -> None:
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(lw if edge else 0.8)
        spine.set_edgecolor(edge or "#D6DEE8")


def badge(ax: plt.Axes, text: str, fc: str, x: float = 0.92, y: float = 0.91) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.15", facecolor=fc, edgecolor="none", alpha=0.96),
        zorder=10,
    )


def card(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, body: str, edge: str, fill: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=fill,
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=PALETTE["ink"])
    ax.text(x + w / 2, y + h * 0.32, body, ha="center", va="center", fontsize=8.5, color=PALETTE["slate"], linespacing=1.35)


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    lw: float = 1.7,
    linestyle: str = "-",
    connectionstyle: str = "arc3,rad=0.0",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        zorder=4,
    )
    ax.add_patch(arrow)


def draw_fig1(assets_root: Path, output_root: Path) -> dict:
    claim_dir = assets_root / "claim_case"
    metrics = load_json(claim_dir / "metrics.json")

    style_matplotlib()
    fig = plt.figure(figsize=(18.0, 8.8))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.65], hspace=0.10)

    fig.text(
        0.05,
        0.97,
        "Figure 1. Retrieval-preferred correction is not equivalent to human-perceptual enhancement.",
        fontsize=17,
        fontweight="bold",
        color=PALETTE["ink"],
        va="top",
    )
    fig.text(
        0.05,
        0.935,
        "A real ATRW query. RetinexNet and Zero-DCE++ change the image appearance strongly, but RIIC-ReID keeps the query in a more retrieval-favorable region.",
        fontsize=10.8,
        color=PALETTE["slate"],
        va="top",
    )

    top = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], wspace=0.08)
    top_panels = [
        ("query_raw.png", "Raw Query", PALETTE["ink"]),
        ("query_retinexnet.png", "RetinexNet", PALETTE["purple"]),
        ("query_zerodcepp.png", "Zero-DCE++", PALETTE["blue"]),
        ("query_riic.png", "RIIC-ReID", PALETTE["orange"]),
    ]
    for idx, (filename, title, color) in enumerate(top_panels):
        ax = fig.add_subplot(top[0, idx])
        image_axes(ax, load_image(claim_dir / filename), edge=color, lw=2.8)
        ax.set_title(title, fontsize=12, color=color, fontweight="bold", pad=10)

    bottom = GridSpecFromSubplotSpec(
        3,
        7,
        subplot_spec=outer[1],
        width_ratios=[0.70, 1, 1, 1, 1, 1, 1.05],
        wspace=0.05,
        hspace=0.18,
    )
    methods = [
        ("retinexnet", "RetinexNet", PALETTE["purple"]),
        ("zerodcepp", "Zero-DCE++", PALETTE["blue"]),
        ("riic", "RIIC-ReID", PALETTE["orange"]),
    ]
    for row_idx, (key, label, color) in enumerate(methods):
        info = metrics["methods"][key]
        ax_label = fig.add_subplot(bottom[row_idx, 0])
        ax_label.axis("off")
        ax_label.text(0.98, 0.58, label, ha="right", va="center", fontsize=11.5, color=color, fontweight="bold")
        ax_label.text(0.98, 0.30, "retrieval", ha="right", va="center", fontsize=8.8, color=PALETTE["slate"])

        for col_idx, ranked in enumerate(info["ranked"][: metrics["topk"]], start=1):
            ax = fig.add_subplot(bottom[row_idx, col_idx])
            image_axes(ax, load_image(Path(ranked["saved_path"])), edge=PALETTE["green"] if ranked["match"] else PALETTE["red"], lw=3.0)
            badge(ax, "T" if ranked["match"] else "F", PALETTE["green"] if ranked["match"] else PALETTE["red"])

        ax_metric = fig.add_subplot(bottom[row_idx, 6])
        ax_metric.axis("off")
        metric_patch = FancyBboxPatch(
            (0.14, 0.18),
            0.76,
            0.64,
            boxstyle="round,pad=0.016,rounding_size=0.03",
            linewidth=1.7,
            edgecolor=color,
            facecolor="white",
        )
        ax_metric.add_patch(metric_patch)
        ax_metric.text(0.52, 0.62, f"AP = {info['ap']:.3f}", ha="center", va="center", fontsize=12, fontweight="bold", color=color)
        ax_metric.text(
            0.52,
            0.38,
            f"correct@{metrics['topk']} = {info['correct_at_5']}/{metrics['topk']}",
            ha="center",
            va="center",
            fontsize=9.6,
            color=PALETTE["slate"],
        )

    fig_note = {
        "query_relpath": metrics["query_relpath"],
        "query_luminance": metrics["query_luminance"],
        "riic_ap": metrics["methods"]["riic"]["ap"],
        "best_perceptual_ap": max(metrics["methods"]["retinexnet"]["ap"], metrics["methods"]["zerodcepp"]["ap"]),
    }
    save_all(fig, output_root / "fig1_claim_mainpaper")
    return fig_note


def draw_fig2(output_root: Path) -> None:
    style_matplotlib()
    fig, ax = plt.subplots(figsize=(15.5, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(
        0.05,
        0.97,
        "Figure 2. RIIC-ReID is a trust-bounded retrieval front-end rather than a generic image enhancer.",
        fontsize=16.5,
        fontweight="bold",
        color=PALETTE["ink"],
        va="top",
    )
    fig.text(
        0.05,
        0.935,
        "Weak photometric priors define a feasible correction set, trust modules regulate risky adaptation, and teacher-guided geometry supervises the final retrieval behavior.",
        fontsize=10.5,
        color=PALETTE["slate"],
        va="top",
    )

    # Background regions.
    regions = [
        (0.05, 0.20, 0.24, 0.62, "Feasible Correction", PALETTE["warm"], PALETTE["amber"]),
        (0.31, 0.20, 0.23, 0.62, "Trust-Controlled Adaptation", "#EAF7F5", PALETTE["teal"]),
        (0.57, 0.20, 0.26, 0.62, "Multi-Branch Encoder", PALETTE["mist"], PALETTE["blue"]),
        (0.34, 0.04, 0.47, 0.11, "Geometry-Guided Training (training only)", PALETTE["salmon"], PALETTE["red"]),
    ]
    for x, y, w, h, title, fill, edge in regions:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.8,
            edgecolor=edge,
            facecolor=fill,
            linestyle="--" if "training only" in title.lower() else "-",
            zorder=0,
        )
        ax.add_patch(patch)
        if "training only" in title.lower():
            ax.text(x + 0.02, y + h + 0.032, title, fontsize=10.2, fontweight="bold", color=edge, ha="left", va="bottom")
        else:
            ax.text(x + 0.02, y + h - 0.035, title, fontsize=11.5, fontweight="bold", color=edge, ha="left", va="center")

    card(ax, 0.03, 0.47, 0.09, 0.15, "Input", "raw image x", PALETTE["ink"], "white")
    card(ax, 0.13, 0.60, 0.15, 0.13, "Coarse Illumination", "luminance Y\nmultiscale L0\nsensitivity S", PALETTE["amber"], "white")
    card(ax, 0.13, 0.36, 0.15, 0.13, "Safe Base Correction", "inverse scaling\nbounded gain\nbounded chroma drift", PALETTE["amber"], "white")
    card(ax, 0.37, 0.58, 0.15, 0.13, "Model-Aware Residual", "M(x, x_base, L, F_mid)", PALETTE["teal"], "white")
    card(ax, 0.37, 0.39, 0.13, 0.12, "Identity Protection", "P_id mask", PALETTE["teal"], "white")
    card(ax, 0.51, 0.39, 0.13, 0.12, "Stripe Rollback", "alpha from illumination,\ncolor risk, gap, F_mid", PALETTE["teal"], "white")
    card(ax, 0.66, 0.62, 0.13, 0.13, "Backbone", "OSNet-AIN", PALETTE["blue"], "white")
    card(ax, 0.65, 0.47, 0.08, 0.10, "Raw", "F_raw", PALETTE["blue"], "white")
    card(ax, 0.75, 0.47, 0.08, 0.10, "Base", "F_base", PALETTE["blue"], "white")
    card(ax, 0.85, 0.47, 0.08, 0.10, "Adapted", "F_adapt", PALETTE["blue"], "white")
    card(ax, 0.68, 0.27, 0.22, 0.12, "Stripe-Aware Branch Attention", "competitive fusion across raw / base / adapted branches", PALETTE["blue"], "white")
    card(ax, 0.90, 0.40, 0.08, 0.14, "Embedding z", "local stripe head\nReID descriptor", PALETTE["green"], "white")

    card(ax, 0.40, 0.058, 0.13, 0.07, "Frozen Teacher", "raw-reference snapshot", PALETTE["red"], "white")
    card(ax, 0.55, 0.058, 0.11, 0.07, "Tube Loss", "same-ID tube", PALETTE["red"], "white")
    card(ax, 0.67, 0.058, 0.11, 0.07, "Separation", "negative margin", PALETTE["red"], "white")
    card(ax, 0.79, 0.058, 0.13, 0.07, "SoftAP + Photo", "ranking + weak prior", PALETTE["red"], "white")

    add_arrow(ax, (0.12, 0.55), (0.13, 0.66), PALETTE["ink"], connectionstyle="arc3,rad=0.12")
    add_arrow(ax, (0.12, 0.54), (0.13, 0.42), PALETTE["ink"])
    add_arrow(ax, (0.205, 0.60), (0.205, 0.49), PALETTE["amber"])
    add_arrow(ax, (0.28, 0.42), (0.37, 0.64), PALETTE["amber"], connectionstyle="arc3,rad=0.02")
    add_arrow(ax, (0.445, 0.58), (0.445, 0.51), PALETTE["teal"])
    add_arrow(ax, (0.50, 0.45), (0.51, 0.45), PALETTE["teal"])
    add_arrow(ax, (0.64, 0.45), (0.85, 0.52), PALETTE["teal"], connectionstyle="arc3,rad=-0.10")
    add_arrow(ax, (0.28, 0.42), (0.75, 0.52), PALETTE["amber"], connectionstyle="arc3,rad=0.04")
    add_arrow(ax, (0.12, 0.54), (0.65, 0.52), PALETTE["slate"], lw=1.2)
    add_arrow(ax, (0.725, 0.62), (0.69, 0.57), PALETTE["blue"])
    add_arrow(ax, (0.79, 0.62), (0.79, 0.57), PALETTE["blue"])
    add_arrow(ax, (0.855, 0.62), (0.89, 0.57), PALETTE["blue"])
    add_arrow(ax, (0.69, 0.47), (0.74, 0.39), PALETTE["blue"])
    add_arrow(ax, (0.79, 0.47), (0.79, 0.39), PALETTE["blue"])
    add_arrow(ax, (0.89, 0.47), (0.84, 0.39), PALETTE["blue"])
    add_arrow(ax, (0.90, 0.33), (0.90, 0.47), PALETTE["blue"], connectionstyle="arc3,rad=-0.08")
    add_arrow(ax, (0.66, 0.68), (0.50, 0.68), PALETTE["teal"], lw=1.1, linestyle="--", connectionstyle="arc3,rad=0.18")
    ax.text(0.58, 0.69, "F_mid", fontsize=8.8, color=PALETTE["teal"], fontweight="bold")

    add_arrow(ax, (0.79, 0.27), (0.60, 0.128), PALETTE["red"], linestyle="--", lw=1.2, connectionstyle="arc3,rad=0.08")
    add_arrow(ax, (0.79, 0.27), (0.72, 0.128), PALETTE["red"], linestyle="--", lw=1.2)
    add_arrow(ax, (0.79, 0.27), (0.85, 0.128), PALETTE["red"], linestyle="--", lw=1.2, connectionstyle="arc3,rad=-0.08")
    add_arrow(ax, (0.53, 0.09), (0.55, 0.09), PALETTE["red"], linestyle="--", lw=1.0)
    add_arrow(ax, (0.53, 0.09), (0.67, 0.09), PALETTE["red"], linestyle="--", lw=1.0, connectionstyle="arc3,rad=-0.04")

    ax.text(0.05, 0.11, "Solid arrows: inference path", fontsize=8.8, color=PALETTE["ink"])

    save_all(fig, output_root / "fig2_method_mainpaper")


def _read_projection(path: Path) -> dict[str, list[tuple[float, float]]]:
    points = {"teacher_positive": [], "hard_negative": [], "query_raw": [], "query_perceptual": [], "query_riic": []}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pt = (float(row["x"]), float(row["y"]))
            if row["role"] == "teacher_positive":
                points["teacher_positive"].append(pt)
            elif row["role"] == "hard_negative":
                points["hard_negative"].append(pt)
            elif row["name"] == "query_raw":
                points["query_raw"].append(pt)
            elif row["name"] == "query_perceptual":
                points["query_perceptual"].append(pt)
            elif row["name"] == "query_riic":
                points["query_riic"].append(pt)
    return points


def draw_fig3(assets_root: Path, output_root: Path) -> dict:
    trust_dir = assets_root / "trust_case"
    geom_dir = assets_root / "geometry_case"
    trust_metrics = load_json(trust_dir / "metrics.json")
    geom_metrics = load_json(geom_dir / "metrics.json")
    projection = _read_projection(geom_dir / "teacher_student_projection.csv")

    style_matplotlib()
    fig = plt.figure(figsize=(17.2, 8.0))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1.28, 1.0], wspace=0.16)

    fig.text(
        0.05,
        0.97,
        "Figure 3. Trust control and teacher-guided geometry create visible, interpretable behavior on real examples.",
        fontsize=16.2,
        fontweight="bold",
        color=PALETTE["ink"],
        va="top",
    )
    fig.text(
        0.05,
        0.935,
        "Left: the trust modules protect identity-critical regions and modulate risky correction. Right: RIIC-ReID keeps the corrected query closer to the teacher-defined same-ID region than a perceptual baseline.",
        fontsize=10.3,
        color=PALETTE["slate"],
        va="top",
    )

    left = GridSpecFromSubplotSpec(3, 3, subplot_spec=outer[0], wspace=0.15, hspace=0.18)
    left_panels = [
        ("raw.png", "Raw", PALETTE["ink"]),
        ("corrected.png", "RIIC-ReID corrected", PALETTE["orange"]),
        (None, f"lum = {trust_metrics['luminance']:.3f}\nrollback mean = {trust_metrics['rollback_mean']:.3f}\ngap mean = {trust_metrics['correction_gap_mean']:.3f}", PALETTE["slate"]),
        ("identity_protection_map.png", "Identity protection P_id", PALETTE["blue"]),
        ("rollback_alpha_map.png", "Rollback alpha", PALETTE["orange"]),
        ("branch_attention.png", "Branch attention", PALETTE["teal"]),
        ("correction_gap.png", "Correction gap", PALETTE["slate"]),
        ("illumination_map.png", "Illumination map", PALETTE["slate"]),
        ("color_risk.png", "Color risk", PALETTE["slate"]),
    ]

    for idx, panel in enumerate(left_panels):
        ax = fig.add_subplot(left[idx // 3, idx % 3])
        filename, title, color = panel
        if filename is None:
            ax.axis("off")
            patch = FancyBboxPatch(
                (0.05, 0.10),
                0.90,
                0.78,
                boxstyle="round,pad=0.015,rounding_size=0.03",
                linewidth=1.6,
                edgecolor=color,
                facecolor="white",
            )
            ax.add_patch(patch)
            ax.text(0.5, 0.66, "Trust summary", ha="center", va="center", fontsize=11, fontweight="bold", color=PALETTE["ink"], transform=ax.transAxes)
            ax.text(0.5, 0.36, title, ha="center", va="center", fontsize=10, color=color, transform=ax.transAxes, linespacing=1.5)
            continue
        image_axes(ax, load_image(trust_dir / filename), edge=color if idx < 6 else None, lw=2.2)
        ax.set_title(title, fontsize=9.8, color=color, fontweight="bold", pad=8)

    right_ax = fig.add_subplot(outer[1])
    right_ax.set_facecolor("white")
    teacher_pts = np.array(projection["teacher_positive"], dtype=np.float32)
    neg_pts = np.array(projection["hard_negative"], dtype=np.float32)
    raw_pt = np.array(projection["query_raw"][0], dtype=np.float32)
    perc_pt = np.array(projection["query_perceptual"][0], dtype=np.float32)
    riic_pt = np.array(projection["query_riic"][0], dtype=np.float32)

    center = teacher_pts.mean(axis=0)
    cov = np.cov(teacher_pts.T) if teacher_pts.shape[0] > 1 else np.eye(2) * 1e-3
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    ellipse = Ellipse(
        xy=center,
        width=3.4 * math.sqrt(max(float(eigvals[0]), 1e-8)),
        height=3.4 * math.sqrt(max(float(eigvals[1]), 1e-8)),
        angle=angle,
        facecolor=PALETTE["green"],
        alpha=0.10,
        edgecolor=PALETTE["green"],
        linewidth=1.8,
        linestyle="--",
        zorder=1,
    )
    right_ax.add_patch(ellipse)

    right_ax.scatter(teacher_pts[:, 0], teacher_pts[:, 1], s=48, color=PALETTE["green"], alpha=0.65, label="teacher positives", zorder=3)
    right_ax.scatter(neg_pts[:, 0], neg_pts[:, 1], s=64, color=PALETTE["red"], marker="D", alpha=0.80, label="hard negatives", zorder=3)
    right_ax.scatter(raw_pt[0], raw_pt[1], s=120, color=PALETTE["slate"], edgecolors="white", linewidths=1.2, label="raw query", zorder=4)
    right_ax.scatter(perc_pt[0], perc_pt[1], s=140, color=PALETTE["blue"], edgecolors="white", linewidths=1.2, label="perceptual query", zorder=4)
    right_ax.scatter(riic_pt[0], riic_pt[1], s=140, color=PALETTE["orange"], edgecolors="white", linewidths=1.2, label="RIIC-ReID query", zorder=4)

    right_ax.annotate("", xy=perc_pt, xytext=raw_pt, arrowprops=dict(arrowstyle="-|>", color=PALETTE["blue"], linewidth=2.0, linestyle="--"))
    right_ax.annotate("", xy=riic_pt, xytext=raw_pt, arrowprops=dict(arrowstyle="-|>", color=PALETTE["orange"], linewidth=2.0))

    right_ax.text(perc_pt[0] + 0.018, perc_pt[1] + 0.02, "perceptual", fontsize=9.5, color=PALETTE["blue"], fontweight="bold")
    right_ax.text(riic_pt[0] + 0.02, riic_pt[1] - 0.03, "RIIC-ReID", fontsize=9.5, color=PALETTE["orange"], fontweight="bold")

    right_ax.set_title("Teacher-centered geometry evidence", fontsize=12, color=PALETTE["ink"], fontweight="bold", pad=12)
    right_ax.legend(frameon=False, fontsize=9, loc="upper left")
    right_ax.grid(alpha=0.18, linestyle="--")
    right_ax.set_xticks([])
    right_ax.set_yticks([])
    for spine in right_ax.spines.values():
        spine.set_edgecolor("#D7DEE7")

    info_box = FancyBboxPatch(
        (0.58, 0.06),
        0.36,
        0.18,
        boxstyle="round,pad=0.016,rounding_size=0.03",
        linewidth=1.4,
        edgecolor=PALETTE["orange"],
        facecolor="white",
        transform=right_ax.transAxes,
    )
    right_ax.add_patch(info_box)
    right_ax.text(0.76, 0.19, "distance to teacher center", transform=right_ax.transAxes, ha="center", va="center", fontsize=9.4, color=PALETTE["ink"], fontweight="bold")
    right_ax.text(
        0.76,
        0.11,
        f"perceptual = {geom_metrics['query_perceptual_dist_to_center']:.3f}\nRIIC-ReID = {geom_metrics['query_riic_dist_to_center']:.3f}",
        transform=right_ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.2,
        color=PALETTE["slate"],
        linespacing=1.35,
    )

    summary = {
        "trust_case_relpath": trust_metrics["sample_relpath"],
        "geometry_case_relpath": geom_metrics["query_relpath"],
        "geometry_gain": geom_metrics["geometry_gain"],
    }
    save_all(fig, output_root / "fig3_mechanism_mainpaper")
    return summary


def write_captions(output_root: Path, fig1_note: dict, fig3_note: dict) -> None:
    text = f"""# RIIC-ReID Main-Paper Figure Captions

## Figure 1
Retrieval-preferred correction is not equivalent to human-perceptual enhancement. The same ATRW query ({fig1_note['query_relpath']}, luminance {fig1_note['query_luminance']:.3f}) is processed by RetinexNet, Zero-DCE++, and RIIC-ReID. Although the perceptual baselines visibly alter brightness and contrast, they produce weaker ranked retrieval lists than RIIC-ReID. This supports the paper's central claim that illumination correction for retrieval should be optimized for the downstream embedding geometry rather than for human-perceptual appearance alone.

## Figure 2
Overview of RIIC-ReID. A bounded feasible correction stage first constructs a safe operating region using coarse illumination estimation, sensitivity modulation, and constrained inverse scaling. Trust-controlled adaptation then applies model-aware residual correction, identity protection, and stripe-wise rollback. The encoder keeps raw, base-corrected, and adapted branches and fuses them using stripe-aware branch attention. During training only, a frozen teacher provides manifold-tube, separation, and ranking supervision.

## Figure 3
Trust and geometry evidence for RIIC-ReID. Left: the trust modules generate interpretable spatial behavior, including identity protection, rollback control, correction-gap localization, illumination estimation, and color-risk detection. Right: in the teacher-centered projection, the RIIC-ReID query remains closer to the teacher-defined same-identity region than a perceptual baseline, illustrating the geometry-guided objective.
"""
    (output_root / "captions.md").write_text(text, encoding="utf-8")


def write_readme(output_root: Path) -> None:
    text = """# RIIC-ReID Main-Paper Figures

Final outputs:

- `final/fig1_claim_mainpaper.{png,pdf,svg}`
- `final/fig2_method_mainpaper.{png,pdf,svg}`
- `final/fig3_mechanism_mainpaper.{png,pdf,svg}`
- `final/captions.md`

These files are intended for the main ACM MM paper rather than supplementary material.
"""
    (output_root.parent / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    assets_root = Path(args.assets_root)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    fig1_note = draw_fig1(assets_root, output_root)
    draw_fig2(output_root)
    fig3_note = draw_fig3(assets_root, output_root)
    write_captions(output_root, fig1_note, fig3_note)
    write_readme(output_root)
    print(f"Saved RIIC-ReID main-paper figures to {output_root}")


if __name__ == "__main__":
    main()
