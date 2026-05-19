#!/usr/bin/env python3
"""Draw polished paper figures for the MM'26 RIFT submission."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon


PALETTE = {
    "ink": "#16324F",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "red": "#E76F51",
    "mist": "#EEF3F7",
    "cloud": "#F8F6F1",
    "slate": "#6B7C93",
    "green": "#7FB069",
    "line": "#2F4858",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _find_sample_images(data_root: Path, count: int = 4) -> List[np.ndarray]:
    candidates: List[Path] = []
    for split in ("query", "gallery", "train"):
        split_root = data_root / split
        if split_root.is_dir():
            candidates.extend(sorted(split_root.rglob("*.jpg")))
        if len(candidates) >= count:
            break

    images: List[np.ndarray] = []
    for image_path in candidates[:count]:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return images


def _add_round_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    fc: str = "white",
    ec: str = PALETTE["line"],
    lw: float = 1.8,
    rounding: float = 0.02,
    title_size: int = 12,
    subtitle_size: int = 10,
    title_color: str = PALETTE["ink"],
    subtitle_color: str = PALETTE["slate"],
    alpha: float = 1.0,
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        zorder=zorder + 1,
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.30,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=subtitle_color,
            zorder=zorder + 1,
            linespacing=1.35,
        )
    return patch


def _add_arrow(
    ax: plt.Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
    color: str = PALETTE["line"],
    lw: float = 2.2,
    style: str = "-|>",
    mutation_scale: float = 16,
    linestyle: str = "-",
    connectionstyle: str = "arc3,rad=0.0",
    alpha: float = 1.0,
    zorder: int = 3,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def _add_section_title(ax: plt.Axes, x: float, y: float, text: str, color: str = PALETTE["ink"]) -> None:
    ax.text(x, y, text, ha="left", va="bottom", fontsize=15, fontweight="bold", color=color)


def _add_panel_badge(ax: plt.Axes, x: float, y: float, label: str, color: str) -> None:
    badge = Circle((x, y), radius=0.022, facecolor=color, edgecolor="white", linewidth=1.5, zorder=5)
    ax.add_patch(badge)
    ax.text(x, y, label, ha="center", va="center", fontsize=11, fontweight="bold", color="white", zorder=6)


def _draw_image_strip(
    ax: plt.Axes,
    images: Sequence[np.ndarray],
    x: float,
    y: float,
    w: float,
    h: float,
    gap: float = 0.012,
) -> None:
    if not images:
        return
    n = len(images)
    card_w = (w - gap * (n - 1)) / n
    for idx, image in enumerate(images):
        x0 = x + idx * (card_w + gap)
        frame = FancyBboxPatch(
            (x0, y),
            card_w,
            h,
            boxstyle="round,pad=0.005,rounding_size=0.012",
            facecolor="none",
            edgecolor=PALETTE["line"],
            linewidth=1.3,
            zorder=3,
        )
        ax.imshow(
            image,
            extent=[x0 + 0.005, x0 + card_w - 0.005, y + 0.005, y + h - 0.005],
            zorder=2,
            aspect="auto",
        )
        ax.add_patch(frame)


def draw_overview_figure(output_dir: Path, sample_images: Sequence[np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.04),
            0.96,
            0.92,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            facecolor=PALETTE["cloud"],
            edgecolor="none",
            zorder=-10,
        )
    )

    _add_section_title(ax, 0.05, 0.92, "RIFT: retrieval-informed illumination front-end")
    ax.text(
        0.05,
        0.885,
        "Weak photometric priors keep the correction feasible; retrieval geometry decides where the correction should move.",
        ha="left",
        va="top",
        fontsize=11,
        color=PALETTE["slate"],
    )

    _add_panel_badge(ax, 0.055, 0.79, "A", PALETTE["teal"])
    _add_section_title(ax, 0.085, 0.775, "Cross-light observations", color=PALETTE["ink"])
    ax.text(
        0.085,
        0.748,
        "Same identity may drift strongly in brightness, shadows, and color cast while stripe structure stays discriminative.",
        ha="left",
        va="top",
        fontsize=10,
        color=PALETTE["slate"],
    )
    _draw_image_strip(ax, sample_images[:4], 0.08, 0.57, 0.28, 0.15)

    _add_panel_badge(ax, 0.42, 0.79, "B", PALETTE["gold"])
    _add_section_title(ax, 0.45, 0.775, "Task-aligned front-end", color=PALETTE["ink"])
    _add_round_box(
        ax,
        0.43,
        0.61,
        0.16,
        0.10,
        "Coarse illumination",
        "luminance map\nsensitivity field",
        fc="#FFF7DF",
        ec=PALETTE["gold"],
    )
    _add_round_box(
        ax,
        0.63,
        0.61,
        0.16,
        0.10,
        "Safe correction",
        "bounded gain\nbounded chroma drift",
        fc="#FFF3EE",
        ec=PALETTE["orange"],
    )
    _add_round_box(
        ax,
        0.53,
        0.45,
        0.16,
        0.10,
        "Identity-aware rollback",
        "risk-conditioned blending\nstripe-sensitive protection",
        fc="#EDF7F5",
        ec=PALETTE["teal"],
    )
    _add_arrow(ax, (0.59, 0.66), (0.63, 0.66), color=PALETTE["gold"])
    _add_arrow(ax, (0.71, 0.61), (0.61, 0.55), color=PALETTE["orange"])

    prior = FancyBboxPatch(
        (0.41, 0.37),
        0.40,
        0.41,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.6,
        edgecolor=PALETTE["gold"],
        facecolor="none",
        linestyle="--",
        zorder=0,
    )
    ax.add_patch(prior)
    ax.text(0.62, 0.785, "Feasible photometric set", ha="center", va="bottom", fontsize=9.5, color=PALETTE["gold"], fontweight="bold")

    _add_panel_badge(ax, 0.055, 0.43, "C", PALETTE["ink"])
    _add_section_title(ax, 0.085, 0.415, "Trust-bounded representation path", color=PALETTE["ink"])
    branch_y = 0.23
    branch_specs = [
        (0.08, "Raw branch", "unaltered appearance", PALETTE["ink"], PALETTE["mist"]),
        (0.23, "Base-corrected", "safe inverse illumination", PALETTE["orange"], "#FFF3EE"),
        (0.38, "Adapted branch", "rollback-aware correction", PALETTE["teal"], "#EDF7F5"),
    ]
    for x, title, subtitle, color, fc in branch_specs:
        _add_round_box(ax, x, branch_y, 0.12, 0.10, title, subtitle, fc=fc, ec=color, title_size=11, subtitle_size=9)

    _add_round_box(
        ax,
        0.56,
        0.22,
        0.15,
        0.12,
        "Trust-bounded fusion",
        "bounded residual trust\nstripe-aware aggregation",
        fc="#EEF3F7",
        ec=PALETTE["ink"],
        title_size=12,
    )
    _add_round_box(
        ax,
        0.79,
        0.22,
        0.14,
        0.12,
        "ReID embedding",
        "identity ranking\ncross-light retrieval",
        fc="white",
        ec=PALETTE["ink"],
        title_size=12,
    )
    for x in (0.20, 0.35, 0.50):
        _add_arrow(ax, (x, 0.28), (0.56, 0.28), color=PALETTE["line"], alpha=0.35, lw=1.8)
    _add_arrow(ax, (0.71, 0.28), (0.79, 0.28), color=PALETTE["ink"])

    _add_panel_badge(ax, 0.80, 0.79, "D", PALETTE["red"])
    ax.text(0.83, 0.775, "Geometry supervision", ha="left", va="center", fontsize=14, fontweight="bold", color=PALETTE["ink"])
    teacher = FancyBboxPatch(
        (0.79, 0.56),
        0.14,
        0.13,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.6,
        edgecolor=PALETTE["red"],
        facecolor="#FFF0EC",
        zorder=1,
    )
    ax.add_patch(teacher)
    ax.text(0.86, 0.635, "Teacher manifold", ha="center", va="center", fontsize=11.5, fontweight="bold", color=PALETTE["ink"])
    ax.text(
        0.86,
        0.585,
        "tube constraint\nnegative separation\nranking geometry",
        ha="center",
        va="center",
        fontsize=9.5,
        color=PALETTE["slate"],
    )
    _add_arrow(ax, (0.86, 0.56), (0.86, 0.34), color=PALETTE["red"], linestyle="--", lw=2.0)
    _add_arrow(ax, (0.76, 0.63), (0.71, 0.32), color=PALETTE["red"], linestyle="--", lw=1.8, alpha=0.7)

    ax.text(
        0.05,
        0.08,
        "Core idea: correction is not optimized to look more natural; it is optimized to become more rankable while staying inside a safe photometric region.",
        ha="left",
        va="center",
        fontsize=11,
        color=PALETTE["ink"],
    )

    for ext in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"rift_overview.{ext}", dpi=320, bbox_inches="tight")
    plt.close(fig)


def draw_frontend_detail_figure(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.05),
            0.96,
            0.90,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            facecolor=PALETTE["cloud"],
            edgecolor="none",
            zorder=-10,
        )
    )

    _add_section_title(ax, 0.05, 0.92, "Safe correction and rollback detail")
    ax.text(
        0.05,
        0.885,
        "The front-end first computes a feasible inverse-illumination proposal, then selectively rolls back risky corrections before they dominate the embedding.",
        ha="left",
        va="top",
        fontsize=11,
        color=PALETTE["slate"],
    )

    stages = [
        (0.05, 0.62, 0.16, 0.15, "Input image x", "RGB observation\ncross-light nuisance", PALETTE["ink"], "white"),
        (0.27, 0.62, 0.18, 0.15, "Coarse estimate", "Y -> L0\nsensitivity S(x)", PALETTE["gold"], "#FFF7DF"),
        (0.51, 0.62, 0.18, 0.15, "Safe correction", "a = 1/(L+eps)\nclamp(a, amin, amax)", PALETTE["orange"], "#FFF3EE"),
        (0.75, 0.62, 0.18, 0.15, "Reflectance base", "photometrically feasible\nstripe detail retained", PALETTE["teal"], "#EDF7F5"),
    ]
    for x, y, w, h, title, subtitle, ec, fc in stages:
        _add_round_box(ax, x, y, w, h, title, subtitle, fc=fc, ec=ec, title_size=12)
    _add_arrow(ax, (0.21, 0.695), (0.27, 0.695), color=PALETTE["line"])
    _add_arrow(ax, (0.45, 0.695), (0.51, 0.695), color=PALETTE["line"])
    _add_arrow(ax, (0.69, 0.695), (0.75, 0.695), color=PALETTE["line"])

    _add_round_box(
        ax,
        0.18,
        0.28,
        0.20,
        0.16,
        "Color-risk estimator",
        "chromatic angle drift\ncorrection gap statistics",
        fc="white",
        ec=PALETTE["red"],
        title_size=12,
    )
    _add_round_box(
        ax,
        0.42,
        0.28,
        0.20,
        0.16,
        "Rollback gate alpha",
        "risk-aware trust\nstripe-level blending",
        fc="#FFF0EC",
        ec=PALETTE["red"],
        title_size=12,
    )
    _add_round_box(
        ax,
        0.66,
        0.28,
        0.22,
        0.16,
        "Adapted reflectance",
        "x' = alpha r + (1-alpha) x\nonly reliable correction survives",
        fc="#EDF7F5",
        ec=PALETTE["teal"],
        title_size=12,
    )

    _add_arrow(ax, (0.60, 0.62), (0.28, 0.44), color=PALETTE["red"], linestyle="--", lw=2.0, alpha=0.65)
    _add_arrow(ax, (0.84, 0.62), (0.52, 0.44), color=PALETTE["red"], linestyle="--", lw=2.0, alpha=0.65)
    _add_arrow(ax, (0.38, 0.36), (0.42, 0.36), color=PALETTE["red"])
    _add_arrow(ax, (0.62, 0.36), (0.66, 0.36), color=PALETTE["red"])

    identity_map = Polygon(
        [[0.10, 0.16], [0.21, 0.22], [0.31, 0.17], [0.29, 0.09], [0.14, 0.08]],
        closed=True,
        facecolor="#D9EFEA",
        edgecolor=PALETTE["teal"],
        linewidth=1.6,
        alpha=0.95,
    )
    ax.add_patch(identity_map)
    ax.text(0.21, 0.145, "identity-sensitive\nstripe support", ha="center", va="center", fontsize=11, color=PALETTE["ink"], fontweight="bold")

    ax.text(0.48, 0.16, r"$R_{\mathrm{safe}} = x + (1-P_{\mathrm{id}})\odot(\hat{R}-x)$", ha="center", va="center", fontsize=15.5, color=PALETTE["ink"])
    ax.text(
        0.78,
        0.13,
        "Rollback prevents the front-end from forcing\nan aggressive but retrieval-unsafe correction.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=PALETTE["slate"],
        linespacing=1.4,
    )

    for ext in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"rift_frontend_detail.{ext}", dpi=320, bbox_inches="tight")
    plt.close(fig)


def draw_geometry_figure(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 8.5))
    ax.set_xlim(-4.3, 4.3)
    ax.set_ylim(-3.2, 3.2)
    ax.axis("off")

    ax.add_patch(
        FancyBboxPatch(
            (-4.15, -3.0),
            8.1,
            5.8,
            boxstyle="round,pad=0.02,rounding_size=0.2",
            facecolor=PALETTE["cloud"],
            edgecolor="none",
            zorder=-20,
        )
    )

    ax.text(-4.0, 2.75, "Correction-aware geometry in the embedding space", fontsize=16, fontweight="bold", color=PALETTE["ink"])
    ax.text(
        -4.0,
        2.45,
        "RIFT allows movement along a nuisance-compatible tube while preserving same-ID compactness and preventing negative-class collapse.",
        fontsize=11,
        color=PALETTE["slate"],
    )

    positive_center = np.array([0.3, 0.0])
    negative_centers = [(-2.4, 0.9), (2.6, 1.2), (2.1, -1.7)]
    neg_colors = [PALETTE["gold"], PALETTE["orange"], PALETTE["red"]]

    for center, color in zip(negative_centers, neg_colors):
        ell = Ellipse(center, width=1.45, height=0.95, angle=20, facecolor=color, edgecolor=color, alpha=0.18, linewidth=2.0)
        ax.add_patch(ell)
        ax.text(center[0], center[1], "negative\nidentity", ha="center", va="center", fontsize=10, color=PALETTE["ink"])

    pos_ell = Ellipse(tuple(positive_center), width=2.35, height=1.25, angle=10, facecolor=PALETTE["teal"], edgecolor=PALETTE["teal"], alpha=0.15, linewidth=2.2)
    ax.add_patch(pos_ell)
    ax.text(positive_center[0], positive_center[1] - 1.15, "same-ID neighborhood", ha="center", va="center", fontsize=11, color=PALETTE["teal"], fontweight="bold")

    tube_x = np.linspace(-1.3, 1.9, 200)
    tube_y = 0.35 * np.sin(0.9 * (tube_x + 0.8)) - 0.10
    tube_upper = np.column_stack([tube_x, tube_y + 0.42])
    tube_lower = np.column_stack([tube_x[::-1], (tube_y - 0.42)[::-1]])
    tube_poly = np.vstack([tube_upper, tube_lower])
    ax.add_patch(Polygon(tube_poly, closed=True, facecolor=PALETTE["ink"], edgecolor=PALETTE["ink"], alpha=0.10, linewidth=1.5))
    ax.plot(tube_x, tube_y, color=PALETTE["ink"], linewidth=2.5, linestyle="--")
    ax.text(1.25, -0.85, "teacher manifold tube", fontsize=11, color=PALETTE["ink"], fontweight="bold")

    raw = np.array([-1.3, -1.25])
    corrected = np.array([0.10, -0.15])
    unsafe = np.array([1.9, -1.95])
    teacher = np.array([0.85, 0.08])

    ax.scatter(*raw, s=180, color=PALETTE["orange"], edgecolors="white", linewidths=1.5, zorder=6)
    ax.scatter(*corrected, s=180, color=PALETTE["teal"], edgecolors="white", linewidths=1.5, zorder=6)
    ax.scatter(*unsafe, s=180, color=PALETTE["red"], edgecolors="white", linewidths=1.5, zorder=6)
    ax.scatter(*teacher, s=180, color=PALETTE["ink"], edgecolors="white", linewidths=1.5, zorder=6)

    ax.text(raw[0] - 0.1, raw[1] - 0.35, "raw feature", ha="center", fontsize=11, color=PALETTE["orange"], fontweight="bold")
    ax.text(corrected[0] + 0.15, corrected[1] - 0.48, "accepted correction", ha="center", fontsize=11, color=PALETTE["teal"], fontweight="bold")
    ax.text(unsafe[0] + 0.05, unsafe[1] - 0.42, "unsafe drift", ha="center", fontsize=11, color=PALETTE["red"], fontweight="bold")
    ax.text(teacher[0], teacher[1] + 0.38, "teacher anchor", ha="center", fontsize=11, color=PALETTE["ink"], fontweight="bold")

    _add_arrow(
        ax,
        tuple(raw),
        tuple(corrected),
        color=PALETTE["teal"],
        lw=2.8,
        mutation_scale=18,
        connectionstyle="arc3,rad=-0.05",
    )
    _add_arrow(
        ax,
        tuple(raw),
        tuple(unsafe),
        color=PALETTE["red"],
        lw=2.0,
        mutation_scale=16,
        linestyle="--",
        alpha=0.75,
        connectionstyle="arc3,rad=0.15",
    )
    _add_arrow(
        ax,
        tuple(corrected),
        tuple(teacher),
        color=PALETTE["ink"],
        lw=2.2,
        mutation_scale=15,
        linestyle="--",
    )

    callouts = [
        (2.35, 1.55, 1.45, 0.82, "Tube term", r"$\mathcal{L}_{tube}$ keeps the corrected feature inside a local same-ID corridor.", PALETTE["ink"], "#EEF3F7"),
        (2.35, 0.48, 1.45, 0.82, "Separation term", r"$\mathcal{L}_{sep}$ prevents crossing into negative neighborhoods.", PALETTE["red"], "#FFF0EC"),
        (2.35, -0.59, 1.45, 0.98, "Trust-bounded fusion", "Raw and corrected branches are mixed as a bounded residual,\nnot as unrestricted replacement.", PALETTE["teal"], "#EDF7F5"),
    ]
    for x, y, w, h, title, subtitle, ec, fc in callouts:
        _add_round_box(ax, x, y, w, h, title, subtitle, fc=fc, ec=ec, title_size=11.5, subtitle_size=10, rounding=0.12)

    for ext in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"rift_geometry.{ext}", dpi=320, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw polished RIFT paper figures.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("references/write/acmart/acmart/figures_generated"),
        help="Output directory for generated figures.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data/processed/atrw"),
        help="ATRW processed root for optional sample images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    _ensure_dir(args.output_dir)
    sample_images = _find_sample_images(args.data_root, count=4)
    draw_overview_figure(args.output_dir, sample_images)
    draw_frontend_detail_figure(args.output_dir)
    draw_geometry_figure(args.output_dir)
    print(f"[INFO] Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
