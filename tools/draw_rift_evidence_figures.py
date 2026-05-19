#!/usr/bin/env python3
"""Generate data-driven evidence figures for the MM'26 RIFT paper.

Figures produced:
  fig1_claim_evidence.{png,pdf}   – retrieval comparison with real images
  fig3a_trust_in_action.{png,pdf} – trust heatmap panels (case_02)
  fig3c_teacher_manifold.{png,pdf}– teacher manifold scatter (projection CSV)

Run from repo root:
  python tools/draw_rift_evidence_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import ConvexHull


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).parent.parent
FIG_ROOT = ROOT / "docs/figures/rift_paper_20260325"
OUT      = FIG_ROOT / "final"
OUT.mkdir(parents=True, exist_ok=True)

SEL      = FIG_ROOT / "fig1_claim/selected_case"
TRUST    = FIG_ROOT / "fig3_trust/case_02"
GEO      = FIG_ROOT / "fig3_geometry"


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
C = {
    "ink":      "#1B2A3B",
    "perc":     "#4078C8",   # blue – generic perceptual
    "rift":     "#E07B39",   # amber – RIFT
    "match":    "#2D8C4E",   # green – correct rank
    "miss":     "#C0392B",   # red – wrong rank
    "teacher":  "#2EAD6A",
    "neg":      "#C0392B",
    "raw_q":    "#6B7C93",
    "slate":    "#6B7C93",
    "light_bg": "#F6F8FA",
}


def _rcparams() -> None:
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.facecolor":    "white",
        "figure.facecolor":  "white",
        "savefig.facecolor": "white",
        "axes.linewidth":    0.8,
    })


def _load(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _ax_img(
    ax: plt.Axes, img: np.ndarray,
    border: str | None = None, lw: float = 3.0,
) -> None:
    """Display image with optional colored border, no ticks."""
    ax.imshow(img, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    bc = border if border else "#D0D0D0"
    bw = lw if border else 0.6
    for sp in ax.spines.values():
        sp.set_edgecolor(bc)
        sp.set_linewidth(bw)


def _add_badge(ax: plt.Axes, text: str, color: str, pos: str = "top-right",
               fontsize: float = 13) -> None:
    """Add a small colored badge (✓/✗) at a corner of an axes."""
    xy = {"top-right": (0.92, 0.92), "top-left": (0.08, 0.92)}[pos]
    ax.text(
        xy[0], xy[1], text,
        transform=ax.transAxes, fontsize=fontsize, fontweight="bold",
        color="white", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.15", facecolor=color, edgecolor="none",
                  alpha=0.92),
        zorder=10,
    )


def _add_panel_label(ax: plt.Axes, label: str, color: str = "#1B2A3B") -> None:
    """Add (a)/(b)/(c) panel label at top-left of axes."""
    ax.text(
        0.04, 0.93, label,
        transform=ax.transAxes, fontsize=12, fontweight="bold",
        color="white", ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor="none",
                  alpha=0.85),
        zorder=10,
    )


# ===========================================================================
# Figure 1 – Multi-method top-10 retrieval comparison
# ===========================================================================

# Method display config: key → (label, color)
METHOD_STYLE = {
    "raw":        ("Matched ReID\nbaseline",  "#6B7C93"),
    "retinexnet": ("RetinexNet",              "#7B2D8E"),
    "zerodcepp":  ("Zero-DCE++",              "#4078C8"),
    "rift":       ("RIFT (ours)",             "#E07B39"),
}

MULTI = FIG_ROOT / "fig1_claim/multimethod"


def draw_fig1() -> None:
    """Figure 1: 4-method × top-10 retrieval comparison with real images."""
    _rcparams()

    # -- Load multimethod metadata --
    with open(MULTI / "multimethod_metrics.json") as f:
        meta = json.load(f)
    lum = meta["query_luminance"]
    topk = meta["topk"]
    methods_data = meta["methods"]

    # -- Load query images --
    q_raw      = _load(MULTI / "query_raw.png")
    q_retinex  = _load(MULTI / "query_retinexnet.png")
    q_zdce     = _load(MULTI / "query_zerodcepp.png")
    q_rift     = _load(MULTI / "query_rift.png")

    # -- Method order for display (skip raw baseline – shown in Table 1) --
    method_keys = ["retinexnet", "zerodcepp", "rift"]

    # -- Figure layout --
    n_methods = len(method_keys)
    fig = plt.figure(figsize=(20, 9.5))
    fig.patch.set_facecolor("white")

    # ===== TOP: 4 query variants =====
    top_y = 0.72
    top_h = 0.22
    mx = 0.065       # left margin for method labels
    gap = 0.012
    n_top = 4
    top_w = (0.93 - mx - gap * (n_top - 1)) / n_top

    query_specs = [
        (q_raw,     "(a) Original query",    C["ink"]),
        (q_retinex, "(b) RetinexNet",        "#7B2D8E"),
        (q_zdce,    "(c) Zero-DCE++",        C["perc"]),
        (q_rift,    "(d) RIFT (ours)",       C["rift"]),
    ]
    for i, (img, label, color) in enumerate(query_specs):
        x = mx + i * (top_w + gap)
        ax = fig.add_axes([x, top_y, top_w, top_h])
        _ax_img(ax, img, border=color, lw=3.0)
        _add_panel_label(ax, label.split(")")[0] + ")", color=color)
        ax.set_xlabel(label.split(") ")[1], fontsize=10, color=color,
                      fontweight="bold", labelpad=4)

    # ===== BOTTOM: 4 rows of top-10 retrieval =====
    row_h = 0.135
    row_gap = 0.015
    bot_top = 0.55    # top of first retrieval row
    label_w = 0.06
    gal_start = mx
    gal_total_w = 0.93 - mx
    gap_gal = 0.005
    gal_w = (gal_total_w - (topk - 1) * gap_gal - 0.08) / topk  # reserve space for AP
    ap_x = mx + topk * (gal_w + gap_gal) + 0.005

    # (Section title removed – use LaTeX \caption instead)

    for ri, mk in enumerate(method_keys):
        mdata = methods_data[mk]
        label, color = METHOD_STYLE[mk]
        ap = mdata["ap"]
        ranked = mdata["ranked"]

        y = bot_top - ri * (row_h + row_gap)

        # Method label (rotated, left side)
        fig.text(mx - 0.01, y + row_h / 2, label,
                 fontsize=9, color=color, fontweight="bold",
                 ha="right", va="center")

        # Gallery thumbnails
        for k, r in enumerate(ranked[:topk]):
            img = _load(ROOT / r["saved_path"])
            is_match = r["match"]
            bc = C["match"] if is_match else C["miss"]
            sym = "✓" if is_match else "✗"

            gx = mx + k * (gal_w + gap_gal)
            ax = fig.add_axes([gx, y, gal_w, row_h])
            _ax_img(ax, img, border=bc, lw=4.0)
            _add_badge(ax, sym, bc, pos="top-right", fontsize=11)

            # Rank number at bottom
            if ri == 0:  # only on first row
                fig.text(gx + gal_w / 2, y + row_h + 0.005,
                         str(k + 1), fontsize=7.5, color=C["slate"],
                         ha="center", va="bottom", fontweight="bold")

        # Correct count + AP badge
        n_correct = sum(1 for r in ranked[:topk] if r["match"])
        fig.text(ap_x, y + row_h * 0.62,
                 f"AP = {ap:.3f}",
                 fontsize=11, color=color, fontweight="bold",
                 ha="left", va="center",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                           edgecolor=color, linewidth=1.5))
        fig.text(ap_x, y + row_h * 0.25,
                 f"{n_correct}/{topk} correct",
                 fontsize=8.5, color=color, ha="left", va="center")

    # Rank column header
    fig.text(mx + topk * (gal_w + gap_gal) / 2, bot_top + 0.025,
             "Rank →", fontsize=9, color=C["slate"], ha="center", va="bottom")

    # (Overall title removed – use LaTeX \caption instead)

    for ext in ("png",):
        fig.savefig(OUT / f"fig1_claim_evidence.{ext}", dpi=300, bbox_inches="tight")
    try:
        fig.savefig(OUT / "fig1_claim_evidence.pdf", dpi=300, bbox_inches="tight")
    except PermissionError:
        print("[WARN] Could not write PDF (file locked), PNG saved successfully")
    plt.close(fig)
    print(f"[Fig1]  → {OUT}/fig1_claim_evidence.{{png,pdf}}")


# ===========================================================================
# Figure 3a – Trust in action (case_02 heatmaps)
# ===========================================================================

def draw_fig3a() -> None:
    _rcparams()

    with open(TRUST / "case_metrics.json") as f:
        cm = json.load(f)
    lum      = cm["luminance"]
    rb_mean  = cm["rollback_mean"]
    pid_mean = cm["identity_protection_mean"]
    gap_mean = cm["correction_gap_mean"]

    raw       = _load(TRUST / "raw.png")
    corrected = _load(TRUST / "corrected.png")
    pid_map   = _load(TRUST / "identity_protection_map.png")
    rb_map    = _load(TRUST / "rollback_alpha_map.png")
    gap_map   = _load(TRUST / "correction_gap.png")
    illum_map = _load(TRUST / "illumination_map.png")
    color_risk = _load(TRUST / "color_risk.png")

    fig = plt.figure(figsize=(15, 4.2))
    fig.patch.set_facecolor("white")

    gs = GridSpec(
        1, 9, figure=fig,
        left=0.01, right=0.99, top=0.82, bottom=0.04,
        wspace=0.12,
        width_ratios=[2, 2, 0.15, 2, 2, 0.15, 1.6, 1.6, 1.6],
    )

    panels = [
        (gs[0, 0], raw,       None,      f"Raw  (lum={lum:.2f})",                    C["ink"]),
        (gs[0, 1], corrected, C["rift"],  "RIFT corrected",                           C["rift"]),
        (gs[0, 3], pid_map,   C["perc"],  f"Identity protection  $P_{{id}}$\n(mean={pid_mean:.2f})", C["perc"]),
        (gs[0, 4], rb_map,    C["rift"],  f"Rollback gate  $\\alpha$\n(mean={rb_mean:.2f})",         C["rift"]),
        (gs[0, 6], gap_map,   C["slate"], f"Correction gap\n(mean={gap_mean:.3f})",   C["slate"]),
        (gs[0, 7], illum_map, C["slate"], "Illumination map",                         C["slate"]),
        (gs[0, 8], color_risk, C["slate"], "Color risk",                              C["slate"]),
    ]

    for spec, img, border, title, tc in panels:
        ax = fig.add_subplot(spec)
        _ax_img(ax, img, border=border, lw=2.5)
        ax.set_title(title, fontsize=8.5, color=tc, pad=4, fontweight="bold")

    # Section dividers
    for col in [2, 5]:
        ax_div = fig.add_subplot(gs[0, col])
        ax_div.axis("off")
        ax_div.text(0.5, 0.5, "│", ha="center", va="center",
                    fontsize=20, color="#CCCCCC", transform=ax_div.transAxes)

    # Section headers
    fig.text(0.115, 0.90, "Image correction", ha="center", fontsize=9.5,
             fontweight="bold", color=C["ink"])
    fig.text(0.455, 0.90, "Trust mechanisms", ha="center", fontsize=9.5,
             fontweight="bold", color=C["ink"])
    fig.text(0.83, 0.90, "Supporting evidence", ha="center", fontsize=9.5,
             fontweight="bold", color=C["slate"])

    # (suptitle removed – use LaTeX \caption)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig3a_trust_in_action.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig3a] → {OUT}/fig3a_trust_in_action.{{png,pdf}}")


# ===========================================================================
# Figure 3c – Teacher manifold geometry
# ===========================================================================

def draw_fig3c() -> None:
    _rcparams()

    with open(GEO / "teacher_manifold_stats.json") as f:
        stats = json.load(f)

    df = pd.read_csv(GEO / "teacher_student_projection.csv")
    tp = df[df["role"] == "teacher_positive"][["x", "y"]].values
    hn = df[df["role"] == "hard_negative"][["x", "y"]].values
    qr = df[df["name"] == "query_raw"][["x", "y"]].values[0]
    qp = df[df["name"] == "query_perceptual"][["x", "y"]].values[0]
    qf = df[df["name"] == "query_rift"][["x", "y"]].values[0]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_aspect("equal")

    # Convex hull of teacher positives
    hull = ConvexHull(tp)
    hull_pts = tp[hull.vertices]
    hull_poly = plt.Polygon(hull_pts, closed=True,
                            facecolor=C["teacher"], alpha=0.10,
                            edgecolor=C["teacher"], linewidth=1.4,
                            linestyle="--", zorder=1)
    ax.add_patch(hull_poly)

    cx, cy = tp.mean(axis=0)
    sx, sy = tp.std(axis=0)
    ell = Ellipse((cx, cy), width=sx * 3.5, height=sy * 3.5,
                  facecolor=C["teacher"], alpha=0.06,
                  edgecolor=C["teacher"], linewidth=0.8,
                  linestyle=":", zorder=1)
    ax.add_patch(ell)

    # Scatter plots
    ax.scatter(tp[:, 0], tp[:, 1], s=28, color=C["teacher"], alpha=0.45, zorder=2)
    ax.scatter(hn[:, 0], hn[:, 1], s=70, color=C["neg"], marker="D", alpha=0.85,
               zorder=4, edgecolors="white", linewidths=0.8)
    ax.scatter(*qr, s=140, color=C["raw_q"], marker="o",
               edgecolors="white", linewidths=1.2, zorder=6)
    ax.scatter(*qp, s=140, color=C["perc"], marker="o",
               edgecolors="white", linewidths=1.2, zorder=6)
    ax.scatter(*qf, s=140, color=C["rift"], marker="D",
               edgecolors="white", linewidths=1.2, zorder=6)

    # Arrows
    arrow_kw = dict(arrowstyle="-|>", lw=1.8, mutation_scale=14)
    ax.annotate("", xy=qp, xytext=qr,
                arrowprops={**arrow_kw, "color": C["perc"],
                            "connectionstyle": "arc3,rad=-0.25",
                            "linestyle": "--"})
    ax.annotate("", xy=qf, xytext=qr,
                arrowprops={**arrow_kw, "color": C["rift"],
                            "connectionstyle": "arc3,rad=0.15"})

    # Labels
    d_raw  = stats["query_raw_dist_to_center"]
    d_perc = stats["query_perceptual_dist_to_center"]
    d_rift = stats["query_rift_dist_to_center"]
    r      = stats["teacher_radius_l2"]

    off = 0.04
    ax.text(qr[0], qr[1] - off * 2.5,
            f"query (raw)\n$d = {d_raw:.3f}$",
            ha="center", va="top", fontsize=8, color=C["raw_q"], fontweight="bold")
    ax.text(qp[0], qp[1] + off * 2.5,
            f"query (Zero-DCE++)\n$d = {d_perc:.3f}$",
            ha="center", va="bottom", fontsize=8, color=C["perc"], fontweight="bold")
    ax.text(qf[0] + 0.08, qf[1] - off * 2.5,
            f"query (RIFT)\n$d = {d_rift:.3f}$",
            ha="center", va="top", fontsize=8, color=C["rift"], fontweight="bold")
    ax.text(cx + sx * 1.9, cy,
            f"teacher manifold\n$(r = {r:.3f})$",
            ha="left", va="center", fontsize=8, color=C["teacher"], fontweight="bold")

    # Legend
    handles = [
        mpatches.Patch(color=C["teacher"], alpha=0.5,
                       label=f"Teacher positives (n={len(tp)})"),
        mpatches.Patch(color=C["neg"], label=f"Hard negatives (n={len(hn)})"),
        mpatches.Patch(color=C["raw_q"], label="Query (raw)"),
        mpatches.Patch(color=C["perc"], label="Query (Zero-DCE++)"),
        mpatches.Patch(color=C["rift"], label="Query (RIFT)"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper left",
              framealpha=0.9, edgecolor="#DDDDDD")

    ax.set_xlabel("PC-1  (2-D PCA of embedding space)", fontsize=9, color=C["slate"])
    ax.set_ylabel("PC-2", fontsize=9, color=C["slate"])
    ax.tick_params(labelsize=7.5, colors=C["slate"])
    for sp in ax.spines.values():
        sp.set_edgecolor("#DDDDDD")

    # (title removed – use LaTeX \caption)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig3c_teacher_manifold.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig3c] → {OUT}/fig3c_teacher_manifold.{{png,pdf}}")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    draw_fig1()
    draw_fig3a()
    draw_fig3c()
    print("[Done] All figures saved to", OUT)
