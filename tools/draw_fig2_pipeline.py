#!/usr/bin/env python3
"""Draw RIFT pipeline architecture diagram (Figure 2) for ACM MM paper.

This is a vector-quality block diagram showing the inference path and
training-only supervision, matching Section 4 of the paper.

Run:  python tools/draw_fig2_pipeline.py
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
import numpy as np

OUT = Path("docs/figures/rift_paper_20260325/final")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------------------
C = {
    "ink":       "#1B2A3B",
    "amber":     "#E8943A",    # illumination front-end
    "amber_bg":  "#FFF5E6",
    "teal":      "#2A9D8F",    # trust mechanisms
    "teal_bg":   "#E8F6F3",
    "blue":      "#3D7ABF",    # encoder
    "blue_bg":   "#E8F0FA",
    "red":       "#D95C4A",    # training-only / geometry
    "red_bg":    "#FFF0EC",
    "green":     "#3A9A5C",    # output
    "slate":     "#6B7C93",
    "light":     "#F7F9FB",
    "white":     "#FFFFFF",
    "purple":    "#7B5DAA",    # losses
}


def _rc():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _box(ax, x, y, w, h, label, sublabel="",
         fc="white", ec="#1B2A3B", lw=1.5, fontsize=9, sub_fontsize=7.5,
         label_color="#1B2A3B", sub_color="#6B7C93",
         rounding=0.02, alpha=1.0, zorder=3, style="round"):
    """Draw a rounded box with title and optional subtitle."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"{style},pad=0.01,rounding_size={rounding}",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w/2, y + h*0.62, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=label_color, zorder=zorder+1)
        ax.text(x + w/2, y + h*0.30, sublabel, ha="center", va="center",
                fontsize=sub_fontsize, color=sub_color, zorder=zorder+1,
                linespacing=1.3)
    else:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=label_color, zorder=zorder+1)
    return box


def _region(ax, x, y, w, h, label="", fc="#F7F9FB", ec="#CCCCCC",
            lw=1.2, ls="-", fontsize=8, label_color="#6B7C93",
            label_pos="top", zorder=0):
    """Draw a background region with optional label."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder,
    )
    ax.add_patch(box)
    if label:
        if label_pos == "top":
            ax.text(x + w/2, y + h + 0.02, label, ha="center", va="bottom",
                    fontsize=fontsize, fontweight="bold", color=label_color, zorder=1)
        elif label_pos == "top-left":
            ax.text(x + 0.02, y + h - 0.02, label, ha="left", va="top",
                    fontsize=fontsize, fontweight="bold", color=label_color, zorder=1)


def _arrow(ax, x0, y0, x1, y1, color="#1B2A3B", lw=1.5, style="-|>",
           ls="-", rad=0.0, ms=12, zorder=5):
    """Draw a fancy arrow."""
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, mutation_scale=ms, linewidth=lw,
        color=color, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(arrow)


def _text(ax, x, y, s, fontsize=8, color="#1B2A3B", **kw):
    ax.text(x, y, s, fontsize=fontsize, color=color,
            ha=kw.get("ha", "center"), va=kw.get("va", "center"),
            fontweight=kw.get("fw", "normal"), zorder=kw.get("zorder", 6),
            fontstyle=kw.get("fs", "normal"))


def draw_fig2():
    _rc()
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_xlim(-0.05, 3.55)
    ax.set_ylim(-0.55, 1.85)
    ax.set_aspect("equal")
    ax.axis("off")

    # ===================================================================
    # REGION BACKGROUNDS
    # ===================================================================

    # Illumination front-end region
    _region(ax, 0.38, 0.55, 1.32, 1.15,
            "Illumination Front-End",
            fc=C["amber_bg"], ec=C["amber"], lw=1.8,
            fontsize=10, label_color=C["amber"])

    # Trust-controlled adaptation region
    _region(ax, 0.82, 0.57, 0.86, 0.55,
            "", fc=C["teal_bg"], ec=C["teal"], lw=1.2, ls="--")
    _text(ax, 1.25, 1.08, "Trust-Controlled Adaptation",
          fontsize=7.5, color=C["teal"], fw="bold")

    # Encoder region
    _region(ax, 1.85, 0.55, 1.15, 1.15,
            "Multi-Branch Encoder",
            fc=C["blue_bg"], ec=C["blue"], lw=1.8,
            fontsize=10, label_color=C["blue"])

    # Training-only region (dashed)
    _region(ax, 1.25, -0.50, 1.75, 0.48,
            "", fc=C["red_bg"], ec=C["red"], lw=1.5, ls="--")
    _text(ax, 2.12, -0.05, "Training-Only Supervision",
          fontsize=8.5, color=C["red"], fw="bold")

    # ===================================================================
    # INPUT
    # ===================================================================
    _box(ax, 0.0, 0.92, 0.30, 0.30, "Input  $x$", "",
         fc=C["light"], ec=C["ink"], fontsize=11, lw=2.0)

    # ===================================================================
    # ILLUMINATION FRONT-END (Section 4.2)
    # ===================================================================

    bw, bh = 0.35, 0.22  # standard box size

    # Coarse illumination
    _box(ax, 0.42, 1.30, bw, bh,
         "Coarse Illumination", "$Y \\to L_0,\\; S_\\phi$",
         fc=C["white"], ec=C["amber"], fontsize=8.5, sub_fontsize=7)

    # Feasible base correction
    _box(ax, 0.42, 0.95, bw, bh,
         "Feasible Base", "bounded gain\nchroma drift $\\leq \\tau_c$",
         fc=C["white"], ec=C["amber"], fontsize=8.5, sub_fontsize=7)
    _text(ax, 0.78, 0.87, "$\\hat{x}_{\\mathrm{base}}$",
          fontsize=8, color=C["amber"], fw="bold")

    # Model-aware refinement
    _box(ax, 0.85, 0.95, bw, bh,
         "Model-Aware", "$M_\\phi(x, \\hat{x}_{\\mathrm{base}}, \\hat{L}, F_{\\mathrm{mid}})$",
         fc=C["white"], ec=C["teal"], fontsize=8.5, sub_fontsize=6.5)
    _text(ax, 1.21, 0.87, "$\\hat{x}_{\\mathrm{att}}$",
          fontsize=8, color=C["teal"], fw="bold")

    # Identity protection
    _box(ax, 0.85, 0.62, bw, bh,
         "Identity Protection", "$P_{\\mathrm{id}}$ mask",
         fc=C["white"], ec=C["teal"], fontsize=8.5, sub_fontsize=7)
    _text(ax, 1.21, 0.57, "$\\hat{x}_{\\mathrm{prot}}$",
          fontsize=8, color=C["teal"], fw="bold")

    # Rollback gate
    _box(ax, 1.30, 0.62, bw, bh,
         "Rollback Gate", "stripe-wise $\\alpha$",
         fc=C["white"], ec=C["teal"], fontsize=8.5, sub_fontsize=7)
    _text(ax, 1.66, 0.57, "$\\tilde{x}$",
          fontsize=8, color=C["teal"], fw="bold")

    # ===================================================================
    # ENCODER (Section 4.3 continued)
    # ===================================================================

    branch_w, branch_h = 0.28, 0.18

    # Backbone
    _box(ax, 1.90, 1.30, 0.45, 0.25,
         "Backbone", "OSNet-AIN",
         fc=C["white"], ec=C["blue"], fontsize=9, sub_fontsize=7.5, lw=2.0)

    # Three branches
    _box(ax, 1.90, 0.97, branch_w, branch_h,
         "Raw Branch", "$F_{\\mathrm{raw}}$",
         fc="#E8EDF5", ec=C["blue"], fontsize=7.5, sub_fontsize=7)

    _box(ax, 2.22, 0.97, branch_w, branch_h,
         "Base Branch", "$F_{\\mathrm{base}}$",
         fc="#E8EDF5", ec=C["blue"], fontsize=7.5, sub_fontsize=7)

    _box(ax, 2.54, 0.97, branch_w, branch_h,
         "Adapted", "$F_{\\mathrm{adapted}}$",
         fc="#E8EDF5", ec=C["blue"], fontsize=7.5, sub_fontsize=7)

    # Stripe-wise attention fusion
    _box(ax, 1.98, 0.62, 0.90, 0.25,
         "Stripe-Wise Attention Fusion",
         "$F^{(s)}_{\\mathrm{fuse}} = \\sum \\alpha^{(s)}_b F^{(s)}_b$",
         fc=C["white"], ec=C["blue"], fontsize=8.5, sub_fontsize=7, lw=2.0)

    # ===================================================================
    # OUTPUT
    # ===================================================================
    _box(ax, 3.10, 0.92, 0.30, 0.30, "$z$", "ReID\nembedding",
         fc=C["white"], ec=C["green"], fontsize=13, sub_fontsize=7.5,
         label_color=C["green"], lw=2.5)

    # ===================================================================
    # TRAINING-ONLY (Section 4.4)
    # ===================================================================

    # Teacher encoder (frozen)
    _box(ax, 1.30, -0.43, 0.50, 0.22,
         "Teacher Encoder", "(frozen snapshot)",
         fc=C["white"], ec=C["red"], fontsize=8, sub_fontsize=7,
         lw=1.5)

    # Losses
    loss_y = -0.40
    _box(ax, 1.92, loss_y, 0.40, 0.18,
         "$\\mathcal{L}_{\\mathrm{tube}}$", "manifold tube",
         fc=C["white"], ec=C["red"], fontsize=8, sub_fontsize=6.5)

    _box(ax, 2.36, loss_y, 0.40, 0.18,
         "$\\mathcal{L}_{\\mathrm{sep}}$", "separation",
         fc=C["white"], ec=C["red"], fontsize=8, sub_fontsize=6.5)

    _box(ax, 1.92, -0.18, 0.40, 0.18,
         "$\\mathcal{L}_{\\mathrm{softap}}$", "ranking",
         fc=C["white"], ec=C["red"], fontsize=8, sub_fontsize=6.5)

    _box(ax, 2.36, -0.18, 0.40, 0.18,
         "$\\mathcal{L}_{\\mathrm{photo}}$", "feasibility",
         fc=C["white"], ec=C["amber"], fontsize=8, sub_fontsize=6.5)

    # ===================================================================
    # ARROWS – Main inference flow
    # ===================================================================

    # Input → Coarse illumination
    _arrow(ax, 0.30, 1.07, 0.42, 1.35, color=C["ink"], rad=0.15)

    # Input → Feasible base (direct path)
    _arrow(ax, 0.30, 1.05, 0.42, 1.06, color=C["ink"])

    # Input → Backbone (raw branch – long horizontal)
    _arrow(ax, 0.30, 1.00, 1.90, 1.42, color=C["slate"], rad=-0.10, lw=1.2)
    _text(ax, 1.02, 1.60, "raw input $x$",
          fontsize=7, color=C["slate"], fs="italic")

    # Coarse → Feasible base
    _arrow(ax, 0.595, 1.30, 0.595, 1.17, color=C["amber"])

    # Feasible base → Model-aware refinement
    _arrow(ax, 0.77, 1.06, 0.85, 1.06, color=C["amber"])

    # Model-aware refinement → Identity protection
    _arrow(ax, 1.025, 0.95, 1.025, 0.84, color=C["teal"])

    # Identity protection → Rollback gate
    _arrow(ax, 1.20, 0.73, 1.30, 0.73, color=C["teal"])

    # Rollback gate → Adapted branch (input to encoder)
    _arrow(ax, 1.65, 0.73, 2.54, 0.97, color=C["teal"], rad=-0.15)

    # Feasible base → Base branch
    _arrow(ax, 0.77, 1.00, 2.22, 1.00, color=C["amber"], lw=1.2, rad=0.08)

    # Backbone → three branches (vertical arrows down)
    for bx in [2.04, 2.36, 2.68]:
        _arrow(ax, bx, 1.30, bx, 1.15, color=C["blue"], lw=1.0)

    # Three branches → Fusion
    for bx in [2.04, 2.36, 2.68]:
        _arrow(ax, bx, 0.97, bx, 0.87, color=C["blue"], lw=1.0)

    # Fusion → Output z
    _arrow(ax, 2.88, 0.75, 3.10, 1.02, color=C["blue"], rad=-0.08, lw=2.0)

    # F_mid feedback: Backbone → Model-aware refinement
    _arrow(ax, 1.90, 1.38, 1.20, 1.06, color=C["teal"], ls="--", lw=1.2, rad=0.15)
    _text(ax, 1.62, 1.30, "$F_{\\mathrm{mid}}$",
          fontsize=7.5, color=C["teal"], fw="bold", fs="italic")

    # ===================================================================
    # ARROWS – Training-only paths (dashed)
    # ===================================================================

    # Encoder output → Losses
    _arrow(ax, 2.43, 0.62, 2.12, 0.0, color=C["red"], ls="--", lw=1.2, rad=0.05)
    _arrow(ax, 2.43, 0.62, 2.56, 0.0, color=C["red"], ls="--", lw=1.2, rad=-0.05)

    # Teacher → Losses
    _arrow(ax, 1.80, -0.32, 1.92, -0.32, color=C["red"], ls="--", lw=1.0)
    _arrow(ax, 1.80, -0.25, 1.92, -0.12, color=C["red"], ls="--", lw=1.0)

    # ===================================================================
    # LEGEND
    # ===================================================================
    legend_x, legend_y = -0.02, -0.15
    legend_items = [
        (C["amber"],  "Feasible correction (Sec. 4.2)"),
        (C["teal"],   "Trust-controlled adaptation (Sec. 4.3)"),
        (C["blue"],   "Encoder & fusion (Sec. 4.3)"),
        (C["red"],    "Geometry-guided training (Sec. 4.4)"),
    ]
    for i, (color, label) in enumerate(legend_items):
        y_pos = legend_y - i * 0.09
        ax.add_patch(FancyBboxPatch(
            (legend_x, y_pos - 0.025), 0.12, 0.05,
            boxstyle="round,pad=0.005,rounding_size=0.01",
            facecolor=color, edgecolor="none", alpha=0.7, zorder=5))
        _text(ax, legend_x + 0.15, y_pos, label,
              fontsize=7, color=C["ink"], ha="left")

    # Inference vs training legend
    _arrow(ax, legend_x, legend_y - 0.40, legend_x + 0.10, legend_y - 0.40,
           color=C["ink"], lw=1.5)
    _text(ax, legend_x + 0.15, legend_y - 0.40, "Inference path",
          fontsize=7, color=C["ink"], ha="left")

    _arrow(ax, legend_x, legend_y - 0.48, legend_x + 0.10, legend_y - 0.48,
           color=C["red"], lw=1.2, ls="--")
    _text(ax, legend_x + 0.15, legend_y - 0.48, "Training only",
          fontsize=7, color=C["red"], ha="left")

    # ===================================================================
    # SAVE
    # ===================================================================
    for ext in ("png", "pdf", "svg"):
        try:
            fig.savefig(OUT / f"fig2_pipeline.{ext}", dpi=350, bbox_inches="tight")
        except PermissionError:
            print(f"[WARN] Could not write {ext}")
    plt.close(fig)
    print(f"[Fig2] → {OUT}/fig2_pipeline.{{png,pdf,svg}}")


if __name__ == "__main__":
    draw_fig2()
