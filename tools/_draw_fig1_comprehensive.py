#!/usr/bin/env python3
"""Draw a comprehensive RIIC-ReID Fig 1 for the main paper.

Layout (single figure):
┌──────────────────────────────────────────────────────────┐
│  (a) Retrieval comparison                                │
│  ┌──────┐  ┌────┐┌────┐┌────┐┌────┐┌────┐               │
│  │Query │  │ R1 ││ R2 ││ R3 ││ R4 ││ R5 │  method + AP  │
│  │ Raw  │  │    ││    ││    ││    ││    │               │
│  └──────┘  └────┘└────┘└────┘└────┘└────┘               │
│  ┌──────┐  ┌────┐┌────┐┌────┐┌────┐┌────┐               │
│  │Query │  │    ││    ││    ││ ✗  ││ ✗  │  RetinexNet   │
│  │Retinx│  │    ││    ││    ││    ││    │               │
│  └──────┘  └────┘└────┘└────┘└────┘└────┘               │
│  ... (Zero-DCE++, RIIC-ReID)                             │
│                                                          │
│  (b) Correction comparison        (c) Embedding geometry │
│  ┌────┐┌────┐┌────┐┌────┐        ┌──────────────┐       │
│  │Raw ││Ret ││ZDC ││RIIC│        │  PCA / t-SNE │       │
│  └────┘└────┘└────┘└────┘        └──────────────┘       │
└──────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

# --- Paths ---
CASE_DIR = Path("docs/figures/riic_reid_fig1_best/case_21_001132")
OUTPUT_DIR = Path("docs/figures/riic_reid_fig1_best/final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Style ---
PALETTE = {
    "green": "#1B9E4B",
    "red": "#D6453D",
    "ink": "#1A2E40",
    "slate": "#5C6E80",
    "muted": "#8A9AAC",
    "bg": "#FFFFFF",
    "panel_bg": "#F7F9FB",
    "blue": "#2563EB",
    "purple": "#7C3AED",
    "orange": "#D97706",
    "teal": "#0D9488",
}

METHOD_COLORS = {
    "raw": PALETTE["slate"],
    "retinexnet": PALETTE["orange"],
    "zerodcepp": PALETTE["purple"],
    "riic": PALETTE["blue"],
}

METHOD_LABELS = {
    "raw": "Matched baseline",
    "retinexnet": "RetinexNet",
    "zerodcepp": "Zero-DCE++",
    "riic": "RIIC-ReID (ours)",
}

BORDER_WIDTH = 5  # px for green/red border


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def add_border(img: np.ndarray, color_hex: str, width: int = BORDER_WIDTH) -> np.ndarray:
    """Add a colored border around an image."""
    r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
    h, w, c = img.shape
    bordered = img.copy()
    # Top / bottom
    bordered[:width, :] = [r, g, b]
    bordered[-width:, :] = [r, g, b]
    # Left / right
    bordered[:, :width] = [r, g, b]
    bordered[:, -width:] = [r, g, b]
    return bordered


def style_matplotlib():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def draw_figure():
    style_matplotlib()

    # Load metrics
    with open(CASE_DIR / "metrics.json", "r") as f:
        meta = json.load(f)

    methods_order = ["raw", "retinexnet", "zerodcepp", "riic"]
    topk = 5

    # --- Create figure ---
    fig = plt.figure(figsize=(12.5, 7.8))

    # Main grid: top section (retrieval, ~72%), bottom section (correction + bar, ~28%)
    gs_main = GridSpec(2, 1, figure=fig, height_ratios=[3.0, 1.1],
                       hspace=0.32, left=0.14, right=0.97, top=0.93, bottom=0.04)

    # --- (a) Retrieval comparison (top) ---
    gs_retrieval = gs_main[0].subgridspec(4, 7, wspace=0.06, hspace=0.18,
                                           width_ratios=[1.3, 0.12, 1, 1, 1, 1, 1])

    for row_idx, mkey in enumerate(methods_order):
        method_meta = meta["methods"][mkey]
        ap = method_meta["ap"]
        ranked = method_meta["ranked"]

        # Query image (column 0)
        ax_q = fig.add_subplot(gs_retrieval[row_idx, 0])
        query_img = load_image(CASE_DIR / f"query_{mkey}.png")
        ax_q.imshow(query_img)
        ax_q.set_xticks([])
        ax_q.set_yticks([])
        for spine in ax_q.spines.values():
            spine.set_visible(True)
            spine.set_color(METHOD_COLORS[mkey])
            spine.set_linewidth(2.0)

        # Method label + AP on the left
        label_text = METHOD_LABELS[mkey]
        if row_idx == 0:
            ax_q.set_title("Query", fontsize=9.5, fontweight="bold", color=PALETTE["ink"], pad=6)

        # Method name to the left of the query image
        fw = "bold" if mkey == "riic" else "semibold"
        mc = METHOD_COLORS[mkey]
        ax_q.text(-0.08, 0.62, label_text, transform=ax_q.transAxes,
                  fontsize=9, fontweight=fw, color=mc,
                  ha="right", va="center", rotation=0)
        ax_q.text(-0.08, 0.35, f"AP = {ap:.3f}", transform=ax_q.transAxes,
                  fontsize=8.5, fontweight="normal", color=mc,
                  ha="right", va="center", rotation=0,
                  fontstyle="italic")

        # Arrow column (column 1) — just draw an arrow
        ax_arrow = fig.add_subplot(gs_retrieval[row_idx, 1])
        ax_arrow.set_xlim(0, 1)
        ax_arrow.set_ylim(0, 1)
        ax_arrow.annotate("", xy=(0.9, 0.5), xytext=(0.1, 0.5),
                          arrowprops=dict(arrowstyle="->", color=PALETTE["slate"],
                                          lw=1.5, mutation_scale=12))
        ax_arrow.axis("off")

        # Gallery images (columns 2-6)
        for col_idx in range(topk):
            ax_g = fig.add_subplot(gs_retrieval[row_idx, col_idx + 2])

            rank_info = ranked[col_idx]
            is_match = rank_info["match"]
            border_color = PALETTE["green"] if is_match else PALETTE["red"]

            gallery_dir = CASE_DIR / f"gallery_{mkey}"
            gallery_file = rank_info["file"]
            gallery_img = load_image(gallery_dir / gallery_file)
            gallery_img = add_border(gallery_img, border_color, width=BORDER_WIDTH)

            ax_g.imshow(gallery_img)
            ax_g.set_xticks([])
            ax_g.set_yticks([])
            for spine in ax_g.spines.values():
                spine.set_visible(False)

            if row_idx == 0:
                ax_g.set_title(f"Rank {col_idx + 1}", fontsize=9,
                              fontweight="bold", color=PALETTE["ink"], pad=6)

            # Add match/mismatch indicator below each gallery image
            if is_match:
                sym_text = "MATCH"
                sym_color = PALETTE["green"]
                sym_size = 6.5
            else:
                sym_text = "WRONG"
                sym_color = PALETTE["red"]
                sym_size = 6.5
            ax_g.text(0.5, -0.04, sym_text, transform=ax_g.transAxes,
                     fontsize=sym_size, fontweight="bold", color=sym_color,
                     ha="center", va="top")

    # --- (b) Correction comparison (bottom left) ---
    gs_bottom = gs_main[1].subgridspec(1, 2, wspace=0.12, width_ratios=[2.5, 1.0])
    gs_correction = gs_bottom[0].subgridspec(1, 4, wspace=0.06)

    correction_labels = [
        ("query_raw.png", "Input"),
        ("query_retinexnet.png", "RetinexNet"),
        ("query_zerodcepp.png", "Zero-DCE++"),
        ("query_riic.png", "RIIC-ReID"),
    ]

    for ci, (fname, label) in enumerate(correction_labels):
        ax = fig.add_subplot(gs_correction[0, ci])
        img = load_image(CASE_DIR / fname)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(label, fontsize=8.5, fontweight="bold" if "RIIC" in label else "normal",
                     color=METHOD_COLORS.get(
                         {"Input": "raw", "RetinexNet": "retinexnet",
                          "Zero-DCE++": "zerodcepp", "RIIC-ReID": "riic"}[label], PALETTE["ink"]
                     ), labelpad=3)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color(PALETTE["muted"])
        if ci == 0:
            ax.set_title("(b) Correction comparison", fontsize=9.5,
                        fontweight="bold", color=PALETTE["ink"], pad=6, loc="left")

    # --- (c) AP bar chart (bottom right) ---
    ax_bar = fig.add_subplot(gs_bottom[1])
    ap_values = [meta["methods"][m]["ap"] for m in methods_order]
    c5_values = [meta["methods"][m]["correct_at_k"] for m in methods_order]
    x_pos = np.arange(len(methods_order))
    bar_colors = [METHOD_COLORS[m] for m in methods_order]
    bar_labels = ["Baseline", "RetinexNet", "Zero-DCE++", "RIIC-ReID"]

    bars = ax_bar.bar(x_pos, ap_values, color=bar_colors, width=0.62, alpha=0.90,
                      edgecolor="white", linewidth=1.0)

    # Add value labels on bars
    for bi, (bar, ap_val, c5) in enumerate(zip(bars, ap_values, c5_values)):
        ypos = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2, ypos + 0.018,
                   f"{ap_val:.3f}", ha="center", va="bottom", fontsize=8,
                   fontweight="bold", color=bar_colors[bi])
        ax_bar.text(bar.get_x() + bar.get_width() / 2, ypos / 2,
                   f"{c5}/{topk}", ha="center", va="center", fontsize=9,
                   fontweight="bold", color="white")

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(bar_labels, fontsize=7.5, rotation=18, ha="right")
    ax_bar.set_ylim(0, 1.12)
    ax_bar.set_ylabel("AP", fontsize=9)
    ax_bar.set_title("(c) Per-query AP", fontsize=9.5,
                    fontweight="bold", color=PALETTE["ink"], pad=6, loc="left")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_color(PALETTE["muted"])
    ax_bar.spines["bottom"].set_color(PALETTE["muted"])
    ax_bar.tick_params(colors=PALETTE["slate"])

    # --- Section title ---
    fig.text(0.14, 0.965, "(a) Retrieval comparison — same query, different illumination corrections",
             fontsize=11, fontweight="bold", color=PALETTE["ink"], va="top")

    # --- Legend for green/red ---
    legend_handles = [
        mpatches.Patch(facecolor=PALETTE["green"], edgecolor="none", label="Correct match"),
        mpatches.Patch(facecolor=PALETTE["red"], edgecolor="none", label="Wrong identity"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=8,
              frameon=True, fancybox=True, framealpha=0.9, edgecolor=PALETTE["muted"],
              bbox_to_anchor=(0.98, 0.97))

    # --- Save ---
    for ext in ("png", "pdf"):
        out_path = OUTPUT_DIR / f"fig1_comprehensive.{ext}"
        fig.savefig(str(out_path), dpi=350, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    draw_figure()
