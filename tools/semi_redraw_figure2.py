#!/usr/bin/env python3
"""Semi-redraw the nano-banana Figure 2 for publication use.

Strategy:
- keep the original generated figure as the structural background
- mask oversized bitmap text regions with locally sampled background color
- re-typeset the labels with compact, consistent academic typography
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "references" / "write" / "acmart" / "acmart" / "figures_generated" / "figure2_2.jpg"
OUT = ROOT / "references" / "write" / "acmart" / "acmart" / "figures_generated" / "figure2_2_refined.png"

FONT_REG = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")


@dataclass
class TextBlock:
    box: tuple[int, int, int, int]
    text: str
    size: int
    weight: Literal["regular", "bold"] = "regular"
    align: Literal["left", "center"] = "center"
    fill: tuple[int, int, int] = (34, 34, 34)
    padding: int = 10
    line_spacing: int = 6
    radius: int = 10
    sample_expand: int = 6


def load_font(size: int, weight: str) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if weight == "bold" else FONT_REG
    return ImageFont.truetype(str(path), size=size)


def sample_fill_color(img: Image.Image, box: tuple[int, int, int, int], expand: int = 6) -> tuple[int, int, int]:
    x0, y0, x1, y1 = box
    x0 = max(0, x0 - expand)
    y0 = max(0, y0 - expand)
    x1 = min(img.width, x1 + expand)
    y1 = min(img.height, y1 + expand)
    patch = img.crop((x0, y0, x1, y1))
    stat = ImageStat.Stat(patch)
    # Median-like estimate; mean is fine here because text occupies a small area.
    return tuple(int(v) for v in stat.median[:3])


def fit_font(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], size: int, weight: str, padding: int, spacing: int) -> ImageFont.FreeTypeFont:
    x0, y0, x1, y1 = box
    max_w = max(10, x1 - x0 - 2 * padding)
    max_h = max(10, y1 - y0 - 2 * padding)
    current = size
    while current >= 12:
        font = load_font(current, weight)
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= max_w and h <= max_h:
            return font
        current -= 1
    return load_font(12, weight)


def draw_block(base: Image.Image, overlay: ImageDraw.ImageDraw, block: TextBlock) -> None:
    x0, y0, x1, y1 = block.box
    bg = sample_fill_color(base, block.box, block.sample_expand)
    overlay.rounded_rectangle(block.box, radius=block.radius, fill=bg)
    font = fit_font(overlay, block.text, block.box, block.size, block.weight, block.padding, block.line_spacing)
    bbox = overlay.multiline_textbbox((0, 0), block.text, font=font, spacing=block.line_spacing, align=block.align)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    if block.align == "left":
        x = x0 + block.padding
    else:
        x = x0 + (x1 - x0 - tw) / 2
    y = y0 + (y1 - y0 - th) / 2 - 1
    overlay.multiline_text(
        (x, y),
        block.text,
        font=font,
        fill=block.fill,
        spacing=block.line_spacing,
        align=block.align,
    )


def main() -> None:
    img = Image.open(SRC).convert("RGB")
    out = img.copy()
    draw = ImageDraw.Draw(out)

    blocks: list[TextBlock] = [
        TextBlock((18, 16, 780, 82), "[Stage 1 – Illumination Decomposition]", 34, "bold", "left", padding=14, radius=16),
        TextBlock((1332, 16, 2028, 92), "[Stage 2 – Feasible Base Correction]", 34, "bold", "left", padding=14, radius=16),
        TextBlock((1768, 68, 2010, 112), "Feasible set F(x)", 20, "regular", "left", padding=10, radius=6),

        TextBlock((724, 116, 1038, 302), "Multi-Scale\nIllumination\nEstimator\nEφ", 30, "bold", radius=18),
        TextBlock((770, 330, 906, 386), "coarse\nillumination", 18, "regular", radius=8),
        TextBlock((420, 470, 708, 648), "Sensitivity\nEstimator\nSφ(x)", 29, "bold", radius=18),
        TextBlock((358, 314, 672, 458), "Y = 0.299·xR +\n0.587·xG + 0.114·xB", 22, "regular", radius=10),
        TextBlock((1168, 338, 1382, 520), "Color Shift\nCorrection\nlearned white-\nbalance gain", 24, "bold", fill=(202, 128, 45), radius=16),
        TextBlock((692, 566, 1278, 640), "Lc = 1 + (L0 - 1) ⊙ Sφ(x)  [Eq. 5]", 20, "regular", radius=8),

        TextBlock((1580, 116, 1912, 166), "Safe Inversion", 26, "bold", radius=14),
        TextBlock((1580, 182, 1892, 262), "Inverse scale: a = 1/(L̂ + ε)", 19, "regular", radius=12),
        TextBlock((1580, 268, 1892, 350), "Saturation bound:\nb = 0.99/(x + ε)", 19, "regular", radius=12),
        TextBlock((1580, 356, 1892, 424), "s = softmin(a, b)", 21, "regular", radius=12),
        TextBlock((1580, 430, 1892, 488), "Gain clamp", 21, "regular", radius=12),
        TextBlock((1580, 494, 1892, 566), "||Δchroma||∞ ≤ τc", 21, "regular", radius=12),
        TextBlock((1740, 624, 2004, 664), "x̂base = x ⊙ s", 20, "regular", radius=8),
        TextBlock((1510, 846, 1992, 888), "x̂base (feasible base-corrected image)", 18, "regular", radius=8),

        TextBlock((128, 690, 904, 760), "[Stage 3 – Trust-Controlled Adaptation]", 34, "bold", "left", padding=14, radius=16),
        TextBlock((176, 796, 722, 938), "Model-Aware Residual Mφ\nEq. 6: responds to encoder’s internal\nrepresentation, not image statistics alone", 24, "bold", radius=16),
        TextBlock((1048, 726, 1418, 822), "Fmid\n(mid-level features guide\nillumination refinement)", 19, "bold", radius=8),
        TextBlock((178, 1028, 744, 1156), "Identity Protection Map Pid\nPid ∈ [0,1], from early backbone activations,\nsuppresses changes in identity-critical regions", 23, "bold", radius=16),
        TextBlock((154, 1188, 782, 1244), "x̂prot = x + (1 − Pid) ⊙ (x̂att − x)  [Eq. 7]", 20, "regular", radius=8),
        TextBlock((182, 1266, 768, 1396), "Stripe-Wise Rollback Gates α\nconditioned on: illumination, color risk,\ncorrection magnitude, Fmid", 23, "bold", radius=16),
        TextBlock((208, 1406, 782, 1458), "x̃ = α ⊙ x̂prot + (1 − α) ⊙ x  [Eq. 8]", 20, "regular", radius=8),
        TextBlock((192, 1496, 796, 1602), "Lightweight Refiner\nsharpens local detail; Pid re-applied to\nprotect identity regions", 23, "bold", radius=16),

        TextBlock((1248, 928, 1900, 992), "[Stage 4 – Three-Branch Fusion]", 34, "bold", "left", padding=14, radius=16),
        TextBlock((1456, 1042, 1710, 1288), "Stripe-Aware\nBranch Attention\nFfuse(s) = Σ αb(s) Fb(s)", 25, "bold", radius=16),
        TextBlock((1020, 1150, 1372, 1246), "OSNet-AIN\nBackbone", 28, "bold", fill=(36, 50, 75), radius=16),
        TextBlock((1340, 1388, 1988, 1498), "α(s) = softmax(MLP(F1; ... ; Fs)/τ)  [Eq. 10]\ngate receives front-end statistics: illumination strength,\ncolor risk, rollback strength, correction magnitude", 18, "regular", radius=10),
        TextBlock((1768, 1138, 1906, 1230), "Local\nStripe\nHead", 22, "bold", radius=12),

        TextBlock((1364, 1568, 1986, 1666), "Frozen Teacher Encoder\n(raw-reference snapshot,\nfrozen after warmup)", 22, "bold", fill=(122, 84, 78), radius=14),
        TextBlock((1370, 1740, 1652, 1938), "Lgeo\nGeometry supervision\nLtube + Lsep + Lsoftap:\nmanifold tube, separation,\ndifferentiable cross-light SoftAP ranking", 20, "bold", radius=16),
        TextBlock((1700, 1742, 1992, 1934), "Laux\nAuxiliary\nnuisance decorrelation,\nmasked identity\npreservation (kept weak)", 20, "bold", radius=16),

        TextBlock((22, 1666, 330, 1718), "Loss Functions", 30, "bold", "left", padding=14, radius=10),
        TextBlock((98, 1742, 378, 1934), "Lreid\nLcls + Lmetric\nCE + Triplet + ArcFace\n+ Center", 20, "bold", radius=16),
        TextBlock((430, 1742, 814, 1938), "Lphoto\nPhotometric feasibility\nreconstruction, smoothness,\nchroma bound, λphoto\nannealed during training", 20, "bold", radius=16),
        TextBlock((862, 1742, 1234, 1932), "Liicl\nIllumination consistency\nLiicl = V⁻¹Σ(1 − cos(z, zv))\non synthetic photometric\nvariants", 20, "bold", radius=16),
        TextBlock((20, 1984, 2028, 2038), "Raw branch   Base-corrected branch   Adapted branch      Ltotal = λreid·Lreid + λphoto·Lphoto + λiicl·Liicl + λgeo·Lgeo + λaux·Laux  [Eq. 12]", 18, "regular", "left", padding=18, radius=8),
    ]

    for block in blocks:
        draw_block(img, draw, block)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
