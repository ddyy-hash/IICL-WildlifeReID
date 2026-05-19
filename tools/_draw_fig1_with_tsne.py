#!/usr/bin/env python3
"""Comprehensive Fig 1 for RIIC-ReID paper: retrieval comparison + correction + t-SNE.

Case 231 (ID 231, lum=0.565):
  Raw: 4/5, AP=0.736 | RetinexNet: 2/5, AP=0.516 | Zero-DCE++: 3/5, AP=0.617 | RIIC: 5/5, AP=0.772
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
# t-SNE implemented without sklearn to avoid dependency issues
def tsne_simple(X, n_components=2, perplexity=30, n_iter=1000, lr=200.0, seed=42):
    """Simple t-SNE using numpy (Barnes-Hut not needed for <200 points)."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    # Pairwise distances
    D = np.sum(X ** 2, axis=1, keepdims=True) - 2 * X @ X.T + np.sum(X ** 2, axis=1)
    D = np.maximum(D, 0)
    # Compute P (joint probabilities)
    P = np.zeros((n, n))
    target_entropy = np.log(perplexity)
    for i in range(n):
        di = D[i].copy(); di[i] = np.inf
        lo, hi = 1e-20, 1e4
        for _ in range(50):
            beta = (lo + hi) / 2
            pi = np.exp(-di * beta); pi[i] = 0
            sp = pi.sum()
            if sp < 1e-30: lo = beta; continue
            pi /= sp
            H = -np.sum(pi * np.log(pi + 1e-30))
            if H > target_entropy: hi = beta
            else: lo = beta
            if abs(H - target_entropy) < 1e-5: break
        P[i] = pi
    P = (P + P.T) / (2.0 * n)
    P = np.maximum(P, 1e-12)
    # Early exaggeration
    P *= 4.0
    Y = rng.randn(n, n_components) * 0.01
    velocity = np.zeros_like(Y)
    gains = np.ones_like(Y)
    for it in range(n_iter):
        # Compute Q
        dY = np.sum(Y ** 2, axis=1, keepdims=True) - 2 * Y @ Y.T + np.sum(Y ** 2, axis=1)
        num = 1.0 / (1.0 + dY)
        np.fill_diagonal(num, 0)
        Q = num / (num.sum() + 1e-30)
        Q = np.maximum(Q, 1e-12)
        # Gradient
        PQ = P - Q
        grad = np.zeros_like(Y)
        for i in range(n):
            grad[i] = 4.0 * np.sum((PQ[i] * num[i])[:, None] * (Y[i] - Y), axis=0)
        # Update
        gains = (gains + 0.2) * ((grad > 0) != (velocity > 0)) + (gains * 0.8) * ((grad > 0) == (velocity > 0))
        gains = np.maximum(gains, 0.01)
        velocity = 0.8 * velocity - lr * gains * grad
        Y += velocity
        if it == 100:
            P /= 4.0  # stop early exaggeration
    return Y

# ---- Paths ----
CASE_DIR = Path("docs/figures/riic_reid_fig1_best/case_231_000011")
OUTPUT_DIR = Path("docs/figures/riic_reid_fig1_best/final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUERY_IDX = 1228  # index in the full query list
QUERY_LABEL = "231"
TOPK = 5
TSNE_GALLERY_IDS = 15  # how many gallery identities to include in t-SNE

# ---- Style ----
PAL = {
    "green": "#1B9E4B", "red": "#D6453D", "ink": "#1A2E40",
    "slate": "#5C6E80", "muted": "#8A9AAC", "light_bg": "#F5F7FA",
    "blue": "#2563EB", "purple": "#7C3AED", "orange": "#D97706", "teal": "#0D9488",
}
METHOD_COLORS = {"raw": PAL["slate"], "retinexnet": PAL["orange"],
                 "zerodcepp": PAL["purple"], "riic": PAL["blue"]}
METHOD_LABELS = {"raw": "Matched baseline", "retinexnet": "RetinexNet",
                 "zerodcepp": "Zero-DCE++", "riic": "RIIC-ReID (ours)"}
BORDER_W = 5


def load_img(p):
    return np.array(Image.open(p).convert("RGB"))


def add_border(img, hex_color, w=BORDER_W):
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    out = img.copy()
    out[:w, :] = out[-w:, :] = [r, g, b]
    out[:, :w] = out[:, -w:] = [r, g, b]
    return out


# ================================================================
# Feature extraction for t-SNE
# ================================================================
import torch, torch.nn.functional as F
from torchvision import transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_split(root):
    samples = []
    for ld in sorted(Path(root).iterdir()):
        if not ld.is_dir():
            continue
        for img in sorted(ld.iterdir()):
            if img.suffix.lower() in IMAGE_EXTS:
                samples.append((str(img), ld.name, img.stem))
    return samples


def resolve_enh(stem, root):
    p = root / (stem + ".jpg")
    return str(p) if p.exists() else None


@torch.no_grad()
def extract_batch(paths, model, tfm, mode, device, bs=16):
    chunks = []
    for i in range(0, len(paths), bs):
        batch = torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths[i:i + bs]], 0).to(device)
        if mode == "riic":
            out = model(batch, return_illuminated=False)
        else:
            out = model.forward_raw_reference(batch, detach=True)
        chunks.append(F.normalize(out["features"], p=2, dim=1).cpu())
    return torch.cat(chunks, 0).numpy()


def build_tsne_data():
    """Extract features for query + nearby gallery identities under all 4 methods."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.core.config import load_config
    from app.core.joint_model import JointReIDModel

    cfg = load_config("config/illumination_config_atrw.yaml")
    tc = cfg.get("training", {})
    h, w = int(tc.get("image_height", 256)), int(tc.get("image_width", 384))
    tfm = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor()])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load("checkpoints/atrw_routeb_theoryB/joint_best.pth",
                       map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    model_cfg = cfg.get("model", {})
    illum_cfg = cfg.get("illumination_module", {})
    ipaid_params = dict(illum_cfg.get("module_params", {}))
    for key in ("feature_fusion", "branch_attention_fusion", "nuisance_head",
                "reid_head", "backbone_random_erasing"):
        sub = model_cfg.get(key, {})
        if sub:
            ipaid_params["_" + key] = sub
    num_classes = 107
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and k.endswith(
                ("classifier.weight", "global_classifier.weight", "arcface.weight")):
            num_classes = int(v.shape[0])
            break
    model = JointReIDModel(num_classes=num_classes, backbone_name="osnet_ain_x1_0",
                            num_stripes=6, pretrained_backbone=False, use_ipaid=True,
                            dropout=0.1, ipaid_params=ipaid_params)
    ms = model.state_dict()
    keep = {k: v for k, v in state.items()
            if k in ms and isinstance(v, torch.Tensor) and v.shape == ms[k].shape}
    model.load_state_dict(keep, strict=False)
    model = model.to(device).eval()

    # Load data
    query_samples = load_split("data/processed/atrw/query")
    gallery_samples = load_split("data/processed/atrw/gallery")

    retinex_root = Path("downloads/westc_perceptual_assets_20260325/root/autodl-tmp/v2_2/"
                         "dog_reid_web/data/perceptual_baselines/atrw/retinexnet/test")
    zerodce_root = Path("downloads/westc_perceptual_assets_20260325/root/autodl-tmp/v2_2/"
                         "dog_reid_web/data/perceptual_baselines/atrw/zerodcepp/test")

    # Get query info
    qs = query_samples[QUERY_IDX]

    # Select gallery samples: same ID + confuser IDs (232, and a few random others for context)
    # First, extract ALL gallery features under raw to find nearest confusers
    gp_raw = [s[0] for s in gallery_samples]
    gf_raw_all = extract_batch(gp_raw, model, tfm, "raw", device)

    # Query features (raw)
    qf_raw = extract_batch([qs[0]], model, tfm, "raw", device)

    # Find closest gallery identities to the query
    dists = 1.0 - gf_raw_all @ qf_raw[0]
    order = np.argsort(dists)

    # Collect the top TSNE_GALLERY_IDS distinct identities from nearest gallery
    seen_ids = set()
    selected_gallery_indices = []
    for gi in order:
        gid = gallery_samples[int(gi)][1]
        if gid not in seen_ids:
            seen_ids.add(gid)
        if len(seen_ids) > TSNE_GALLERY_IDS:
            break
        selected_gallery_indices.append(int(gi))

    # Also ensure we have all samples of the query identity
    for gi, gs in enumerate(gallery_samples):
        if gs[1] == QUERY_LABEL and gi not in selected_gallery_indices:
            selected_gallery_indices.append(gi)

    selected_gallery_indices = sorted(set(selected_gallery_indices))

    # Now extract features under all 4 methods for selected gallery + query
    sel_gallery = [gallery_samples[i] for i in selected_gallery_indices]
    sel_labels = [s[1] for s in sel_gallery]

    results = {}
    for mkey, gpath_fn, mode in [
        ("raw", lambda s: s[0], "raw"),
        ("retinexnet", lambda s: resolve_enh(s[2], retinex_root) or s[0], "raw"),
        ("zerodcepp", lambda s: resolve_enh(s[2], zerodce_root) or s[0], "raw"),
        ("riic", lambda s: s[0], "riic"),
    ]:
        g_paths = [gpath_fn(s) for s in sel_gallery]
        q_path = resolve_enh(qs[2], retinex_root) if mkey == "retinexnet" else (
            resolve_enh(qs[2], zerodce_root) if mkey == "zerodcepp" else qs[0])
        if q_path is None:
            q_path = qs[0]

        g_feats = extract_batch(g_paths, model, tfm, mode, device)
        q_feat = extract_batch([q_path], model, tfm, mode, device)

        results[mkey] = {"gallery_feats": g_feats, "query_feat": q_feat[0]}

    return results, sel_labels


def draw_figure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.facecolor": "white", "figure.facecolor": "white",
    })

    with open(CASE_DIR / "metrics.json") as f:
        meta = json.load(f)

    methods_order = ["raw", "retinexnet", "zerodcepp", "riic"]

    # ---- Build embedding distance data ----
    print("Extracting features for embedding analysis...")
    tsne_data, tsne_labels = build_tsne_data()

    # Compute cosine similarity: query → same-ID gallery centroid, query → confuser centroid
    same_id_mask = np.array([l == QUERY_LABEL for l in tsne_labels])
    confuser_mask = np.array([l == "232" for l in tsne_labels])

    sim_data = {}  # {method: {same_avg, confuser_avg, same_max, confuser_max, margin}}
    for mkey in methods_order:
        td = tsne_data[mkey]
        qf = td["query_feat"]  # shape (dim,)
        gf = td["gallery_feats"]  # shape (N, dim)

        cos_sim = gf @ qf  # shape (N,)

        same_sims = cos_sim[same_id_mask]
        conf_sims = cos_sim[confuser_mask]

        sim_data[mkey] = {
            "same_avg": float(same_sims.mean()),
            "confuser_avg": float(conf_sims.mean()),
            "same_min": float(same_sims.min()),
            "confuser_max": float(conf_sims.max()),
            "margin": float(same_sims.mean() - conf_sims.mean()),
        }
        print(f"  {mkey:>12s}: sim(same)={same_sims.mean():.4f}  sim(confuser)={conf_sims.mean():.4f}  margin={same_sims.mean()-conf_sims.mean():.4f}")

    # ================================================================
    # Draw the figure
    # ================================================================
    fig = plt.figure(figsize=(15.5, 10.5))

    # Push grid down to create clear space for title at top
    # right=0.945 leaves room so legend can sit in the 0.947-0.998 strip without overlapping Rank 5
    gs_main = GridSpec(2, 1, figure=fig, height_ratios=[2.8, 2.2],
                       hspace=0.28, left=0.115, right=0.945, top=0.87, bottom=0.04)

    # ---- (a) Retrieval comparison (top) ----
    gs_ret = gs_main[0].subgridspec(4, 7, wspace=0.04, hspace=0.16,
                                     width_ratios=[1.2, 0.14, 1, 1, 1, 1, 1])

    for row, mkey in enumerate(methods_order):
        mm = meta["methods"][mkey]
        ap = mm["ap"]
        ranked = mm["ranked"]

        # Query
        ax_q = fig.add_subplot(gs_ret[row, 0])
        ax_q.imshow(load_img(CASE_DIR / f"query_{mkey}.png"))
        ax_q.set_xticks([]); ax_q.set_yticks([])
        for sp in ax_q.spines.values():
            sp.set_visible(True); sp.set_color(METHOD_COLORS[mkey]); sp.set_linewidth(2.2)
        if row == 0:
            ax_q.set_title("Query", fontsize=10, fontweight="bold", color=PAL["ink"], pad=8)
        fw = "bold" if mkey == "riic" else "semibold"
        ax_q.text(-0.10, 0.65, METHOD_LABELS[mkey], transform=ax_q.transAxes,
                  fontsize=9.5, fontweight=fw, color=METHOD_COLORS[mkey], ha="right", va="center")
        ax_q.text(-0.10, 0.35, f"AP = {ap:.3f}", transform=ax_q.transAxes,
                  fontsize=9, color=METHOD_COLORS[mkey], ha="right", va="center", fontstyle="italic")

        # Arrow
        ax_a = fig.add_subplot(gs_ret[row, 1])
        ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1)
        ax_a.annotate("", xy=(0.9, 0.5), xytext=(0.1, 0.5),
                       arrowprops=dict(arrowstyle="->", color=PAL["slate"], lw=1.5, mutation_scale=12))
        ax_a.axis("off")

        # Gallery top-k
        for ci in range(TOPK):
            ax_g = fig.add_subplot(gs_ret[row, ci + 2])
            ri = ranked[ci]
            match = ri["match"]
            bc = PAL["green"] if match else PAL["red"]
            gimg = add_border(load_img(CASE_DIR / f"gallery_{mkey}" / ri["file"]), bc)
            ax_g.imshow(gimg); ax_g.set_xticks([]); ax_g.set_yticks([])
            for sp in ax_g.spines.values():
                sp.set_visible(False)
            if row == 0:
                ax_g.set_title(f"Rank {ci+1}", fontsize=10, fontweight="bold",
                              color=PAL["ink"], pad=8)
            tag = "ID " + ri["label"]
            tc = PAL["green"] if match else PAL["red"]
            ax_g.text(0.5, -0.06, tag, transform=ax_g.transAxes, fontsize=7,
                     fontweight="bold", color=tc, ha="center", va="top")

    # ---- Bottom section: (b) correction + (c) bar chart ----
    gs_bot = gs_main[1].subgridspec(1, 2, wspace=0.14, width_ratios=[1.0, 1.4])

    # (b) Correction comparison (bottom left)
    gs_corr = gs_bot[0].subgridspec(2, 2, wspace=0.08, hspace=0.25)
    corr_items = [
        ("query_raw.png", "Input (raw)", "raw"),
        ("query_retinexnet.png", "RetinexNet", "retinexnet"),
        ("query_zerodcepp.png", "Zero-DCE++", "zerodcepp"),
        ("query_riic.png", "RIIC-ReID", "riic"),
    ]
    for ci, (fname, label, mk) in enumerate(corr_items):
        r, c = ci // 2, ci % 2
        ax = fig.add_subplot(gs_corr[r, c])
        ax.imshow(load_img(CASE_DIR / fname))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(label, fontsize=9, fontweight="bold" if mk == "riic" else "normal",
                     color=METHOD_COLORS[mk], labelpad=4)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(1.2); sp.set_color(METHOD_COLORS[mk])
        if ci == 0:
            ax.set_title("(b) Illumination correction", fontsize=11,
                        fontweight="bold", color=PAL["ink"], pad=10, loc="left")

    # (c) t-SNE is drawn directly on gs_bot[1], no subgrid needed

    # (c) Cosine similarity analysis: same-ID vs confuser
    ax_sim = fig.add_subplot(gs_bot[1])

    x_pos = np.arange(len(methods_order))
    bar_width = 0.32

    same_vals = [sim_data[m]["same_avg"] for m in methods_order]
    conf_vals = [sim_data[m]["confuser_avg"] for m in methods_order]
    margins = [sim_data[m]["margin"] for m in methods_order]

    BLUE_BAR  = "#3A74C8"
    RED_BAR   = "#C85A50"
    GREY_LINE = "#9AAABB"

    bars_same = ax_sim.bar(x_pos - bar_width / 2, same_vals, bar_width,
                           color=BLUE_BAR, alpha=0.88, label=f"Same ID ({QUERY_LABEL})",
                           edgecolor="white", linewidth=1.0)
    bars_conf = ax_sim.bar(x_pos + bar_width / 2, conf_vals, bar_width,
                           color=RED_BAR, alpha=0.82, label="Confuser (ID 232)",
                           edgecolor="white", linewidth=1.0)

    # Zoom y-axis
    y_min = min(min(same_vals), min(conf_vals)) - 0.018
    y_max = max(max(same_vals), max(conf_vals)) + 0.038
    ax_sim.set_ylim(y_min, y_max)

    # Value labels on top of bars
    for bi in range(len(methods_order)):
        ax_sim.text(x_pos[bi] - bar_width / 2, same_vals[bi] + 0.002,
                   f"{same_vals[bi]:.3f}", ha="center", va="bottom",
                   fontsize=8, fontweight="bold", color=BLUE_BAR)
        ax_sim.text(x_pos[bi] + bar_width / 2, conf_vals[bi] + 0.002,
                   f"{conf_vals[bi]:.3f}", ha="center", va="bottom",
                   fontsize=8, fontweight="bold", color=RED_BAR)

    # Margin: clean bracket + plain text, no colored box
    for bi, m in enumerate(methods_order):
        margin_val = margins[bi]
        top_y = max(same_vals[bi], conf_vals[bi]) + 0.012
        sign = "+" if margin_val >= 0 else ""
        delta_color = "#1B7E3C" if margin_val > 0 else "#B03030"
        bx_l = x_pos[bi] - bar_width / 2 - 0.04
        bx_r = x_pos[bi] + bar_width / 2 + 0.04
        brace_y = top_y + 0.003
        ax_sim.plot([bx_l, bx_l, bx_r, bx_r],
                   [brace_y - 0.003, brace_y, brace_y, brace_y - 0.003],
                   color=GREY_LINE, lw=0.9, solid_capstyle="round")
        ax_sim.text(x_pos[bi], brace_y + 0.003,
                   f"\u0394 {sign}{margin_val:.3f}",
                   ha="center", va="bottom", fontsize=8.5,
                   fontweight="bold", color=delta_color)

    bar_labels = [METHOD_LABELS[m] for m in methods_order]
    ax_sim.set_xticks(x_pos)
    ax_sim.set_xticklabels(bar_labels, fontsize=9)
    ax_sim.set_ylabel("Cosine similarity to query", fontsize=10)
    ax_sim.set_title("(c) Embedding proximity: same ID vs. confuser",
                    fontsize=11, fontweight="bold", color=PAL["ink"], pad=10, loc="left")
    ax_sim.spines["top"].set_visible(False)
    ax_sim.spines["right"].set_visible(False)
    ax_sim.spines["left"].set_color(GREY_LINE)
    ax_sim.spines["bottom"].set_color(GREY_LINE)
    ax_sim.tick_params(colors=PAL["slate"])
    ax_sim.legend(fontsize=9, frameon=True, fancybox=False, framealpha=0.95,
                 edgecolor="#CCCCCC", loc="upper left")

    # Highlight: RIIC has largest margin
    best_m = max(range(len(margins)), key=lambda i: margins[i])
    ax_sim.get_xticklabels()[best_m].set_fontweight("bold")
    ax_sim.get_xticklabels()[best_m].set_color(METHOD_COLORS[methods_order[best_m]])

    # ---- Section title + legend in a clean title bar above the grid ----
    fig.text(0.115, 0.924,
             "(a) Retrieval results under different illumination corrections",
             fontsize=12, fontweight="bold", color=PAL["ink"], va="bottom")

    # Legend placed to the RIGHT of the grid (x > 0.945), vertical, small
    ret_handles = [
        mpatches.Patch(facecolor=PAL["green"], edgecolor="none", label="Correct match"),
        mpatches.Patch(facecolor=PAL["red"],   edgecolor="none", label="Wrong identity"),
    ]
    fig.legend(handles=ret_handles,
               fontsize=8, frameon=True, fancybox=False, framealpha=0.95,
               edgecolor="#CCCCCC", ncol=1,
               bbox_to_anchor=(0.948, 0.870),   # top-right corner just outside grid
               loc="upper left")

    # No separate t-SNE legend needed

    # ---- Save ----
    for ext in ("png", "pdf"):
        out = OUTPUT_DIR / f"fig1_comprehensive_tsne.{ext}"
        fig.savefig(str(out), dpi=350, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    draw_figure()
