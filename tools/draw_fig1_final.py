#!/usr/bin/env python3
"""Draw Figure 1 & Figure 2 for RIIC-ReID (ACM MM 2026).

Fig 1: Ranked retrieval comparison grid  (full-width)
Fig 2: Embedding geometry analysis       (full-width, scatter + distance bar)

Usage:  conda activate dog_train && python tools/draw_fig1_final.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse, FancyArrowPatch
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.core.config import load_config
from app.core.joint_model import JointReIDModel

# ── paths ──────────────────────────────────────────────────────
CFG   = "config/illumination_config_atrw.yaml"
CKPT  = "checkpoints/atrw_routeb_theoryB/joint_best.pth"
QDIR  = "data/processed/atrw/query"
GDIR  = "data/processed/atrw/gallery"
RDIR  = ("downloads/westc_perceptual_assets_20260325/root/autodl-tmp/v2_2/"
         "dog_reid_web/data/perceptual_baselines/atrw/retinexnet/test")
ZDIR  = ("downloads/westc_perceptual_assets_20260325/root/autodl-tmp/v2_2/"
         "dog_reid_web/data/perceptual_baselines/atrw/zerodcepp/test")
META  = "docs/figures/riic_reid_main_paper_20260327/assets/claim_case/metrics.json"
RIIC_Q = "docs/figures/riic_reid_main_paper_20260327/assets/claim_case/query_riic.png"
ODIR  = "docs/figures/riic_reid_main_paper_20260327/final"
EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}

# ── global style ───────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "figure.dpi": 350,
})

# Academic muted palette
C_RAW  = "#555555"
C_RET  = "#C44E52"
C_ZDC  = "#4C72B0"
C_RIIC = "#217844"
C_POS  = "#217844"
C_NEG  = "#C44E52"

# ── model helpers ──────────────────────────────────────────────
def _nc(sd):
    for k,v in sd.items():
        if isinstance(v,torch.Tensor) and v.ndim==2 and k.endswith(
            ("classifier.weight","global_classifier.weight","arcface.weight")):
            return int(v.shape[0])
    return 107

def _load_model(dev):
    cfg=load_config(CFG); ck=torch.load(CKPT,map_location="cpu",weights_only=False)
    st=ck.get("model_state_dict",ck)
    mc=cfg.get("model",{}); ic=cfg.get("illumination_module",{})
    lc=mc.get("local_extractor",{})
    ip=dict(mc.get("illumination_module",{}).get("module_params") or ic.get("module_params",{}))
    for k in ("feature_fusion","branch_attention_fusion","nuisance_head",
              "reid_head","backbone_random_erasing"):
        sc=mc.get(k,{})
        if sc: ip[f"_{k}"]=sc
    m=JointReIDModel(num_classes=_nc(st),backbone_name=mc.get("backbone","osnet_ain_x1_0"),
        num_stripes=int(lc.get("num_parts",6)),pretrained_backbone=False,
        use_ipaid=True,dropout=float(lc.get("dropout",0.0)),ipaid_params=ip)
    ms=m.state_dict()
    m.load_state_dict({k:v for k,v in st.items()
        if k in ms and isinstance(v,torch.Tensor) and v.shape==ms[k].shape},strict=False)
    return m.to(dev).eval()

def _tfm():
    c=load_config(CFG).get("training",{})
    return transforms.Compose([transforms.Resize(
        (int(c.get("image_height",256)),int(c.get("image_width",384)))),
        transforms.ToTensor()])

def _rgb(p,t): return t(Image.open(p).convert("RGB"))
def _split(d):
    r=Path(d); o=[]
    for ld in sorted(r.iterdir()):
        if not ld.is_dir(): continue
        for ip in sorted(ld.iterdir()):
            if ip.suffix.lower() in EXTS: o.append((ld.name,str(ip)))
    return o
def _enh(stem,root):
    p=Path(root)/f"{stem}.jpg"
    if p.exists(): return str(p)
    ms=sorted(Path(root).glob(f"{stem}.*"))
    return str(ms[0]) if ms else ""

@torch.no_grad()
def _feat(model,tensor,mode,dev):
    x=tensor.unsqueeze(0).to(dev)
    o=model(x,return_illuminated=False) if mode=="riic" else model.forward_raw_reference(x,detach=True)
    return F.normalize(o["features"],p=2,dim=1).cpu().squeeze(0).numpy()


def bordered_img(path, color, bw=5, size=(160,106)):
    """Load image → resize → draw coloured border → numpy array."""
    img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for i in range(bw):
        draw.rectangle([i, i, w-1-i, h-1-i], outline=color)
    return np.array(img)


# ================================================================
#  FIGURE 1 — Ranked retrieval comparison
# ================================================================
def draw_fig1(claim, q_imgs, row_labels, row_colors, aps, ranks, gpaths, out_dir):
    """4-row retrieval grid: query | rank1-5 | AP."""
    N_ROWS = 4
    N_RANK = 5
    THUMB  = (150, 100)
    BW     = 5

    fig = plt.figure(figsize=(7.16, 3.8))

    # gridspec: 4 rows × 7 cols (label | query | r1-r5 | AP)
    gs = gridspec.GridSpec(N_ROWS, N_RANK + 2,
        width_ratios=[0.22, 0.18, 0.155, 0.155, 0.155, 0.155, 0.12],
        wspace=0.04, hspace=0.12,
        left=0.0, right=1.0, top=0.92, bottom=0.06)

    # Column headers
    headers = ["Query", "Rank 1", "Rank 2", "Rank 3", "Rank 4", "Rank 5"]
    for ci, hdr in enumerate(headers):
        bbox = gs[0, ci + 1].get_position(fig)
        fig.text((bbox.x0 + bbox.x1) / 2, 0.95, hdr,
                 fontsize=7.5, fontweight="bold", color="#444444",
                 va="bottom", ha="center")

    for ri in range(N_ROWS):
        # ── method label (col 0) ──
        ax_lbl = fig.add_subplot(gs[ri, 0])
        ax_lbl.axis("off")
        ax_lbl.text(0.95, 0.5, row_labels[ri],
                    transform=ax_lbl.transAxes, fontsize=8,
                    fontweight="bold", color=row_colors[ri],
                    va="center", ha="right", linespacing=1.35)

        # ── query image (col 1) ──
        ax_q = fig.add_subplot(gs[ri, 1])
        ax_q.imshow(bordered_img(q_imgs[ri], row_colors[ri], BW, THUMB))
        ax_q.axis("off")

        # ── gallery rank 1-5 (cols 2-6) ──
        for ci in range(N_RANK):
            ax_g = fig.add_subplot(gs[ri, ci + 2])
            bc = C_POS if ranks[ri][ci] else C_NEG
            gp = str(ROOT / gpaths[ri][ci])
            ax_g.imshow(bordered_img(gp, bc, BW, THUMB))
            ax_g.axis("off")

    # ── AP values (col 7, overlaid as text) ──
    for ri in range(N_ROWS):
        bbox = gs[ri, N_RANK + 1].get_position(fig)
        cy = (bbox.y0 + bbox.y1) / 2
        cx = (bbox.x0 + bbox.x1) / 2
        fig.text(cx, cy, f"AP\n{aps[ri]:.2f}",
                 fontsize=9.5, fontweight="bold", color=row_colors[ri],
                 va="center", ha="center", linespacing=1.3,
                 fontfamily="monospace")

    # ── legend at bottom ──
    import matplotlib.patches as mpatches
    p1 = mpatches.Patch(facecolor=C_POS, edgecolor="white", label="Correct match")
    p2 = mpatches.Patch(facecolor=C_NEG, edgecolor="white", label="Wrong identity")
    fig.legend(handles=[p1, p2], loc="lower center", ncol=2,
               fontsize=7.5, frameon=False, handletextpad=0.4,
               bbox_to_anchor=(0.5, 0.0))

    for fmt in ("png", "pdf"):
        p = out_dir / f"fig1_retrieval.{fmt}"
        fig.savefig(p, dpi=350, bbox_inches="tight", facecolor="white")
        print(f"Saved: {p}")
    plt.close(fig)


# ================================================================
#  FIGURE 2 — Embedding geometry
# ================================================================
def draw_fig2(dists, nn_dist, t_rad, pp, np_, qp, nP, out_dir):
    """Left: PCA scatter.  Right: distance-to-centre bar chart."""

    fig = plt.figure(figsize=(7.16, 3.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.25, 1.0], wspace=0.32,
                           left=0.06, right=0.97, top=0.90, bottom=0.12)

    # ── (a) PCA scatter ───────────────────────────────────────
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_title("(a) Feature embedding (PCA projection)",
                   fontsize=10, fontweight="bold", pad=8, loc="left")

    # Positive cluster
    pm = pp.mean(0); ps = pp.std(0)
    ell = Ellipse(xy=pm, width=4.2*ps[0], height=4.2*ps[1],
                  facecolor=C_POS, alpha=0.06, edgecolor=C_POS,
                  linestyle="--", linewidth=0.9)
    ax_a.add_patch(ell)
    ax_a.scatter(pp[:,0], pp[:,1], c=C_POS, s=30, marker="o", alpha=0.50,
                 zorder=3, edgecolors="white", linewidths=0.3,
                 label=f"Same-identity gallery ($n$={nP})")
    ax_a.scatter(np_[:,0], np_[:,1], c=C_NEG, s=35, marker="x", alpha=0.55,
                 zorder=3, linewidths=1.1,
                 label="Hard negatives")

    # Query points
    qspec = [
        (0, "^", C_RAW, 65,  "Query: Raw (input)"),
        (1, "s", C_RET, 55,  "Query: RetinexNet"),
        (2, "D", C_ZDC, 50,  "Query: Zero-DCE++"),
        (3, "*", C_RIIC,160, "Query: RIIC-ReID (ours)"),
    ]
    for i, mk, cc, sz, nm in qspec:
        ax_a.scatter(qp[i,0], qp[i,1], c=cc, s=sz, marker=mk,
                     zorder=5, edgecolors="black", linewidths=0.6, label=nm)

    # Arrows
    arrow_kw = dict(lw=1.2, alpha=0.50)
    ax_a.annotate("", xy=qp[1], xytext=qp[0],
        arrowprops={**arrow_kw, "arrowstyle":"-|>", "color":C_RET,
                    "linestyle":"--", "connectionstyle":"arc3,rad=0.22"})
    ax_a.annotate("", xy=qp[2], xytext=qp[0],
        arrowprops={**arrow_kw, "arrowstyle":"-|>", "color":C_ZDC,
                    "linestyle":"--", "connectionstyle":"arc3,rad=0.12"})
    ax_a.annotate("", xy=qp[3], xytext=qp[0],
        arrowprops={**arrow_kw, "arrowstyle":"-|>", "color":C_RIIC,
                    "linestyle":"-", "connectionstyle":"arc3,rad=-0.18"})

    # "pushed away" / "stays close" annotations
    mid_ret = qp[0] + 0.55 * (qp[1] - qp[0])
    ax_a.text(mid_ret[0] - 0.03, mid_ret[1] + 0.05,
              "pushed away",
              fontsize=7, color=C_RET, fontstyle="italic",
              ha="center", va="bottom", alpha=0.85)
    mid_riic = qp[0] + 0.5 * (qp[3] - qp[0])
    ax_a.text(mid_riic[0] + 0.08, mid_riic[1] - 0.015,
              "stays close",
              fontsize=7, color=C_RIIC, fontstyle="italic",
              ha="left", va="top", alpha=0.85)

    ax_a.legend(loc="upper left", framealpha=0.88, borderpad=0.4,
                labelspacing=0.35, handletextpad=0.4, edgecolor="#cccccc",
                fontsize=7)
    ax_a.grid(alpha=0.10, linestyle="-", linewidth=0.4)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.set_xlabel("PC 1")
    ax_a.set_ylabel("PC 2")

    # ── (b) Distance bar chart ─────────────────────────────────
    ax_b = fig.add_subplot(gs[1])
    ax_b.set_title("(b) Distance to same-identity centre",
                   fontsize=10, fontweight="bold", pad=8, loc="left")

    bar_keys   = ["riic", "raw", "zdc", "ret"]
    bar_labels = ["RIIC-ReID\n(ours)", "Raw\n(input)", "Zero-DCE++", "RetinexNet"]
    bar_colors = [C_RIIC, C_RAW, C_ZDC, C_RET]
    bar_vals   = [dists[k] for k in bar_keys]

    y_pos = np.arange(len(bar_keys))
    bars = ax_b.barh(y_pos, bar_vals, height=0.55, color=bar_colors,
                     edgecolor="white", linewidth=0.8, zorder=3)

    # Value labels on bars
    for i, (bar, val) in enumerate(zip(bars, bar_vals)):
        ax_b.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                  f"{val:.3f}", fontsize=8, fontweight="bold",
                  va="center", ha="left", color=bar_colors[i])

    # Nearest negative line
    ax_b.axvline(nn_dist, color=C_NEG, linewidth=1.5, linestyle="-",
                 alpha=0.7, zorder=4)
    ax_b.text(nn_dist + 0.005, len(bar_keys) - 0.15,
              f"Nearest\nnegative\n({nn_dist:.3f})",
              fontsize=6.5, color=C_NEG, va="top", ha="left",
              fontweight="bold", linespacing=1.2)

    # Teacher radius line
    ax_b.axvline(t_rad, color=C_POS, linewidth=1.2, linestyle="--",
                 alpha=0.5, zorder=4)
    ax_b.text(t_rad - 0.005, -0.45,
              f"Teacher\nradius\n({t_rad:.3f})",
              fontsize=6.5, color=C_POS, va="top", ha="right",
              fontstyle="italic", linespacing=1.2)

    # Margin annotation between RIIC and nearest negative
    riic_d = dists["riic"]
    margin_riic = nn_dist - riic_d
    ret_d = dists["ret"]
    margin_ret = nn_dist - ret_d
    # Small brace / arrow for RIIC margin
    mid_y_riic = y_pos[0]
    ax_b.annotate("", xy=(nn_dist, mid_y_riic - 0.35),
                  xytext=(riic_d, mid_y_riic - 0.35),
                  arrowprops=dict(arrowstyle="<->", color=C_RIIC,
                                  lw=1.0, alpha=0.7))
    ax_b.text((riic_d + nn_dist)/2, mid_y_riic - 0.48,
              f"margin={margin_riic:.3f}",
              fontsize=6, ha="center", va="top", color=C_RIIC,
              fontweight="bold")

    # Margin for RetinexNet
    mid_y_ret = y_pos[3]
    ax_b.annotate("", xy=(nn_dist, mid_y_ret + 0.35),
                  xytext=(ret_d, mid_y_ret + 0.35),
                  arrowprops=dict(arrowstyle="<->", color=C_RET,
                                  lw=1.0, alpha=0.7))
    ax_b.text((ret_d + nn_dist)/2, mid_y_ret + 0.48,
              f"margin={margin_ret:.3f}",
              fontsize=6, ha="center", va="bottom", color=C_RET,
              fontweight="bold")

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(bar_labels, fontsize=8, fontweight="bold")
    for i, tick in enumerate(ax_b.get_yticklabels()):
        tick.set_color(bar_colors[i])
    ax_b.set_xlabel("$L_2$ distance to positive gallery centroid", fontsize=9)
    ax_b.set_xlim(0.58, nn_dist + 0.06)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(axis="x", alpha=0.12, linestyle="-", linewidth=0.4)
    ax_b.invert_yaxis()

    for fmt in ("png", "pdf"):
        p = out_dir / f"fig2_geometry.{fmt}"
        fig.savefig(p, dpi=350, bbox_inches="tight", facecolor="white")
        print(f"Saved: {p}")
    plt.close(fig)


# ================================================================
#  MAIN
# ================================================================
def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    claim = json.loads(Path(META).read_text("utf-8"))
    qlabel = claim["query_label"]; qstem = Path(claim["query_relpath"]).stem
    tfm = _tfm(); model = _load_model(dev)

    qpath = next(p for l,p in _split(QDIR) if l==qlabel and Path(p).stem==qstem)
    rpath = _enh(qstem, RDIR); zpath = _enh(qstem, ZDIR)

    gallery = _split(GDIR)
    pos_g = [(l,p) for l,p in gallery if l==qlabel]
    neg_g = [(l,p) for l,p in gallery if l!=qlabel]

    # ── embeddings ─────────────────────────────────────────────
    print("Extracting gallery features...")
    pos_f = np.stack([_feat(model,_rgb(p,tfm),"raw",dev) for _,p in pos_g])
    pos_c = pos_f.mean(0,keepdims=True)
    t_rad = float(np.linalg.norm(pos_f-pos_c,axis=1).max())
    neg_f = np.stack([_feat(model,_rgb(p,tfm),"raw",dev) for _,p in neg_g])
    neg_d = np.linalg.norm(neg_f-pos_c,axis=1)
    nn_dist = float(neg_d.min())

    qt = _rgb(qpath,tfm)
    feats = {"raw": _feat(model,qt,"raw",dev),
             "riic":_feat(model,qt,"riic",dev),
             "ret": _feat(model,_rgb(rpath,tfm),"raw",dev),
             "zdc": _feat(model,_rgb(zpath,tfm),"raw",dev)}
    dists = {k:float(np.linalg.norm(v-pos_c.squeeze())) for k,v in feats.items()}
    for k in dists:
        print(f"  {k}: dist={dists[k]:.4f}  margin={nn_dist-dists[k]:.4f}")

    # PCA
    hi = np.argsort(neg_d)[:6]; hn_f = neg_f[hi]
    af = np.concatenate([pos_f, hn_f,
                         feats["raw"][None], feats["ret"][None],
                         feats["zdc"][None], feats["riic"][None]])
    xc = af - af.mean(0,keepdims=True)
    _,_,vt = np.linalg.svd(xc, full_matrices=False)
    pr = xc @ vt[:2].T
    nP=len(pos_f); nN=len(hn_f)
    pp=pr[:nP]; np_=pr[nP:nP+nN]; qp=pr[nP+nN:]

    # ── claim metadata ─────────────────────────────────────────
    mk_order = ["raw","retinexnet","zerodcepp","riic"]
    mm = claim["methods"]
    aps   = [mm[k]["ap"] for k in mk_order]
    ranks = [[r["match"] for r in mm[k]["ranked"]] for k in mk_order]
    gpaths= [[r["saved_path"] for r in mm[k]["ranked"]] for k in mk_order]
    q_imgs = [qpath, rpath, zpath, RIIC_Q]
    row_labels = ["Raw\n(input)", "RetinexNet", "Zero-DCE++", "RIIC-ReID\n(ours)"]
    row_colors = [C_RAW, C_RET, C_ZDC, C_RIIC]

    out = Path(ODIR); out.mkdir(parents=True, exist_ok=True)

    # ── draw ───────────────────────────────────────────────────
    print("\n=== Drawing Figure 1 (retrieval) ===")
    draw_fig1(claim, q_imgs, row_labels, row_colors, aps, ranks, gpaths, out)

    print("\n=== Drawing Figure 2 (geometry) ===")
    draw_fig2(dists, nn_dist, t_rad, pp, np_, qp, nP, out)

    # ── save metrics ───────────────────────────────────────────
    metrics = {"query_label":qlabel, "distances":dists,
               "margins":{k:nn_dist-v for k,v in dists.items()},
               "teacher_radius":t_rad, "nearest_neg":nn_dist,
               "ap":dict(zip(mk_order,aps))}
    mp = out / "fig1_fig2_metrics.json"
    mp.write_text(json.dumps(metrics, indent=2), "utf-8")
    print(f"Saved: {mp}\nDone!")

if __name__ == "__main__":
    main()
