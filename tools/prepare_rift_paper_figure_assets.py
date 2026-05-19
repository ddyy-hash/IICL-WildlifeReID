#!/usr/bin/env python3
"""Prepare code-grounded figure assets for the RIFT paper."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Circle, Ellipse
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import load_config
from app.core.joint_model import JointReIDModel


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Sample:
    split: str
    label: str
    path: str
    relpath: str


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rift_ckpt", default="checkpoints/atrw_routeb_theoryB/joint_best.pth")
    p.add_argument("--config", default="config/illumination_config_atrw.yaml")
    p.add_argument("--query_dir", default="data/processed/atrw/query")
    p.add_argument("--gallery_dir", default="data/processed/atrw/gallery")
    p.add_argument("--output_dir", default="docs/figures/rift_paper_20260325")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--max_query_scan", type=int, default=60)
    p.add_argument("--num_trust_cases", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_query", type=str, default=None,
                   help="Force a specific query by relpath, e.g. 'query/112/003457.jpg'. "
                        "Skips automatic case selection.")
    return p.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data):
    ensure_dir(path.parent)

    def _default(v):
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, Path):
            return str(v)
        raise TypeError(type(v))

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_default)


def load_split(root_dir: str, split: str):
    root = Path(root_dir)
    samples = []
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        for img_path in sorted(label_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            relpath = str((Path(split) / label_dir.name / img_path.name).as_posix())
            samples.append(Sample(split, label_dir.name, str(img_path), relpath))
    return samples


def build_transform(config_path: str):
    cfg = load_config(config_path)
    train_cfg = cfg.get("training", {})
    h = int(train_cfg.get("image_height", 256))
    w = int(train_cfg.get("image_width", 384))
    return transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor()])


def load_rgb(sample: Sample, tfm):
    return tfm(Image.open(sample.path).convert("RGB"))


def gamma_correct(x: torch.Tensor, gamma: float):
    return x.clamp(1e-3, 1.0).pow(gamma).clamp(0.0, 1.0)


def to_u8(x: torch.Tensor):
    arr = x.detach().cpu().clamp(0.0, 1.0).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def save_tensor(path: Path, x: torch.Tensor):
    ensure_dir(path.parent)
    Image.fromarray(to_u8(x)).save(path)


def save_heatmap(path: Path, x: torch.Tensor | None, title: str):
    if x is None:
        return
    arr = x.detach().cpu().float().squeeze().numpy()
    if arr.ndim != 2:
        return
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    hm = cv2.applyColorMap(arr, cv2.COLORMAP_TURBO)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(hm)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def luminance(x: torch.Tensor):
    return float((0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).mean().item())


def infer_num_classes(state):
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and k.endswith(("classifier.weight", "global_classifier.weight", "arcface.weight")):
            return int(v.shape[0])
    return 107


def load_model(ckpt_path: str, config_path: str, device: torch.device):
    cfg = load_config(config_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model_cfg = cfg.get("model", {})
    illum_cfg = cfg.get("illumination_module", {})
    local_cfg = model_cfg.get("local_extractor", {})
    ipaid_params = dict(model_cfg.get("illumination_module", {}).get("module_params") or illum_cfg.get("module_params", {}))
    for key in ("feature_fusion", "branch_attention_fusion", "nuisance_head", "reid_head", "backbone_random_erasing"):
        sub = model_cfg.get(key, {})
        if sub:
            ipaid_params[f"_{key}"] = sub
    model = JointReIDModel(
        num_classes=infer_num_classes(state),
        backbone_name=model_cfg.get("backbone", "osnet_ain_x1_0"),
        num_stripes=int(local_cfg.get("num_parts", 6)),
        pretrained_backbone=False,
        use_ipaid=True,
        dropout=float(local_cfg.get("dropout", 0.0)),
        ipaid_params=ipaid_params,
    )
    model_state = model.state_dict()
    keep = {k: v for k, v in state.items() if k in model_state and isinstance(v, torch.Tensor) and v.shape == model_state[k].shape}
    model.load_state_dict(keep, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def forward_rift(model: JointReIDModel, batch: torch.Tensor):
    out = model(batch, return_illuminated=False)
    return F.normalize(out["features"], p=2, dim=1)


@torch.no_grad()
def forward_raw(model: JointReIDModel, batch: torch.Tensor):
    out = model.forward_raw_reference(batch, detach=True)
    return F.normalize(out["features"], p=2, dim=1)


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def extract_features(samples, tfm, device, model, batch_size, mode: str, gamma: float):
    feats = []
    for chunk in batched(samples, batch_size):
        xs = [load_rgb(s, tfm) for s in chunk]
        if mode == "gamma":
            xs = [gamma_correct(x, gamma) for x in xs]
        batch = torch.stack(xs, 0).to(device)
        if mode == "rift":
            feat = forward_rift(model, batch)
        else:
            feat = forward_raw(model, batch)
        feats.append(feat.cpu())
    return torch.cat(feats, 0).numpy()


def compute_ap(dists, gallery_labels, query_label):
    order = np.argsort(dists)
    matches = (gallery_labels[order] == query_label).astype(np.float32)
    if matches.sum() <= 0:
        return 0.0
    prec = np.cumsum(matches) / (np.arange(matches.size, dtype=np.float32) + 1.0)
    return float((prec * matches).sum() / matches.sum())


def pca2(x):
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    y = x @ vt[:2].T
    if y.shape[1] < 2:
        y = np.pad(y, ((0, 0), (0, 2 - y.shape[1])))
    return y


def choose_case(query_samples, query_lums, gallery_labels, gallery_rift, gallery_gamma, tfm, device, model, gamma, max_scan):
    order = np.argsort(query_lums)[:max_scan]
    records = []
    best = None
    for rank, qi in enumerate(order):
        q = query_samples[int(qi)]
        q_rift = extract_features([q], tfm, device, model, 1, "rift", gamma)[0]
        q_gamma = extract_features([q], tfm, device, model, 1, "gamma", gamma)[0]
        d_rift = 1.0 - gallery_rift @ q_rift
        d_gamma = 1.0 - gallery_gamma @ q_gamma
        ap_rift = compute_ap(d_rift, gallery_labels, q.label)
        ap_gamma = compute_ap(d_gamma, gallery_labels, q.label)
        rec = {
            "query_index": int(qi),
            "query_relpath": q.relpath,
            "query_label": q.label,
            "luminance": float(query_lums[qi]),
            "ap_gamma": ap_gamma,
            "ap_rift": ap_rift,
            "delta_ap": ap_rift - ap_gamma,
            "top1_gamma_match": bool(gallery_labels[np.argmin(d_gamma)] == q.label),
            "top1_rift_match": bool(gallery_labels[np.argmin(d_rift)] == q.label),
            "scan_rank": rank,
        }
        records.append(rec)
        if best is None or rec["delta_ap"] > best["delta_ap"]:
            best = rec
    records.sort(key=lambda x: x["delta_ap"], reverse=True)
    best_out = dict(best)
    best_out["candidates"] = [dict(x) for x in records[:12]]
    return best_out


def first_idx(order, gallery_labels, query_label, positive):
    for idx in order:
        if bool(gallery_labels[idx] == query_label) == positive:
            return int(idx)
    return int(order[0])


def filter_display_order(order, gallery_samples, query_sample):
    filtered = []
    query_stem = Path(query_sample.path).stem
    for idx in order:
        sample = gallery_samples[int(idx)]
        same_file = sample.label == query_sample.label and Path(sample.path).stem == query_stem
        if same_file:
            continue
        filtered.append(int(idx))
    return np.asarray(filtered, dtype=np.int64)


def draw_retrieval_board(path: Path, query_raw, query_gamma, query_rift, gallery_samples, order_gamma, order_rift, gallery_labels, query_label, tfm, topk, ap_gamma, ap_rift):
    fig = plt.figure(figsize=(11.5, 5.4))
    gs = fig.add_gridspec(2, 7, hspace=0.28, wspace=0.10)
    rows = [
        (query_gamma, "Generic perceptual correction", order_gamma, ap_gamma, "#2563EB"),
        (query_rift, "RIFT correction", order_rift, ap_rift, "#D97706"),
    ]
    for ri, (corr, title, order, apv, color) in enumerate(rows):
        ax = fig.add_subplot(gs[ri, 0]); ax.imshow(to_u8(query_raw)); ax.set_title("Query", fontsize=9); ax.axis("off")
        ax = fig.add_subplot(gs[ri, 1]); ax.imshow(to_u8(corr)); ax.set_title(title, fontsize=9, color=color); ax.axis("off")
        for k in range(topk):
            gi = int(order[k])
            gx = load_rgb(gallery_samples[gi], tfm)
            match = bool(gallery_labels[gi] == query_label)
            border = "#16A34A" if match else "#DC2626"
            ax = fig.add_subplot(gs[ri, 2 + k])
            ax.imshow(to_u8(gx))
            for sp in ax.spines.values():
                sp.set_edgecolor(border); sp.set_linewidth(3.0)
            ax.set_title(f"#{k + 1} {'match' if match else 'distractor'}", fontsize=7, color=border)
            ax.axis("off")
        fig.text(0.95, 0.73 if ri == 0 else 0.28, f"AP={apv:.2f}", ha="right", va="center", fontsize=10, fontweight="bold")
    fig.text(0.975, 0.50, "same tiger,\ndifferent correction,\ndifferent retrieval", rotation=270, ha="center", va="center", fontsize=8, color="#6B7280", style="italic")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_ranked(out_dir: Path, gallery_samples, order, gallery_labels, query_label, tfm, topk):
    ensure_dir(out_dir)
    rows = []
    for k in range(topk):
        gi = int(order[k])
        s = gallery_samples[gi]
        x = load_rgb(s, tfm)
        match = bool(gallery_labels[gi] == query_label)
        name = f"rank{k + 1:02d}_{'pos' if match else 'neg'}_{s.label}_{Path(s.path).stem}.png"
        save_tensor(out_dir / name, x)
        rows.append({"rank": k + 1, "gallery_relpath": s.relpath, "gallery_label": s.label, "match": match, "saved_path": str((out_dir / name).as_posix())})
    return rows


def save_branch(case_dir: Path, branch):
    if branch is None:
        return {}
    x = branch.detach().cpu().float().squeeze(0).numpy()
    if x.ndim != 2:
        return {}
    csv_path = case_dir / "branch_attention.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("stripe,raw,base,adapted\n")
        for i, row in enumerate(x):
            f.write(f"{i + 1},{row[0]:.6f},{row[1]:.6f},{row[2]:.6f}\n")
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    colors = ["#2563EB", "#16A34A", "#D97706"]
    labels = ["raw", "base", "adapted"]
    left = np.zeros(x.shape[0], dtype=np.float32)
    ys = np.arange(x.shape[0])
    for b in range(x.shape[1]):
        ax.barh(ys, x[:, b], left=left, color=colors[b], label=labels[b], height=0.74)
        left += x[:, b]
    ax.set_xlim(0, 1); ax.set_yticks(ys); ax.set_yticklabels([f"s{i + 1}" for i in ys], fontsize=8); ax.invert_yaxis(); ax.legend(fontsize=8, loc="lower right"); ax.set_title("Stripe-wise branch attention")
    fig.tight_layout()
    fig.savefig(case_dir / "branch_attention.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return {"csv": str(csv_path.as_posix()), "png": str((case_dir / "branch_attention.png").as_posix())}


def draw_trust_board(path: Path, title: str, raw_x, corr_x, maps):
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 6.3))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    axes[0, 0].imshow(to_u8(raw_x)); axes[0, 0].set_title("raw"); axes[0, 0].axis("off")
    axes[0, 1].imshow(to_u8(corr_x)); axes[0, 1].set_title("rift corrected"); axes[0, 1].axis("off")
    slots = [axes[0, 2], axes[0, 3], axes[1, 0], axes[1, 1], axes[1, 2], axes[1, 3]]
    for ax in axes.flat[2:]:
        ax.axis("off")
    for ax, (label, tensor) in zip(slots, maps):
        if tensor is None:
            continue
        arr = tensor.detach().cpu().float().squeeze().numpy()
        if arr.ndim != 2:
            continue
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        ax.imshow(arr, cmap="turbo")
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def ellipse_spec(points):
    if points.shape[0] < 2:
        return None
    center = points.mean(0)
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
    angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    w = 2.0 * math.sqrt(max(float(eigvals[0]), 1e-8))
    h = 2.0 * math.sqrt(max(float(eigvals[1]), 1e-8))
    return center, w, h, angle


def main():
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_root = Path(args.output_dir)
    fig1_dir = out_root / "fig1_claim"
    trust_dir = out_root / "fig3_trust"
    geom_dir = out_root / "fig3_geometry"
    comp_dir = out_root / "fig2_components"
    for d in (fig1_dir, trust_dir, geom_dir, comp_dir):
        ensure_dir(d)

    tfm = build_transform(args.config)
    query_samples = load_split(args.query_dir, "query")
    gallery_samples = load_split(args.gallery_dir, "gallery")
    gallery_labels = np.array([s.label for s in gallery_samples], dtype=object)
    model = load_model(args.rift_ckpt, args.config, device)

    query_lums = np.array([luminance(load_rgb(s, tfm)) for s in query_samples], dtype=np.float32)
    gallery_rift = extract_features(gallery_samples, tfm, device, model, args.batch_size, "rift", args.gamma)
    gallery_gamma = extract_features(gallery_samples, tfm, device, model, args.batch_size, "gamma", args.gamma)

    if args.force_query:
        # Find the forced query by relpath match
        matched = [i for i, s in enumerate(query_samples) if s.relpath == args.force_query]
        if not matched:
            # Try partial match (e.g. "112/003457" without "query/" prefix)
            matched = [i for i, s in enumerate(query_samples) if args.force_query in s.relpath]
        if not matched:
            raise ValueError(f"Could not find query matching '{args.force_query}' in {len(query_samples)} samples")
        qi = matched[0]
        q_feat_rift = extract_features([query_samples[qi]], tfm, device, model, 1, "rift", args.gamma)[0]
        q_feat_gamma = extract_features([query_samples[qi]], tfm, device, model, 1, "gamma", args.gamma)[0]
        d_rift_q = 1.0 - gallery_rift @ q_feat_rift
        d_gamma_q = 1.0 - gallery_gamma @ q_feat_gamma
        selected = {
            "query_index": qi,
            "query_relpath": query_samples[qi].relpath,
            "query_label": query_samples[qi].label,
            "luminance": float(query_lums[qi]),
            "ap_gamma": compute_ap(d_gamma_q, gallery_labels, query_samples[qi].label),
            "ap_rift": compute_ap(d_rift_q, gallery_labels, query_samples[qi].label),
            "delta_ap": compute_ap(d_rift_q, gallery_labels, query_samples[qi].label) - compute_ap(d_gamma_q, gallery_labels, query_samples[qi].label),
            "candidates": [],
        }
        print(f"[force_query] Using {query_samples[qi].relpath} | lum={query_lums[qi]:.3f} | ΔAP={selected['delta_ap']:.3f}")
    else:
        selected = choose_case(query_samples, query_lums, gallery_labels, gallery_rift, gallery_gamma, tfm, device, model, args.gamma, args.max_query_scan)
    save_json(fig1_dir / "candidate_summary.json", {"candidates": selected.get("candidates", [])})

    q = query_samples[selected["query_index"]]
    q_raw = load_rgb(q, tfm)
    q_gamma = gamma_correct(q_raw, args.gamma)
    with torch.no_grad():
        q_out = model(q_raw.unsqueeze(0).to(device), return_illuminated=True)
    q_rift = q_out.get("illuminated", q_raw.unsqueeze(0).to(device))[0].detach().cpu().clamp(0.0, 1.0)
    q_feat_rift = extract_features([q], tfm, device, model, 1, "rift", args.gamma)[0]
    q_feat_gamma = extract_features([q], tfm, device, model, 1, "gamma", args.gamma)[0]
    d_rift = 1.0 - gallery_rift @ q_feat_rift
    d_gamma = 1.0 - gallery_gamma @ q_feat_gamma
    order_rift = np.argsort(d_rift)
    order_gamma = np.argsort(d_gamma)
    display_order_rift = filter_display_order(order_rift, gallery_samples, q)
    display_order_gamma = filter_display_order(order_gamma, gallery_samples, q)

    case_dir = fig1_dir / "selected_case"
    ensure_dir(case_dir)
    save_tensor(case_dir / "query_raw.png", q_raw)
    save_tensor(case_dir / "query_perceptual.png", q_gamma)
    save_tensor(case_dir / "query_rift.png", q_rift)
    ranked_gamma = save_ranked(case_dir / "gallery_topk_perceptual", gallery_samples, display_order_gamma, gallery_labels, q.label, tfm, args.topk)
    ranked_rift = save_ranked(case_dir / "gallery_topk_rift", gallery_samples, display_order_rift, gallery_labels, q.label, tfm, args.topk)
    draw_retrieval_board(case_dir / "retrieval_comparison_board.png", q_raw, q_gamma, q_rift, gallery_samples, display_order_gamma, display_order_rift, gallery_labels, q.label, tfm, args.topk, compute_ap(d_gamma, gallery_labels, q.label), compute_ap(d_rift, gallery_labels, q.label))

    pos_idx = first_idx(display_order_rift, gallery_labels, q.label, True)
    neg_idx = first_idx(display_order_gamma, gallery_labels, q.label, False)
    pos_feat = extract_features([gallery_samples[pos_idx]], tfm, device, model, 1, "raw", args.gamma)[0]
    neg_feat = extract_features([gallery_samples[neg_idx]], tfm, device, model, 1, "raw", args.gamma)[0]
    emb = pca2(np.stack([extract_features([q], tfm, device, model, 1, "raw", args.gamma)[0], q_feat_gamma, q_feat_rift, pos_feat, neg_feat], 0))
    with (case_dir / "embedding_projection.csv").open("w", encoding="utf-8") as f:
        f.write("name,role,x,y\n")
        for name, role, pt in zip(["query_raw", "query_perceptual", "query_rift", "correct_match", "distractor"], ["query", "query", "query", "positive", "negative"], emb):
            f.write(f"{name},{role},{pt[0]:.6f},{pt[1]:.6f}\n")
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    colors = {"query_raw": "#6B7280", "query_perceptual": "#2563EB", "query_rift": "#D97706", "correct_match": "#16A34A", "distractor": "#DC2626"}
    for name, pt in zip(["query_raw", "query_perceptual", "query_rift", "correct_match", "distractor"], emb):
        ax.scatter(pt[0], pt[1], s=90, c=colors[name], edgecolors="black", linewidths=0.8)
        ax.text(pt[0] + 0.02, pt[1] + 0.02, name, fontsize=8)
    ax.set_title("Feature-space sketch points"); ax.set_xticks([]); ax.set_yticks([]); fig.tight_layout(); fig.savefig(case_dir / "embedding_projection.png", dpi=260, bbox_inches="tight"); plt.close(fig)
    fig1_metrics = {
        "query_relpath": q.relpath,
        "query_label": q.label,
        "query_luminance": float(query_lums[selected["query_index"]]),
        "ap_perceptual": compute_ap(d_gamma, gallery_labels, q.label),
        "ap_rift": compute_ap(d_rift, gallery_labels, q.label),
        "delta_ap": compute_ap(d_rift, gallery_labels, q.label) - compute_ap(d_gamma, gallery_labels, q.label),
        "top1_perceptual_match": bool(gallery_labels[order_gamma[0]] == q.label),
        "top1_rift_match": bool(gallery_labels[order_rift[0]] == q.label),
        "display_filter": "exclude duplicated query basename in gallery for visualization only",
        "ranked_perceptual": ranked_gamma,
        "ranked_rift": ranked_rift,
        "positive_example": gallery_samples[pos_idx].relpath,
        "distractor_example": gallery_samples[neg_idx].relpath,
    }
    save_json(case_dir / "metrics.json", fig1_metrics)

    trust_idxs = sorted(set(np.linspace(0, len(query_samples) - 1, num=max(args.num_trust_cases, 1), dtype=int).tolist()))
    lum_order = np.argsort(query_lums)
    trust_records = []
    for ci, idx_pos in enumerate(trust_idxs):
        qi = int(lum_order[idx_pos])
        s = query_samples[qi]
        case_dir = trust_dir / f"case_{ci + 1:02d}"
        ensure_dir(case_dir)
        raw_x = load_rgb(s, tfm)
        with torch.no_grad():
            out = model(raw_x.unsqueeze(0).to(device), return_illuminated=True)
        corr_x = out.get("illuminated", raw_x.unsqueeze(0).to(device))[0].detach().cpu().clamp(0.0, 1.0)
        ipaid = out.get("ipaid_details") or {}
        save_tensor(case_dir / "raw.png", raw_x); save_tensor(case_dir / "corrected.png", corr_x)
        if isinstance(ipaid.get("reflectance_base"), torch.Tensor):
            save_tensor(case_dir / "reflectance_base.png", ipaid["reflectance_base"][0].detach().cpu().clamp(0.0, 1.0))
        if isinstance(ipaid.get("reflectance_att"), torch.Tensor):
            save_tensor(case_dir / "reflectance_att.png", ipaid["reflectance_att"][0].detach().cpu().clamp(0.0, 1.0))
        save_heatmap(case_dir / "illumination_map.png", ipaid.get("illumination"), "illumination")
        save_heatmap(case_dir / "identity_protection_map.png", ipaid.get("identity_protection_map"), "identity protection")
        save_heatmap(case_dir / "rollback_alpha_map.png", ipaid.get("rollback_alpha_map"), "rollback alpha")
        save_heatmap(case_dir / "correction_gap.png", ipaid.get("correction_gap"), "correction gap")
        save_heatmap(case_dir / "color_risk.png", ipaid.get("color_risk"), "color risk")
        branch_info = save_branch(case_dir, ipaid.get("branch_attention_weights"))
        draw_trust_board(case_dir / "trust_board.png", f"{s.relpath} | lum={query_lums[qi]:.3f}", raw_x, corr_x, [("P_id", ipaid.get("identity_protection_map")), ("rollback", ipaid.get("rollback_alpha_map")), ("gap", ipaid.get("correction_gap")), ("illum", ipaid.get("illumination")), ("color risk", ipaid.get("color_risk")), ("base", ipaid.get("reflectance_base"))])
        rec = {"sample_relpath": s.relpath, "label": s.label, "luminance": float(query_lums[qi]), "rollback_mean": float(ipaid["rollback_alpha"].mean().item()) if isinstance(ipaid.get("rollback_alpha"), torch.Tensor) else None, "identity_protection_mean": float(ipaid["identity_protection_map"].mean().item()) if isinstance(ipaid.get("identity_protection_map"), torch.Tensor) else None, "correction_gap_mean": float(ipaid["correction_gap"].mean().item()) if isinstance(ipaid.get("correction_gap"), torch.Tensor) else None, "branch_attention": branch_info}
        save_json(case_dir / "case_metrics.json", rec)
        trust_records.append(rec)
    save_json(trust_dir / "summary.json", {"cases": trust_records})

    pos_gallery = [s for s in gallery_samples if s.label == q.label]
    neg_gallery = [s for s in gallery_samples if s.label != q.label]
    teacher_feats = extract_features(pos_gallery, tfm, device, model, args.batch_size, "raw", args.gamma)
    neg_feats = extract_features(neg_gallery, tfm, device, model, args.batch_size, "raw", args.gamma)
    teacher_center = teacher_feats.mean(axis=0, keepdims=True)
    neg_d = np.linalg.norm(neg_feats - teacher_center, axis=1)
    nn_idx = np.argsort(neg_d)[:6]
    hard_neg_feats = neg_feats[nn_idx]
    hard_neg_samples = [neg_gallery[int(i)] for i in nn_idx]
    proj = pca2(np.concatenate([teacher_feats, hard_neg_feats, extract_features([q], tfm, device, model, 1, "raw", args.gamma), q_feat_gamma[None, :], q_feat_rift[None, :]], axis=0))
    teacher_proj = proj[: teacher_feats.shape[0]]
    neg_proj = proj[teacher_feats.shape[0] : teacher_feats.shape[0] + hard_neg_feats.shape[0]]
    q_proj = proj[-3:]
    with (geom_dir / "teacher_student_projection.csv").open("w", encoding="utf-8") as f:
        f.write("name,role,label,x,y\n")
        for i, s in enumerate(pos_gallery):
            f.write(f"teacher_pos_{i + 1},teacher_positive,{s.label},{teacher_proj[i,0]:.6f},{teacher_proj[i,1]:.6f}\n")
        for i, s in enumerate(hard_neg_samples):
            f.write(f"hard_neg_{i + 1},hard_negative,{s.label},{neg_proj[i,0]:.6f},{neg_proj[i,1]:.6f}\n")
        for name, pt in zip(["query_raw", "query_perceptual", "query_rift"], q_proj):
            f.write(f"{name},query,{q.label},{pt[0]:.6f},{pt[1]:.6f}\n")
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.scatter(teacher_proj[:, 0], teacher_proj[:, 1], c="#16A34A", s=46, alpha=0.80, label="teacher positives")
    ax.scatter(neg_proj[:, 0], neg_proj[:, 1], c="#DC2626", s=50, alpha=0.80, label="hard negatives")
    for name, color, pt in zip(["query_raw", "query_perceptual", "query_rift"], ["#6B7280", "#2563EB", "#D97706"], q_proj):
        ax.scatter(pt[0], pt[1], c=color, s=110, edgecolors="black", linewidths=0.9, label=name); ax.text(pt[0] + 0.02, pt[1] + 0.02, name, fontsize=8)
    if teacher_proj.shape[0] >= 2:
        center = teacher_proj.mean(0)
        cov = np.cov(teacher_proj.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
        angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
        ell = Ellipse(xy=center, width=2.4 * 2.0 * math.sqrt(max(float(eigvals[0]), 1e-8)), height=2.4 * 2.0 * math.sqrt(max(float(eigvals[1]), 1e-8)), angle=angle, facecolor="#16A34A", alpha=0.12, edgecolor="#15803D", linewidth=2.0)
        ax.add_patch(ell)
        circ = Circle(center, radius=np.linalg.norm(teacher_proj - center, axis=1).max(), fill=False, color="#15803D", linestyle="--", linewidth=1.4, alpha=0.8)
        ax.add_patch(circ)
    ax.set_title("Teacher manifold projection"); ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=8, loc="best"); fig.tight_layout(); fig.savefig(geom_dir / "teacher_student_projection.png", dpi=260, bbox_inches="tight"); plt.close(fig)
    geom_stats = {
        "query_relpath": q.relpath,
        "query_label": q.label,
        "num_teacher_positives": len(pos_gallery),
        "teacher_radius_l2": float(np.linalg.norm(teacher_feats - teacher_center, axis=1).max()),
        "query_raw_dist_to_center": float(np.linalg.norm(extract_features([q], tfm, device, model, 1, "raw", args.gamma)[0] - teacher_center.squeeze(0))),
        "query_perceptual_dist_to_center": float(np.linalg.norm(q_feat_gamma - teacher_center.squeeze(0))),
        "query_rift_dist_to_center": float(np.linalg.norm(q_feat_rift - teacher_center.squeeze(0))),
        "nearest_negative_dist_to_center": float(neg_d[nn_idx[0]]),
        "hard_negative_relpaths": [s.relpath for s in hard_neg_samples],
    }
    save_json(geom_dir / "teacher_manifold_stats.json", geom_stats)

    pipeline_note = """# Figure 2 Component Manifest

## Inference path
1. Input image
2. Coarse illumination guidance
3. Feasible correction with identity-protection map `P_id`
4. Raw / base-corrected / adapted branches
5. Stripe-wise branch attention fusion
6. Descriptor head

## Training-only blocks
- frozen phase-3 teacher snapshot
- trust-bounded rollback
- ranking-aware identity terms
- teacher-manifold tube loss
- teacher-manifold separation loss
- local ranking topology preservation
"""
    (comp_dir / "pipeline_manifest.md").write_text(pipeline_note, encoding="utf-8")

    save_json(out_root / "manifest.json", {
        "rift_checkpoint": args.rift_ckpt,
        "config": args.config,
        "device": str(device),
        "gamma": args.gamma,
        "query_count": len(query_samples),
        "gallery_count": len(gallery_samples),
        "fig1": fig1_metrics,
        "fig3_trust": {"cases": trust_records},
        "fig3_geometry": geom_stats,
        "fig2_manifest": str((comp_dir / "pipeline_manifest.md").as_posix()),
    })
    print(f"Saved asset bundle to {out_root}")


if __name__ == "__main__":
    main()
