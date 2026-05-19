#!/usr/bin/env python3
"""Generate multi-method top-10 retrieval data for Figure 1.

Methods compared:
  raw       – raw encoder, no correction (matched ReID baseline proxy)
  retinexnet– RetinexNet-enhanced images through raw encoder
  zerodcepp – Zero-DCE++-enhanced images through raw encoder
  rift      – full RIFT pipeline

Usage:
  python tools/prepare_fig1_multimethod.py --query_relpath query/112/003457.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import load_config
from app.core.joint_model import JointReIDModel


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rift_ckpt", default="checkpoints/atrw_routeb_theoryB/joint_best.pth")
    p.add_argument("--config", default="config/illumination_config_atrw.yaml")
    p.add_argument("--query_dir", default="data/processed/atrw/query")
    p.add_argument("--gallery_dir", default="data/processed/atrw/gallery")
    p.add_argument("--retinexnet_dir",
                   default="downloads/westc_perceptual_assets_20260325/root/autodl-tmp/v2_2/"
                           "dog_reid_web/data/perceptual_baselines/atrw/retinexnet/test")
    p.add_argument("--zerodcepp_dir",
                   default="downloads/westc_perceptual_assets_20260325/root/autodl-tmp/v2_2/"
                           "dog_reid_web/data/perceptual_baselines/atrw/zerodcepp/test")
    p.add_argument("--output_dir", default="docs/figures/rift_paper_20260325/fig1_claim/multimethod")
    p.add_argument("--query_relpath", default="query/112/003457.jpg")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--topk", type=int, default=10)
    return p.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data):
    ensure_dir(path.parent)
    def _conv(v):
        if isinstance(v, (np.generic, np.integer, np.floating)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, Path):
            return str(v)
        raise TypeError(type(v))
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_conv)


def load_split(root_dir: str, split: str):
    root = Path(root_dir)
    samples = []
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        for img_path in sorted(label_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            relpath = f"{split}/{label_dir.name}/{img_path.name}"
            samples.append({"split": split, "label": label_dir.name,
                            "path": str(img_path), "relpath": relpath,
                            "stem": img_path.stem})
    return samples


def build_transform(config_path: str):
    cfg = load_config(config_path)
    tc = cfg.get("training", {})
    h, w = int(tc.get("image_height", 256)), int(tc.get("image_width", 384))
    return transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor()])


def load_rgb(path: str, tfm) -> torch.Tensor:
    return tfm(Image.open(path).convert("RGB"))


def to_u8(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().clamp(0.0, 1.0).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def save_tensor(path: Path, x: torch.Tensor):
    ensure_dir(path.parent)
    Image.fromarray(to_u8(x)).save(path)


def infer_num_classes(state):
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and k.endswith(
                ("classifier.weight", "global_classifier.weight", "arcface.weight")):
            return int(v.shape[0])
    return 107


def load_model(ckpt_path, config_path, device):
    cfg = load_config(config_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    mc = cfg.get("model", {})
    ic = cfg.get("illumination_module", {})
    lc = mc.get("local_extractor", {})
    ip = dict(mc.get("illumination_module", {}).get("module_params") or ic.get("module_params", {}))
    for key in ("feature_fusion", "branch_attention_fusion", "nuisance_head",
                "reid_head", "backbone_random_erasing"):
        sub = mc.get(key, {})
        if sub:
            ip[f"_{key}"] = sub
    model = JointReIDModel(
        num_classes=infer_num_classes(state),
        backbone_name=mc.get("backbone", "osnet_ain_x1_0"),
        num_stripes=int(lc.get("num_parts", 6)),
        pretrained_backbone=False,
        use_ipaid=True,
        dropout=float(lc.get("dropout", 0.0)),
        ipaid_params=ip,
    )
    ms = model.state_dict()
    keep = {k: v for k, v in state.items()
            if k in ms and isinstance(v, torch.Tensor) and v.shape == ms[k].shape}
    model.load_state_dict(keep, strict=False)
    return model.to(device).eval()


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ---------------------------------------------------------------------------
# Feature extraction for each method
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_raw(model, batch):
    """Raw encoder path (no illumination correction)."""
    out = model.forward_raw_reference(batch, detach=True)
    return F.normalize(out["features"], p=2, dim=1)


@torch.no_grad()
def extract_rift(model, batch):
    """Full RIFT pipeline."""
    out = model(batch, return_illuminated=False)
    return F.normalize(out["features"], p=2, dim=1)


def extract_features_from_paths(paths: list[str], tfm, device, model, batch_size,
                                 forward_fn) -> np.ndarray:
    """Extract features from a list of image file paths."""
    feats = []
    for chunk in batched(paths, batch_size):
        xs = torch.stack([load_rgb(p, tfm) for p in chunk], 0).to(device)
        feat = forward_fn(model, xs)
        feats.append(feat.cpu())
    return torch.cat(feats, 0).numpy()


def compute_ap(dists, gallery_labels, query_label):
    order = np.argsort(dists)
    matches = (gallery_labels[order] == query_label).astype(np.float32)
    if matches.sum() <= 0:
        return 0.0
    prec = np.cumsum(matches) / (np.arange(matches.size, dtype=np.float32) + 1.0)
    return float((prec * matches).sum() / matches.sum())


def luminance(x: torch.Tensor):
    return float((0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).mean().item())


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    ensure_dir(out)

    tfm = build_transform(args.config)
    model = load_model(args.rift_ckpt, args.config, device)

    # -- Load gallery samples --
    gallery_samples = load_split(args.gallery_dir, "gallery")
    gallery_labels = np.array([s["label"] for s in gallery_samples], dtype=object)
    gallery_paths_raw = [s["path"] for s in gallery_samples]
    n_gal = len(gallery_samples)
    print(f"Gallery: {n_gal} images, {len(set(gallery_labels))} identities")

    # -- Build enhanced gallery paths --
    retinexnet_dir = Path(args.retinexnet_dir)
    zerodcepp_dir = Path(args.zerodcepp_dir)

    gallery_paths_retinex = []
    gallery_paths_zdce = []
    missing_retinex = 0
    missing_zdce = 0
    for s in gallery_samples:
        stem = s["stem"]
        rp = retinexnet_dir / f"{stem}.jpg"
        zp = zerodcepp_dir / f"{stem}.jpg"
        gallery_paths_retinex.append(str(rp) if rp.exists() else s["path"])
        gallery_paths_zdce.append(str(zp) if zp.exists() else s["path"])
        if not rp.exists():
            missing_retinex += 1
        if not zp.exists():
            missing_zdce += 1

    if missing_retinex:
        print(f"[WARN] {missing_retinex}/{n_gal} gallery images missing RetinexNet version, using raw fallback")
    if missing_zdce:
        print(f"[WARN] {missing_zdce}/{n_gal} gallery images missing Zero-DCE++ version, using raw fallback")

    # -- Find query --
    query_samples = load_split(args.query_dir, "query")
    qi = None
    for i, s in enumerate(query_samples):
        if s["relpath"] == args.query_relpath or args.query_relpath in s["relpath"]:
            qi = i
            break
    if qi is None:
        raise ValueError(f"Query {args.query_relpath} not found")

    q = query_samples[qi]
    q_label = q["label"]
    q_stem = q["stem"]
    q_raw_tensor = load_rgb(q["path"], tfm)
    q_lum = luminance(q_raw_tensor)
    print(f"Query: {q['relpath']} | label={q_label} | lum={q_lum:.3f}")

    # -- Query enhanced versions --
    q_retinex_path = retinexnet_dir / f"{q_stem}.jpg"
    q_zdce_path = zerodcepp_dir / f"{q_stem}.jpg"
    assert q_retinex_path.exists(), f"Missing RetinexNet query: {q_retinex_path}"
    assert q_zdce_path.exists(), f"Missing Zero-DCE++ query: {q_zdce_path}"

    # RIFT corrected query image
    with torch.no_grad():
        q_out = model(q_raw_tensor.unsqueeze(0).to(device), return_illuminated=True)
    q_rift_tensor = q_out.get("illuminated", q_raw_tensor.unsqueeze(0).to(device))[0].cpu().clamp(0, 1)

    # Save query images
    save_tensor(out / "query_raw.png", q_raw_tensor)
    save_tensor(out / "query_retinexnet.png", load_rgb(str(q_retinex_path), tfm))
    save_tensor(out / "query_zerodcepp.png", load_rgb(str(q_zdce_path), tfm))
    save_tensor(out / "query_rift.png", q_rift_tensor)
    print("Saved query images")

    # -- Extract gallery features for each method --
    print("Extracting gallery features [raw]...")
    gf_raw = extract_features_from_paths(gallery_paths_raw, tfm, device, model,
                                          args.batch_size, extract_raw)

    print("Extracting gallery features [retinexnet]...")
    gf_retinex = extract_features_from_paths(gallery_paths_retinex, tfm, device, model,
                                              args.batch_size, extract_raw)

    print("Extracting gallery features [zerodcepp]...")
    gf_zdce = extract_features_from_paths(gallery_paths_zdce, tfm, device, model,
                                           args.batch_size, extract_raw)

    print("Extracting gallery features [rift]...")
    gf_rift = extract_features_from_paths(gallery_paths_raw, tfm, device, model,
                                           args.batch_size, extract_rift)

    # -- Extract query features for each method --
    print("Extracting query features...")
    qf_raw = extract_features_from_paths([q["path"]], tfm, device, model, 1, extract_raw)[0]
    qf_retinex = extract_features_from_paths([str(q_retinex_path)], tfm, device, model, 1, extract_raw)[0]
    qf_zdce = extract_features_from_paths([str(q_zdce_path)], tfm, device, model, 1, extract_raw)[0]
    qf_rift = extract_features_from_paths([q["path"]], tfm, device, model, 1, extract_rift)[0]

    # -- Compute distances and AP for each method --
    methods = {
        "raw":        (qf_raw,     gf_raw,     "Matched ReID baseline"),
        "retinexnet": (qf_retinex, gf_retinex, "RetinexNet enhancement"),
        "zerodcepp":  (qf_zdce,    gf_zdce,    "Zero-DCE++ enhancement"),
        "rift":       (qf_rift,    gf_rift,    "RIFT (ours)"),
    }

    results = {}
    for method_key, (qf, gf, label) in methods.items():
        dists = 1.0 - gf @ qf
        order = np.argsort(dists)

        # Filter out query duplicate in gallery (same stem)
        display_order = [int(idx) for idx in order
                         if gallery_samples[int(idx)]["stem"] != q_stem][:args.topk]

        ap = compute_ap(dists, gallery_labels, q_label)

        # Save top-k gallery images
        ranked = []
        gal_dir = out / f"gallery_topk_{method_key}"
        for k, gi in enumerate(display_order):
            s = gallery_samples[gi]
            match = bool(s["label"] == q_label)
            tag = "pos" if match else "neg"
            name = f"rank{k+1:02d}_{tag}_{s['label']}_{s['stem']}.png"

            # Load the appropriate version of the gallery image
            if method_key == "retinexnet":
                gimg = load_rgb(gallery_paths_retinex[gi], tfm)
            elif method_key == "zerodcepp":
                gimg = load_rgb(gallery_paths_zdce[gi], tfm)
            else:
                gimg = load_rgb(s["path"], tfm)
            save_tensor(gal_dir / name, gimg)

            ranked.append({
                "rank": k + 1,
                "gallery_relpath": s["relpath"],
                "gallery_label": s["label"],
                "match": match,
                "saved_path": str((gal_dir / name).as_posix()),
            })

        results[method_key] = {
            "label": label,
            "ap": ap,
            "ranked": ranked,
        }
        n_correct = sum(1 for r in ranked if r["match"])
        print(f"  {label:30s}  AP={ap:.4f}  top-{args.topk}: {n_correct}/{args.topk} correct")

    # -- Save combined metrics --
    meta = {
        "query_relpath": q["relpath"],
        "query_label": q_label,
        "query_luminance": q_lum,
        "topk": args.topk,
        "methods": results,
    }
    save_json(out / "multimethod_metrics.json", meta)
    print(f"\nAll data saved to {out}")


if __name__ == "__main__":
    main()
