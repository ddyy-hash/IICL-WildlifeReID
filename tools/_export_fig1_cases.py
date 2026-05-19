#!/usr/bin/env python3
"""Export two best Fig 1 cases with retrieval comparison assets."""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import load_config
from app.core.joint_model import JointReIDModel

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
TOPK = 5
TARGET_CASES = [1228, 1696, 694, 797]  # Best cases where c5_riic > c5_raw


def load_split(root):
    samples = []
    root = Path(root)
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        for img in sorted(label_dir.iterdir()):
            if img.suffix.lower() in IMAGE_EXTS:
                samples.append((str(img), label_dir.name, img.stem))
    return samples


def resolve_enhanced(stem, root):
    p = root / (stem + ".jpg")
    if p.exists():
        return str(p)
    m = sorted(root.glob(stem + ".*"))
    return str(m[0]) if m else None


def save_tensor(path, x):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = x.detach().cpu().clamp(0.0, 1.0).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


@torch.no_grad()
def extract_all(paths, model, tfm, mode, bs=16):
    chunks = []
    for i in range(0, len(paths), bs):
        batch = torch.stack(
            [tfm(Image.open(p).convert("RGB")) for p in paths[i : i + bs]], 0
        ).cuda()
        if mode == "riic":
            out = model(batch, return_illuminated=False)
            feat = F.normalize(out["features"], p=2, dim=1)
        else:
            out = model.forward_raw_reference(batch, detach=True)
            feat = F.normalize(out["features"], p=2, dim=1)
        chunks.append(feat.cpu())
    return torch.cat(chunks, 0).numpy()


def main():
    cfg = load_config("config/illumination_config_atrw.yaml")
    train_cfg = cfg.get("training", {})
    h = int(train_cfg.get("image_height", 256))
    w = int(train_cfg.get("image_width", 384))
    tfm = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor()])

    # Load model
    ckpt = torch.load(
        "checkpoints/atrw_routeb_theoryB/joint_best.pth",
        map_location="cpu",
        weights_only=False,
    )
    state = ckpt["model_state_dict"]
    model_cfg = cfg.get("model", {})
    illum_cfg = cfg.get("illumination_module", {})
    local_cfg = model_cfg.get("local_extractor", {})
    ipaid_params = dict(illum_cfg.get("module_params", {}))
    for key in (
        "feature_fusion",
        "branch_attention_fusion",
        "nuisance_head",
        "reid_head",
        "backbone_random_erasing",
    ):
        sub = model_cfg.get(key, {})
        if sub:
            ipaid_params["_" + key] = sub

    num_classes = 107
    for k, v in state.items():
        if (
            isinstance(v, torch.Tensor)
            and v.ndim == 2
            and k.endswith(
                ("classifier.weight", "global_classifier.weight", "arcface.weight")
            )
        ):
            num_classes = int(v.shape[0])
            break

    model = JointReIDModel(
        num_classes=num_classes,
        backbone_name="osnet_ain_x1_0",
        num_stripes=6,
        pretrained_backbone=False,
        use_ipaid=True,
        dropout=0.1,
        ipaid_params=ipaid_params,
    )
    ms = model.state_dict()
    keep = {
        k: v
        for k, v in state.items()
        if k in ms and isinstance(v, torch.Tensor) and v.shape == ms[k].shape
    }
    model.load_state_dict(keep, strict=False)
    model = model.cuda().eval()

    # Load data
    query_samples = load_split("data/processed/atrw/query")
    gallery_samples = load_split("data/processed/atrw/gallery")
    gallery_labels = np.array([s[1] for s in gallery_samples])

    retinex_root = Path(
        "downloads/westc_perceptual_assets_20260325/root/autodl-tmp/v2_2/"
        "dog_reid_web/data/perceptual_baselines/atrw/retinexnet/test"
    )
    zerodce_root = Path(
        "downloads/westc_perceptual_assets_20260325/root/autodl-tmp/v2_2/"
        "dog_reid_web/data/perceptual_baselines/atrw/zerodcepp/test"
    )

    # Extract features
    print("Extracting features...")
    gp_raw = [s[0] for s in gallery_samples]
    gp_ret = [resolve_enhanced(s[2], retinex_root) or s[0] for s in gallery_samples]
    gp_zdc = [resolve_enhanced(s[2], zerodce_root) or s[0] for s in gallery_samples]

    gf_raw = extract_all(gp_raw, model, tfm, "raw")
    gf_ret = extract_all(gp_ret, model, tfm, "raw")
    gf_zdc = extract_all(gp_zdc, model, tfm, "raw")
    gf_riic = extract_all(gp_raw, model, tfm, "riic")

    qp_raw = [s[0] for s in query_samples]
    qp_ret = [resolve_enhanced(s[2], retinex_root) or s[0] for s in query_samples]
    qp_zdc = [resolve_enhanced(s[2], zerodce_root) or s[0] for s in query_samples]

    qf_raw = extract_all(qp_raw, model, tfm, "raw")
    qf_ret = extract_all(qp_ret, model, tfm, "raw")
    qf_zdc = extract_all(qp_zdc, model, tfm, "raw")
    qf_riic = extract_all(qp_raw, model, tfm, "riic")

    for case_idx in TARGET_CASES:
        qs = query_samples[case_idx]
        qlabel = qs[1]
        qstem = qs[2]
        out_dir = Path("docs/figures/riic_reid_fig1_best") / (
            "case_" + qlabel + "_" + qstem
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save query images (raw, retinexnet, zerodcepp, riic)
        raw_t = tfm(Image.open(qs[0]).convert("RGB"))
        ret_p = resolve_enhanced(qstem, retinex_root) or qs[0]
        zdc_p = resolve_enhanced(qstem, zerodce_root) or qs[0]
        ret_t = tfm(Image.open(ret_p).convert("RGB"))
        zdc_t = tfm(Image.open(zdc_p).convert("RGB"))

        with torch.no_grad():
            riic_out = model(raw_t.unsqueeze(0).cuda(), return_illuminated=True)
        riic_t = riic_out["illuminated"][0].detach().cpu().clamp(0.0, 1.0)

        save_tensor(str(out_dir / "query_raw.png"), raw_t)
        save_tensor(str(out_dir / "query_retinexnet.png"), ret_t)
        save_tensor(str(out_dir / "query_zerodcepp.png"), zdc_t)
        save_tensor(str(out_dir / "query_riic.png"), riic_t)

        lum = float((0.299 * raw_t[0] + 0.587 * raw_t[1] + 0.114 * raw_t[2]).mean())

        # Compute distances and save top-k gallery
        feat_dict = {
            "raw": (gf_raw, qf_raw),
            "retinexnet": (gf_ret, qf_ret),
            "zerodcepp": (gf_zdc, qf_zdc),
            "riic": (gf_riic, qf_riic),
        }
        meta = {
            "query": qs[0],
            "label": qlabel,
            "luminance": round(lum, 3),
            "topk": TOPK,
            "methods": {},
        }

        for mkey, (gf, qf) in feat_dict.items():
            dists = 1.0 - gf @ qf[case_idx]
            order = np.argsort(dists)
            display = [
                int(i)
                for i in order
                if gallery_samples[int(i)][2] != qstem
            ][:TOPK]

            method_dir = out_dir / ("gallery_" + mkey)
            method_dir.mkdir(parents=True, exist_ok=True)
            ranked = []
            for rank, gi in enumerate(display, 1):
                gs = gallery_samples[gi]
                match = gs[1] == qlabel
                tag = "pos" if match else "neg"
                fname = f"rank{rank:02d}_{tag}_{gs[1]}_{gs[2]}.png"
                save_tensor(
                    str(method_dir / fname),
                    tfm(Image.open(gs[0]).convert("RGB")),
                )
                ranked.append(
                    {"rank": rank, "label": gs[1], "match": match, "file": fname}
                )

            # Compute AP
            order_all = np.argsort(dists)
            matches = (gallery_labels[order_all] == qlabel).astype(np.float32)
            prec = np.cumsum(matches) / (
                np.arange(matches.size, dtype=np.float32) + 1.0
            )
            ap = float((prec * matches).sum() / max(matches.sum(), 1))

            meta["methods"][mkey] = {
                "ap": round(ap, 4),
                "correct_at_k": sum(1 for r in ranked if r["match"]),
                "ranked": ranked,
            }

        with open(str(out_dir / "metrics.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Exported case {case_idx} (ID {qlabel}, lum={lum:.3f}) to {out_dir}")
        for mkey in ("raw", "retinexnet", "zerodcepp", "riic"):
            m = meta["methods"][mkey]
            print(
                f"  {mkey:>12s}: AP={m['ap']:.4f}  C@{TOPK}={m['correct_at_k']}/{TOPK}"
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
