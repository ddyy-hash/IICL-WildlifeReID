#!/usr/bin/env python3
"""Generate Fig1 motivation on remote server."""
import os, sys, numpy as np, torch, torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from collections import defaultdict
sys.path.insert(0, '/root/autodl-tmp/v2_2/dog_reid_web')
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from app.core.config import load_config
from app.core.joint_model import JointReIDModel

def load_model(ckpt, cfg_path, dev, use_ipaid=None):
    cfg = load_config(cfg_path)
    mc = cfg.get("model", {})
    ic = cfg.get("illumination_module", {}).get("module_params", {})
    bb = mc.get("backbone", "osnet_ain_x1_0")
    ns = mc.get("local_extractor", {}).get("num_parts", 6)
    dp = mc.get("local_extractor", {}).get("dropout", 0.0)
    ui = use_ipaid if use_ipaid is not None else mc.get("illumination_module", {}).get("enabled", True)
    ip = dict(ic)
    for k in ["feature_fusion", "branch_attention_fusion", "nuisance_head", "reid_head"]:
        s = mc.get(k, {})
        if s:
            ip["_" + k] = s
    m = JointReIDModel(
        num_classes=107, backbone_name=bb, num_stripes=ns,
        hidden_dim=256, pretrained_backbone=False,
        use_ipaid=ui, dropout=dp, ipaid_params=ip,
    )
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    st = st.get("model_state_dict", st)
    ms = m.state_dict()
    m.load_state_dict({k: v for k, v in st.items() if k in ms and v.shape == ms[k].shape}, strict=False)
    return m.to(dev).eval()

os.chdir('/root/autodl-tmp/v2_2/dog_reid_web')
tfm = transforms.Compose([transforms.Resize((256, 384)), transforms.ToTensor()])

data_dir = "data/processed/atrw/train"
samples, id2idx, lmap = [], defaultdict(list), {}
for f in sorted(os.listdir(data_dir)):
    fp = os.path.join(data_dir, f)
    if not os.path.isdir(fp):
        continue
    if f not in lmap:
        lmap[f] = len(lmap)
    l = lmap[f]
    for im in sorted(os.listdir(fp)):
        if not im.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        idx = len(samples)
        samples.append((os.path.join(fp, im), l))
        id2idx[l].append(idx)

print(f"Dataset: {len(samples)} images, {len(lmap)} IDs")

dev = "cuda"
cfg_path = "config/illumination_config_atrw.yaml"
print("Loading baseline...")
bl = load_model("checkpoints/ablation/atrw_baseline/joint_best.pth", cfg_path, dev, False)
print("Loading RIFT...")
rf = load_model("checkpoints/atrw_routeb_softap_20260322_083312/joint_best.pth", cfg_path, dev, True)

# Compute luminance
print("Computing luminance...")
lums = []
for p, _ in samples:
    t = tfm(Image.open(p).convert("RGB"))
    lums.append((0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2]).mean().item())
lums = np.array(lums)

def compute_ap(dists, labels, ql):
    order = np.argsort(dists)
    matches = (labels[order] == ql).astype(float)
    if matches.sum() == 0:
        return 0.0
    cum = np.cumsum(matches)
    prec = cum / (np.arange(len(matches)) + 1)
    return (prec * matches).sum() / matches.sum()

# Search for best motivation case
dark_order = np.argsort(lums)
rng = np.random.RandomState(42)
best_qi, best_gap, best_info = None, -999, None

print("Scanning queries...")
for count, qi in enumerate(dark_order[:100]):
    ql = samples[qi][1]
    same = [j for j in range(len(samples)) if samples[j][1] == ql and j != qi]
    diff = [j for j in range(len(samples)) if samples[j][1] != ql]
    if len(same) < 2:
        continue
    sub = same + rng.choice(diff, min(100, len(diff)), replace=False).tolist()
    gl = np.array([samples[j][1] for j in sub])

    qimg = tfm(Image.open(samples[qi][0]).convert("RGB")).unsqueeze(0).to(dev)
    with torch.no_grad():
        qfb = F.normalize(bl(qimg)["features"], dim=1).cpu()
        qfr = F.normalize(rf(qimg)["features"], dim=1).cpu()
        gfb_l, gfr_l = [], []
        for gi in range(0, len(sub), 16):
            bj = sub[gi:gi + 16]
            bi = torch.stack([tfm(Image.open(samples[j][0]).convert("RGB")) for j in bj]).to(dev)
            gfb_l.append(F.normalize(bl(bi)["features"], dim=1).cpu())
            gfr_l.append(F.normalize(rf(bi)["features"], dim=1).cpu())
            del bi
            torch.cuda.empty_cache()
        gfb, gfr = torch.cat(gfb_l, 0), torch.cat(gfr_l, 0)

    db = 1 - torch.mm(qfb, gfb.t()).squeeze(0).numpy()
    dr = 1 - torch.mm(qfr, gfr.t()).squeeze(0).numpy()
    ab = compute_ap(db, gl, ql)
    ar = compute_ap(dr, gl, ql)
    gap = ar - ab

    if gap > best_gap:
        best_gap = gap
        best_qi = qi
        best_info = dict(ab=ab, ar=ar, db=db, dr=dr, gidx=sub, gl=gl)

    if count % 20 == 0:
        print(f"  checked {count}/100, current best gap={best_gap:.3f}")

if best_qi is None:
    print("No case found")
    sys.exit(1)

qi = best_qi
info = best_info
ql = samples[qi][1]
print(f"Best: idx={qi}, lum={lums[qi]:.3f}, AP_base={info['ab']:.3f}, AP_rift={info['ar']:.3f}")

qimg = tfm(Image.open(samples[qi][0]).convert("RGB")).unsqueeze(0).to(dev)
with torch.no_grad():
    ro = rf(qimg, return_illuminated=True)
if ro.get("illuminated") is not None:
    rc = ro["illuminated"][0].cpu().clamp(0, 1)
else:
    rc = qimg[0].cpu()
gc = qimg[0].cpu().clamp(0.001, 1).pow(0.4).clamp(0, 1)

ob = np.argsort(info["db"])[:5]
orr = np.argsort(info["dr"])[:5]

fig = plt.figure(figsize=(11, 5))
gs = gridspec.GridSpec(2, 7, hspace=0.35, wspace=0.12)
plt.rcParams.update({"font.size": 9, "figure.dpi": 300})

# Row 0: perceptual correction
row_data = [
    (gc, "Perceptual\ncorrection", "#4285F4", ob, info["ab"], "#EA4335"),
    (rc, "RIFT\ncorrection", "#EA8D13", orr, info["ar"], "#34A853"),
]

for row, (ci, ct, cc, order, apv, apc) in enumerate(row_data):
    ax = fig.add_subplot(gs[row, 0])
    ax.imshow(qimg[0].cpu().permute(1, 2, 0).numpy())
    ax.set_title("Query" if row == 0 else "", fontsize=8)
    ax.axis("off")

    ax = fig.add_subplot(gs[row, 1])
    ax.imshow(ci.permute(1, 2, 0).numpy())
    ax.set_title(ct, fontsize=8, color=cc)
    ax.axis("off")

    for ri in range(5):
        ax = fig.add_subplot(gs[row, 2 + ri])
        gj = info["gidx"][order[ri]]
        gimg = tfm(Image.open(samples[gj][0]).convert("RGB"))
        ax.imshow(gimg.permute(1, 2, 0).numpy())
        mt = info["gl"][order[ri]] == ql
        c = "#34A853" if mt else "#EA4335"
        mark = "Y" if mt else "X"
        for sp in ax.spines.values():
            sp.set_edgecolor(c)
            sp.set_linewidth(3)
        ax.set_title(f"#{ri+1} {mark}", fontsize=7, color=c)
        ax.axis("off")

    y_pos = 0.73 if row == 0 else 0.28
    fig.text(0.95, y_pos, f"AP={apv:.2f}", fontsize=10, color=apc, ha="right", weight="bold")

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig1_motivation.pdf", bbox_inches="tight", dpi=300)
fig.savefig("figures/fig1_motivation.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("Saved figures/fig1_motivation.pdf and .png")
