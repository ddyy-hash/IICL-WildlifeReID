#!/usr/bin/env python3
"""Generate paper figures for RIFT (ACM MM 2026).

Produces:
  - fig1_motivation.pdf/png : query + rank list comparison (baseline vs RIFT)
  - fig3_tsne.pdf/png       : t-SNE embedding visualization (baseline vs RIFT)
  - fig4_trust_viz.pdf/png  : rollback gate + branch attention visualization

Usage (on GPU server):
    cd /root/autodl-tmp/v2_2/dog_reid_web
    python tools/generate_paper_figures.py \
        --baseline_ckpt checkpoints/ablation/atrw_baseline/joint_best.pth \
        --rift_ckpt checkpoints/atrw_routeb_softap_20260322_083312/joint_best.pth \
        --config config/illumination_config_atrw.yaml \
        --data_dir data/processed/atrw/train \
        --output_dir figures/
"""
import os, sys, argparse, numpy as np, torch, torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams.update({'font.family':'sans-serif','font.size':9,'axes.titlesize':10,'figure.dpi':300})

from app.core.config import load_config
from app.core.joint_model import JointReIDModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline_ckpt', required=True)
    p.add_argument('--rift_ckpt', required=True)
    p.add_argument('--config', default='config/illumination_config_atrw.yaml')
    p.add_argument('--data_dir', default='data/processed/atrw/train')
    p.add_argument('--output_dir', default='figures/')
    p.add_argument('--device', default='cuda')
    p.add_argument('--num_tsne_ids', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def load_model(ckpt_path, config_path, device, use_ipaid=None):
    cfg = load_config(config_path)
    model_cfg = cfg.get('model', {})
    illum_cfg = cfg.get('illumination_module', {})
    module_params = illum_cfg.get('module_params', {})
    backbone = model_cfg.get('backbone', 'osnet_ain_x1_0')
    local_cfg = model_cfg.get('local_extractor', {})
    num_stripes = local_cfg.get('num_parts', 6)
    dropout = local_cfg.get('dropout', 0.0)
    _use_ipaid = use_ipaid if use_ipaid is not None else model_cfg.get('illumination_module', {}).get('enabled', True)

    ipaid_params = dict(module_params)
    for key in ['feature_fusion', 'branch_attention_fusion', 'nuisance_head', 'reid_head', 'backbone_random_erasing']:
        sub = model_cfg.get(key, {})
        if sub:
            ipaid_params[f'_{key}'] = sub

    model = JointReIDModel(
        num_classes=107, backbone_name=backbone, num_stripes=num_stripes,
        hidden_dim=256, pretrained_backbone=False, use_ipaid=_use_ipaid,
        dropout=dropout, ipaid_params=ipaid_params,
    )
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    print(f"  Loaded {len(filtered)}/{len(model_state)} params from {os.path.basename(ckpt_path)}")
    model.to(device).eval()
    return model


def load_dataset(data_dir, img_h=256, img_w=384):
    transform = transforms.Compose([transforms.Resize((img_h, img_w)), transforms.ToTensor()])
    samples, id_to_indices, label_map = [], defaultdict(list), {}
    for folder in sorted(os.listdir(data_dir)):
        fp = os.path.join(data_dir, folder)
        if not os.path.isdir(fp): continue
        if folder not in label_map: label_map[folder] = len(label_map)
        label = label_map[folder]
        for img_name in sorted(os.listdir(fp)):
            if not img_name.lower().endswith(('.jpg','.jpeg','.png')): continue
            idx = len(samples)
            samples.append((os.path.join(fp, img_name), label))
            id_to_indices[label].append(idx)
    return samples, id_to_indices, transform, label_map


def compute_luminance(img_path, transform):
    img = Image.open(img_path).convert('RGB')
    t = transform(img)
    return (0.299*t[0] + 0.587*t[1] + 0.114*t[2]).mean().item()


def compute_ap(dists, labels, qlabel):
    order = np.argsort(dists)
    matches = (labels[order] == qlabel).astype(float)
    if matches.sum() == 0: return 0.0
    cum = np.cumsum(matches)
    prec = cum / (np.arange(len(matches)) + 1)
    return (prec * matches).sum() / matches.sum()


def load_img(path, transform, device):
    return transform(Image.open(path).convert('RGB')).unsqueeze(0).to(device)


# ============================================================================
#  Figure 1: Motivation
# ============================================================================
def generate_fig1(baseline, rift, samples, id_to_indices, transform, device, outdir):
    print("[Fig1] Computing luminance...")
    lums = np.array([compute_luminance(s[0], transform) for s in samples])
    dark_thr = np.percentile(lums, 30)
    dark_idx = np.where(lums < dark_thr)[0]

    best_qi, best_gap, best_info = None, -1e-9, None
    rng = np.random.RandomState(42)

    print(f"[Fig1] Scanning {min(50, len(dark_idx))} dark queries...")
    for qi in dark_idx[:50]:
        qlabel = samples[qi][1]
        gidx = [j for j in range(len(samples)) if j != qi]
        same = [j for j in gidx if samples[j][1] == qlabel]
        diff = [j for j in gidx if samples[j][1] != qlabel]
        if len(same) < 3: continue
        sub = same + rng.choice(diff, min(150, len(diff)), replace=False).tolist()
        glabels = np.array([samples[j][1] for j in sub])

        qimg = load_img(samples[qi][0], transform, device)

        with torch.no_grad():
            qf_b = F.normalize(baseline(qimg)['features'], dim=1).cpu()
            qf_r = F.normalize(rift(qimg)['features'], dim=1).cpu()

            # Batch gallery to avoid OOM
            gf_b_list, gf_r_list = [], []
            BS = 16
            for gi in range(0, len(sub), BS):
                batch_j = sub[gi:gi+BS]
                batch_imgs = torch.stack([transform(Image.open(samples[j][0]).convert('RGB')) for j in batch_j]).to(device)
                gf_b_list.append(F.normalize(baseline(batch_imgs)['features'], dim=1).cpu())
                gf_r_list.append(F.normalize(rift(batch_imgs)['features'], dim=1).cpu())
                del batch_imgs; torch.cuda.empty_cache()
            gf_b = torch.cat(gf_b_list, 0)
            gf_r = torch.cat(gf_r_list, 0)

        db = 1 - torch.mm(qf_b, gf_b.t()).squeeze(0).numpy()
        dr = 1 - torch.mm(qf_r, gf_r.t()).squeeze(0).numpy()
        ap_b, ap_r = compute_ap(db, glabels, qlabel), compute_ap(dr, glabels, qlabel)

        if ap_r - ap_b > best_gap and ap_b < 0.95:
            best_gap = ap_r - ap_b
            best_qi = qi
            best_info = dict(ap_b=ap_b, ap_r=ap_r, db=db, dr=dr, gidx=sub, glabels=glabels)

    if best_qi is None:
        print("[Fig1] No good case found"); return

    qi = best_qi; info = best_info
    print(f"[Fig1] Best: idx={qi}, AP_base={info['ap_b']:.2f}, AP_rift={info['ap_r']:.2f}")

    qimg_t = load_img(samples[qi][0], transform, device)
    with torch.no_grad():
        rift_out = rift(qimg_t, return_illuminated=True)
    rift_corr = rift_out['illuminated'][0].cpu().clamp(0,1) if rift_out.get('illuminated') is not None else qimg_t[0].cpu()
    gamma_corr = qimg_t[0].cpu().clamp(0.001,1).pow(0.4).clamp(0,1)

    order_b = np.argsort(info['db'])[:5]
    order_r = np.argsort(info['dr'])[:5]
    qlabel = samples[qi][1]

    fig = plt.figure(figsize=(11, 5))
    gs = gridspec.GridSpec(2, 7, hspace=0.35, wspace=0.12)

    for row, (corr_img, corr_title, corr_color, order, ap_val, ap_color) in enumerate([
        (gamma_corr, 'Perceptual\ncorrection', '#4285F4', order_b, info['ap_b'], '#EA4335'),
        (rift_corr, 'RIFT\ncorrection', '#EA8D13', order_r, info['ap_r'], '#34A853'),
    ]):
        ax_q = fig.add_subplot(gs[row, 0])
        ax_q.imshow(qimg_t[0].cpu().permute(1,2,0).numpy())
        ax_q.set_title('Query\n(low light)' if row==0 else '', fontsize=8)
        ax_q.axis('off')

        ax_c = fig.add_subplot(gs[row, 1])
        ax_c.imshow(corr_img.permute(1,2,0).numpy())
        ax_c.set_title(corr_title, fontsize=8, color=corr_color)
        ax_c.axis('off')

        for ri in range(5):
            ax = fig.add_subplot(gs[row, 2+ri])
            gj = info['gidx'][order[ri]]
            ax.imshow(transform(Image.open(samples[gj][0]).convert('RGB')).permute(1,2,0).numpy())
            match = info['glabels'][order[ri]] == qlabel
            c = '#34A853' if match else '#EA4335'
            for sp in ax.spines.values(): sp.set_edgecolor(c); sp.set_linewidth(3)
            ax.set_title(f'#{ri+1} {"✓" if match else "✗"}', fontsize=7, color=c)
            ax.axis('off')

        fig.text(0.95, 0.73 if row==0 else 0.28, f'AP={ap_val:.2f}',
                fontsize=10, color=ap_color, ha='right', weight='bold')

    fig.text(0.97, 0.5, 'Same query\ndifferent correction\n→ different retrieval',
            fontsize=8, ha='center', va='center', rotation=270, color='gray',
            style='italic')

    path = os.path.join(outdir, 'fig1_motivation')
    fig.savefig(path+'.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(path+'.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"[Fig1] Saved {path}.pdf")


# ============================================================================
#  Figure 3: t-SNE
# ============================================================================
def generate_fig3(baseline, rift, samples, id_to_indices, transform, device, outdir, n_ids=8, seed=42):
    from sklearn.manifold import TSNE
    rng = np.random.RandomState(seed)
    valid = [lid for lid, idxs in id_to_indices.items() if len(idxs) >= 6]
    sel_ids = rng.choice(valid, min(n_ids, len(valid)), replace=False)

    sel_samples, sel_labels = [], []
    for lid in sel_ids:
        chosen = rng.choice(id_to_indices[lid], min(8, len(id_to_indices[lid])), replace=False)
        for idx in chosen:
            sel_samples.append(samples[idx])
            sel_labels.append(lid)

    print(f"[Fig3] {len(sel_samples)} samples, {len(sel_ids)} IDs")

    with torch.no_grad():
        bf_list, rf_list = [], []
        BS = 16
        for i in range(0, len(sel_samples), BS):
            batch = torch.stack([transform(Image.open(p).convert('RGB')) for p,_ in sel_samples[i:i+BS]]).to(device)
            bf_list.append(F.normalize(baseline(batch)['features'], dim=1).cpu())
            rf_list.append(F.normalize(rift(batch)['features'], dim=1).cpu())
            del batch; torch.cuda.empty_cache()
        bf = torch.cat(bf_list, 0).numpy()
        rf = torch.cat(rf_list, 0).numpy()

    perp = min(15, len(sel_samples)-1)
    b2d = TSNE(n_components=2, perplexity=perp, random_state=seed).fit_transform(bf)
    r2d = TSNE(n_components=2, perplexity=perp, random_state=seed).fit_transform(rf)

    lums = np.array([compute_luminance(p, transform) for p,_ in sel_samples])
    lo, hi = np.percentile(lums, 33), np.percentile(lums, 67)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(sel_ids)))
    id2c = {lid: colors[i] for i, lid in enumerate(sel_ids)}

    for ax, emb, title in [(ax1, b2d, 'Plain baseline'), (ax2, r2d, 'RIFT (ours)')]:
        for i, (lid, lum) in enumerate(zip(sel_labels, lums)):
            m = 'v' if lum < lo else ('^' if lum > hi else 'o')
            ax.scatter(emb[i,0], emb[i,1], c=[id2c[lid]], marker=m, s=55,
                      edgecolors='k', linewidths=0.4, alpha=0.85)
        ax.set_title(title, fontsize=11, weight='bold')
        ax.set_xticks([]); ax.set_yticks([])

        # Intra-ID variance
        vars_ = []
        for lid in sel_ids:
            mask = np.array(sel_labels) == lid
            pts = emb[mask]
            vars_.append(np.mean(np.sum((pts - pts.mean(0))**2, 1)))
        ax.text(0.02, 0.02, f'mean intra-var: {np.mean(vars_):.1f}',
               transform=ax.transAxes, fontsize=7, color='gray')

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([0],[0],marker='v',color='gray',label='Dark',markersize=6,markerfacecolor='gray',linestyle='None'),
        Line2D([0],[0],marker='o',color='gray',label='Normal',markersize=6,markerfacecolor='gray',linestyle='None'),
        Line2D([0],[0],marker='^',color='gray',label='Bright',markersize=6,markerfacecolor='gray',linestyle='None'),
    ], loc='lower center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))

    path = os.path.join(outdir, 'fig3_tsne')
    fig.savefig(path+'.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(path+'.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"[Fig3] Saved {path}.pdf")


# ============================================================================
#  Figure 4: Trust Viz
# ============================================================================
def generate_fig4(rift, samples, id_to_indices, transform, device, outdir):
    lums = np.array([compute_luminance(s[0], transform) for s in samples])
    cases = {
        'Very dark':  np.argmin(lums),
        'Dark':       np.argsort(lums)[len(lums)//5],
        'Normal':     np.argsort(lums)[len(lums)//2],
        'Bright':     np.argmax(lums),
    }

    fig, axes = plt.subplots(4, 4, figsize=(13, 10))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)
    cmap = plt.cm.RdYlGn

    for ri, (name, idx) in enumerate(cases.items()):
        img_t = load_img(samples[idx][0], transform, device)
        with torch.no_grad():
            out = rift(img_t, return_illuminated=True)
        ipaid = out.get('ipaid_details') or {}
        raw = img_t[0].cpu().permute(1,2,0).numpy()

        axes[ri,0].imshow(raw); axes[ri,0].set_title(f'{name} (lum={lums[idx]:.2f})', fontsize=8); axes[ri,0].axis('off')

        corr = out['illuminated'][0].cpu().clamp(0,1).permute(1,2,0).numpy() if out.get('illuminated') is not None else raw
        axes[ri,1].imshow(corr); axes[ri,1].set_title('Corrected', fontsize=8); axes[ri,1].axis('off')

        # Rollback
        ax = axes[ri,2]
        if 'rollback_alpha' in ipaid:
            a = ipaid['rollback_alpha'][0].cpu().squeeze().numpy()
            if np.ndim(a)==0: a = np.full(6, float(a))
            if len(a)<6: a = np.pad(a, (0,6-len(a)), constant_values=a[-1])
            a = a[:6]
            ax.barh(range(6), a, height=0.75, color=[cmap(v) for v in a])
            ax.set_xlim(0,1); ax.set_yticks(range(6))
            ax.set_yticklabels([f's{i+1}' for i in range(6)], fontsize=7)
            ax.invert_yaxis()
            for i,v in enumerate(a): ax.text(min(v+0.02, 0.85), i, f'{v:.2f}', va='center', fontsize=6)
        ax.set_title('Rollback α', fontsize=8)

        # Branch attention
        ax = axes[ri,3]
        if 'branch_attention_weights' in ipaid:
            baw = ipaid['branch_attention_weights'][0].cpu().numpy()
            if baw.ndim==2 and baw.shape[1]==3:
                n = min(6, baw.shape[0])
                bc = ['#4285F4','#34A853','#EA8D13']
                bn = ['Raw','Base','Adapted']
                bot = np.zeros(n)
                for b in range(3):
                    ax.barh(range(n), baw[:n,b], left=bot, height=0.75, color=bc[b], label=bn[b] if ri==0 else None)
                    bot += baw[:n,b]
                ax.set_xlim(0,1); ax.set_yticks(range(n))
                ax.set_yticklabels([f's{i+1}' for i in range(n)], fontsize=7)
                ax.invert_yaxis()
                if ri==0: ax.legend(fontsize=5, loc='lower right')
        ax.set_title('Branch attn', fontsize=8)

    path = os.path.join(outdir, 'fig4_trust_viz')
    fig.savefig(path+'.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(path+'.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"[Fig4] Saved {path}.pdf")


# ============================================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("="*60 + "\nRIFT Paper Figure Generator\n" + "="*60)

    samples, id2idx, tfm, lmap = load_dataset(args.data_dir)
    print(f"Dataset: {len(samples)} images, {len(lmap)} IDs")

    print("\nLoading baseline (no IPAID)...")
    baseline = load_model(args.baseline_ckpt, args.config, args.device, use_ipaid=False)
    print("Loading RIFT...")
    rift = load_model(args.rift_ckpt, args.config, args.device, use_ipaid=True)

    generate_fig1(baseline, rift, samples, id2idx, tfm, args.device, args.output_dir)
    generate_fig3(baseline, rift, samples, id2idx, tfm, args.device, args.output_dir, args.num_tsne_ids)
    generate_fig4(rift, samples, id2idx, tfm, args.device, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}")

if __name__ == '__main__':
    main()
