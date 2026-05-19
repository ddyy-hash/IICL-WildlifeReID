#!/usr/bin/env python3
"""


    python tools/generate_comparison_table.py \
        --baseline_results checkpoints/baselines/baseline_results.json \
        --our_checkpoint checkpoints/joint_atrw_ipaid/joint_best.pth \
        --query_dir data/processed/atrw/query \
        --gallery_dir data/processed/atrw/gallery \
        --output outputs/comparison_table.txt
"""

import os
import sys
import json
import argparse
from typing import Dict

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.core.joint_model import JointReIDModel

try:
    from reranking import re_ranking
    RERANKING_AVAILABLE = True
except ImportError:
    RERANKING_AVAILABLE = False


class SimpleDataset:
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        
        for pid_folder in sorted(os.listdir(root)):
            pid_path = os.path.join(root, pid_folder)
            if not os.path.isdir(pid_path):
                continue
            for img_name in sorted(os.listdir(pid_path)):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(pid_path, img_name), pid_folder))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        import cv2
        img_path, pid = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, pid


def evaluate_our_model(checkpoint_path, query_dir, gallery_dir, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = checkpoint.get("num_classes", 107)
    
    model = JointReIDModel(
        num_classes=num_classes,
        num_stripes=6,
        pretrained_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    query_dataset = SimpleDataset(query_dir, transform)
    gallery_dataset = SimpleDataset(gallery_dir, transform)
    
    query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False, num_workers=0)
    gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    def extract_features(loader):
        feats, pids = [], []
        with torch.no_grad():
            for imgs, pid_list in loader:
                imgs = imgs.to(device)
                output = model(imgs, boxes_list=None)
                features = output["features"]
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                feats.append(features.cpu().numpy())
                pids.extend(list(pid_list))
        return np.concatenate(feats, axis=0), pids
    
    query_feats, query_pids = extract_features(query_loader)
    gallery_feats, gallery_pids = extract_features(gallery_loader)
    
    def compute_metrics(qf, qp, gf, gp, use_rerank=False):
        if use_rerank and RERANKING_AVAILABLE:
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        else:
            q_norm = np.sum(qf ** 2, axis=1, keepdims=True)
            g_norm = np.sum(gf ** 2, axis=1, keepdims=True).T
            distmat = q_norm + g_norm - 2 * np.dot(qf, gf.T)
        
        qp = np.array(qp)
        gp = np.array(gp)
        
        indices = np.argsort(distmat, axis=1)
        matches = (gp[indices] == qp[:, np.newaxis]).astype(np.int32)
        
        all_cmc, all_AP = [], []
        for q_idx in range(len(qp)):
            raw_cmc = matches[q_idx]
            if not np.any(raw_cmc):
                continue
            cmc = raw_cmc.cumsum()
            cmc[cmc > 1] = 1
            all_cmc.append(cmc[0])
            
            num_rel = raw_cmc.sum()
            tmp_cmc = raw_cmc.cumsum()
            precision = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            precision = np.array(precision) * raw_cmc
            AP = precision.sum() / num_rel
            all_AP.append(AP)
        
        rank1 = np.mean(all_cmc) * 100
        mAP = np.mean(all_AP) * 100
        return rank1, mAP
    
    rank1, mAP = compute_metrics(query_feats, query_pids, gallery_feats, gallery_pids)
    rank1_rk, mAP_rk = compute_metrics(query_feats, query_pids, gallery_feats, gallery_pids, use_rerank=True)
    
    return {
        'rank1': rank1,
        'mAP': mAP,
        'rank1_rk': rank1_rk,
        'mAP_rk': mAP_rk,
    }


def generate_latex_table(results: Dict, output_path: str):
    
    latex = r"""
\begin{table}[t]
\centering
\caption{Comparison with state-of-the-art methods on ATRW dataset.}
\label{tab:comparison}
\begin{tabular}{l|cc|cc}
\toprule
\multirow{2}{*}{Method} & \multicolumn{2}{c|}{w/o Re-ranking} & \multicolumn{2}{c}{w/ Re-ranking} \\
 & Rank-1 & mAP & Rank-1 & mAP \\
\midrule
"""
    
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('mAP', 0))
    
    for name, res in sorted_results:
        rank1 = res.get('rank1', 0)
        mAP = res.get('mAP', 0)
        rank1_rk = res.get('rank1_rk', '-')
        mAP_rk = res.get('mAP_rk', '-')
        
        display_name = res.get('name', name)
        
        if isinstance(rank1_rk, float):
            latex += f"{display_name} & {rank1:.2f} & {mAP:.2f} & {rank1_rk:.2f} & {mAP_rk:.2f} \\\\\n"
        else:
            latex += f"{display_name} & {rank1:.2f} & {mAP:.2f} & - & - \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"LaTeX table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_results', type=str, 
                        default='checkpoints/baselines/baseline_results.json')
    parser.add_argument('--our_checkpoint', type=str,
                        default='checkpoints/joint_atrw_ipaid/joint_best.pth')
    parser.add_argument('--query_dir', type=str, default='data/processed/atrw/query')
    parser.add_argument('--gallery_dir', type=str, default='data/processed/atrw/gallery')
    parser.add_argument('--output', type=str, default='outputs/comparison_table.tex')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load baseline results.
    all_results = {}
    if os.path.exists(args.baseline_results):
        with open(args.baseline_results, 'r') as f:
            baseline_results = json.load(f)
        all_results.update(baseline_results)
        print(f"[INFO] Loaded baseline results for {len(baseline_results)} models")
    
    # Evaluate our model outputs.
    if os.path.exists(args.our_checkpoint):
        print("[INFO] Evaluating our models...")
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        our_results = evaluate_our_model(
            args.our_checkpoint, 
            args.query_dir, 
            args.gallery_dir, 
            device
        )
        all_results['ours'] = {
            'name': 'Ours (IPAID + OSNet-AIN)',
            **our_results
        }
        print(f"  Rank-1: {our_results['rank1']:.2f}%, mAP: {our_results['mAP']:.2f}%")
        print(f"  Rank-1(RK): {our_results['rank1_rk']:.2f}%, mAP(RK): {our_results['mAP_rk']:.2f}%")
    
    # Print the summary table.
    print("\n" + "=" * 80)
    print("ATRW comparison results")
    print("=" * 80)
    print(f"{'Method':<35} {'Rank-1':>10} {'mAP':>10} {'Rank-1(RK)':>12} {'mAP(RK)':>10}")
    print("-" * 80)
    
    # Sort by mAP.
    sorted_results = sorted(all_results.items(), key=lambda x: x[1].get('mAP', 0), reverse=True)
    
    for name, res in sorted_results:
        display_name = res.get('name', name)
        rank1 = res.get('rank1', 0)
        mAP = res.get('mAP', 0)
        rank1_rk = res.get('rank1_rk', None)
        mAP_rk = res.get('mAP_rk', None)
        
        if rank1_rk is not None:
            print(f"{display_name:<35} {rank1:>9.2f}% {mAP:>9.2f}% {rank1_rk:>11.2f}% {mAP_rk:>9.2f}%")
        else:
            print(f"{display_name:<35} {rank1:>9.2f}% {mAP:>9.2f}% {'-':>12} {'-':>10}")
    
    print("=" * 80)
    
    # Generate the LaTeX table.
    generate_latex_table(all_results, args.output)
    
    # Save the combined results.
    combined_path = args.output.replace('.tex', '_all.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Combined results saved to: {combined_path}")


if __name__ == '__main__':
    main()
