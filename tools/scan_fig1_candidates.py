#!/usr/bin/env python3
"""Scan ATRW query candidates for a clearer Figure 1 example."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.prepare_rift_paper_figure_assets import (
    build_transform,
    compute_ap,
    extract_features,
    load_model,
    load_rgb,
    load_split,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/illumination_config_atrw.yaml")
    parser.add_argument("--query_dir", default="data/processed/atrw/query")
    parser.add_argument("--gallery_dir", default="data/processed/atrw/gallery")
    parser.add_argument("--rift_ckpt", default="checkpoints/atrw_routeb_theoryB/joint_best.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.40)
    parser.add_argument("--lum_min", type=float, default=0.10)
    parser.add_argument("--lum_max", type=float, default=0.50)
    parser.add_argument("--topn", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    tfm = build_transform(args.config)
    query_samples = load_split(args.query_dir, "query")
    gallery_samples = load_split(args.gallery_dir, "gallery")
    gallery_labels = np.array([sample.label for sample in gallery_samples], dtype=object)

    model = load_model(args.rift_ckpt, args.config, device)
    gallery_rift = extract_features(
        gallery_samples, tfm, device, model, args.batch_size, "rift", args.gamma
    )
    gallery_gamma = extract_features(
        gallery_samples, tfm, device, model, args.batch_size, "gamma", args.gamma
    )

    query_lums = []
    for sample in query_samples:
        x = load_rgb(sample, tfm)
        query_lums.append(float((0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).mean().item()))
    query_lums = np.array(query_lums, dtype=np.float32)

    rows = []
    idxs = np.where((query_lums >= args.lum_min) & (query_lums <= args.lum_max))[0]
    print(f"Scanning {len(idxs)} queries in luminance range [{args.lum_min}, {args.lum_max}]")
    for idx, qi in enumerate(idxs):
        query = query_samples[int(qi)]
        q_rift = extract_features([query], tfm, device, model, 1, "rift", args.gamma)[0]
        q_gamma = extract_features([query], tfm, device, model, 1, "gamma", args.gamma)[0]
        d_rift = 1.0 - gallery_rift @ q_rift
        d_gamma = 1.0 - gallery_gamma @ q_gamma
        ap_rift = compute_ap(d_rift, gallery_labels, query.label)
        ap_gamma = compute_ap(d_gamma, gallery_labels, query.label)
        rows.append(
            {
                "query_index": int(qi),
                "query_relpath": query.relpath,
                "label": query.label,
                "lum": float(query_lums[qi]),
                "ap_gamma": float(ap_gamma),
                "ap_rift": float(ap_rift),
                "delta": float(ap_rift - ap_gamma),
                "top1_gamma": bool(gallery_labels[np.argmin(d_gamma)] == query.label),
                "top1_rift": bool(gallery_labels[np.argmin(d_rift)] == query.label),
            }
        )
        if (idx + 1) % 100 == 0:
            print(f"  processed {idx + 1}/{len(idxs)}")

    rows.sort(key=lambda item: item["delta"], reverse=True)
    for item in rows[: args.topn]:
        print(item)


if __name__ == "__main__":
    main()
