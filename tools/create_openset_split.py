#!/usr/bin/env python3
"""Create open-set splits from existing processed ReID datasets.

Design summary:
    In open-set evaluation, the test pool contains identities that never appear
    in the training split. This helper simulates that setting by removing a
    subset of training identities while keeping the existing query/gallery split
    unchanged.

Input:
    data/processed/{dataset}/train, query, gallery

Output:
    data/processed/{dataset}_openset/train, query, gallery

Strategy:
    1. Randomly keep X% of training identities as seen identities.
    2. Remove the remaining identities from the training split.
    3. Copy query/gallery unchanged so the evaluation pool still contains all
       identities, including unseen ones.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from typing import Dict, List, Set

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASETS = ["atrw", "stripespotter", "gzgc_zebra", "gzgc_giraffe"]
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def get_dataset_stats(dataset_path: str) -> Dict[str, Dict[str, object]]:
    """Collect compact split statistics for a processed dataset."""
    stats: Dict[str, Dict[str, object]] = {}
    for split in ["train", "query", "gallery"]:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            continue

        ids: Set[str] = set()
        img_count: Dict[str, int] = {}
        for pid in os.listdir(split_path):
            pid_path = os.path.join(split_path, pid)
            if os.path.isdir(pid_path):
                ids.add(pid)
                imgs = [f for f in os.listdir(pid_path) if f.lower().endswith(VALID_EXTENSIONS)]
                img_count[pid] = len(imgs)

        stats[split] = {
            "ids": ids,
            "img_count": img_count,
            "total_imgs": sum(img_count.values()),
        }
    return stats


def create_openset_split(
    dataset_name: str,
    seen_ratio: float = 0.7,
    seed: int = 42,
    base_dir: str = "data/processed",
) -> Dict[str, object]:
    """Create an open-set split from an existing closed-set processed dataset."""
    random.seed(seed)

    src_path = os.path.join(base_dir, dataset_name)
    dst_path = os.path.join(base_dir, f"{dataset_name}_openset")

    if not os.path.exists(src_path):
        print(f"[ERROR] Source dataset does not exist: {src_path}")
        return {}

    print(f"\n{'=' * 60}")
    print(f"Creating open-set split: {dataset_name}")
    print(f"  Source:      {src_path}")
    print(f"  Destination: {dst_path}")
    print(f"  Seen ratio:  {seen_ratio * 100:.0f}%")
    print(f"{'=' * 60}")

    stats = get_dataset_stats(src_path)
    train_ids = stats["train"]["ids"]
    query_ids = stats.get("query", {}).get("ids", set())
    gallery_ids = stats.get("gallery", {}).get("ids", set())

    print("\nOriginal closed-set statistics:")
    print(f"  Train:   {len(train_ids)} identities")
    print(f"  Query:   {len(query_ids)} identities")
    print(f"  Gallery: {len(gallery_ids)} identities")

    all_ids = list(train_ids)
    random.shuffle(all_ids)

    num_seen = max(1, int(len(all_ids) * seen_ratio))
    seen_ids = set(all_ids[:num_seen])
    unseen_ids = set(all_ids[num_seen:])

    print("\nOpen-set partition:")
    print(f"  Seen identities:   {len(seen_ids)}")
    print(f"  Unseen identities: {len(unseen_ids)}")

    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    os.makedirs(dst_path)

    train_dst = os.path.join(dst_path, "train")
    os.makedirs(train_dst)
    train_src = os.path.join(src_path, "train")

    seen_train_imgs = 0
    for pid in seen_ids:
        src_pid_path = os.path.join(train_src, pid)
        dst_pid_path = os.path.join(train_dst, pid)
        if os.path.exists(src_pid_path):
            shutil.copytree(src_pid_path, dst_pid_path)
            seen_train_imgs += len(
                [f for f in os.listdir(dst_pid_path) if f.lower().endswith(VALID_EXTENSIONS)]
            )

    print("\nCopied training split:")
    print(f"  Train: {len(seen_ids)} identities, {seen_train_imgs} images")

    query_src = os.path.join(src_path, "query")
    query_dst = os.path.join(dst_path, "query")
    if os.path.exists(query_src):
        shutil.copytree(query_src, query_dst)
        query_imgs = sum(
            len([f for f in os.listdir(os.path.join(query_dst, pid)) if f.lower().endswith(VALID_EXTENSIONS)])
            for pid in os.listdir(query_dst)
            if os.path.isdir(os.path.join(query_dst, pid))
        )
        print(f"  Query: {len(query_ids)} identities, {query_imgs} images (unchanged)")

    gallery_src = os.path.join(src_path, "gallery")
    gallery_dst = os.path.join(dst_path, "gallery")
    if os.path.exists(gallery_src):
        shutil.copytree(gallery_src, gallery_dst)
        gallery_imgs = sum(
            len([f for f in os.listdir(os.path.join(gallery_dst, pid)) if f.lower().endswith(VALID_EXTENSIONS)])
            for pid in os.listdir(gallery_dst)
            if os.path.isdir(os.path.join(gallery_dst, pid))
        )
        print(f"  Gallery: {len(gallery_ids)} identities, {gallery_imgs} images (unchanged)")

    split_info: Dict[str, object] = {
        "dataset": dataset_name,
        "seen_ratio": seen_ratio,
        "seed": seed,
        "seen_ids": sorted(list(seen_ids)),
        "unseen_ids": sorted(list(unseen_ids)),
        "stats": {
            "train_ids": len(seen_ids),
            "train_imgs": seen_train_imgs,
            "query_ids": len(query_ids),
            "gallery_ids": len(gallery_ids),
            "total_ids": len(all_ids),
            "seen_count": len(seen_ids),
            "unseen_count": len(unseen_ids),
        },
    }

    info_path = os.path.join(dst_path, "openset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f"\nSaved split metadata to: {info_path}")

    return split_info


def verify_openset_split(dataset_path: str) -> bool:
    """Verify that an open-set split satisfies the expected constraints."""
    print(f"\nVerifying open-set split: {dataset_path}")

    info_path = os.path.join(dataset_path, "openset_info.json")
    if not os.path.exists(info_path):
        print("  [ERROR] openset_info.json is missing.")
        return False

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    seen_ids = set(info["seen_ids"])
    unseen_ids = set(info["unseen_ids"])

    train_path = os.path.join(dataset_path, "train")
    train_ids = {
        pid for pid in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, pid))
    }

    if train_ids != seen_ids:
        extra = train_ids - seen_ids
        missing = seen_ids - train_ids
        print("  [ERROR] Train identities do not match the recorded seen-id set.")
        if extra:
            print(f"    Extra identities:   {sorted(extra)}")
        if missing:
            print(f"    Missing identities: {sorted(missing)}")
        return False

    leaked = train_ids & unseen_ids
    if leaked:
        print(f"  [ERROR] Train split still contains unseen identities: {sorted(leaked)}")
        return False

    query_path = os.path.join(dataset_path, "query")
    gallery_path = os.path.join(dataset_path, "gallery")
    query_ids = {
        pid for pid in os.listdir(query_path) if os.path.isdir(os.path.join(query_path, pid))
    }
    gallery_ids = {
        pid for pid in os.listdir(gallery_path) if os.path.isdir(os.path.join(gallery_path, pid))
    }

    if not unseen_ids.issubset(query_ids):
        print("  [ERROR] Query split is missing some unseen identities.")
        return False
    if not unseen_ids.issubset(gallery_ids):
        print("  [ERROR] Gallery split is missing some unseen identities.")
        return False

    print(f"  [OK] Train contains only seen identities: {len(seen_ids)}")
    print(f"  [OK] Query/gallery still cover all identities: {len(query_ids)}")
    print(f"  [OK] Unseen identities do not leak into train: {len(unseen_ids)}")
    print("  Verification passed.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Create open-set dataset splits")
    parser.add_argument("--dataset", type=str, choices=DATASETS, help="Dataset name")
    parser.add_argument("--all", action="store_true", help="Process all supported datasets")
    parser.add_argument("--seen_ratio", type=float, default=0.7, help="Seen-identity ratio (default: 0.7)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--base_dir", type=str, default="data/processed", help="Processed dataset root")
    parser.add_argument("--verify", action="store_true", help="Only verify existing open-set splits")
    args = parser.parse_args()

    if args.verify:
        datasets = DATASETS if args.all else ([args.dataset] if args.dataset else [])
        for ds in datasets:
            ds_path = os.path.join(args.base_dir, f"{ds}_openset")
            if os.path.exists(ds_path):
                verify_openset_split(ds_path)
            else:
                print(f"Skipping {ds}: open-set split does not exist.")
        return

    if not args.all and not args.dataset:
        parser.print_help()
        print("\nPlease specify either --dataset or --all.")
        return

    datasets = DATASETS if args.all else [args.dataset]
    results: Dict[str, Dict[str, object]] = {}
    for ds in datasets:
        result = create_openset_split(
            ds,
            seen_ratio=args.seen_ratio,
            seed=args.seed,
            base_dir=args.base_dir,
        )
        if result:
            results[ds] = result
            verify_openset_split(os.path.join(args.base_dir, f"{ds}_openset"))

    print(f"\n{'=' * 60}")
    print("Open-set split summary")
    print(f"{'=' * 60}")
    print(f"{'Dataset':<20} {'Total IDs':>10} {'Seen':>8} {'Unseen':>8} {'Ratio':>8}")
    print("-" * 60)
    for ds, info in results.items():
        stats = info["stats"]
        print(
            f"{ds:<20} {stats['total_ids']:>10} {stats['seen_count']:>8} "
            f"{stats['unseen_count']:>8} {info['seen_ratio'] * 100:>7.0f}%"
        )

    print(f"\nOutput root: {args.base_dir}/{{dataset}}_openset/")
    print("\nUsage example:")
    print("  # Train on the open-set ATRW split")
    print("  python tools/train_joint.py --data_dir data/processed/atrw_openset/train \\")
    print("      --query_dir data/processed/atrw_openset/query \\")
    print("      --gallery_dir data/processed/atrw_openset/gallery")


if __name__ == "__main__":
    main()
