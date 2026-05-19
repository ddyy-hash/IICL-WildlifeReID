#!/usr/bin/env python3
"""Prepare the official ATRW train/query/gallery split used by the paper package."""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import defaultdict
from typing import Dict, List, Set

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_ROOT = os.path.join(PROJECT_ROOT, "orignal_data")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "data", "processed")


def prepare_atrw_official() -> None:
    """Create the official paper-facing ATRW split from the raw data bundle."""
    train_images_dir = os.path.join(
        ORIGINAL_ROOT,
        "Amur Tiger Re-identification",
        "atrw_reid_train",
        "train",
    )
    train_csv_path = os.path.join(
        ORIGINAL_ROOT,
        "Amur Tiger Re-identification",
        "atrw_anno_reid_train",
        "reid_list_train.csv",
    )
    test_images_dir = os.path.join(
        ORIGINAL_ROOT,
        "Amur Tiger Re-identification",
        "atrw_reid_test",
        "test",
    )
    test_json_path = os.path.join(
        PROJECT_ROOT,
        "ATRWEvalScript-main",
        "annotations",
        "gt_test_plain.json",
    )

    train_out = os.path.join(OUTPUT_ROOT, "atrw", "train")
    query_out = os.path.join(OUTPUT_ROOT, "atrw", "query")
    gallery_out = os.path.join(OUTPUT_ROOT, "atrw", "gallery")

    if not os.path.exists(train_images_dir):
        raise FileNotFoundError(f"Training-image directory not found: {train_images_dir}")
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Training-annotation file not found: {train_csv_path}")
    if not os.path.exists(test_images_dir):
        raise FileNotFoundError(f"Test-image directory not found: {test_images_dir}")
    if not os.path.exists(test_json_path):
        raise FileNotFoundError(f"Official test-annotation file not found: {test_json_path}")

    for out_dir in [train_out, query_out, gallery_out]:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Preparing the official ATRW split")
    print("=" * 70)

    print("\n[1/3] Preparing the training split...")
    id_to_files: Dict[str, List[str]] = defaultdict(list)
    with open(train_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            tiger_id = row[0].strip()
            filename = row[1].strip()
            if tiger_id and filename:
                id_to_files[tiger_id].append(filename)

    print(f"  Found {len(id_to_files)} training identities")

    num_train = 0
    for tiger_id, files in id_to_files.items():
        id_train_dir = os.path.join(train_out, tiger_id)
        os.makedirs(id_train_dir, exist_ok=True)

        for fname in files:
            src = os.path.join(train_images_dir, fname)
            if not os.path.exists(src):
                print(f"  [WARN] Missing training image: {fname}")
                continue
            shutil.copy2(src, os.path.join(id_train_dir, fname))
            num_train += 1

    print(f"  [OK] Copied {num_train} training images")

    print("\n[2/3] Reading the official test annotations...")
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_annotations = json.load(f)

    test_image_files = sorted(
        [f for f in os.listdir(test_images_dir) if f.endswith((".jpg", ".png"))]
    )
    imgid_to_filename = {int(os.path.splitext(img_file)[0]): img_file for img_file in test_image_files}

    print(f"  Found {len(test_image_files)} test images")
    print(f"  Loaded {len(test_annotations)} test annotations")

    query_annotations = [a for a in test_annotations if a["query"] == "sing"]
    gallery_annotations = [a for a in test_annotations if a["query"] == "multi"]

    print(f"  Query annotations:   {len(query_annotations)}")
    print(f"  Gallery annotations: {len(gallery_annotations)}")

    print("\n[3/3] Copying query and gallery images...")
    num_query = 0
    query_entity_ids: Set[int] = set()
    for anno in query_annotations:
        imgid = anno["imgid"]
        entity_id = str(anno["entityid"])
        filename = imgid_to_filename.get(imgid)
        if filename is None:
            print(f"  [WARN] Query imgid {imgid} is missing from the test directory")
            continue

        src = os.path.join(test_images_dir, filename)
        if not os.path.exists(src):
            print(f"  [WARN] Missing query image: {filename}")
            continue

        id_query_dir = os.path.join(query_out, entity_id)
        os.makedirs(id_query_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(id_query_dir, filename))
        num_query += 1
        query_entity_ids.add(anno["entityid"])

    num_gallery = 0
    gallery_entity_ids: Set[int] = set()
    for anno in gallery_annotations:
        imgid = anno["imgid"]
        entity_id = str(anno["entityid"])
        filename = imgid_to_filename.get(imgid)
        if filename is None:
            print(f"  [WARN] Gallery imgid {imgid} is missing from the test directory")
            continue

        src = os.path.join(test_images_dir, filename)
        if not os.path.exists(src):
            print(f"  [WARN] Missing gallery image: {filename}")
            continue

        id_gallery_dir = os.path.join(gallery_out, entity_id)
        os.makedirs(id_gallery_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(id_gallery_dir, filename))
        num_gallery += 1
        gallery_entity_ids.add(anno["entityid"])

    print(f"  [OK] Copied {num_query} query images across {len(query_entity_ids)} identities")
    print(f"  [OK] Copied {num_gallery} gallery images across {len(gallery_entity_ids)} identities")

    print("\n" + "=" * 70)
    print("Official ATRW split prepared successfully")
    print("=" * 70)
    print(f"Training images: {num_train}")
    print(f"Query images:    {num_query} ({len(query_entity_ids)} identities)")
    print(f"Gallery images:  {num_gallery} ({len(gallery_entity_ids)} identities)")
    print(f"Total test-side images: {num_query + num_gallery}")
    print("=" * 70)
    print("\nOutput directories:")
    print(f"  Train:   {train_out}")
    print(f"  Query:   {query_out}")
    print(f"  Gallery: {gallery_out}")


if __name__ == "__main__":
    prepare_atrw_official()
