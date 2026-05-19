#!/usr/bin/env python3
"""Prepare the older ATRW evaluation split where query and gallery share the same test pool."""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import defaultdict
from typing import Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_ROOT = os.path.join(PROJECT_ROOT, "orignal_data")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "data", "processed")


def prepare_atrw_correct() -> None:
    """Create the legacy ATRW split used by earlier closed-set experiments."""
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
    test_csv_path = os.path.join(
        ORIGINAL_ROOT,
        "Amur Tiger Re-identification",
        "atrw_anno_reid_test",
        "reid_list_test.csv",
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
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test-annotation file not found: {test_csv_path}")

    for out_dir in [train_out, query_out, gallery_out]:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Preparing the legacy ATRW split")
    print("=" * 70)

    print("\n[1/2] Preparing the training split...")
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

    print("\n[2/2] Preparing the legacy query/gallery test split...")
    test_files: List[str] = []
    with open(test_csv_path, "r", encoding="utf-8") as f:
        for line in f:
            filename = line.strip()
            if filename:
                test_files.append(filename)

    print(f"  Found {len(test_files)} test images")

    test_json_path = os.path.join(
        PROJECT_ROOT,
        "ATRWEvalScript-main",
        "annotations",
        "gt_test_plain.json",
    )

    if not os.path.exists(test_json_path):
        print(f"  [WARN] Official annotation file not found: {test_json_path}")
        print("  Falling back to temporary per-image identity folders.")
        num_test = 0
        for filename in test_files:
            src = os.path.join(test_images_dir, filename)
            if not os.path.exists(src):
                continue
            img_num = os.path.splitext(filename)[0]

            id_query_dir = os.path.join(query_out, f"test_{img_num}")
            os.makedirs(id_query_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(id_query_dir, filename))

            id_gallery_dir = os.path.join(gallery_out, f"test_{img_num}")
            os.makedirs(id_gallery_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(id_gallery_dir, filename))
            num_test += 1
    else:
        with open(test_json_path, "r", encoding="utf-8") as f:
            test_annotations = json.load(f)

        imgid_to_entity = {anno["imgid"]: anno["entityid"] for anno in test_annotations}
        test_image_files = sorted(
            [f for f in os.listdir(test_images_dir) if f.endswith((".jpg", ".png"))]
        )
        imgid_to_filename = {
            int(os.path.splitext(img_file)[0]): img_file for img_file in test_image_files
        }

        num_test = 0
        entity_ids = set()
        for imgid, entity_id in imgid_to_entity.items():
            filename = imgid_to_filename.get(imgid)
            if filename is None:
                continue

            src = os.path.join(test_images_dir, filename)
            if not os.path.exists(src):
                continue

            id_query_dir = os.path.join(query_out, str(entity_id))
            os.makedirs(id_query_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(id_query_dir, filename))

            id_gallery_dir = os.path.join(gallery_out, str(entity_id))
            os.makedirs(id_gallery_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(id_gallery_dir, filename))

            num_test += 1
            entity_ids.add(entity_id)

        print(f"  [OK] Copied {num_test} legacy test images across {len(entity_ids)} identities")
        print("  [OK] Query and gallery contain the same test pool.")

    print("\n" + "=" * 70)
    print("Legacy ATRW split prepared successfully")
    print("=" * 70)
    print(f"Training images: {num_train}")
    print(f"Query images:    {num_test}")
    print(f"Gallery images:  {num_test} (same pool as query)")
    print("=" * 70)
    print("\nOutput directories:")
    print(f"  Train:   {train_out}")
    print(f"  Query:   {query_out}")
    print(f"  Gallery: {gallery_out}")
    print("\nEvaluation notes:")
    print("  - Query and gallery use the same underlying test pool.")
    print("  - Evaluation code should exclude the query image itself.")
    print("  - Every query should still have same-identity matches in the gallery pool.")


if __name__ == "__main__":
    prepare_atrw_correct()
