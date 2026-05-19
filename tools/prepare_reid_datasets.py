#!/usr/bin/env python3
"""Prepare multiple ReID datasets into the project's train/query/gallery layout."""

import os
import csv
import json
import shutil
import argparse
import re
import random
import subprocess
import time
from collections import defaultdict
from typing import Dict, List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_ROOT = os.path.join(PROJECT_ROOT, "orignal_data")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "data", "processed")


def _safe_rmtree(path: str, retries: int = 5, delay: float = 0.2) -> None:
    """Best-effort recursive delete that is a bit more tolerant on Windows."""
    if not os.path.exists(path):
        return

    last_error = None
    for _ in range(retries):
        try:
            shutil.rmtree(path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(delay)

    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "rmdir", "/s", "/q", path],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not os.path.exists(path):
            return
    if last_error is not None:
        raise last_error


# =============================================================================
# Amur Tiger Re-identification
# =============================================================================

def prepare_atrw_train(seed: int = 42, train_ratio: float = 0.7) -> None:
    """Prepare the ATRW closed-set split from the official training annotations."""
    images_dir = os.path.join(
        ORIGINAL_ROOT,
        "Amur Tiger Re-identification",
        "atrw_reid_train",
        "train",
    )
    csv_path = os.path.join(
        ORIGINAL_ROOT,
        "Amur Tiger Re-identification",
        "atrw_anno_reid_train",
        "reid_list_train.csv",
    )
    
    train_out = os.path.join(OUTPUT_ROOT, "atrw", "train")
    query_out = os.path.join(OUTPUT_ROOT, "atrw", "query")
    gallery_out = os.path.join(OUTPUT_ROOT, "atrw", "gallery")

    for out_dir in [train_out, query_out, gallery_out]:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"ATRW training-image directory does not exist: {images_dir}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ATRW training-annotation file does not exist: {csv_path}")

    id_to_files: Dict[str, List[str]] = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                continue
            tiger_id = row[0].strip()
            filename = row[1].strip()
            if not tiger_id or not filename:
                continue
            id_to_files[tiger_id].append(filename)

    print(f"[ATRW] Total identities: {len(id_to_files)}")
    
    random.seed(seed)
    
    num_train = 0
    num_query = 0
    num_gallery = 0
    
    for tiger_id, files in id_to_files.items():
        files = sorted(files)
        random.shuffle(files)
        
        n_train = max(1, int(len(files) * train_ratio))
        n_test = len(files) - n_train
        
        if n_test < 2 and len(files) >= 3:
            n_train = len(files) - 2
            n_test = 2
        
        train_files = files[:n_train]
        test_files = files[n_train:]
        
        id_train_dir = os.path.join(train_out, tiger_id)
        os.makedirs(id_train_dir, exist_ok=True)
        for fname in train_files:
            src = os.path.join(images_dir, fname)
            if not os.path.exists(src):
                print(f"[ATRW][WARN] Image not found: {src}")
                continue
            dst = os.path.join(id_train_dir, fname)
            shutil.copy2(src, dst)
            num_train += 1
        
        if test_files:
            query_file = test_files[0]
            gallery_files = test_files[1:]
            
            id_query_dir = os.path.join(query_out, tiger_id)
            os.makedirs(id_query_dir, exist_ok=True)
            src_q = os.path.join(images_dir, query_file)
            if os.path.exists(src_q):
                shutil.copy2(src_q, os.path.join(id_query_dir, query_file))
                num_query += 1
            
            if gallery_files:
                id_gallery_dir = os.path.join(gallery_out, tiger_id)
                os.makedirs(id_gallery_dir, exist_ok=True)
                for fname in gallery_files:
                    src_g = os.path.join(images_dir, fname)
                    if os.path.exists(src_g):
                        shutil.copy2(src_g, os.path.join(id_gallery_dir, fname))
                        num_gallery += 1

    print(f"[ATRW] Standard 70:30 split complete (seed={seed}):")
    print(f"  - Train: {num_train} images ({num_train/1887*100:.1f}%)")
    print(f"  - Query: {num_query} images")
    print(f"  - Gallery: {num_gallery} images")
    print(f"  - Total test images: {num_query + num_gallery} ({(num_query+num_gallery)/1887*100:.1f}%)")
    print("  - Train and test do not overlap: yes")


def build_atrw_query_gallery(seed: int = 42) -> None:
    """Deprecated ATRW helper kept for compatibility with older workflows."""
    print("[ATRW] build_atrw_query_gallery() is deprecated; the split is already created in prepare_atrw_train().")
    return


def _old_build_atrw_query_gallery(seed: int = 42) -> None:
    """Legacy ATRW query/gallery builder kept for historical reference."""
    train_root = os.path.join(OUTPUT_ROOT, "atrw", "train")
    query_root = os.path.join(OUTPUT_ROOT, "atrw", "query")
    gallery_root = os.path.join(OUTPUT_ROOT, "atrw", "gallery")

    if not os.path.exists(train_root):
        print(f"[ATRW][WARN] Train directory does not exist; skipping query/gallery construction: {train_root}")
        return

    os.makedirs(query_root, exist_ok=True)
    os.makedirs(gallery_root, exist_ok=True)

    random.seed(seed)

    num_query = 0
    num_gallery = 0

    for tiger_id in sorted(os.listdir(train_root)):
        id_train_dir = os.path.join(train_root, tiger_id)
        if not os.path.isdir(id_train_dir):
            continue

        files = [
            f
            for f in os.listdir(id_train_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not files:
            continue

        files = sorted(files)
        random.shuffle(files)

        query_file = files[0]
        gallery_files = files[1:]

        src_q = os.path.join(id_train_dir, query_file)
        id_query_dir = os.path.join(query_root, tiger_id)
        os.makedirs(id_query_dir, exist_ok=True)
        dst_q = os.path.join(id_query_dir, query_file)
        shutil.copy2(src_q, dst_q)
        num_query += 1

        if gallery_files:
            id_gallery_dir = os.path.join(gallery_root, tiger_id)
            os.makedirs(id_gallery_dir, exist_ok=True)
            for fname in gallery_files:
                src_g = os.path.join(id_train_dir, fname)
                dst_g = os.path.join(id_gallery_dir, fname)
                shutil.copy2(src_g, dst_g)
                num_gallery += 1

    print(
        f"[ATRW] Query/gallery construction complete: query={num_query}, gallery={num_gallery} "
        f"→ query: {query_root}, gallery: {gallery_root}"
    )


# =============================================================================
# DukeMTMC-reID
# =============================================================================

PID_PATTERN = re.compile(r"^([\-0-9]{4})_")


def _collect_duke_split(split_name: str, src_dir: str, out_root: str) -> None:
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Duke split directory does not exist: {src_dir}")

    out_dir = os.path.join(out_root, split_name)
    os.makedirs(out_dir, exist_ok=True)

    num_files = 0
    skipped = 0

    for fname in os.listdir(src_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        m = PID_PATTERN.match(fname)
        if not m:
            print(f"[Duke][WARN] Unable to parse PID from filename: {fname}")
            skipped += 1
            continue
        pid_str = m.group(1)
        if pid_str == "-1":
            skipped += 1
            continue
        pid = str(int(pid_str))

        id_dir = os.path.join(out_dir, pid)
        os.makedirs(id_dir, exist_ok=True)

        src = os.path.join(src_dir, fname)
        dst = os.path.join(id_dir, fname)
        shutil.copy2(src, dst)
        num_files += 1

    print(
        f"[Duke] split={split_name}: copied {num_files} images to {out_dir} (skipped {skipped})"
    )


def prepare_duke() -> None:
    duke_root = os.path.join(ORIGINAL_ROOT, "DukeMTMC-reID")
    train_src = os.path.join(duke_root, "bounding_box_train")
    query_src = os.path.join(duke_root, "query")
    gallery_src = os.path.join(duke_root, "bounding_box_test")

    out_root = os.path.join(OUTPUT_ROOT, "duke")
    os.makedirs(out_root, exist_ok=True)

    _collect_duke_split("train", train_src, out_root)
    _collect_duke_split("query", query_src, out_root)
    _collect_duke_split("gallery", gallery_src, out_root)


# =============================================================================
# =============================================================================

def prepare_nyala(seed: int = 42) -> None:
    """Prepare the Nyala dataset with a simple query/gallery split from the test pool."""
    nyala_root = os.path.join(
        ORIGINAL_ROOT, "wildlife_reidentification", "Nyala_Data_Zero"
    )
    train_src = os.path.join(nyala_root, "train")
    test_src = os.path.join(nyala_root, "test")
    
    out_root = os.path.join(OUTPUT_ROOT, "nyala")
    train_out = os.path.join(out_root, "train")
    query_out = os.path.join(out_root, "query")
    gallery_out = os.path.join(out_root, "gallery")
    
    if not os.path.exists(nyala_root):
        print(f"[Nyala][WARN] Dataset directory does not exist: {nyala_root}")
        return
    
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(query_out, exist_ok=True)
    os.makedirs(gallery_out, exist_ok=True)
    
    num_train = 0
    if os.path.exists(train_src):
        for identity in sorted(os.listdir(train_src)):
            id_src = os.path.join(train_src, identity)
            if not os.path.isdir(id_src):
                continue
            id_dst = os.path.join(train_out, identity)
            os.makedirs(id_dst, exist_ok=True)
            
            for fname in os.listdir(id_src):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                shutil.copy2(os.path.join(id_src, fname), os.path.join(id_dst, fname))
                num_train += 1
    
    print(f"[Nyala] Training split: {num_train} images -> {train_out}")
    
    random.seed(seed)
    num_query = 0
    num_gallery = 0
    
    if os.path.exists(test_src):
        for identity in sorted(os.listdir(test_src)):
            id_src = os.path.join(test_src, identity)
            if not os.path.isdir(id_src):
                continue
            
            files = [f for f in os.listdir(id_src) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not files:
                continue
            
            files = sorted(files)
            random.shuffle(files)
            
            query_file = files[0]
            gallery_files = files[1:]
            
            id_query_dir = os.path.join(query_out, identity)
            os.makedirs(id_query_dir, exist_ok=True)
            shutil.copy2(
                os.path.join(id_src, query_file),
                os.path.join(id_query_dir, query_file)
            )
            num_query += 1
            
            if gallery_files:
                id_gallery_dir = os.path.join(gallery_out, identity)
                os.makedirs(id_gallery_dir, exist_ok=True)
                for fname in gallery_files:
                    shutil.copy2(
                        os.path.join(id_src, fname),
                        os.path.join(id_gallery_dir, fname)
                    )
                    num_gallery += 1
    
    print(f"[Nyala] query/gallery: query={num_query}, gallery={num_gallery}")


# =============================================================================
# =============================================================================

def prepare_lion(seed: int = 42) -> None:
    """Prepare the Lion dataset with query/gallery images derived from the validation pool."""
    lion_root = os.path.join(
        ORIGINAL_ROOT, "wildlife_reidentification", "Lion_Data_Zero"
    )
    train_src = os.path.join(lion_root, "train")
    val_src = os.path.join(lion_root, "val")
    
    out_root = os.path.join(OUTPUT_ROOT, "lion")
    train_out = os.path.join(out_root, "train")
    query_out = os.path.join(out_root, "query")
    gallery_out = os.path.join(out_root, "gallery")
    
    if not os.path.exists(lion_root):
        print(f"[Lion][WARN] Dataset directory does not exist: {lion_root}")
        return
    
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(query_out, exist_ok=True)
    os.makedirs(gallery_out, exist_ok=True)
    
    num_train = 0
    if os.path.exists(train_src):
        for identity in sorted(os.listdir(train_src)):
            id_src = os.path.join(train_src, identity)
            if not os.path.isdir(id_src):
                continue
            id_dst = os.path.join(train_out, identity)
            os.makedirs(id_dst, exist_ok=True)
            
            for fname in os.listdir(id_src):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                shutil.copy2(os.path.join(id_src, fname), os.path.join(id_dst, fname))
                num_train += 1
    
    print(f"[Lion] Training split: {num_train} images -> {train_out}")
    
    random.seed(seed)
    num_query = 0
    num_gallery = 0
    
    if os.path.exists(val_src):
        for identity in sorted(os.listdir(val_src)):
            id_src = os.path.join(val_src, identity)
            if not os.path.isdir(id_src):
                continue
            
            files = [f for f in os.listdir(id_src) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not files:
                continue
            
            files = sorted(files)
            random.shuffle(files)
            
            query_file = files[0]
            gallery_files = files[1:]
            
            id_query_dir = os.path.join(query_out, identity)
            os.makedirs(id_query_dir, exist_ok=True)
            shutil.copy2(
                os.path.join(id_src, query_file),
                os.path.join(id_query_dir, query_file)
            )
            num_query += 1
            
            if gallery_files:
                id_gallery_dir = os.path.join(gallery_out, identity)
                os.makedirs(id_gallery_dir, exist_ok=True)
                for fname in gallery_files:
                    shutil.copy2(
                        os.path.join(id_src, fname),
                        os.path.join(id_gallery_dir, fname)
                    )
                    num_gallery += 1
    
    print(f"[Lion] query/gallery: query={num_query}, gallery={num_gallery}")


# =============================================================================
# Market-1501
# =============================================================================

MARKET_PID_PATTERN = re.compile(r"^([\-0-9]{4})_")


def prepare_market() -> None:
    market_root = os.path.join(ORIGINAL_ROOT, "Market-1501-v15.09.15")
    train_src = os.path.join(market_root, "bounding_box_train")
    query_src = os.path.join(market_root, "query")
    gallery_src = os.path.join(market_root, "bounding_box_test")
    
    out_root = os.path.join(OUTPUT_ROOT, "market")
    
    if not os.path.exists(market_root):
        print(f"[Market][WARN] Dataset directory does not exist: {market_root}")
        return
    
    os.makedirs(out_root, exist_ok=True)
    
    _collect_market_split("train", train_src, out_root)
    _collect_market_split("query", query_src, out_root)
    _collect_market_split("gallery", gallery_src, out_root)


def _collect_market_split(split_name: str, src_dir: str, out_root: str) -> None:
    if not os.path.exists(src_dir):
        print(f"[Market][WARN] Split directory does not exist: {src_dir}")
        return
    
    out_dir = os.path.join(out_root, split_name)
    os.makedirs(out_dir, exist_ok=True)
    
    num_files = 0
    skipped = 0
    
    for fname in os.listdir(src_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        m = MARKET_PID_PATTERN.match(fname)
        if not m:
            skipped += 1
            continue
        pid_str = m.group(1)
        if pid_str in ("-1", "0000", "-001"):
            skipped += 1
            continue
        pid = str(int(pid_str))
        
        id_dir = os.path.join(out_dir, pid)
        os.makedirs(id_dir, exist_ok=True)
        
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(id_dir, fname))
        num_files += 1
    
    print(f"[Market] split={split_name}: {num_files} images -> {out_dir} (skipped {skipped})")


# =============================================================================
# =============================================================================

def prepare_sealid() -> None:
    """Prepare the SealID dataset from its annotation CSV into train/query/gallery splits."""
    import csv
    
    sealid_root = os.path.join(ORIGINAL_ROOT, "SealID")
    patches_dir = os.path.join(sealid_root, "patches", "patches")
    source_dir = os.path.join(patches_dir, "source")
    annotation_file = os.path.join(patches_dir, "annotation.csv")
    
    out_root = os.path.join(OUTPUT_ROOT, "sealid")
    train_out = os.path.join(out_root, "train")
    query_out = os.path.join(out_root, "query")
    gallery_out = os.path.join(out_root, "gallery")
    
    if not os.path.exists(sealid_root):
        print(f"[SealID][WARN] Dataset directory does not exist: {sealid_root}")
        return
    
    if not os.path.exists(annotation_file):
        print(f"[SealID][WARN] annotation.csv does not exist: {annotation_file}")
        return
    
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(query_out, exist_ok=True)
    os.makedirs(gallery_out, exist_ok=True)
    
    num_train = 0
    num_query = 0
    num_gallery = 0
    skipped = 0
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for row in reader:
            if len(row) < 4:
                skipped += 1
                continue
            
            class_id, filename, split, testing_split = row[0], row[1], row[2], row[3]
            
            if '_' in class_id:
                seal_id = class_id.split('_')[0]
            else:
                seal_id = class_id
            
            src_path = os.path.join(source_dir, filename)
            if not os.path.exists(src_path):
                skipped += 1
                continue
            
            if split == "training":
                out_dir = train_out
                target_dir = os.path.join(out_dir, seal_id)
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy2(src_path, os.path.join(target_dir, filename))
                num_train += 1
            elif split == "testing":
                if testing_split == "query":
                    out_dir = query_out
                    target_dir = os.path.join(out_dir, seal_id)
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.copy2(src_path, os.path.join(target_dir, filename))
                    num_query += 1
                elif testing_split == "database":
                    out_dir = gallery_out
                    target_dir = os.path.join(out_dir, seal_id)
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.copy2(src_path, os.path.join(target_dir, filename))
                    num_gallery += 1
                else:
                    skipped += 1
            else:
                skipped += 1
    
    train_ids = len(os.listdir(train_out)) if os.path.exists(train_out) else 0
    query_ids = len(os.listdir(query_out)) if os.path.exists(query_out) else 0
    gallery_ids = len(os.listdir(gallery_out)) if os.path.exists(gallery_out) else 0
    
    print(f"[SealID] train: {num_train} images, {train_ids} identities")
    print(f"[SealID] query: {num_query} images, {query_ids} identities")
    print(f"[SealID] gallery: {num_gallery} images, {gallery_ids} identities")
    print(f"[SealID] skipped records: {skipped}")


# =============================================================================
# Leopard ID Dataset (Wild Me COCO Format)
# =============================================================================

def prepare_leopard(seed: int = 42, train_ratio: float = 0.7) -> None:
    """Prepare the Leopard-ID dataset from the WildMe COCO-style annotations."""
    import json
    from PIL import Image
    
    leopard_root = os.path.join(ORIGINAL_ROOT, "leopard.coco")
    anno_path = os.path.join(leopard_root, "annotations", "instances_train2022.json")
    images_dir = os.path.join(leopard_root, "images", "train2022")
    
    out_root = os.path.join(OUTPUT_ROOT, "leopard")
    train_out = os.path.join(out_root, "train")
    query_out = os.path.join(out_root, "query")
    gallery_out = os.path.join(out_root, "gallery")
    
    if not os.path.exists(anno_path):
        print(f"[Leopard][WARN] Annotation file does not exist: {anno_path}")
        return
    
    print("[Leopard] Loading annotation file...")
    with open(anno_path, 'r') as f:
        data = json.load(f)
    
    id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    
    individual_annotations = defaultdict(list)
    for ann in data['annotations']:
        if 'name' not in ann or not ann['name']:
            continue
        individual_id = ann['name']
        individual_annotations[individual_id].append(ann)
    
    print(f"[Leopard] Total individuals: {len(individual_annotations)}, annotations: {len(data['annotations'])}")
    
    random.seed(seed)
    all_individuals = sorted(individual_annotations.keys())
    random.shuffle(all_individuals)
    
    n_train = int(len(all_individuals) * train_ratio)
    train_individuals = set(all_individuals[:n_train])
    test_individuals = set(all_individuals[n_train:])
    
    print(f"[Leopard] Training individuals: {len(train_individuals)}, test individuals: {len(test_individuals)}")
    
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(query_out, exist_ok=True)
    os.makedirs(gallery_out, exist_ok=True)
    
    num_train = 0
    num_query = 0
    num_gallery = 0
    skipped = 0
    
    individual_to_numid = {ind: str(i) for i, ind in enumerate(sorted(all_individuals))}
    
    def crop_and_save(ann, out_path):
        image_id = ann['image_id']
        if image_id not in id_to_filename:
            return False
        
        filename = id_to_filename[image_id]
        src_path = os.path.join(images_dir, filename)
        
        if not os.path.exists(src_path):
            return False
        
        try:
            img = Image.open(src_path)
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(img.width, int(x + w))
            y2 = min(img.height, int(y + h))
            
            if x2 <= x1 or y2 <= y1:
                return False
            
            cropped = img.crop((x1, y1, x2, y2))
            cropped.save(out_path, "JPEG", quality=95)
            return True
        except Exception as e:
            print(f"[Leopard][WARN] Crop failed for {filename}: {e}")
            return False
    
    for ind_id in train_individuals:
        annotations = individual_annotations[ind_id]
        num_id = individual_to_numid[ind_id]
        id_dir = os.path.join(train_out, num_id)
        os.makedirs(id_dir, exist_ok=True)
        
        for i, ann in enumerate(annotations):
            out_path = os.path.join(id_dir, f"{num_id}_{i:04d}.jpg")
            if crop_and_save(ann, out_path):
                num_train += 1
            else:
                skipped += 1
    
    for ind_id in test_individuals:
        annotations = individual_annotations[ind_id]
        num_id = individual_to_numid[ind_id]
        
        if len(annotations) == 0:
            continue
        
        random.shuffle(annotations)
        query_ann = annotations[0]
        gallery_anns = annotations[1:]
        
        # Query
        id_query_dir = os.path.join(query_out, num_id)
        os.makedirs(id_query_dir, exist_ok=True)
        out_path = os.path.join(id_query_dir, f"{num_id}_query.jpg")
        if crop_and_save(query_ann, out_path):
            num_query += 1
        else:
            skipped += 1
        
        # Gallery
        if gallery_anns:
            id_gallery_dir = os.path.join(gallery_out, num_id)
            os.makedirs(id_gallery_dir, exist_ok=True)
            for i, ann in enumerate(gallery_anns):
                out_path = os.path.join(id_gallery_dir, f"{num_id}_{i:04d}.jpg")
                if crop_and_save(ann, out_path):
                    num_gallery += 1
                else:
                    skipped += 1
    
    print(f"[Leopard] train: {num_train} images, {len(train_individuals)} identities")
    print(f"[Leopard] query: {num_query} images")
    print(f"[Leopard] gallery: {num_gallery} images")
    print(f"[Leopard] skipped annotations: {skipped}")


# =============================================================================
# Whale Shark ID Dataset (Wild Me COCO Format)
# =============================================================================

def prepare_whaleshark(seed: int = 42, train_ratio: float = 0.7) -> None:
    """Prepare the WhaleShark-ID dataset from the WildMe COCO-style annotations."""
    import json
    from PIL import Image
    
    whaleshark_root = os.path.join(ORIGINAL_ROOT, "whaleshark.coco", "whaleshark.coco")
    anno_path = os.path.join(whaleshark_root, "annotations", "instances_train2020.json")
    images_dir = os.path.join(whaleshark_root, "images", "train2020")
    
    out_root = os.path.join(OUTPUT_ROOT, "whaleshark")
    train_out = os.path.join(out_root, "train")
    query_out = os.path.join(out_root, "query")
    gallery_out = os.path.join(out_root, "gallery")
    
    if not os.path.exists(anno_path):
        print(f"[WhaleShark][WARN] Annotation file does not exist: {anno_path}")
        return
    
    print("[WhaleShark] Loading annotation file...")
    with open(anno_path, 'r') as f:
        data = json.load(f)
    
    id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    
    individual_annotations = defaultdict(list)
    for ann in data['annotations']:
        if 'name' not in ann or not ann['name']:
            continue
        individual_id = ann['name']
        individual_annotations[individual_id].append(ann)
    
    print(f"[WhaleShark] Total individuals: {len(individual_annotations)}, annotations: {len(data['annotations'])}")
    
    random.seed(seed)
    all_individuals = sorted(individual_annotations.keys())
    random.shuffle(all_individuals)
    
    n_train = int(len(all_individuals) * train_ratio)
    train_individuals = set(all_individuals[:n_train])
    test_individuals = set(all_individuals[n_train:])
    
    print(f"[WhaleShark] Training individuals: {len(train_individuals)}, test individuals: {len(test_individuals)}")
    
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(query_out, exist_ok=True)
    os.makedirs(gallery_out, exist_ok=True)
    
    num_train = 0
    num_query = 0
    num_gallery = 0
    skipped = 0
    
    individual_to_numid = {ind: str(i) for i, ind in enumerate(sorted(all_individuals))}
    
    def crop_and_save(ann, out_path):
        image_id = ann['image_id']
        if image_id not in id_to_filename:
            return False
        
        filename = id_to_filename[image_id]
        src_path = os.path.join(images_dir, filename)
        
        if not os.path.exists(src_path):
            return False
        
        try:
            img = Image.open(src_path)
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(img.width, int(x + w))
            y2 = min(img.height, int(y + h))
            
            if x2 <= x1 or y2 <= y1:
                return False
            
            cropped = img.crop((x1, y1, x2, y2))
            cropped.save(out_path, "JPEG", quality=95)
            return True
        except Exception as e:
            print(f"[WhaleShark][WARN] Crop failed for {filename}: {e}")
            return False
    
    for ind_id in train_individuals:
        annotations = individual_annotations[ind_id]
        num_id = individual_to_numid[ind_id]
        id_dir = os.path.join(train_out, num_id)
        os.makedirs(id_dir, exist_ok=True)
        
        for i, ann in enumerate(annotations):
            out_path = os.path.join(id_dir, f"{num_id}_{i:04d}.jpg")
            if crop_and_save(ann, out_path):
                num_train += 1
            else:
                skipped += 1
    
    for ind_id in test_individuals:
        annotations = individual_annotations[ind_id]
        num_id = individual_to_numid[ind_id]
        
        if len(annotations) == 0:
            continue
        
        random.shuffle(annotations)
        query_ann = annotations[0]
        gallery_anns = annotations[1:]
        
        # Query
        id_query_dir = os.path.join(query_out, num_id)
        os.makedirs(id_query_dir, exist_ok=True)
        out_path = os.path.join(id_query_dir, f"{num_id}_query.jpg")
        if crop_and_save(query_ann, out_path):
            num_query += 1
        else:
            skipped += 1
        
        # Gallery
        if gallery_anns:
            id_gallery_dir = os.path.join(gallery_out, num_id)
            os.makedirs(id_gallery_dir, exist_ok=True)
            for i, ann in enumerate(gallery_anns):
                out_path = os.path.join(id_gallery_dir, f"{num_id}_{i:04d}.jpg")
                if crop_and_save(ann, out_path):
                    num_gallery += 1
                else:
                    skipped += 1
    
    print(f"[WhaleShark] train: {num_train} images, {len(train_individuals)} identities")
    print(f"[WhaleShark] query: {num_query} images")
    print(f"[WhaleShark] gallery: {num_gallery} images")
    print(f"[WhaleShark] skipped annotations: {skipped}")


# =============================================================================
# =============================================================================

def prepare_ipanda50(seed: int = 42, split_idx: int = 0) -> None:
    """Prepare iPanda50 using the official train/test split files."""
    images_root = os.path.join(ORIGINAL_ROOT, "iPanda50", "iPanda50-images")
    split_dir = os.path.join(ORIGINAL_ROOT, "iPanda50", "iPanda50-split")
    
    train_split_file = os.path.join(split_dir, f"split{split_idx}_train.txt")
    test_split_file = os.path.join(split_dir, f"split{split_idx}_test.txt")
    
    train_out = os.path.join(OUTPUT_ROOT, "ipanda50", "train")
    test_out = os.path.join(OUTPUT_ROOT, "ipanda50", "test")
    
    if not os.path.exists(images_root):
        raise FileNotFoundError(f"iPanda50 image directory does not exist: {images_root}")
    if not os.path.exists(train_split_file):
        raise FileNotFoundError(f"iPanda50 split file does not exist: {train_split_file}")
    
    for out_dir in [train_out, test_out]:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
    
    for old_dir in ["query", "gallery"]:
        old_path = os.path.join(OUTPUT_ROOT, "ipanda50", old_dir)
        if os.path.exists(old_path):
            shutil.rmtree(old_path)
            print(f"[iPanda50] Removed legacy directory: {old_path}")
    
    identity_dirs = sorted([d for d in os.listdir(images_root) 
                           if os.path.isdir(os.path.join(images_root, d)) and not d.startswith('.')])
    print(f"[iPanda50] Total identities: {len(identity_dirs)}")
    
    id_to_dir = {}
    for id_dir in identity_dirs:
        id_prefix = id_dir.split('_')[0]
        id_to_dir[id_prefix] = id_dir
    
    all_actual_files = {}  # {id_prefix: {filename: full_path}}
    for id_prefix, id_dir in id_to_dir.items():
        dir_path = os.path.join(images_root, id_dir)
        all_actual_files[id_prefix] = {}
        for fname in os.listdir(dir_path):
            if fname.endswith('.jpg'):
                all_actual_files[id_prefix][fname] = os.path.join(dir_path, fname)
    
    def read_split_file(filepath):
        files = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    files.append(line)
        return files
    
    train_list = read_split_file(train_split_file)
    test_list = read_split_file(test_split_file)
    
    print(f"[iPanda50] Split {split_idx}: official train={len(train_list)}, test={len(test_list)}")
    
    def find_file(id_prefix: str, split_filename: str) -> str:
        if id_prefix not in all_actual_files:
            return None
        
        if split_filename in all_actual_files[id_prefix]:
            return all_actual_files[id_prefix][split_filename]
        
        parts = split_filename.split('_')
        if len(parts) >= 3:
            date_prefix = parts[1]  # 20150303-110300
            import re
            match = re.search(r'_(\d+)\.jpg$', split_filename)
            if match:
                frame_num = match.group(1)
                for actual_fname, actual_path in all_actual_files[id_prefix].items():
                    if date_prefix in actual_fname and actual_fname.endswith(f'_{frame_num}.jpg'):
                        return actual_path
        
        return None
    
    random.seed(seed)
    
    num_train = 0
    train_by_id = defaultdict(list)
    
    for split_fname in train_list:
        id_prefix = split_fname.split('_')[0]
        src_path = find_file(id_prefix, split_fname)
        
        if src_path is None:
            continue
        
        seq = len(train_by_id[id_prefix])
        train_by_id[id_prefix].append(src_path)
        
        dst_dir = os.path.join(train_out, id_prefix)
        os.makedirs(dst_dir, exist_ok=True)
        out_name = f"{id_prefix}_{seq:04d}.jpg"
        shutil.copy2(src_path, os.path.join(dst_dir, out_name))
        num_train += 1
    
    test_by_id = defaultdict(list)
    for split_fname in test_list:
        id_prefix = split_fname.split('_')[0]
        src_path = find_file(id_prefix, split_fname)
        if src_path:
            test_by_id[id_prefix].append(src_path)
    
    num_test = 0
    
    for id_prefix, file_list in test_by_id.items():
        dst_dir = os.path.join(test_out, id_prefix)
        os.makedirs(dst_dir, exist_ok=True)
        
        for i, src_path in enumerate(file_list):
            out_name = f"{id_prefix}_{i:04d}.jpg"
            shutil.copy2(src_path, os.path.join(dst_dir, out_name))
            num_test += 1
    
    total = num_train + num_test
    expected_total = len(train_list) + len(test_list)
    match_rate = total / expected_total * 100 if expected_total > 0 else 0
    
    print(f"[iPanda50] Processing complete (split {split_idx}, seed={seed}):")
    print(f"  - Train: {num_train} images (official: {len(train_list)})")
    print(f"  - Test: {num_test} images (official: {len(test_list)}, all-vs-all evaluation)")
    print(f"  - Match rate: {match_rate:.1f}%")
    print(f"  - Output directory: {os.path.dirname(train_out)}")
    print(f"\n  [Note] Use tools/evaluate_ipanda50.py --test_dir {test_out} for evaluation.")


# =============================================================================
# CLI
# =============================================================================


def prepare_czechlynx(split_type: str = "time_closed", seed: int = 42) -> None:
    """Prepare CzechLynx according to one of the published protocol columns."""
    import pandas as pd
    random.seed(seed)
    
    csv_path = os.path.join(ORIGINAL_ROOT, "CzechLynxDataset-Metadata-Real.csv")
    base_dir = ORIGINAL_ROOT
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CzechLynx metadata file does not exist: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"[CzechLynx] Total samples: {len(df)}, identities: {df['unique_name'].nunique()}")
    
    split_col = f"split-{split_type}"
    if split_col not in df.columns:
        raise ValueError(f"Invalid split_type: {split_type}. Options: geo_aware, time_open, time_closed, pose")
    
    df = df[df[split_col].notna()]
    print(f"[CzechLynx] Using protocol column {split_col}:")
    print(f"  Train: {len(df[df[split_col] == 'train'])}")
    print(f"  Test: {len(df[df[split_col] == 'test'])}")
    
    train_out = os.path.join(OUTPUT_ROOT, "czechlynx", "train")
    test_out = os.path.join(OUTPUT_ROOT, "czechlynx", "test")
    query_out = os.path.join(OUTPUT_ROOT, "czechlynx", "query")
    gallery_out = os.path.join(OUTPUT_ROOT, "czechlynx", "gallery")
    protocol_info_path = os.path.join(OUTPUT_ROOT, "czechlynx", "protocol_info.json")

    for out_dir in [train_out, test_out, query_out, gallery_out]:
        if os.path.exists(out_dir):
            _safe_rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
    
    all_ids = sorted(df['unique_name'].unique())
    id_to_num = {name: str(i).zfill(4) for i, name in enumerate(all_ids)}
    print(f"[CzechLynx] Total identities: {len(id_to_num)}")
    
    train_df = df[df[split_col] == 'train']
    train_count = 0
    for _, row in train_df.iterrows():
        identity = row['unique_name']
        rel_path = row['path']
        
        src_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(src_path):
            continue
        
        num_id = id_to_num[identity]
        dst_dir = os.path.join(train_out, num_id)
        os.makedirs(dst_dir, exist_ok=True)
        
        filename = os.path.basename(rel_path)
        dst_path = os.path.join(dst_dir, filename)
        shutil.copy2(src_path, dst_path)
        train_count += 1
    
    print(f"[CzechLynx] Train: {train_count} images")
    
    test_df = df[df[split_col] == 'test']
    test_ids = test_df['unique_name'].unique()

    test_count = 0
    query_count = 0
    gallery_count = 0

    for identity in test_ids:
        id_samples = test_df[test_df['unique_name'] == identity]
        sample_list = sorted(id_samples.to_dict('records'), key=lambda item: str(item['path']))

        if len(sample_list) == 0:
            continue

        num_id = id_to_num[identity]

        for sample in sample_list:
            rel_path = sample['path']
            src_path = os.path.join(base_dir, rel_path)
            if os.path.exists(src_path):
                dst_dir = os.path.join(test_out, num_id)
                os.makedirs(dst_dir, exist_ok=True)
                filename = os.path.basename(rel_path)
                dst_path = os.path.join(dst_dir, filename)
                shutil.copy2(src_path, dst_path)
                test_count += 1

        query_sample = sample_list[0]
        rel_path = query_sample['path']
        src_path = os.path.join(base_dir, rel_path)
        if os.path.exists(src_path):
            dst_dir = os.path.join(query_out, num_id)
            os.makedirs(dst_dir, exist_ok=True)
            filename = os.path.basename(rel_path)
            dst_path = os.path.join(dst_dir, filename)
            shutil.copy2(src_path, dst_path)
            query_count += 1
        
        for sample in sample_list[1:]:
            rel_path = sample['path']
            src_path = os.path.join(base_dir, rel_path)
            if os.path.exists(src_path):
                dst_dir = os.path.join(gallery_out, num_id)
                os.makedirs(dst_dir, exist_ok=True)
                filename = os.path.basename(rel_path)
                dst_path = os.path.join(dst_dir, filename)
                shutil.copy2(src_path, dst_path)
                gallery_count += 1
    
    protocol_info = {
        "dataset": "czechlynx",
        "source": "official",
        "split_type": split_type,
        "split_column": split_col,
        "seed": seed,
        "train_dir": train_out,
        "test_dir": test_out,
        "query_dir": query_out,
        "gallery_dir": gallery_out,
        "query_gallery_derivation": {
            "derived_from": "official_test_partition",
            "policy": "deterministic_path_sorted_single_query",
            "query_per_identity": 1,
            "gallery_per_identity": "remaining_test_images",
        },
        "counts": {
            "train_images": train_count,
            "test_images": test_count,
            "query_images": query_count,
            "gallery_images": gallery_count,
            "num_identities": len(id_to_num),
            "num_query_identities": int(query_count),
        },
    }
    with open(protocol_info_path, "w", encoding="utf-8") as f:
        json.dump(protocol_info, f, ensure_ascii=False, indent=2)

    print(f"[CzechLynx] Test: {test_count} images")
    print(f"[CzechLynx] Query: {query_count} images")
    print(f"[CzechLynx] Gallery: {gallery_count} images")
    print(f"[CzechLynx] Protocol info: {protocol_info_path}")
    print(f"[CzechLynx] Output directory: {os.path.join(OUTPUT_ROOT, 'czechlynx')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ReID datasets into identity-directory layout")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "atrw", "duke", "nyala", "lion", "market", "sealid", "wildlife", "leopard", "whaleshark", "coco", "ipanda50", "czechlynx"],
        help="Dataset type to prepare (wildlife = nyala + lion + sealid, coco = leopard + whaleshark)",
    )
    parser.add_argument(
        "--czechlynx_split",
        type=str,
        default="time_closed",
        choices=["geo_aware", "time_open", "time_closed", "pose"],
        help="CzechLynx split protocol (default: time_closed)",
    )

    args = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    if args.dataset in ("all", "atrw"):
        print("========== Preparing Amur Tiger Re-identification ==========")
        prepare_atrw_train()

    if args.dataset in ("all", "duke"):
        print("========== Preparing DukeMTMC-reID ==========")
        prepare_duke()

    if args.dataset in ("all", "nyala", "wildlife"):
        print("========== Preparing Nyala ==========")
        prepare_nyala()

    if args.dataset in ("all", "lion", "wildlife"):
        print("========== Preparing Lion ==========")
        prepare_lion()

    if args.dataset in ("all", "market"):
        print("========== Preparing Market-1501 ==========")
        prepare_market()

    if args.dataset in ("all", "sealid", "wildlife"):
        print("========== Preparing SealID ==========")
        prepare_sealid()

    if args.dataset in ("all", "leopard", "coco"):
        print("========== Preparing Leopard ID ==========")
        prepare_leopard()

    if args.dataset in ("all", "whaleshark", "coco"):
        print("========== Preparing Whale Shark ID ==========")
        prepare_whaleshark()

    if args.dataset in ("all", "ipanda50"):
        print("========== Preparing iPanda50 ==========")
        prepare_ipanda50()

    if args.dataset in ("all", "czechlynx"):
        print("========== Preparing CzechLynx ==========")
        prepare_czechlynx(split_type=args.czechlynx_split)
    
    print("\n========== Preparation complete ==========")
    print(f"Output directory: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
