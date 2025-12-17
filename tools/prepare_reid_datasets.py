#!/usr/bin/env python3
"""ReID 数据预处理脚本

用途：
- 将 Amur Tiger Re-identification 训练集整理成 identity_id/image.jpg 结构
- 将 DukeMTMC-reID 的 train/query/gallery 整理成 identity_id/image.jpg 结构
- 将 Nyala (羚羊) 数据集整理成 identity_id/image.jpg 结构
- 将 Lion (狮子) 数据集整理成 identity_id/image.jpg 结构
- 将 Market-1501 的 train/query/gallery 整理成 identity_id/image.jpg 结构

整理后的目录示例（相对于项目根目录）：

- data/processed/atrw/train/<tiger_id>/<image>.jpg
- data/processed/duke/train/<pid>/<image>.jpg
- data/processed/nyala/train/<nyala_id>/<image>.jpg
- data/processed/nyala/query/<nyala_id>/<image>.jpg
- data/processed/nyala/gallery/<nyala_id>/<image>.jpg
- data/processed/lion/train/<lion_id>/<image>.jpg
- data/processed/lion/query/<lion_id>/<image>.jpg
- data/processed/lion/gallery/<lion_id>/<image>.jpg
- data/processed/market/train/<pid>/<image>.jpg

这样可以直接给：
- `tools/train_joint.py` 作为 `--data_dir` 使用
- `tools/evaluate_reid.py` 作为 `--query_dir` / `--gallery_dir` 使用
"""

import os
import csv
import shutil
import argparse
import re
import random
from collections import defaultdict
from typing import Dict, List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_ROOT = os.path.join(PROJECT_ROOT, "orignal_data")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "data", "processed")


# =============================================================================
# Amur Tiger Re-identification
# =============================================================================

def prepare_atrw_train(seed: int = 42, train_ratio: float = 0.7) -> None:
    """根据官方 csv，把 ATRW 训练图像整理为 identity 目录结构
    
    采用标准 7:3 图像级划分协议（与 SMFFEN 2024 等论文一致）：
    - 每个 identity 内，70% 图像用于训练，30% 图像用于测试
    - 测试集中，每个 identity 抽 1 张作为 query，其余作为 gallery
    - Train 和 Test 完全不重叠
    
    参数:
        seed: 随机种子，确保可复现
        train_ratio: 训练集比例，默认 0.7 (70%)
    """
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

    # 清理旧数据
    for out_dir in [train_out, query_out, gallery_out]:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"ATRW 训练图像目录不存在: {images_dir}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ATRW 训练标注文件不存在: {csv_path}")

    # 统计每个 identity 拥有的图片
    id_to_files: Dict[str, List[str]] = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            # 格式: <id>,<filename>
            if not row:
                continue
            if len(row) < 2:
                continue
            tiger_id = row[0].strip()
            filename = row[1].strip()
            if not tiger_id or not filename:
                continue
            id_to_files[tiger_id].append(filename)

    print(f"[ATRW] 共 {len(id_to_files)} 个 identity")
    
    random.seed(seed)
    
    num_train = 0
    num_query = 0
    num_gallery = 0
    
    for tiger_id, files in id_to_files.items():
        # 打乱文件列表
        files = sorted(files)
        random.shuffle(files)
        
        # 7:3 划分：每个身份内 70% 训练，30% 测试
        n_train = max(1, int(len(files) * train_ratio))  # 至少保留 1 张训练
        n_test = len(files) - n_train
        
        # 确保测试集至少有 2 张（1 query + 1 gallery）
        if n_test < 2 and len(files) >= 3:
            n_train = len(files) - 2
            n_test = 2
        
        train_files = files[:n_train]
        test_files = files[n_train:]
        
        # 复制训练集
        id_train_dir = os.path.join(train_out, tiger_id)
        os.makedirs(id_train_dir, exist_ok=True)
        for fname in train_files:
            src = os.path.join(images_dir, fname)
            if not os.path.exists(src):
                print(f"[ATRW][WARN] 找不到图像: {src}")
                continue
            dst = os.path.join(id_train_dir, fname)
            shutil.copy2(src, dst)
            num_train += 1
        
        # 处理测试集：1 张 query，其余 gallery
        if test_files:
            query_file = test_files[0]
            gallery_files = test_files[1:]
            
            # 复制 query
            id_query_dir = os.path.join(query_out, tiger_id)
            os.makedirs(id_query_dir, exist_ok=True)
            src_q = os.path.join(images_dir, query_file)
            if os.path.exists(src_q):
                shutil.copy2(src_q, os.path.join(id_query_dir, query_file))
                num_query += 1
            
            # 复制 gallery
            if gallery_files:
                id_gallery_dir = os.path.join(gallery_out, tiger_id)
                os.makedirs(id_gallery_dir, exist_ok=True)
                for fname in gallery_files:
                    src_g = os.path.join(images_dir, fname)
                    if os.path.exists(src_g):
                        shutil.copy2(src_g, os.path.join(id_gallery_dir, fname))
                        num_gallery += 1

    print(f"[ATRW] 标准 7:3 划分完成 (seed={seed}):")
    print(f"  - Train: {num_train} 张图像 ({num_train/1887*100:.1f}%)")
    print(f"  - Query: {num_query} 张图像")
    print(f"  - Gallery: {num_gallery} 张图像")
    print(f"  - Test 总计: {num_query + num_gallery} 张 ({(num_query+num_gallery)/1887*100:.1f}%)")
    print(f"  - Train 和 Test 不重叠: ✓")


def build_atrw_query_gallery(seed: int = 42) -> None:
    """[已废弃] 此函数已合并到 prepare_atrw_train() 中
    
    保留此函数是为了兼容性，但不再执行任何操作。
    新的标准 7:3 划分直接在 prepare_atrw_train() 中完成。
    """
    print("[ATRW] build_atrw_query_gallery() 已废弃，划分已在 prepare_atrw_train() 中完成")
    return


def _old_build_atrw_query_gallery(seed: int = 42) -> None:
    """[旧版本-已废弃] 从 processed 的 ATRW train 构建简单的 query/gallery 划分
    
    问题：这种方式会导致 Train 和 Test 数据重叠，不符合标准评估协议。
    请使用新的 prepare_atrw_train() 函数。

    规则：
    - 对每个 identity，在其 train 目录中打乱文件列表
    - 抽 1 张作为 query
    - 其余作为 gallery（如果只有 1 张，则该 identity 在 gallery 中缺失）
    """
    train_root = os.path.join(OUTPUT_ROOT, "atrw", "train")
    query_root = os.path.join(OUTPUT_ROOT, "atrw", "query")
    gallery_root = os.path.join(OUTPUT_ROOT, "atrw", "gallery")

    if not os.path.exists(train_root):
        print(f"[ATRW][WARN] train 目录不存在，跳过 query/gallery 构建: {train_root}")
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

        # 抽 1 张为 query
        query_file = files[0]
        gallery_files = files[1:]

        # 复制 query
        src_q = os.path.join(id_train_dir, query_file)
        id_query_dir = os.path.join(query_root, tiger_id)
        os.makedirs(id_query_dir, exist_ok=True)
        dst_q = os.path.join(id_query_dir, query_file)
        shutil.copy2(src_q, dst_q)
        num_query += 1

        # 复制 gallery
        if gallery_files:
            id_gallery_dir = os.path.join(gallery_root, tiger_id)
            os.makedirs(id_gallery_dir, exist_ok=True)
            for fname in gallery_files:
                src_g = os.path.join(id_train_dir, fname)
                dst_g = os.path.join(id_gallery_dir, fname)
                shutil.copy2(src_g, dst_g)
                num_gallery += 1

    print(
        f"[ATRW] query/gallery 构建完成: query={num_query}, gallery={num_gallery} "
        f"→ query: {query_root}, gallery: {gallery_root}"
    )


# =============================================================================
# DukeMTMC-reID
# =============================================================================

PID_PATTERN = re.compile(r"^([\-0-9]{4})_")  # 如 0002_c1s1_... 或 -1_c1s1_...


def _collect_duke_split(split_name: str, src_dir: str, out_root: str) -> None:
    """将 Duke 某个 split（train/query/gallery）整理为 identity 目录结构"""
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Duke split 目录不存在: {src_dir}")

    out_dir = os.path.join(out_root, split_name)
    os.makedirs(out_dir, exist_ok=True)

    num_files = 0
    skipped = 0

    for fname in os.listdir(src_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        m = PID_PATTERN.match(fname)
        if not m:
            print(f"[Duke][WARN] 文件名无法解析 PID: {fname}")
            skipped += 1
            continue
        pid_str = m.group(1)
        # PID 为 -1 表示垃圾帧，跳过
        if pid_str == "-1":
            skipped += 1
            continue
        # 去掉左侧 0，例如 0002 -> 2，这样更通用，也可保留原样
        pid = str(int(pid_str))

        id_dir = os.path.join(out_dir, pid)
        os.makedirs(id_dir, exist_ok=True)

        src = os.path.join(src_dir, fname)
        dst = os.path.join(id_dir, fname)
        shutil.copy2(src, dst)
        num_files += 1

    print(
        f"[Duke] split={split_name}: 复制 {num_files} 张图像到 {out_dir} (跳过 {skipped} 张)"
    )


def prepare_duke() -> None:
    """整理 DukeMTMC-reID 的 train/query/gallery"""
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
# Nyala (羚羊) Dataset
# =============================================================================

def prepare_nyala(seed: int = 42) -> None:
    """整理 Nyala 羚羊数据集
    
    原始结构: Nyala_Data_Zero/train/<id>/*.jpg, Nyala_Data_Zero/test/<id>/*.jpg
    目标结构: nyala/train/<id>/*.jpg, nyala/query/<id>/*.jpg, nyala/gallery/<id>/*.jpg
    
    策略:
    - train 目录直接复制
    - test 目录: 每个 identity 抽 1 张作为 query，其余作为 gallery
    """
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
        print(f"[Nyala][WARN] 数据集目录不存在: {nyala_root}")
        return
    
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(query_out, exist_ok=True)
    os.makedirs(gallery_out, exist_ok=True)
    
    # 1. 复制 train 目录
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
    
    print(f"[Nyala] 训练集: {num_train} 张图像 → {train_out}")
    
    # 2. 处理 test 目录 -> query + gallery
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
            
            # 抽 1 张为 query
            query_file = files[0]
            gallery_files = files[1:]
            
            # 复制 query
            id_query_dir = os.path.join(query_out, identity)
            os.makedirs(id_query_dir, exist_ok=True)
            shutil.copy2(
                os.path.join(id_src, query_file),
                os.path.join(id_query_dir, query_file)
            )
            num_query += 1
            
            # 复制 gallery
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
# Lion (狮子) Dataset
# =============================================================================

def prepare_lion(seed: int = 42) -> None:
    """整理 Lion 狮子数据集
    
    原始结构: Lion_Data_Zero/train/<id>/*.jpg, Lion_Data_Zero/val/<id>/*.jpg
    目标结构: lion/train/<id>/*.jpg, lion/query/<id>/*.jpg, lion/gallery/<id>/*.jpg
    
    策略:
    - train 目录直接复制用于训练
    - val 目录: 每个 identity 抽 1 张作为 query，其余作为 gallery
    """
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
        print(f"[Lion][WARN] 数据集目录不存在: {lion_root}")
        return
    
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(query_out, exist_ok=True)
    os.makedirs(gallery_out, exist_ok=True)
    
    # 1. 复制 train 目录
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
    
    print(f"[Lion] 训练集: {num_train} 张图像 → {train_out}")
    
    # 2. 处理 val 目录 -> query + gallery
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
            
            # 抽 1 张为 query
            query_file = files[0]
            gallery_files = files[1:]
            
            # 复制 query
            id_query_dir = os.path.join(query_out, identity)
            os.makedirs(id_query_dir, exist_ok=True)
            shutil.copy2(
                os.path.join(id_src, query_file),
                os.path.join(id_query_dir, query_file)
            )
            num_query += 1
            
            # 复制 gallery
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

MARKET_PID_PATTERN = re.compile(r"^([\-0-9]{4})_")  # 如 0001_c1s1_...


def prepare_market() -> None:
    """整理 Market-1501 的 train/query/gallery"""
    market_root = os.path.join(ORIGINAL_ROOT, "Market-1501-v15.09.15")
    train_src = os.path.join(market_root, "bounding_box_train")
    query_src = os.path.join(market_root, "query")
    gallery_src = os.path.join(market_root, "bounding_box_test")
    
    out_root = os.path.join(OUTPUT_ROOT, "market")
    
    if not os.path.exists(market_root):
        print(f"[Market][WARN] 数据集目录不存在: {market_root}")
        return
    
    os.makedirs(out_root, exist_ok=True)
    
    _collect_market_split("train", train_src, out_root)
    _collect_market_split("query", query_src, out_root)
    _collect_market_split("gallery", gallery_src, out_root)


def _collect_market_split(split_name: str, src_dir: str, out_root: str) -> None:
    """将 Market-1501 某个 split 整理为 identity 目录结构"""
    if not os.path.exists(src_dir):
        print(f"[Market][WARN] split 目录不存在: {src_dir}")
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
        # PID 为 -1 或 0000 表示垃圾帧/干扰项，跳过
        if pid_str in ("-1", "0000", "-001"):
            skipped += 1
            continue
        pid = str(int(pid_str))
        
        id_dir = os.path.join(out_dir, pid)
        os.makedirs(id_dir, exist_ok=True)
        
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(id_dir, fname))
        num_files += 1
    
    print(f"[Market] split={split_name}: {num_files} 张图像 → {out_dir} (跳过 {skipped})")


# =============================================================================
# SealID (海豹) Dataset
# =============================================================================

def prepare_sealid() -> None:
    """整理 SealID 海豹数据集
    
    原始结构:
    - SealID/patches/patches/source/<filename>.png  (所有图像)
    - SealID/patches/patches/annotation.csv  (class_id,file,split,testing_split)
    
    annotation.csv 格式:
    - class_id: "XXX_Y" 其中 XXX 是海豹ID，Y 是位置编号 (如 "096_1")
    - file: 图像文件名 (如 "curxlu.png")
    - split: "training" 或 "testing"
    - testing_split: "database" (gallery) 或 "query" (仅 testing 有效)
    
    目标结构:
    - sealid/train/<seal_id>/*.png
    - sealid/query/<seal_id>/*.png
    - sealid/gallery/<seal_id>/*.png
    """
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
        print(f"[SealID][WARN] 数据集目录不存在: {sealid_root}")
        return
    
    if not os.path.exists(annotation_file):
        print(f"[SealID][WARN] annotation.csv 不存在: {annotation_file}")
        return
    
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(query_out, exist_ok=True)
    os.makedirs(gallery_out, exist_ok=True)
    
    num_train = 0
    num_query = 0
    num_gallery = 0
    skipped = 0
    
    # 读取 annotation.csv
    with open(annotation_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头: class_id,file,split,testing_split
        
        for row in reader:
            if len(row) < 4:
                skipped += 1
                continue
            
            class_id, filename, split, testing_split = row[0], row[1], row[2], row[3]
            
            # 从 class_id 提取 seal_id (去掉位置后缀)
            # 格式: "XXX_Y" -> "XXX"
            if '_' in class_id:
                seal_id = class_id.split('_')[0]
            else:
                seal_id = class_id
            
            # 源图像路径
            src_path = os.path.join(source_dir, filename)
            if not os.path.exists(src_path):
                skipped += 1
                continue
            
            # 根据 split 决定目标目录
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
    
    # 统计
    train_ids = len(os.listdir(train_out)) if os.path.exists(train_out) else 0
    query_ids = len(os.listdir(query_out)) if os.path.exists(query_out) else 0
    gallery_ids = len(os.listdir(gallery_out)) if os.path.exists(gallery_out) else 0
    
    print(f"[SealID] train: {num_train} 张图像, {train_ids} 个身份")
    print(f"[SealID] query: {num_query} 张图像, {query_ids} 个身份")
    print(f"[SealID] gallery: {num_gallery} 张图像, {gallery_ids} 个身份")
    print(f"[SealID] 跳过: {skipped} 条记录")


# =============================================================================
# Leopard ID Dataset (Wild Me COCO Format)
# =============================================================================

def prepare_leopard(seed: int = 42, train_ratio: float = 0.7) -> None:
    """整理 Leopard ID 豹子数据集 (Wild Me COCO 格式)
    
    原始结构: leopard.coco/annotations/instances_train2022.json + images/
    COCO 格式字段:
    - images: id, file_name, uuid
    - annotations: id, image_id, bbox, name (个体UUID)
    
    目标结构: leopard/train/<id>/*.jpg, leopard/query/<id>/*.jpg, leopard/gallery/<id>/*.jpg
    
    策略 (因为原始只有 train 集):
    - 按个体划分: 70% 个体用于训练，30% 个体用于测试
    - 测试个体: 每个抽 1 张作为 query，其余作为 gallery
    """
    import json
    from PIL import Image
    
    leopard_root = os.path.join(ORIGINAL_ROOT, "leopard.coco")
    anno_path = os.path.join(leopard_root, "annotations", "instances_train2022.json")
    # 图像在 images/train2022/ 子目录下
    images_dir = os.path.join(leopard_root, "images", "train2022")
    
    out_root = os.path.join(OUTPUT_ROOT, "leopard")
    train_out = os.path.join(out_root, "train")
    query_out = os.path.join(out_root, "query")
    gallery_out = os.path.join(out_root, "gallery")
    
    if not os.path.exists(anno_path):
        print(f"[Leopard][WARN] 标注文件不存在: {anno_path}")
        return
    
    print("[Leopard] 加载标注文件...")
    with open(anno_path, 'r') as f:
        data = json.load(f)
    
    # 构建 image_id -> filename 映射
    id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    
    # 按个体收集标注
    individual_annotations = defaultdict(list)
    for ann in data['annotations']:
        if 'name' not in ann or not ann['name']:
            continue
        individual_id = ann['name']  # UUID 格式
        individual_annotations[individual_id].append(ann)
    
    print(f"[Leopard] 共 {len(individual_annotations)} 个个体, {len(data['annotations'])} 个标注")
    
    # 按个体划分 train/test
    random.seed(seed)
    all_individuals = sorted(individual_annotations.keys())
    random.shuffle(all_individuals)
    
    n_train = int(len(all_individuals) * train_ratio)
    train_individuals = set(all_individuals[:n_train])
    test_individuals = set(all_individuals[n_train:])
    
    print(f"[Leopard] 训练个体: {len(train_individuals)}, 测试个体: {len(test_individuals)}")
    
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(query_out, exist_ok=True)
    os.makedirs(gallery_out, exist_ok=True)
    
    num_train = 0
    num_query = 0
    num_gallery = 0
    skipped = 0
    
    # 为每个个体分配数字ID (更短的目录名)
    individual_to_numid = {ind: str(i) for i, ind in enumerate(sorted(all_individuals))}
    
    def crop_and_save(ann, out_path):
        """裁剪 bbox 区域并保存"""
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
            # 确保边界框在图像范围内
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
            print(f"[Leopard][WARN] 裁剪失败 {filename}: {e}")
            return False
    
    # 处理训练个体
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
    
    # 处理测试个体 -> query + gallery
    for ind_id in test_individuals:
        annotations = individual_annotations[ind_id]
        num_id = individual_to_numid[ind_id]
        
        if len(annotations) == 0:
            continue
        
        # 打乱并划分
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
    
    print(f"[Leopard] train: {num_train} 张图像, {len(train_individuals)} 个身份")
    print(f"[Leopard] query: {num_query} 张图像")
    print(f"[Leopard] gallery: {num_gallery} 张图像")
    print(f"[Leopard] 跳过: {skipped} 个标注")


# =============================================================================
# Whale Shark ID Dataset (Wild Me COCO Format)
# =============================================================================

def prepare_whaleshark(seed: int = 42, train_ratio: float = 0.7) -> None:
    """整理 Whale Shark ID 鲸鲨数据集 (Wild Me COCO 格式)
    
    原始结构: whaleshark.coco/whaleshark.coco/annotations/instances_train2020.json + images/
    COCO 格式字段:
    - images: id, file_name, uuid
    - annotations: id, image_id, bbox, name (个体UUID)
    
    目标结构: whaleshark/train/<id>/*.jpg, whaleshark/query/<id>/*.jpg, whaleshark/gallery/<id>/*.jpg
    
    策略 (因为原始只有 train 集):
    - 按个体划分: 70% 个体用于训练，30% 个体用于测试
    - 测试个体: 每个抽 1 张作为 query，其余作为 gallery
    """
    import json
    from PIL import Image
    
    whaleshark_root = os.path.join(ORIGINAL_ROOT, "whaleshark.coco", "whaleshark.coco")
    anno_path = os.path.join(whaleshark_root, "annotations", "instances_train2020.json")
    # 图像在 images/train2020/ 子目录下
    images_dir = os.path.join(whaleshark_root, "images", "train2020")
    
    out_root = os.path.join(OUTPUT_ROOT, "whaleshark")
    train_out = os.path.join(out_root, "train")
    query_out = os.path.join(out_root, "query")
    gallery_out = os.path.join(out_root, "gallery")
    
    if not os.path.exists(anno_path):
        print(f"[WhaleShark][WARN] 标注文件不存在: {anno_path}")
        return
    
    print("[WhaleShark] 加载标注文件...")
    with open(anno_path, 'r') as f:
        data = json.load(f)
    
    # 构建 image_id -> filename 映射
    id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    
    # 按个体收集标注
    individual_annotations = defaultdict(list)
    for ann in data['annotations']:
        if 'name' not in ann or not ann['name']:
            continue
        individual_id = ann['name']  # UUID 格式
        individual_annotations[individual_id].append(ann)
    
    print(f"[WhaleShark] 共 {len(individual_annotations)} 个个体, {len(data['annotations'])} 个标注")
    
    # 按个体划分 train/test
    random.seed(seed)
    all_individuals = sorted(individual_annotations.keys())
    random.shuffle(all_individuals)
    
    n_train = int(len(all_individuals) * train_ratio)
    train_individuals = set(all_individuals[:n_train])
    test_individuals = set(all_individuals[n_train:])
    
    print(f"[WhaleShark] 训练个体: {len(train_individuals)}, 测试个体: {len(test_individuals)}")
    
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(query_out, exist_ok=True)
    os.makedirs(gallery_out, exist_ok=True)
    
    num_train = 0
    num_query = 0
    num_gallery = 0
    skipped = 0
    
    # 为每个个体分配数字ID (更短的目录名)
    individual_to_numid = {ind: str(i) for i, ind in enumerate(sorted(all_individuals))}
    
    def crop_and_save(ann, out_path):
        """裁剪 bbox 区域并保存"""
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
            # 确保边界框在图像范围内
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
            print(f"[WhaleShark][WARN] 裁剪失败 {filename}: {e}")
            return False
    
    # 处理训练个体
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
    
    # 处理测试个体 -> query + gallery
    for ind_id in test_individuals:
        annotations = individual_annotations[ind_id]
        num_id = individual_to_numid[ind_id]
        
        if len(annotations) == 0:
            continue
        
        # 打乱并划分
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
    
    print(f"[WhaleShark] train: {num_train} 张图像, {len(train_individuals)} 个身份")
    print(f"[WhaleShark] query: {num_query} 张图像")
    print(f"[WhaleShark] gallery: {num_gallery} 张图像")
    print(f"[WhaleShark] 跳过: {skipped} 个标注")


# =============================================================================
# iPanda50 (大熊猫)
# =============================================================================

def prepare_ipanda50(seed: int = 42, split_idx: int = 0) -> None:
    """处理 iPanda50 数据集 - 严格遵循官方协议
    
    官方协议 (TIP 2021):
    - Train:Test = 3:2 (4106:2768)
    - 5次蒙特卡洛试验 (split0-4)
    - 报告5次平均 ± 标准差
    - **评估方式: Test 集内 All-vs-All 互检索**
    
    处理流程:
    1. 按官方 split 文件划分 train/test
    2. Test 集整体作为评估集（All-vs-All，不再划分 query/gallery）
    3. 输出文件统一命名为 {id}_{序号}.jpg (解决乱码问题)
    
    参数:
        seed: 随机种子
        split_idx: 使用哪个官方划分 (0-4)，默认使用 split0
    """
    images_root = os.path.join(ORIGINAL_ROOT, "iPanda50", "iPanda50-images")
    split_dir = os.path.join(ORIGINAL_ROOT, "iPanda50", "iPanda50-split")
    
    train_split_file = os.path.join(split_dir, f"split{split_idx}_train.txt")
    test_split_file = os.path.join(split_dir, f"split{split_idx}_test.txt")
    
    train_out = os.path.join(OUTPUT_ROOT, "ipanda50", "train")
    test_out = os.path.join(OUTPUT_ROOT, "ipanda50", "test")  # 官方协议：整个 test 集用于评估
    
    # 检查源目录
    if not os.path.exists(images_root):
        raise FileNotFoundError(f"iPanda50 图像目录不存在: {images_root}")
    if not os.path.exists(train_split_file):
        raise FileNotFoundError(f"iPanda50 划分文件不存在: {train_split_file}")
    
    # 清理旧数据
    for out_dir in [train_out, test_out]:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
    
    # 兼容旧目录结构：也清理 query/gallery（如果存在）
    for old_dir in ["query", "gallery"]:
        old_path = os.path.join(OUTPUT_ROOT, "ipanda50", old_dir)
        if os.path.exists(old_path):
            shutil.rmtree(old_path)
            print(f"[iPanda50] 已删除旧目录: {old_path}")
    
    # 获取所有身份文件夹
    identity_dirs = sorted([d for d in os.listdir(images_root) 
                           if os.path.isdir(os.path.join(images_root, d)) and not d.startswith('.')])
    print(f"[iPanda50] 共 {len(identity_dirs)} 个身份")
    
    # 建立 identity_id -> 文件夹名 的映射
    id_to_dir = {}
    for id_dir in identity_dirs:
        id_prefix = id_dir.split('_')[0]
        id_to_dir[id_prefix] = id_dir
    
    # 建立 (实际文件夹中) 所有文件的索引
    # 因为 Windows 乱码，需要遍历实际文件来匹配
    all_actual_files = {}  # {id_prefix: {filename: full_path}}
    for id_prefix, id_dir in id_to_dir.items():
        dir_path = os.path.join(images_root, id_dir)
        all_actual_files[id_prefix] = {}
        for fname in os.listdir(dir_path):
            if fname.endswith('.jpg'):
                all_actual_files[id_prefix][fname] = os.path.join(dir_path, fname)
    
    # 读取官方 split
    def read_split_file(filepath):
        """读取 split 文件，返回文件名列表"""
        files = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    files.append(line)
        return files
    
    train_list = read_split_file(train_split_file)
    test_list = read_split_file(test_split_file)
    
    print(f"[iPanda50] Split {split_idx}: 官方 Train {len(train_list)} 张, Test {len(test_list)} 张")
    
    def find_file(id_prefix: str, split_filename: str) -> str:
        """在实际文件中查找匹配的文件路径"""
        if id_prefix not in all_actual_files:
            return None
        
        # 1. 直接匹配
        if split_filename in all_actual_files[id_prefix]:
            return all_actual_files[id_prefix][split_filename]
        
        # 2. 模糊匹配 - 提取日期和帧号
        # split: 49_20150303-110300_幼年_xxx_925.jpg
        # actual: 49_20150303-110300_乱码_925.jpg
        parts = split_filename.split('_')
        if len(parts) >= 3:
            date_prefix = parts[1]  # 20150303-110300
            # 提取最后的数字（帧号）
            import re
            match = re.search(r'_(\d+)\.jpg$', split_filename)
            if match:
                frame_num = match.group(1)
                # 在实际文件中查找同日期同帧号的文件
                for actual_fname, actual_path in all_actual_files[id_prefix].items():
                    if date_prefix in actual_fname and actual_fname.endswith(f'_{frame_num}.jpg'):
                        return actual_path
        
        return None
    
    random.seed(seed)
    
    # 处理训练集
    num_train = 0
    train_by_id = defaultdict(list)  # 按 ID 分组计数，用于生成序号
    
    for split_fname in train_list:
        id_prefix = split_fname.split('_')[0]
        src_path = find_file(id_prefix, split_fname)
        
        if src_path is None:
            continue
        
        # 输出: {id_prefix}/{id_prefix}_{序号}.jpg
        seq = len(train_by_id[id_prefix])
        train_by_id[id_prefix].append(src_path)
        
        dst_dir = os.path.join(train_out, id_prefix)
        os.makedirs(dst_dir, exist_ok=True)
        out_name = f"{id_prefix}_{seq:04d}.jpg"
        shutil.copy2(src_path, os.path.join(dst_dir, out_name))
        num_train += 1
    
    # 处理测试集: 官方协议 - 整个 test 集用于 All-vs-All 评估
    # 不再划分 query/gallery，所有测试图像放入 test 目录
    test_by_id = defaultdict(list)
    for split_fname in test_list:
        id_prefix = split_fname.split('_')[0]
        src_path = find_file(id_prefix, split_fname)
        if src_path:
            test_by_id[id_prefix].append(src_path)
    
    num_test = 0
    
    # 官方协议: 所有测试图像放入 test 目录，评估时 All-vs-All 互检索
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
    
    print(f"[iPanda50] 处理完成 (Split {split_idx}, seed={seed}):")
    print(f"  - Train: {num_train} 张 (官方: {len(train_list)})")
    print(f"  - Test: {num_test} 张 (官方: {len(test_list)}, All-vs-All 评估)")
    print(f"  - 匹配率: {match_rate:.1f}%")
    print(f"  - 输出目录: {os.path.dirname(train_out)}")
    print(f"\n  [注意] 评估时使用 tools/evaluate_ipanda50.py --test_dir {test_out}")


# =============================================================================
# CLI
# =============================================================================


def prepare_czechlynx(split_type: str = "time_closed", seed: int = 42) -> None:
    """处理 CzechLynx 数据集
    
    CzechLynx 官方提供了4种 split 协议：
    - split-geo_aware: 地理感知划分（train 21763, test 17997）
    - split-time_open: 时间开放划分（train 27587, test 12173）
    - split-time_closed: 时间封闭划分（train 27836, test 11924）- 推荐用于 Closed-Set ReID
    - split-pose: 姿态划分（train 2080, test 3126）- 较小子集
    
    参数:
        split_type: 使用的 split 协议，可选 "geo_aware", "time_open", "time_closed", "pose"
        seed: 随机种子
    
    参考:
    - Picek et al., "CzechLynx: A Large-Scale Dataset for Lynx Re-Identification" (2024)
    - Kaggle: https://www.kaggle.com/datasets/picekl/czechlynx/
    """
    import pandas as pd
    random.seed(seed)
    
    csv_path = os.path.join(ORIGINAL_ROOT, "CzechLynxDataset-Metadata-Real.csv")
    base_dir = ORIGINAL_ROOT  # path 列以 "CzechLynx/" 开头
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CzechLynx 元数据文件不存在: {csv_path}")
    
    # 读取元数据
    df = pd.read_csv(csv_path)
    print(f"[CzechLynx] 总样本: {len(df)}, 身份数: {df['unique_name'].nunique()}")
    
    # 选择 split 列
    split_col = f"split-{split_type}"
    if split_col not in df.columns:
        raise ValueError(f"无效的 split_type: {split_type}. 可选: geo_aware, time_open, time_closed, pose")
    
    # 过滤掉没有 split 标注的样本
    df = df[df[split_col].notna()]
    print(f"[CzechLynx] 使用 {split_col} 协议:")
    print(f"  Train: {len(df[df[split_col] == 'train'])}")
    print(f"  Test: {len(df[df[split_col] == 'test'])}")
    
    # 输出目录
    train_out = os.path.join(OUTPUT_ROOT, "czechlynx", "train")
    query_out = os.path.join(OUTPUT_ROOT, "czechlynx", "query")
    gallery_out = os.path.join(OUTPUT_ROOT, "czechlynx", "gallery")
    
    for out_dir in [train_out, query_out, gallery_out]:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
    
    # 创建身份 ID 映射（lynx_XXX -> 数字ID）
    all_ids = sorted(df['unique_name'].unique())
    id_to_num = {name: str(i).zfill(4) for i, name in enumerate(all_ids)}
    print(f"[CzechLynx] 共 {len(id_to_num)} 个身份")
    
    # 处理训练集
    train_df = df[df[split_col] == 'train']
    train_count = 0
    for _, row in train_df.iterrows():
        identity = row['unique_name']
        rel_path = row['path']  # 如 "CzechLynx/foe_carpaths/lynx_296/00000_lynx_296.jpg"
        
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
    
    print(f"[CzechLynx] Train: {train_count} 张图像")
    
    # 处理测试集：每个 ID 抽 1 张 query，其余 gallery
    test_df = df[df[split_col] == 'test']
    test_ids = test_df['unique_name'].unique()
    
    query_count = 0
    gallery_count = 0
    
    for identity in test_ids:
        id_samples = test_df[test_df['unique_name'] == identity]
        sample_list = id_samples.to_dict('records')
        
        if len(sample_list) == 0:
            continue
        
        random.shuffle(sample_list)
        
        num_id = id_to_num[identity]
        
        # 第一张作为 query
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
        
        # 其余作为 gallery
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
    
    print(f"[CzechLynx] Query: {query_count} 张图像")
    print(f"[CzechLynx] Gallery: {gallery_count} 张图像")
    print(f"[CzechLynx] 输出目录: {os.path.join(OUTPUT_ROOT, 'czechlynx')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="预处理 ReID 数据集为 identity 目录结构")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "atrw", "duke", "nyala", "lion", "market", "sealid", "wildlife", "leopard", "whaleshark", "coco", "ipanda50", "czechlynx"],
        help="要处理的数据集类型 (wildlife = nyala + lion + sealid, coco = leopard + whaleshark)",
    )
    parser.add_argument(
        "--czechlynx_split",
        type=str,
        default="time_closed",
        choices=["geo_aware", "time_open", "time_closed", "pose"],
        help="CzechLynx 使用的 split 协议 (默认: time_closed)",
    )

    args = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    if args.dataset in ("all", "atrw"):
        print("========== 处理 Amur Tiger Re-identification ==========")
        prepare_atrw_train()

    if args.dataset in ("all", "duke"):
        print("========== 处理 DukeMTMC-reID ==========")
        prepare_duke()

    if args.dataset in ("all", "nyala", "wildlife"):
        print("========== 处理 Nyala (羚羊) ==========")
        prepare_nyala()

    if args.dataset in ("all", "lion", "wildlife"):
        print("========== 处理 Lion (狮子) ==========")
        prepare_lion()

    if args.dataset in ("all", "market"):
        print("========== 处理 Market-1501 ==========")
        prepare_market()

    if args.dataset in ("all", "sealid", "wildlife"):
        print("========== 处理 SealID (海豹) ==========")
        prepare_sealid()

    if args.dataset in ("all", "leopard", "coco"):
        print("========== 处理 Leopard ID (豹子) ==========")
        prepare_leopard()

    if args.dataset in ("all", "whaleshark", "coco"):
        print("========== 处理 Whale Shark ID (鲸鲨) ==========")
        prepare_whaleshark()

    if args.dataset in ("all", "ipanda50"):
        print("========== 处理 iPanda50 (大熊猫) ==========")
        prepare_ipanda50()

    if args.dataset in ("all", "czechlynx"):
        print("========== 处理 CzechLynx (猞猁) ==========")
        prepare_czechlynx(split_type=args.czechlynx_split)
    
    print("\n========== 处理完成 ==========")
    print(f"输出目录: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
