"""
StripeSpotter (Zebra) 数据预处理脚本
转换为 ReID 标准格式: train/query/gallery 目录结构

数据集特点:
- 824 条记录, 45 个斑马个体
- 已提供 ROI 裁剪图像 (roi-*.jpg)
- 包含左右侧面信息 (flank)
- 数据质量标注 (photo_quality)
"""

import os
import shutil
import random
import pandas as pd
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import argparse


def load_sighting_data(csv_path):
    """加载 SightingData.csv"""
    # 读取时跳过注释行
    df = pd.read_csv(csv_path, comment='#', header=None,
                     names=['imgindex', 'original_filepath', 'roi', 'animal_name',
                            'sighting_id', 'flank', 'notes', 'photo_quality',
                            'sighting_date', 'sighting_time', 'exposure_time',
                            'focal_length', 'aperture', 'camera_info'])
    
    # 按个体分组
    identity_images = defaultdict(list)
    for _, row in df.iterrows():
        animal_name = str(row['animal_name']).strip()
        if not animal_name or animal_name == 'nan':
            continue
        
        imgindex = int(row['imgindex'])
        roi_filename = f"roi-{imgindex:07d}.jpg"
        
        identity_images[animal_name].append({
            'imgindex': imgindex,
            'roi_filename': roi_filename,
            'flank': row['flank'] if pd.notna(row['flank']) else 'unknown',
            'quality': row['photo_quality'] if pd.notna(row['photo_quality']) else 'unknown'
        })
    
    return identity_images


def adaptive_split(n_images):
    """
    自适应划分策略
    - 样本>=10张: 标准 70/15/15
    - 样本 5-9张: 80/10/10
    - 样本 3-4张: 复用模式 (全部train, query/gallery复用)
    - 样本 <3张: 跳过
    """
    if n_images >= 10:
        n_train = int(n_images * 0.70)
        n_query = max(1, int(n_images * 0.15))
        n_gallery = n_images - n_train - n_query
        return n_train, n_query, n_gallery, 'standard'
    elif n_images >= 5:
        n_train = int(n_images * 0.80)
        n_query = max(1, int(n_images * 0.10))
        n_gallery = n_images - n_train - n_query
        if n_gallery < 1:
            n_train -= 1
            n_gallery = 1
        return n_train, n_query, n_gallery, 'compact'
    elif n_images >= 3:
        # 复用模式: 全部用于train, query和gallery各复用1张
        return n_images, 1, 1, 'reuse'
    else:
        return 0, 0, 0, 'skip'


def process_identity(args):
    """处理单个身份的所有图像"""
    identity, images, src_dir, output_dir = args
    
    n_images = len(images)
    n_train, n_query, n_gallery, mode = adaptive_split(n_images)
    
    if mode == 'skip':
        return identity, 0, 0, 0, 'skipped'
    
    # 打乱图像顺序
    random.shuffle(images)
    
    # 创建身份目录
    train_dir = os.path.join(output_dir, 'train', identity)
    query_dir = os.path.join(output_dir, 'query', identity)
    gallery_dir = os.path.join(output_dir, 'gallery', identity)
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(gallery_dir, exist_ok=True)
    
    if mode == 'reuse':
        # 复用模式: 全部作为train, query/gallery复用前两张
        for i, img_info in enumerate(images):
            src_path = os.path.join(src_dir, img_info['roi_filename'])
            if not os.path.exists(src_path):
                continue
            
            # 命名: identity_camid_seqid.jpg
            dst_name = f"{identity}_c1_{i:04d}.jpg"
            
            # 复制到train
            shutil.copy2(src_path, os.path.join(train_dir, dst_name))
            
            # 前两张也复制到query和gallery
            if i == 0:
                shutil.copy2(src_path, os.path.join(query_dir, dst_name))
            elif i == 1:
                shutil.copy2(src_path, os.path.join(gallery_dir, dst_name))
        
        if n_images == 3:
            # 只有3张时,第三张也放gallery
            img_info = images[2]
            src_path = os.path.join(src_dir, img_info['roi_filename'])
            if os.path.exists(src_path):
                dst_name = f"{identity}_c1_0002.jpg"
                shutil.copy2(src_path, os.path.join(gallery_dir, dst_name))
        
        return identity, n_images, 1, min(2, n_images-1), mode
    
    else:
        # 标准划分
        train_images = images[:n_train]
        query_images = images[n_train:n_train + n_query]
        gallery_images = images[n_train + n_query:]
        
        actual_train = 0
        actual_query = 0
        actual_gallery = 0
        
        # 复制训练图像
        for i, img_info in enumerate(train_images):
            src_path = os.path.join(src_dir, img_info['roi_filename'])
            if not os.path.exists(src_path):
                continue
            dst_name = f"{identity}_c1_{i:04d}.jpg"
            shutil.copy2(src_path, os.path.join(train_dir, dst_name))
            actual_train += 1
        
        # 复制query图像
        for i, img_info in enumerate(query_images):
            src_path = os.path.join(src_dir, img_info['roi_filename'])
            if not os.path.exists(src_path):
                continue
            dst_name = f"{identity}_c2_{i:04d}.jpg"
            shutil.copy2(src_path, os.path.join(query_dir, dst_name))
            actual_query += 1
        
        # 复制gallery图像
        for i, img_info in enumerate(gallery_images):
            src_path = os.path.join(src_dir, img_info['roi_filename'])
            if not os.path.exists(src_path):
                continue
            dst_name = f"{identity}_c3_{i:04d}.jpg"
            shutil.copy2(src_path, os.path.join(gallery_dir, dst_name))
            actual_gallery += 1
        
        return identity, actual_train, actual_query, actual_gallery, mode


def main():
    parser = argparse.ArgumentParser(description='Preprocess StripeSpotter dataset')
    parser.add_argument('--input_dir', type=str, 
                        default='orignal_data/StripeSpotter/data',
                        help='Input directory containing SightingData.csv and images/')
    parser.add_argument('--output_dir', type=str,
                        default='data/processed/stripespotter',
                        help='Output directory for processed data')
    parser.add_argument('--min_samples', type=int, default=3,
                        help='Minimum samples per identity')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # 路径设置
    csv_path = os.path.join(args.input_dir, 'SightingData.csv')
    images_dir = os.path.join(args.input_dir, 'images')
    
    print("=" * 60)
    print("StripeSpotter Dataset Preprocessing")
    print("=" * 60)
    
    # 加载数据
    print("\n[1/3] Loading sighting data...")
    identity_images = load_sighting_data(csv_path)
    print(f"Found {len(identity_images)} unique identities")
    print(f"Total images: {sum(len(v) for v in identity_images.values())}")
    
    # 过滤小样本个体
    valid_identities = {k: v for k, v in identity_images.items() 
                        if len(v) >= args.min_samples}
    skipped = len(identity_images) - len(valid_identities)
    print(f"\nFiltered: {len(valid_identities)} identities (skipped {skipped} with <{args.min_samples} samples)")
    
    # 创建输出目录
    print("\n[2/3] Creating output directories...")
    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'query'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'gallery'), exist_ok=True)
    
    # 处理每个身份
    print("\n[3/3] Processing identities...")
    results = []
    stats = {'standard': 0, 'compact': 0, 'reuse': 0, 'skipped': 0}
    
    for identity, images in tqdm(valid_identities.items()):
        result = process_identity((identity, images, images_dir, args.output_dir))
        results.append(result)
        stats[result[4]] += 1
    
    # 统计结果
    total_train = sum(r[1] for r in results)
    total_query = sum(r[2] for r in results)
    total_gallery = sum(r[3] for r in results)
    valid_ids = sum(1 for r in results if r[4] != 'skipped')
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"\nValid identities: {valid_ids}")
    print(f"  - Standard split (>=10 samples): {stats['standard']}")
    print(f"  - Compact split (5-9 samples): {stats['compact']}")
    print(f"  - Reuse mode (3-4 samples): {stats['reuse']}")
    print(f"\nDataset statistics:")
    print(f"  - Train: {total_train} images")
    print(f"  - Query: {total_query} images")
    print(f"  - Gallery: {total_gallery} images")
    print(f"  - Total: {total_train + total_query + total_gallery} images")
    print(f"\nAverage samples per identity:")
    print(f"  - Train: {total_train / valid_ids:.1f}")
    
    # PK采样有效率
    print("\n" + "=" * 60)
    print("PK Sampling Effectiveness")
    print("=" * 60)
    train_counts = defaultdict(int)
    for r in results:
        if r[4] != 'skipped':
            train_counts[r[0]] = r[1]
    
    for k in [2, 3, 4, 5]:
        valid = sum(1 for v in train_counts.values() if v >= k)
        print(f"K={k}: {valid}/{valid_ids} ({valid/valid_ids*100:.1f}%) effective")
    
    print(f"\nOutput saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
