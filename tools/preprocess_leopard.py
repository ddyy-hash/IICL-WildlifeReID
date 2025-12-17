"""
LeopardID2022 数据预处理脚本 v2
自适应划分策略 - 处理长尾分布

策略:
- 样本>=10张: 标准 70/15/15 划分
- 样本 5-9张: 80/10/10 划分
- 样本 3-4张: train全部, query/gallery各复用1张
"""

import os
import json
import shutil
import random
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def load_coco_annotations(ann_file):
    """加载COCO格式标注"""
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    id2filename = {img['id']: img['file_name'] for img in data['images']}
    
    identity_images = defaultdict(list)
    for ann in data['annotations']:
        if 'name' in ann and ann['name']:
            identity_id = ann['name']
            image_id = ann['image_id']
            filename = id2filename.get(image_id)
            if filename:
                bbox = ann.get('bbox', None)
                viewpoint = ann.get('viewpoint', 'unknown')
                identity_images[identity_id].append({
                    'filename': filename,
                    'bbox': bbox,
                    'viewpoint': viewpoint
                })
    
    return identity_images


def crop_image(img_path, bbox, output_path, padding=0.1):
    """从原图裁剪出目标区域"""
    try:
        if not bbox or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            shutil.copy2(img_path, output_path)
            return True
        
        img = Image.open(img_path)
        width, height = img.size
        
        x, y, bw, bh = bbox
        pad_w = bw * padding
        pad_h = bh * padding
        
        x1 = max(0, int(x - pad_w))
        y1 = max(0, int(y - pad_h))
        x2 = min(width, int(x + bw + pad_w))
        y2 = min(height, int(y + bh + pad_h))
        
        if (x2 - x1) >= width * 0.9 and (y2 - y1) >= height * 0.9:
            img.save(output_path, quality=95)
        else:
            cropped = img.crop((x1, y1, x2, y2))
            cropped.save(output_path, quality=95)
        
        img.close()
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False


def adaptive_split(n_images):
    """
    自适应划分策略
    返回 (train_ratio, query_count, gallery_count)
    """
    if n_images >= 10:
        # 标准划分
        n_train = int(n_images * 0.70)
        n_query = max(1, int(n_images * 0.15))
        n_gallery = n_images - n_train - n_query
        return n_train, n_query, n_gallery
    elif n_images >= 5:
        # 80/10/10
        n_train = int(n_images * 0.80)
        n_query = max(1, int(n_images * 0.10))
        n_gallery = n_images - n_train - n_query
        if n_gallery < 1:
            n_train -= 1
            n_gallery = 1
        return n_train, n_query, n_gallery
    elif n_images >= 3:
        # 全部放train，query/gallery各复用1张
        # 返回特殊标记 -1 表示复用模式
        return -1, 1, 1
    else:
        return 0, 0, 0  # 跳过


def process_dataset_v2(src_dir, dst_dir, seed=42, num_workers=8):
    """
    处理LeopardID2022数据集 v2 - 自适应划分
    """
    random.seed(seed)
    
    ann_file = os.path.join(src_dir, 'annotations', 'instances_train2022.json')
    images_dir = os.path.join(src_dir, 'images', 'train2022')
    
    print(f"Loading annotations from {ann_file}...")
    print(f"Images directory: {images_dir}")
    print(f"Using adaptive split strategy")
    identity_images = load_coco_annotations(ann_file)
    
    print(f"Found {len(identity_images)} identities")
    
    # 创建输出目录
    train_dir = os.path.join(dst_dir, 'train')
    query_dir = os.path.join(dst_dir, 'query')
    gallery_dir = os.path.join(dst_dir, 'gallery')
    
    for d in [train_dir, query_dir, gallery_dir]:
        os.makedirs(d, exist_ok=True)
    
    tasks = []
    stats = {
        'train': {'images': 0, 'identities': 0},
        'query': {'images': 0, 'identities': 0},
        'gallery': {'images': 0, 'identities': 0},
        'skipped': 0,
        'reuse_mode': 0  # 复用模式的个体数
    }
    
    print("Preparing tasks with adaptive split...")
    for identity_id, images_info in identity_images.items():
        n_images = len(images_info)
        
        n_train, n_query, n_gallery = adaptive_split(n_images)
        
        if n_train == 0:
            stats['skipped'] += n_images
            continue
        
        random.shuffle(images_info)
        
        safe_id = identity_id.replace('/', '_').replace('\\', '_')
        train_id_dir = os.path.join(train_dir, safe_id)
        query_id_dir = os.path.join(query_dir, safe_id)
        gallery_id_dir = os.path.join(gallery_dir, safe_id)
        
        if n_train == -1:
            # 复用模式: 全部放train，从train中选1张到query和gallery
            stats['reuse_mode'] += 1
            os.makedirs(train_id_dir, exist_ok=True)
            os.makedirs(query_id_dir, exist_ok=True)
            os.makedirs(gallery_id_dir, exist_ok=True)
            
            for idx, info in enumerate(images_info):
                src_path = os.path.join(images_dir, info['filename'])
                dst_path = os.path.join(train_id_dir, f"{idx:04d}.jpg")
                if os.path.exists(src_path):
                    tasks.append((src_path, info['bbox'], dst_path))
                    stats['train']['images'] += 1
            
            # 复用第一张到query
            src_path = os.path.join(images_dir, images_info[0]['filename'])
            dst_path = os.path.join(query_id_dir, "q_0000.jpg")
            if os.path.exists(src_path):
                tasks.append((src_path, images_info[0]['bbox'], dst_path))
                stats['query']['images'] += 1
            
            # 复用最后一张到gallery
            src_path = os.path.join(images_dir, images_info[-1]['filename'])
            dst_path = os.path.join(gallery_id_dir, "g_0000.jpg")
            if os.path.exists(src_path):
                tasks.append((src_path, images_info[-1]['bbox'], dst_path))
                stats['gallery']['images'] += 1
            
            stats['train']['identities'] += 1
            stats['query']['identities'] += 1
            stats['gallery']['identities'] += 1
        else:
            # 标准划分
            train_images = images_info[:n_train]
            query_images = images_info[n_train:n_train + n_query]
            gallery_images = images_info[n_train + n_query:]
            
            if train_images:
                os.makedirs(train_id_dir, exist_ok=True)
                for idx, info in enumerate(train_images):
                    src_path = os.path.join(images_dir, info['filename'])
                    dst_path = os.path.join(train_id_dir, f"{idx:04d}.jpg")
                    if os.path.exists(src_path):
                        tasks.append((src_path, info['bbox'], dst_path))
                        stats['train']['images'] += 1
                stats['train']['identities'] += 1
            
            if query_images:
                os.makedirs(query_id_dir, exist_ok=True)
                for idx, info in enumerate(query_images):
                    src_path = os.path.join(images_dir, info['filename'])
                    dst_path = os.path.join(query_id_dir, f"q_{idx:04d}.jpg")
                    if os.path.exists(src_path):
                        tasks.append((src_path, info['bbox'], dst_path))
                        stats['query']['images'] += 1
                stats['query']['identities'] += 1
            
            if gallery_images:
                os.makedirs(gallery_id_dir, exist_ok=True)
                for idx, info in enumerate(gallery_images):
                    src_path = os.path.join(images_dir, info['filename'])
                    dst_path = os.path.join(gallery_id_dir, f"g_{idx:04d}.jpg")
                    if os.path.exists(src_path):
                        tasks.append((src_path, info['bbox'], dst_path))
                        stats['gallery']['images'] += 1
                stats['gallery']['identities'] += 1
    
    print(f"Total tasks: {len(tasks)}")
    print(f"Reuse mode identities: {stats['reuse_mode']}")
    
    # 多进程并行处理
    success = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(crop_image, src, bbox, dst): (src, dst) 
                   for src, bbox, dst in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            if future.result():
                success += 1
            else:
                failed += 1
    
    # 打印统计
    print("\n" + "=" * 50)
    print("LeopardID2022 预处理完成 (v2 自适应划分)")
    print("=" * 50)
    print(f"Train:   {stats['train']['images']:5d} images, {stats['train']['identities']:3d} identities")
    print(f"Query:   {stats['query']['images']:5d} images, {stats['query']['identities']:3d} identities")
    print(f"Gallery: {stats['gallery']['images']:5d} images, {stats['gallery']['identities']:3d} identities")
    print(f"Skipped: {stats['skipped']:5d} images (identities with <3 images)")
    print(f"Reuse mode: {stats['reuse_mode']} identities (3-4 samples)")
    print("=" * 50)
    print(f"Output directory: {dst_dir}")
    
    return stats


if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description='Preprocess LeopardID2022 for ReID (v2)')
    parser.add_argument('--src', type=str, default='orignal_data/leopard.coco',
                        help='Source directory (COCO format)')
    parser.add_argument('--dst', type=str, default='data/processed/leopard',
                        help='Destination directory (ReID format)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    
    args = parser.parse_args()
    
    process_dataset_v2(
        src_dir=args.src,
        dst_dir=args.dst,
        seed=args.seed,
        num_workers=args.workers
    )
