"""
GZGC (Great Zebra and Giraffe Count) 数据预处理脚本
转换为 ReID 标准格式: train/query/gallery 目录结构

数据集特点:
- 4948 images, 6925 annotations
- 斑马: 6286 annotations, 1905 individuals
- 长颈鹿: 639 annotations, 151 individuals
- COCO JSON 格式, 包含 bbox 和 individual name
- 很多个体只有1个样本 (challenging dataset)

对于 IPAID 验证, 我们只使用斑马子集 (zebra_plains)
"""

import os
import json
import random
import shutil
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def load_coco_annotations(json_path, species='zebra'):
    """
    加载 COCO JSON 标注
    
    Args:
        json_path: COCO JSON 文件路径
        species: 'zebra' 或 'giraffe' 或 'all'
    
    Returns:
        identity_images: {individual_name: [{image_id, file_name, bbox, ...}, ...]}
        images_dict: {image_id: image_info}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 建立 image_id -> image_info 映射
    images_dict = {img['id']: img for img in data['images']}
    
    # 确定要使用的 category_id
    # category_id = 0: giraffe_masai
    # category_id = 1: zebra_plains
    if species == 'zebra':
        target_category = 1
    elif species == 'giraffe':
        target_category = 0
    else:
        target_category = None  # 使用所有
    
    # 按个体分组
    identity_images = defaultdict(list)
    
    for ann in data['annotations']:
        # 过滤物种
        if target_category is not None and ann['category_id'] != target_category:
            continue
        
        name = ann.get('name', '')
        if not name:
            continue
        
        image_id = ann['image_id']
        if image_id not in images_dict:
            continue
        
        image_info = images_dict[image_id]
        
        # COCO bbox: [x, y, width, height]
        bbox = ann['bbox']
        
        identity_images[name].append({
            'image_id': image_id,
            'file_name': image_info['file_name'],
            'bbox': bbox,  # [x, y, w, h]
            'viewpoint': ann.get('viewpoint', 'unknown'),
            'annotation_id': ann['id']
        })
    
    return identity_images, images_dict


def adaptive_split(n_images):
    """
    自适应划分策略
    - 样本>=10张: 标准 70/15/15
    - 样本 5-9张: 80/10/10
    - 样本 3-4张: 复用模式 (全部train, query/gallery复用)
    - 样本 2张: 最小模式 (1 train, 1 query, 1 gallery复用)
    - 样本 <2张: 跳过
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
    elif n_images == 2:
        # 最小模式: 2张图, 1 train, 1同时作为query和gallery
        return 1, 1, 1, 'minimal'
    else:
        return 0, 0, 0, 'skip'


def crop_and_save(src_path, dst_path, bbox, margin=0.1):
    """
    从原图裁剪 bbox 区域并保存
    
    Args:
        src_path: 源图像路径
        dst_path: 目标保存路径
        bbox: [x, y, width, height] COCO 格式
        margin: 边界扩展比例
    """
    try:
        img = Image.open(src_path)
        img_w, img_h = img.size
        
        x, y, w, h = bbox
        
        # 扩展边界
        margin_w = w * margin
        margin_h = h * margin
        
        x1 = max(0, int(x - margin_w))
        y1 = max(0, int(y - margin_h))
        x2 = min(img_w, int(x + w + margin_w))
        y2 = min(img_h, int(y + h + margin_h))
        
        # 裁剪
        cropped = img.crop((x1, y1, x2, y2))
        
        # 保存
        cropped.save(dst_path, quality=95)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


def sanitize_name(name):
    """清理个体名称,使其适合作为文件夹名"""
    # 移除或替换不合法字符
    return name.replace(' ', '_').replace('/', '_').replace('\\', '_').replace(':', '_')


def process_single_crop(task):
    """处理单个裁剪任务 (用于多进程)"""
    src_path, dst_path, bbox, margin = task
    try:
        img = Image.open(src_path)
        img_w, img_h = img.size
        
        x, y, w, h = bbox
        margin_w = w * margin
        margin_h = h * margin
        
        x1 = max(0, int(x - margin_w))
        y1 = max(0, int(y - margin_h))
        x2 = min(img_w, int(x + w + margin_w))
        y2 = min(img_h, int(y + h + margin_h))
        
        cropped = img.crop((x1, y1, x2, y2))
        cropped.save(dst_path, quality=95)
        return dst_path, True
    except Exception as e:
        return dst_path, False


def process_dataset(args):
    """处理整个数据集 - 多进程优化版"""
    
    # 路径设置
    json_path = os.path.join(args.data_root, 'annotations', 'instances_train2020.json')
    images_dir = os.path.join(args.data_root, 'images', 'train2020')
    output_dir = args.output_dir
    
    print(f"Loading annotations from {json_path}")
    identity_images, images_dict = load_coco_annotations(json_path, species=args.species)
    
    print(f"\nTotal individuals: {len(identity_images)}")
    
    # 统计样本分布
    sample_counts = [len(imgs) for imgs in identity_images.values()]
    print(f"Sample distribution:")
    print(f"  Min: {min(sample_counts)}, Max: {max(sample_counts)}, Mean: {sum(sample_counts)/len(sample_counts):.2f}")
    print(f"  1 sample: {sum(1 for c in sample_counts if c == 1)}")
    print(f"  2 samples: {sum(1 for c in sample_counts if c == 2)}")
    print(f"  3-4 samples: {sum(1 for c in sample_counts if 3 <= c <= 4)}")
    print(f"  5+ samples: {sum(1 for c in sample_counts if c >= 5)}")
    
    # 创建输出目录
    for split in ['train', 'query', 'gallery']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # 创建ID映射 (名称 -> 数字ID)
    valid_identities = {name: imgs for name, imgs in identity_images.items() 
                       if len(imgs) >= args.min_samples}
    
    print(f"\nValid individuals (>={args.min_samples} samples): {len(valid_identities)}")
    
    # 按样本数排序,分配数字ID
    sorted_identities = sorted(valid_identities.keys(), 
                               key=lambda x: len(valid_identities[x]), reverse=True)
    name_to_id = {name: i for i, name in enumerate(sorted_identities)}
    
    # 保存ID映射
    id_mapping_path = os.path.join(output_dir, 'id_mapping.json')
    with open(id_mapping_path, 'w') as f:
        json.dump({'name_to_id': name_to_id, 'id_to_name': {v: k for k, v in name_to_id.items()}}, f, indent=2)
    print(f"Saved ID mapping to {id_mapping_path}")
    
    # 第一步: 收集所有裁剪任务
    print("\nCollecting crop tasks...")
    crop_tasks = []  # (src_path, dst_path, bbox, margin)
    copy_tasks = []  # (src_path, dst_path) 后续复制
    
    stats = {'train': 0, 'query': 0, 'gallery': 0, 'skipped': 0, 'modes': defaultdict(int)}
    
    random.seed(42)
    
    for name in sorted_identities:
        images = valid_identities[name]
        pid = name_to_id[name]
        n_images = len(images)
        
        n_train, n_query, n_gallery, mode = adaptive_split(n_images)
        stats['modes'][mode] += 1
        
        if mode == 'skip':
            stats['skipped'] += 1
            continue
        
        random.shuffle(images)
        
        pid_str = f"{pid:04d}"
        train_dir = os.path.join(output_dir, 'train', pid_str)
        query_dir = os.path.join(output_dir, 'query', pid_str)
        gallery_dir = os.path.join(output_dir, 'gallery', pid_str)
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(query_dir, exist_ok=True)
        os.makedirs(gallery_dir, exist_ok=True)
        
        if mode == 'minimal':
            for i, img_info in enumerate(images):
                src_path = os.path.join(images_dir, img_info['file_name'])
                if not os.path.exists(src_path):
                    continue
                if i == 0:
                    dst_name = f"{pid_str}_c1_{i:04d}.jpg"  # train用c1
                    dst_path = os.path.join(train_dir, dst_name)
                    crop_tasks.append((src_path, dst_path, img_info['bbox'], args.margin, 'train'))
                else:
                    dst_name_q = f"{pid_str}_c2_{i:04d}.jpg"  # query用c2
                    dst_name_g = f"{pid_str}_c3_{i:04d}.jpg"  # gallery用c3
                    dst_path_q = os.path.join(query_dir, dst_name_q)
                    dst_path_g = os.path.join(gallery_dir, dst_name_g)
                    crop_tasks.append((src_path, dst_path_q, img_info['bbox'], args.margin, 'query'))
                    crop_tasks.append((src_path, dst_path_g, img_info['bbox'], args.margin, 'gallery'))
        
        elif mode == 'reuse':
            for i, img_info in enumerate(images):
                src_path = os.path.join(images_dir, img_info['file_name'])
                if not os.path.exists(src_path):
                    continue
                dst_name_train = f"{pid_str}_c1_{i:04d}.jpg"  # train用c1
                dst_path = os.path.join(train_dir, dst_name_train)
                crop_tasks.append((src_path, dst_path, img_info['bbox'], args.margin, 'train'))
                if i == 0:
                    dst_name_q = f"{pid_str}_c2_{i:04d}.jpg"  # query用c2
                    copy_tasks.append((dst_path, os.path.join(query_dir, dst_name_q), 'query'))
                elif i == 1:
                    dst_name_g = f"{pid_str}_c3_{i:04d}.jpg"  # gallery用c3
                    copy_tasks.append((dst_path, os.path.join(gallery_dir, dst_name_g), 'gallery'))
        
        else:
            train_images = images[:n_train]
            query_images = images[n_train:n_train + n_query]
            gallery_images = images[n_train + n_query:]
            
            for i, img_info in enumerate(train_images):
                src_path = os.path.join(images_dir, img_info['file_name'])
                if not os.path.exists(src_path):
                    continue
                dst_name = f"{pid_str}_c1_{i:04d}.jpg"  # train用c1
                crop_tasks.append((src_path, os.path.join(train_dir, dst_name), img_info['bbox'], args.margin, 'train'))
            
            for i, img_info in enumerate(query_images):
                src_path = os.path.join(images_dir, img_info['file_name'])
                if not os.path.exists(src_path):
                    continue
                dst_name = f"{pid_str}_c2_{n_train + i:04d}.jpg"  # query用c2
                crop_tasks.append((src_path, os.path.join(query_dir, dst_name), img_info['bbox'], args.margin, 'query'))
            
            for i, img_info in enumerate(gallery_images):
                src_path = os.path.join(images_dir, img_info['file_name'])
                if not os.path.exists(src_path):
                    continue
                dst_name = f"{pid_str}_c3_{n_train + n_query + i:04d}.jpg"  # gallery用c3
                crop_tasks.append((src_path, os.path.join(gallery_dir, dst_name), img_info['bbox'], args.margin, 'gallery'))
    
    print(f"Total crop tasks: {len(crop_tasks)}")
    print(f"Total copy tasks: {len(copy_tasks)}")
    
    # 第二步: 多进程执行裁剪
    n_workers = min(multiprocessing.cpu_count(), 8)
    print(f"\nProcessing with {n_workers} workers...")
    
    success_count = {'train': 0, 'query': 0, 'gallery': 0}
    
    # 转换任务格式
    tasks_for_pool = [(t[0], t[1], t[2], t[3]) for t in crop_tasks]
    task_types = [t[4] for t in crop_tasks]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(executor.map(process_single_crop, tasks_for_pool), 
                           total=len(tasks_for_pool), desc="Cropping"))
    
    for (dst_path, success), task_type in zip(results, task_types):
        if success:
            success_count[task_type] += 1
    
    # 第三步: 执行复制任务
    print("\nCopying files...")
    for src, dst, task_type in tqdm(copy_tasks, desc="Copying"):
        if os.path.exists(src):
            shutil.copy2(src, dst)
            success_count[task_type] += 1
    
    # 打印统计
    print("\n" + "="*50)
    print("Processing Complete!")
    print("="*50)
    print(f"Train images: {success_count['train']}")
    print(f"Query images: {success_count['query']}")
    print(f"Gallery images: {success_count['gallery']}")
    print(f"Skipped individuals: {stats['skipped']}")
    print(f"\nSplit modes used:")
    for mode, count in stats['modes'].items():
        print(f"  {mode}: {count}")
    
    # 保存统计信息
    stats_path = os.path.join(output_dir, 'dataset_stats.json')
    final_stats = {
        'train': success_count['train'],
        'query': success_count['query'],
        'gallery': success_count['gallery'],
        'skipped': stats['skipped'],
        'modes': dict(stats['modes'])
    }
    with open(stats_path, 'w') as f:
        json.dump(final_stats, f, indent=2)
    print(f"\nSaved statistics to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description='GZGC Dataset Preprocessing')
    parser.add_argument('--data_root', type=str, 
                        default='orignal_data/gzgc.coco',
                        help='Path to GZGC dataset root')
    parser.add_argument('--output_dir', type=str,
                        default='data/processed/gzgc_zebra',
                        help='Output directory')
    parser.add_argument('--species', type=str, default='zebra',
                        choices=['zebra', 'giraffe', 'all'],
                        help='Species to process')
    parser.add_argument('--min_samples', type=int, default=2,
                        help='Minimum samples per identity')
    parser.add_argument('--margin', type=float, default=0.1,
                        help='Margin ratio for bbox expansion')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("="*50)
    print("GZGC Dataset Preprocessing")
    print("="*50)
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Species: {args.species}")
    print(f"Min samples: {args.min_samples}")
    print(f"Margin: {args.margin}")
    print("="*50)
    
    process_dataset(args)


if __name__ == '__main__':
    main()
