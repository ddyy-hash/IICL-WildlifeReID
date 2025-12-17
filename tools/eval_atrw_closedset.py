#!/usr/bin/env python3
"""
ATRW Closed-Set 评估 (7:3 Split)

与 SMFFEN 2024 等论文使用相同的评估协议：
- 使用官方训练集 (107 IDs, 1887 images)
- 随机按 7:3 划分为 train/val
- 同一 ID 同时出现在 train 和 val 中 (Closed-Set)

这样可以与以下论文直接对比:
- SMFFEN 2024: mAP 78.70%, Rank-1 96.30%
- ResNet50: mAP 68.40%, Rank-1 91.70%
- PCB: mAP 71.20%, Rank-1 94.70%
"""

import os
import sys
import json
import random
import shutil
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.joint_model import JointReIDModel


class ATRWDataset(Dataset):
    """ATRW ReID 数据集"""
    
    def __init__(self, image_list, transform=None):
        """
        Args:
            image_list: [(img_path, label), ...]
            transform: 图像变换
        """
        self.samples = image_list
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            # 注意: 不做 Normalize，模型内部会自己归一化
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label, img_path


def load_atrw_train_data(data_root):
    """
    加载官方 ATRW 训练集
    
    Args:
        data_root: 数据根目录，如 'orignal_data/Amur Tiger Re-identification'
    
    Returns:
        samples: [(img_path, entity_id), ...]
        id_to_samples: {entity_id: [sample_indices]}
    """
    # 支持两种目录结构:
    # 1. 服务器结构: data_root/train/, data_root/reid_list_train.csv
    # 2. 本地结构: data_root/atrw_reid_train/train/, data_root/atrw_anno_reid_train/reid_list_train.csv
    
    # 尝试服务器结构
    train_dir = os.path.join(data_root, 'train')
    anno_file = os.path.join(data_root, 'reid_list_train.csv')
    
    # 如果不存在，尝试本地结构
    if not os.path.exists(train_dir):
        train_dir = os.path.join(data_root, 'atrw_reid_train', 'train')
    if not os.path.exists(anno_file):
        anno_file = os.path.join(data_root, 'atrw_anno_reid_train', 'reid_list_train.csv')
    
    print(f"训练目录: {train_dir}")
    print(f"标注文件: {anno_file}")
    
    if not os.path.exists(anno_file):
        raise FileNotFoundError(f"找不到标注文件: {anno_file}")
    
    # 解析 CSV: entityid,filename
    filename_to_id = {}
    with open(anno_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('entityid'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                entity_id = int(parts[0])
                filename = parts[1]
                filename_to_id[filename] = entity_id
    
    # 加载所有图片
    samples = []
    id_to_samples = defaultdict(list)
    
    for img_name in os.listdir(train_dir):
        if not img_name.endswith(('.jpg', '.png', '.jpeg')):
            continue
        
        if img_name not in filename_to_id:
            continue
        
        entity_id = filename_to_id[img_name]
        img_path = os.path.join(train_dir, img_name)
        idx = len(samples)
        samples.append((img_path, entity_id))
        id_to_samples[entity_id].append(idx)
    
    print(f"加载了 {len(samples)} 张图片, {len(id_to_samples)} 个身份")
    return samples, id_to_samples


def split_train_val(samples, id_to_samples, train_ratio=0.7, seed=42):
    """
    按照论文方式进行 7:3 划分
    
    每个 ID 的图片按 7:3 划分到 train 和 val
    """
    random.seed(seed)
    np.random.seed(seed)
    
    train_samples = []
    val_samples = []
    
    # 重新映射标签 (0, 1, 2, ...)
    unique_ids = sorted(id_to_samples.keys())
    id_to_label = {eid: i for i, eid in enumerate(unique_ids)}
    
    for entity_id, indices in id_to_samples.items():
        label = id_to_label[entity_id]
        
        # 打乱该 ID 的所有样本
        random.shuffle(indices)
        
        # 按 7:3 划分
        n_train = max(1, int(len(indices) * train_ratio))
        
        for i, idx in enumerate(indices):
            img_path, _ = samples[idx]
            if i < n_train:
                train_samples.append((img_path, label))
            else:
                val_samples.append((img_path, label))
    
    print(f"划分完成: 训练集 {len(train_samples)}, 验证集 {len(val_samples)}")
    return train_samples, val_samples, len(unique_ids)


def extract_features(model, dataloader, device):
    """提取特征"""
    model.eval()
    
    all_features = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            
            # 获取特征
            outputs = model(images)
            if isinstance(outputs, dict):
                features = outputs.get('features', outputs.get('global_feat'))
            elif isinstance(outputs, (tuple, list)):
                features = outputs[0]
            else:
                features = outputs
            
            features = F.normalize(features, dim=1)
            
            all_features.append(features.cpu())
            all_labels.extend(labels.numpy() if hasattr(labels, 'numpy') else labels)
            all_paths.extend(paths)
    
    features = torch.cat(all_features, dim=0)
    labels = np.array(all_labels)
    
    return features, labels, all_paths


def compute_distance_matrix(query_features, gallery_features):
    """计算欧氏距离矩阵"""
    # 使用余弦距离
    similarity = torch.mm(query_features, gallery_features.t())
    distance = 1 - similarity
    return distance.numpy()


def evaluate_rank(distmat, query_labels, gallery_labels, max_rank=50):
    """
    计算 CMC 和 mAP
    """
    num_query = distmat.shape[0]
    
    all_cmc = []
    all_AP = []
    
    for q_idx in range(num_query):
        q_label = query_labels[q_idx]
        
        # 排序
        order = np.argsort(distmat[q_idx])
        
        # 找到匹配的 gallery 样本
        matches = (gallery_labels[order] == q_label).astype(np.int32)
        
        if not np.any(matches):
            continue
        
        # CMC
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        
        # AP
        num_rel = matches.sum()
        tmp_cmc = matches.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * matches
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_query
    mAP = np.mean(all_AP)
    
    return {
        'rank1': all_cmc[0] * 100,
        'rank5': all_cmc[4] * 100 if len(all_cmc) > 4 else 0,
        'rank10': all_cmc[9] * 100 if len(all_cmc) > 9 else 0,
        'mAP': mAP * 100
    }


def main():
    parser = argparse.ArgumentParser(description='ATRW Closed-Set Evaluation (7:3 Split)')
    parser.add_argument('--data_root', type=str, 
                        default='orignal_data/Amur Tiger Re-identification',
                        help='ATRW 数据根目录')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型 checkpoint 路径')
    parser.add_argument('--backbone', type=str, default='osnet_ain_x1_0',
                        help='Backbone 类型')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (影响 7:3 划分)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='训练集比例')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 加载数据
    print("\n" + "="*60)
    print("Step 1: 加载 ATRW 官方训练集")
    print("="*60)
    samples, id_to_samples = load_atrw_train_data(args.data_root)
    
    # 2. 7:3 划分
    print("\n" + "="*60)
    print(f"Step 2: 按 {int(args.train_ratio*100)}:{int((1-args.train_ratio)*100)} 划分 (seed={args.seed})")
    print("="*60)
    train_samples, val_samples, num_classes = split_train_val(
        samples, id_to_samples, args.train_ratio, args.seed
    )
    
    # 3. 创建数据集
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        # 注意: 不做 Normalize，模型内部会自己归一化
    ])
    
    # 验证集同时作为 query 和 gallery
    val_dataset = ATRWDataset(val_samples, transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # 4. 加载模型
    print("\n" + "="*60)
    print("Step 3: 加载模型")
    print("="*60)
    
    model = JointReIDModel(
        num_classes=num_classes,
        backbone_name=args.backbone,
        pretrained_backbone=False  # 不下载预训练，直接加载 checkpoint
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 处理可能的 key 不匹配
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() 
                       if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    
    print(f"加载了 {len(pretrained_dict)}/{len(state_dict)} 个参数")
    
    model = model.to(device)
    model.eval()
    
    # 5. 提取特征
    print("\n" + "="*60)
    print("Step 4: 提取验证集特征")
    print("="*60)
    
    features, labels, paths = extract_features(model, val_loader, device)
    print(f"特征维度: {features.shape}")
    
    # 6. 评估 (每个样本轮流作为 query)
    print("\n" + "="*60)
    print("Step 5: 计算距离矩阵并评估")
    print("="*60)
    
    # 计算距离矩阵
    distmat = compute_distance_matrix(features, features)
    
    # 将对角线设为无穷大 (排除自己)
    np.fill_diagonal(distmat, np.inf)
    
    # 评估
    results = evaluate_rank(distmat, labels, labels)
    
    # 7. 输出结果
    print("\n" + "="*60)
    print("Closed-Set 评估结果 (7:3 Split, 与论文对比)")
    print("="*60)
    print(f"  Rank-1:  {results['rank1']:.2f}%")
    print(f"  Rank-5:  {results['rank5']:.2f}%")
    print(f"  Rank-10: {results['rank10']:.2f}%")
    print(f"  mAP:     {results['mAP']:.2f}%")
    
    print("\n" + "-"*60)
    print("与其他方法对比 (ATRW Closed-Set):")
    print("-"*60)
    print(f"{'Method':<25} {'Rank-1':<12} {'Rank-5':<12} {'mAP':<12}")
    print("-"*60)
    print(f"{'ResNet50':<25} {'91.70%':<12} {'97.90%':<12} {'68.40%':<12}")
    print(f"{'PCB':<25} {'94.70%':<12} {'98.40%':<12} {'71.20%':<12}")
    print(f"{'SMFFEN 2024':<25} {'96.30%':<12} {'98.90%':<12} {'78.70%':<12}")
    print(f"{'Ours (IPAID+IICL)':<25} {results['rank1']:.2f}%{'':<6} {results['rank5']:.2f}%{'':<6} {results['mAP']:.2f}%")
    print("-"*60)
    
    return results


if __name__ == '__main__':
    main()
