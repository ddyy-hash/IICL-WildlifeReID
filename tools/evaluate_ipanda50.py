#!/usr/bin/env python3
"""
iPanda50 官方评估协议脚本

iPanda50 官方协议特点：
1. 测试集内所有图像互相检索（test-to-test retrieval）
2. Query 也加入 Gallery 进行检索
3. 每个 query 排除自己（最近邻匹配）
4. 计算所有 query 的平均 Rank-1、Rank-5、Rank-10 和 mAP

参考：iPanda-50 官方论文评估方法
"""

import os
import sys
import argparse
from typing import List, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import cv2
from tqdm import tqdm

# 项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.core.joint_model import JointReIDModel


class iPanda50Dataset(Dataset):
    """iPanda50 测试集数据加载器
    
    支持单目录或多目录（合并 query + gallery）
    目录结构：root/panda_id/image.jpg
    """
    
    def __init__(self, roots: List[str], transform=None):
        """
        Args:
            roots: 目录列表（可以是单个目录或多个目录合并）
            transform: 图像变换
        """
        if isinstance(roots, str):
            roots = [roots]
        self.roots = roots
        self.transform = transform
        self.samples: List[Tuple[str, str, int]] = []  # (img_path, identity, index)
        self.identity_to_idx: Dict[str, int] = {}
        self._scan()
    
    def _scan(self):
        # 先收集所有身份
        all_identities = set()
        for root in self.roots:
            if not os.path.isdir(root):
                raise FileNotFoundError(f"数据目录不存在: {root}")
            for identity in os.listdir(root):
                identity_dir = os.path.join(root, identity)
                if os.path.isdir(identity_dir):
                    all_identities.add(identity)
        
        # 建立身份到索引的映射
        for idx, identity in enumerate(sorted(all_identities)):
            self.identity_to_idx[identity] = idx
        
        # 扫描所有目录
        for root in self.roots:
            for identity in sorted(os.listdir(root)):
                identity_dir = os.path.join(root, identity)
                if not os.path.isdir(identity_dir):
                    continue
                
                idx = self.identity_to_idx[identity]
                
                for fname in sorted(os.listdir(identity_dir)):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(identity_dir, fname)
                        self.samples.append((img_path, identity, idx))
        
        print(f"[INFO] 加载 iPanda50 测试集: {len(self.samples)} 张图像, {len(self.identity_to_idx)} 个熊猫")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, identity, id_idx = self.samples[idx]
        
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"无法读取图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        return img, id_idx, img_path


def build_transform(img_size=(256, 256)):
    """构建评估时的图像变换"""
    return T.Compose([
        T.ToPILImage(),
        T.Resize(img_size),
        T.ToTensor(),
        # 注意：不做 Normalize，模型内部会处理
    ])


def extract_features(model, dataloader, device):
    """提取所有图像的特征
    
    Returns:
        features: [N, D] numpy array
        labels: [N] list of identity indices
        paths: [N] list of image paths
    """
    model.eval()
    
    all_features = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for imgs, labels, paths in tqdm(dataloader, desc="提取特征"):
            imgs = imgs.to(device)
            
            # Forward
            output = model(imgs, boxes_list=None)
            features = output['features']
            
            # L2 归一化
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            all_features.append(features.cpu())
            all_labels.extend(labels.tolist())
            all_paths.extend(paths)
    
    all_features = torch.cat(all_features, dim=0).numpy()
    
    return all_features, all_labels, all_paths


def compute_ipanda50_metrics(features, labels, max_rank=10):
    """
    iPanda50 官方评估协议
    
    每张图像作为 query，在所有图像（包括自己）中检索
    排除自己后计算指标
    
    Args:
        features: [N, D] 所有图像的特征
        labels: [N] 所有图像的标签
        max_rank: 最大 rank
    
    Returns:
        cmc: [max_rank] CMC曲线
        mAP: 平均精度
        rank1, rank5, rank10: 各个 rank 的准确率
    """
    N = len(labels)
    labels = np.array(labels)
    
    # 计算距离矩阵 [N, N]
    # 使用欧氏距离平方
    feat_norm = np.sum(features ** 2, axis=1, keepdims=True)
    distmat = feat_norm + feat_norm.T - 2 * np.dot(features, features.T)
    
    print(f"\n[INFO] 距离矩阵形状: {distmat.shape}")
    print(f"[INFO] 开始计算 iPanda50 指标...")
    
    # 初始化
    cmc = np.zeros(max_rank, dtype=float)
    all_ap = []
    
    # 每张图像作为 query
    for i in tqdm(range(N), desc="计算指标"):
        query_label = labels[i]
        
        # 获取距离并排序
        dist = distmat[i]
        order = np.argsort(dist)
        
        # 排除自己（距离为0或最小的那个）
        # 通常第一个就是自己
        if order[0] == i:
            order = order[1:]  # 去掉自己
        else:
            # 如果第一个不是自己，找到自己并移除
            self_idx = np.where(order == i)[0]
            if len(self_idx) > 0:
                order = np.delete(order, self_idx[0])
        
        # 找到所有同 ID 的图像（排除自己）
        matches = (labels[order] == query_label).astype(np.int32)
        
        num_gt = matches.sum()
        if num_gt == 0:
            # 该 query 没有其他同 ID 图像（只有自己）
            continue
        
        # CMC: 第一个匹配的位置
        first_match_positions = np.where(matches == 1)[0]
        if len(first_match_positions) > 0:
            first_match = first_match_positions[0]
            if first_match < max_rank:
                cmc[first_match:] += 1
        
        # AP: Average Precision
        # precision@k = (累积正样本数) / k
        tmp_cmc = matches.cumsum()
        precision_at_k = tmp_cmc * matches / (np.arange(len(matches)) + 1)
        ap = precision_at_k.sum() / num_gt
        all_ap.append(ap)
    
    if len(all_ap) == 0:
        raise RuntimeError("没有有效的 query（可能所有熊猫只有1张图）")
    
    # 归一化 CMC
    cmc = cmc / len(all_ap)
    mAP = float(np.mean(all_ap))
    
    # 提取 Rank-1, 5, 10
    rank1 = cmc[0]
    rank5 = cmc[4] if len(cmc) >= 5 else cmc[-1]
    rank10 = cmc[9] if len(cmc) >= 10 else cmc[-1]
    
    return cmc, mAP, rank1, rank5, rank10


def main():
    parser = argparse.ArgumentParser(description="iPanda50 官方评估协议")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--test_dir", type=str, default=None, help="iPanda50 测试集目录（官方协议）")
    parser.add_argument("--query_dir", type=str, default=None, help="Query 目录（兼容旧数据）")
    parser.add_argument("--gallery_dir", type=str, default=None, help="Gallery 目录（兼容旧数据）")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--img_size", type=int, default=256, help="图像尺寸")
    parser.add_argument("--device", type=str, default="auto", help="设备: auto/cpu/cuda")
    
    args = parser.parse_args()
    
    # 确定测试目录
    test_dirs = []
    if args.test_dir and os.path.exists(args.test_dir):
        test_dirs = [args.test_dir]
        print(f"[INFO] 使用 test 目录: {args.test_dir}")
    elif args.query_dir and args.gallery_dir:
        if os.path.exists(args.query_dir) and os.path.exists(args.gallery_dir):
            test_dirs = [args.query_dir, args.gallery_dir]
            print(f"[INFO] 合并 query + gallery 目录进行 All-vs-All 评估")
        else:
            raise FileNotFoundError(f"目录不存在: query={args.query_dir}, gallery={args.gallery_dir}")
    else:
        raise ValueError("请指定 --test_dir 或同时指定 --query_dir 和 --gallery_dir")
    
    args = parser.parse_args()
    
    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*60}")
    print(f"iPanda50 官方评估协议")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test Dirs: {test_dirs}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # 加载 checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint 不存在: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_classes = checkpoint.get("num_classes", 50)
    
    print(f"[INFO] 类别数: {num_classes}")
    
    # 初始化模型
    model = JointReIDModel(
        num_classes=num_classes,
        num_stripes=6,
        pretrained_backbone=False,
        soft_mask_temperature=10.0,
        soft_mask_type="sigmoid",
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    print("[INFO] 模型加载完成\n")
    
    # 构建数据集（支持单目录或多目录合并）
    transform = build_transform(img_size=(args.img_size, args.img_size))
    test_dataset = iPanda50Dataset(test_dirs, transform=transform)
    
    # Linux 多进程，Windows 单进程
    import platform
    num_workers = 4 if platform.system() != 'Windows' else 0
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    # 提取特征
    print("[INFO] 开始提取特征...")
    features, labels, paths = extract_features(model, test_loader, device)
    print(f"[INFO] 特征提取完成: {features.shape}\n")
    
    # 计算指标
    cmc, mAP, rank1, rank5, rank10 = compute_ipanda50_metrics(
        features, labels, max_rank=10
    )
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"iPanda50 评估结果（官方协议）")
    print(f"{'='*60}")
    print(f"Rank-1  : {rank1 * 100:.2f}%")
    print(f"Rank-5  : {rank5 * 100:.2f}%")
    print(f"Rank-10 : {rank10 * 100:.2f}%")
    print(f"mAP     : {mAP * 100:.2f}%")
    print(f"{'='*60}\n")
    
    # 保存结果
    output_file = os.path.join(os.path.dirname(args.checkpoint), "ipanda50_results.txt")
    with open(output_file, 'w') as f:
        f.write(f"iPanda50 Evaluation Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test Dir: {args.test_dir}\n")
        f.write(f"Num Images: {len(features)}\n")
        f.write(f"Num Pandas: {len(set(labels))}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Rank-1  : {rank1 * 100:.2f}%\n")
        f.write(f"Rank-5  : {rank5 * 100:.2f}%\n")
        f.write(f"Rank-10 : {rank10 * 100:.2f}%\n")
        f.write(f"mAP     : {mAP * 100:.2f}%\n")
    
    print(f"[INFO] 结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
