#!/usr/bin/env python3
"""简单的 ReID 检索评估脚本

使用已训练好的 JointReIDModel，对给定的 query / gallery 目录进行检索评估，
计算 Rank-1 / Rank-5 / Rank-10 和 mAP。

支持 k-reciprocal re-ranking 后处理，通常可以提升 mAP 5-10%！

目录结构约定：
    root_query/ID_1/img1.jpg
               /ID_1/img2.jpg
               /ID_2/img3.jpg
    root_gallery/ID_1/img4.jpg
                 /ID_3/img5.jpg
等，即每个身份一个子目录，子目录名为身份 ID（字符串即可）。

用法示例：
    # 普通评估
    python tools/evaluate_reid.py \
        --checkpoint ./checkpoints/joint/joint_best.pth \
        --query_dir ./data/reid_eval/query \
        --gallery_dir ./data/reid_eval/gallery \
        --batch_size 32
    
    # 使用 re-ranking 评估（推荐！）
    python tools/evaluate_reid.py \
        --checkpoint ./checkpoints/joint/joint_best.pth \
        --query_dir ./data/reid_eval/query \
        --gallery_dir ./data/reid_eval/gallery \
        --rerank
"""

import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2

# 确保可以从项目根目录导入 app.core
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.core.joint_model import JointReIDModel
from tools.train_baselines import BaselineReIDModel

# 导入 re-ranking 模块
try:
    from tools.reranking import re_ranking
    RERANKING_AVAILABLE = True
except ImportError:
    try:
        from reranking import re_ranking
        RERANKING_AVAILABLE = True
    except ImportError:
        RERANKING_AVAILABLE = False
        print("[WARNING] re-ranking 模块不可用，将使用普通评估")


class ReIDImageFolder(Dataset):
    """基于文件夹结构的简单 ReID 数据集

    目录结构：root/identity_id/image.jpg
    返回：(image_tensor, identity_id(str), image_path)
    """

    def __init__(self, root: str, transform=None, extensions=(".jpg", ".jpeg", ".png")):
        self.root = root
        self.transform = transform
        self.extensions = extensions
        self.samples: List[Tuple[str, str]] = []  # (img_path, identity_id)

        self._scan()

    def _scan(self):
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"数据目录不存在: {self.root}")

        for identity_name in sorted(os.listdir(self.root)):
            identity_dir = os.path.join(self.root, identity_name)
            if not os.path.isdir(identity_dir):
                continue

            for fname in sorted(os.listdir(identity_dir)):
                if any(fname.lower().endswith(ext) for ext in self.extensions):
                    img_path = os.path.join(identity_dir, fname)
                    self.samples.append((img_path, identity_name))

        print(f"[INFO] 加载数据集 {self.root}, 共 {len(self.samples)} 张图像, {len(self.identities)} 个身份")

    @property
    def identities(self) -> List[str]:
        return sorted({id_ for _, id_ in self.samples})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, identity_id = self.samples[idx]

        img = cv2.imread(img_path)
        if img is None:
            # 容错：读图失败时使用噪声图
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        return img, identity_id, img_path


def build_transform() -> transforms.Compose:
    """评估阶段使用的图像变换
    
    犬类ReID使用正方形 (256, 256)，与训练一致。
    
    注意：训练时先做 ImageNet 归一化，然后在模型内部反归一化回 [0,1] 给光照模块。
    但评估时我们直接输入 [0,1] 范围的图像，因为 extract_features 中
    model(imgs, boxes_list=None) 期望的输入就是 [0,1] 范围。
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # 犬类适合正方形
        transforms.ToTensor(),
        # 与训练时保持一致：使用 [0,1] 范围图像
        # 光照模块期望 [0,1] 输入
    ])


@torch.no_grad()
def extract_features(
    model: JointReIDModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, List[str]]:
    """对一个 dataloader 提取特征

    返回：
        features: (N, D) 的 numpy 数组
        ids:     长度 N 的身份 ID 列表（字符串）
    """
    model.eval()
    feats: List[torch.Tensor] = []
    ids: List[str] = []

    for imgs, pid_list, _ in dataloader:
        imgs = imgs.to(device)

        # 不使用 YOLO 软掩码，直接 boxes_list=None
        output = model(imgs, boxes_list=None, return_illuminated=False)
        batch_feats = output["features"]  # (B, D)

        # L2 归一化
        batch_feats = torch.nn.functional.normalize(batch_feats, p=2, dim=1)

        feats.append(batch_feats.cpu())
        ids.extend(list(pid_list))

    feats_tensor = torch.cat(feats, dim=0) if feats else torch.empty(0, model.local_extractor.hidden_dim)
    return feats_tensor.numpy(), ids


def compute_cmc_map(
    query_feats: np.ndarray,
    query_ids: List[str],
    gallery_feats: np.ndarray,
    gallery_ids: List[str],
    max_rank: int = 10,
    distmat: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    """根据特征和 ID 计算 CMC 和 mAP

    CMC: 累积匹配特性曲线，返回长度 max_rank 的数组
    mAP: mean Average Precision（标量）
    
    Args:
        query_feats: query 特征矩阵
        query_ids: query 身份列表
        gallery_feats: gallery 特征矩阵
        gallery_ids: gallery 身份列表
        max_rank: 最大排名
        distmat: 可选的预计算距离矩阵（用于 re-ranking）
    """
    if query_feats.shape[0] == 0 or gallery_feats.shape[0] == 0:
        raise ValueError("query 或 gallery 特征为空，无法评估")

    num_q = query_feats.shape[0]
    num_g = gallery_feats.shape[0]

    # 计算距离矩阵 (num_q, num_g)
    if distmat is None:
        # 使用欧氏距离平方（去掉 sqrt 能保持排序不变）
        q_norm = np.sum(query_feats ** 2, axis=1, keepdims=True)  # (num_q, 1)
        g_norm = np.sum(gallery_feats ** 2, axis=1, keepdims=True).T  # (1, num_g)
        distmat = q_norm + g_norm - 2 * np.dot(query_feats, gallery_feats.T)

    # 初始化 CMC 和 AP
    max_rank = min(max_rank, num_g)
    cmc = np.zeros(max_rank, dtype=float)
    all_ap = []

    gallery_ids = np.asarray(gallery_ids)

    for i in range(num_q):
        qid = query_ids[i]
        dist = distmat[i]
        # 按距离从小到大排序
        order = np.argsort(dist)
        matches = (gallery_ids[order] == qid).astype(np.int32)  # 1 表示匹配

        if matches.sum() == 0:
            # 该 query 在 gallery 中没有正样本，跳过（不计入 mAP 和 CMC）
            continue

        # CMC
        first_match_idx = np.where(matches == 1)[0][0]
        if first_match_idx < max_rank:
            cmc[first_match_idx:] += 1

        # AP 计算
        # precision@k = 累积正样本数 / k
        # AP = 对所有正样本位置的 precision 取平均
        num_rel = matches.sum()
        tmp_cmc = matches.cumsum()
        precision_at_k = tmp_cmc * matches / (np.arange(len(matches)) + 1)
        ap = precision_at_k.sum() / num_rel
        all_ap.append(ap)

    if len(all_ap) == 0:
        raise RuntimeError("所有 query 在 gallery 中都没有对应 ID，检查数据划分是否正确")

    cmc = cmc / len(all_ap)
    mAP = float(np.mean(all_ap))
    return cmc, mAP


def main():
    parser = argparse.ArgumentParser(description="使用 JointReIDModel 进行 ReID 检索评估")
    parser.add_argument("--checkpoint", type=str, required=True, help="joint_best.pth 的路径")
    parser.add_argument("--query_dir", type=str, required=True, help="query 图像根目录")
    parser.add_argument("--gallery_dir", type=str, required=True, help="gallery 图像根目录")
    parser.add_argument("--batch_size", type=int, default=32, help="评估 batch 大小")
    parser.add_argument("--device", type=str, default="auto", help="设备: auto / cpu / cuda")
    parser.add_argument("--baseline", action="store_true", help="使用 BaselineReIDModel（无光照模块）")
    parser.add_argument("--rerank", action="store_true", help="使用 k-reciprocal re-ranking（推荐！可提升 mAP 5-10%%）")
    parser.add_argument("--rerank_k1", type=int, default=20, help="re-ranking 参数 k1")
    parser.add_argument("--rerank_k2", type=int, default=6, help="re-ranking 参数 k2")
    parser.add_argument("--rerank_lambda", type=float, default=0.3, help="re-ranking 参数 lambda")

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] 使用设备: {device}")

    # 加载 checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint 不存在: {args.checkpoint}")

    # PyTorch 2.6+ 需要设置 weights_only=False 来加载包含 numpy 数组的 checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_classes = checkpoint.get("num_classes", 100)

    # 初始化模型
    if args.baseline:
        # 使用 BaselineReIDModel（无光照模块）
        model = BaselineReIDModel(
            num_classes=num_classes,
            backbone_name="osnet_ain_x1_0",
            pretrained_backbone=False,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("[INFO] BaselineReIDModel 加载完成（无光照模块）")
    else:
        # 使用 JointReIDModel（包含光照模块）
        model = JointReIDModel(
            num_classes=num_classes,
            num_stripes=6,
            pretrained_backbone=False,
            soft_mask_temperature=10.0,
            soft_mask_type="sigmoid",
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("[INFO] JointReIDModel 加载完成")

    # 构建数据加载器
    transform = build_transform()

    query_dataset = ReIDImageFolder(args.query_dir, transform=transform)
    gallery_dataset = ReIDImageFolder(args.gallery_dir, transform=transform)

    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 提取特征
    print("[INFO] 提取 query 特征...")
    q_feats, q_ids = extract_features(model, query_loader, device)

    print("[INFO] 提取 gallery 特征...")
    g_feats, g_ids = extract_features(model, gallery_loader, device)

    # 计算指标
    print("[INFO] 计算 ReID 指标 (CMC & mAP)...")
    
    # 是否使用 re-ranking
    distmat = None
    if args.rerank:
        if not RERANKING_AVAILABLE:
            print("[WARNING] re-ranking 模块不可用，使用普通评估")
        else:
            print("[INFO] 使用 k-reciprocal re-ranking 优化排序...")
            distmat = re_ranking(
                q_feats, g_feats,
                k1=args.rerank_k1,
                k2=args.rerank_k2,
                lambda_value=args.rerank_lambda,
            )
    
    cmc, mAP = compute_cmc_map(q_feats, q_ids, g_feats, g_ids, max_rank=10, distmat=distmat)

    rank1 = cmc[0]
    rank5 = cmc[4] if cmc.shape[0] >= 5 else cmc[-1]
    rank10 = cmc[9] if cmc.shape[0] >= 10 else cmc[-1]

    print("\n===== ReID Evaluation Results =====")
    if args.rerank and RERANKING_AVAILABLE:
        print("(with k-reciprocal re-ranking)")
    print(f"Rank-1  : {rank1 * 100:.2f}%")
    print(f"Rank-5  : {rank5 * 100:.2f}%")
    print(f"Rank-10 : {rank10 * 100:.2f}%")
    print(f"mAP     : {mAP * 100:.2f}%")
    print("===================================\n")


if __name__ == "__main__":
    main()
