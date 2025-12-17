#!/usr/bin/env python3
"""
IICL: Illumination-Invariant Contrastive Learning for Wildlife ReID

=== 核心创新: 光照不变对比学习 (IICL) ===

    问题: 野生动物ReID面临"光照变化大 + 样本稀少"的双重挑战。
          传统对比学习用随机颜色抖动生成正样本对 → 可能破坏花纹/斑点等关键身份特征。
          
    方案: 利用Retinex物理分解自动生成"同一目标不同光照"的正样本对:
          
          原图 I = R × L  (Retinex分解)
               │
               ├── R (反射率，保持不变) ← 花纹/斑点等身份特征
               │
               ├── L → L' (光照变换: gamma/缩放/局部扰动)
               │
               └── I' = R × L' (光照变体)
          
          对比学习: f(I) ≈ f(I')  (同一图像的光照变体应特征相近)

    优势:
    1. 保留纹理特征 (R不变) vs SimCLR的颜色抖动可能破坏花纹
    2. 物理合理的数据增强 (1张→N张光照变体)
    3. 无需真实光照配对数据
    4. 端到端可微训练

【训练流程】

    完整原始图像
         │
         ▼
    IPAID光照分解 ──────► L_illum (正则化)
         │ (R = I/L)
         ├── R (反射层)
         └── L (光照图) → 变换 → L'_1, L'_2
                           │
                           ▼
                      I'_1 = R×L'_1  (光照变体)
                      I'_2 = R×L'_2
         │
         ▼
    骨干网络 + 特征提取
         │
         ├── f(R)      → L_reid (度量学习)
         ├── f(I'_1)   ┐
         └── f(I'_2)   ┴→ L_IICL (对比学习: ||f(R)-f(I')||²)
    
    联合损失: L = L_reid + λ_illum * L_illum + λ_iicl * L_IICL

=== 两阶段训练策略 ===

Phase 1 - Backbone Warmup (10 epochs):
    - 冻结光照模块
    - 只训练 Backbone + LocalExtractor
    - loss = L_reid only

Phase 2 - IICL联合训练 (100 epochs):
    - 解冻所有参数
    - loss = L_reid + 0.2 * L_illum + 0.5 * L_IICL
    - 每张图生成2个光照变体进行对比学习

=== 使用方法 ===

python tools/train_joint.py \\
    --data_dir ./data/processed/nyala/train \\
    --output_dir ./checkpoints/iicl_nyala \\
    --use_iicl --iicl_weight 0.5 --iicl_variants 2
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime
from torchvision import transforms
from typing import Optional, List, Tuple, Dict, Union
from ultralytics import YOLO

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.illumination_module_v2 import (
    IPAIDModule,
    IPAIDLoss,
    IdentityPreservingLoss,
    IlluminationContrastiveLoss,  # IICL: 光照不变对比学习损失
)
from app.core.metric_losses import TripletLoss, ArcFaceLoss, CircleLoss
from app.core.joint_model import JointReIDModel, SUPPORTED_BACKBONES, get_backbone_dim


# ============================================================================
#                           YOLO检测器包装
# ============================================================================

class YOLODetectorWrapper:
    """
    YOLO检测器包装器
    
    用于在训练时获取边界框（detached，不参与反向传播）
    """
    
    def __init__(self, model_path: str = './fea_data/yolov8m-seg.pt', conf: float = 0.5):
        self.model = YOLO(model_path)
        self.conf = conf
    
    @torch.no_grad()
    def detect_batch(self, images: torch.Tensor) -> List[Optional[torch.Tensor]]:
        """
        批量检测图像中的目标（不再限制为狗）
        
        Args:
            images: (B, 3, H, W) 归一化后的图像张量
        
        Returns:
            boxes_list: 每张图像的检测框列表
        """
        B = images.shape[0]
        device = images.device
        
        # 转换为numpy用于YOLO
        images_np = images.cpu().numpy()
        images_np = (images_np * 255).astype(np.uint8)
        images_np = images_np.transpose(0, 2, 3, 1)  # (B, H, W, 3)
        
        boxes_list = []
        
        for i in range(B):
            img = images_np[i]
            # BGR格式
            if img.shape[-1] == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img

            results = self.model.predict(img_bgr, conf=self.conf, verbose=False)

            det_boxes = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                det_boxes.append([x1, y1, x2, y2])

            if det_boxes:
                boxes = torch.tensor(det_boxes, dtype=torch.float32, device=device)
            else:
                # YOLO未检测到任何目标时，退化为整图一个框，兼容老虎等未标注类别
                H, W = img_bgr.shape[:2]
                boxes = torch.tensor([[0.0, 0.0, float(W - 1), float(H - 1)]], dtype=torch.float32, device=device)

            boxes_list.append(boxes)
        
        return boxes_list


# ============================================================================
#                           数据集
# ============================================================================

class FullImageDataset(Dataset):
    """
    完整图像数据集（用于联合训练）
    
    加载原始完整图像，包含身份标签
    """
    
    def __init__(
        self,
        data_dir: str,
        transform=None,
        extensions: tuple = ('.jpg', '.jpeg', '.png')
    ):
        self.data_dir = data_dir
        self.transform = transform
        
        self.samples = []  # (image_path, identity_id)
        self.identity_to_idx = {}
        self.idx_to_samples = {}  # 新增：用于 PK 采样
        
        self._load_samples(data_dir, extensions)
        
        print(f"[INFO] 数据集加载完成: {len(self.samples)} 张图像, {len(self.identity_to_idx)} 个身份")
    
    def _load_samples(self, data_dir: str, extensions: tuple):
        idx = 0
        for identity_dir in sorted(os.listdir(data_dir)):
            identity_path = os.path.join(data_dir, identity_dir)
            if not os.path.isdir(identity_path):
                continue
            
            identity_id = identity_dir
            
            if identity_id not in self.identity_to_idx:
                self.identity_to_idx[identity_id] = idx
                self.idx_to_samples[idx] = []
                idx += 1
            
            label_idx = self.identity_to_idx[identity_id]
            
            for img_file in os.listdir(identity_path):
                if img_file.lower().endswith(extensions):
                    img_path = os.path.join(identity_path, img_file)
                    sample_idx = len(self.samples)
                    self.samples.append((img_path, identity_id))
                    self.idx_to_samples[label_idx].append(sample_idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, str]:
        img_path, identity_id = self.samples[idx]
        label = self.identity_to_idx[identity_id]
        
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path
    
    @property
    def num_classes(self):
        return len(self.identity_to_idx)


# ============================================================================
#                           PK 采样器（提升 mAP 关键）
# ============================================================================

class PKSampler(torch.utils.data.Sampler):
    """
    PK 采样器：每个 batch 采样 P 个身份，每个身份采样 K 张图片
    
    优势：
    1. 保证每个 batch 有足够的正样本对（同 ID）
    2. 让 Triplet/Circle Loss 的 hard mining 更有效
    3. 直接提升 mAP
    """
    
    def __init__(self, dataset: FullImageDataset, p: int = 8, k: int = 4):
        """
        Args:
            dataset: 数据集（需要有 idx_to_samples 属性）
            p: 每个 batch 的身份数
            k: 每个身份的样本数
        """
        self.dataset = dataset
        self.p = p
        self.k = k
        self.batch_size = p * k
        
        # 过滤掉样本数少于 k 的身份
        self.valid_ids = [
            idx for idx, samples in dataset.idx_to_samples.items()
            if len(samples) >= k
        ]
        
        if len(self.valid_ids) < p:
            print(f"[WARNING] 有效身份数 {len(self.valid_ids)} < P={p}，将使用所有身份")
            self.valid_ids = list(dataset.idx_to_samples.keys())
        
        print(f"[INFO] PK采样器: P={p}, K={k}, 有效身份数={len(self.valid_ids)}")
    
    def __iter__(self):
        # 每个 epoch 重新打乱
        batch_indices = []
        
        # 计算需要多少个 batch
        num_batches = len(self.dataset) // self.batch_size
        
        for _ in range(num_batches):
            # 随机选择 P 个身份
            selected_ids = np.random.choice(self.valid_ids, size=min(self.p, len(self.valid_ids)), replace=False)
            
            batch = []
            for pid in selected_ids:
                samples = self.dataset.idx_to_samples[pid]
                # 随机选择 K 个样本（可能需要重复采样）
                if len(samples) >= self.k:
                    selected = np.random.choice(samples, size=self.k, replace=False)
                else:
                    selected = np.random.choice(samples, size=self.k, replace=True)
                batch.extend(selected.tolist())
            
            batch_indices.extend(batch)
        
        return iter(batch_indices)
    
    def __len__(self):
        return (len(self.dataset) // self.batch_size) * self.batch_size


# ============================================================================
#                           Center Loss（提升 mAP 关键）
# ============================================================================

class CenterLoss(nn.Module):
    """
    Center Loss: 让同类特征向类中心聚拢
    
    论文: A Discriminative Feature Learning Approach for Deep Face Recognition (ECCV 2016)
    
    对 mAP 的提升：
    - 让同 ID 的所有样本特征更紧凑
    - 直接改善 ranking，提升 mAP
    """
    
    def __init__(self, num_classes: int, feat_dim: int = 256, lr_center: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lr_center = lr_center  # 中心更新率
        
        # 类中心（可学习参数）
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, feat_dim] 特征
            labels: [B] 标签
        
        Returns:
            loss: Center Loss
        """
        batch_size = features.size(0)
        
        # 获取每个样本对应的类中心
        centers_batch = self.centers[labels]  # [B, feat_dim]
        
        # 计算到类中心的距离
        loss = F.mse_loss(features, centers_batch)
        
        return loss


# ============================================================================
#                           联合训练器
# ============================================================================

class JointTrainer:
    """
    端到端联合训练器
    
    实现两阶段训练策略：
    - Phase 1: 光照预训练（冻结光照网络）
    - Phase 2: 联合微调（解冻所有参数）
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        config_path: Optional[str] = None,
        backbone: str = "osnet_ain_x1_0",
        batch_size: int = 28,
        phase1_epochs: int = 50,
        phase2_epochs: int = 150,
        learning_rate: float = 3e-4,
        num_stripes: int = 6,
        device: str = 'auto',
        logger: Optional[logging.Logger] = None,
        resume_checkpoint: Optional[str] = None,
        query_dir: Optional[str] = None,
        gallery_dir: Optional[str] = None,
        eval_interval: int = 5,
        p_size: Optional[int] = None,
        k_size: int = 4,
        circle_gamma: int = 256,
        img_height: int = 256,
        img_width: int = 256,
        # IICL 参数（小样本场景核心度量学习信号）
        use_iicl: bool = True,
        iicl_weight: float = 1.0,  # 禁用triplet/circle后，IICL成为主要度量学习损失
        iicl_num_variants: int = 2,
        # 数据加载参数
        num_workers: int = 4,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.backbone = backbone
        self.batch_size = batch_size
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs
        self.learning_rate = learning_rate
        self.num_stripes = num_stripes
        self.logger = logger or logging.getLogger('JointTraining')
        self.resume_checkpoint = resume_checkpoint
        self.resume_phase: Optional[int] = None
        self.resume_epoch: int = 0
        self.resume_optimizer_state: Optional[dict] = None
        self.resume_scheduler_state: Optional[dict] = None
        self.best_acc: float = 0.0
        self.best_rank1: float = 0.0
        self.best_map: float = 0.0
        
        # 评估相关参数
        self.query_dir = query_dir
        self.gallery_dir = gallery_dir
        self.eval_interval = eval_interval
        
        # PK采样和损失函数参数
        self.p_size = p_size
        self.k_size = k_size
        self.circle_gamma = circle_gamma
        
        # 图像尺寸参数（支持非正方形）
        self.img_height = img_height
        self.img_width = img_width
        
        # IICL 参数（光照不变对比学习）
        self.use_iicl_arg = use_iicl
        self.iicl_weight_arg = iicl_weight
        self.iicl_num_variants_arg = iicl_num_variants
        
        # 数据加载参数
        self.num_workers = num_workers
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"使用设备: {self.device}")
        self.logger.info(f"骨干网络: {self.backbone}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化组件
        self._init_dataloader()
        self._init_model()
        self._init_losses()
        self._init_yolo()
        self._maybe_resume()
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """加载配置文件，包含完整的训练超参数"""
        # 默认配置（针对 ATRW Tiger ReID 优化）
        default_config = {
            'illumination_module': {
                'loss_params': {
                    # IPAID 损失参数
                    'lambda_recon': 1.0,       # 重建损失
                    'lambda_smooth': 0.15,     # 光照平滑（TV正则）
                    'lambda_edge': 0.08,       # 反射层边缘保持
                    'lambda_sensitivity': 0.02, # 敏感度正则化
                    'lambda_identity': 0.1,    # 身份保持损失
                },
                'module_params': {
                    'base_channels': 32,
                    'num_scales': 3,
                    'refine_iterations': 2,
                    'use_sensitivity': True,
                    'use_refinement': True,
                }
            },
            'training': {
                'learning_rate': 3.5e-4,
                'weight_decay': 5e-4,
                'gradient_clip': 1.0,
                # PK 采样器参数
                # 注意：禁用Triplet/Circle后，PK采样不再必要，用随机采样更均匀
                'pk_sampler': {
                    'enabled': False,  # 禁用：ArcFace+IICL不需要batch内正样本对
                    'p': 8,   # 每 batch 采样 P 个身份
                    'k': 2,   # 每身份采样 K 张 (8*2=16)
                },
                # Center Loss 参数
                'center_loss': {
                    'enabled': True,
                    'weight': 0.0005,
                    'feat_dim': 256,
                    'lr_scale': 0.5,
                },
                # 度量学习损失权重
                # 对于小样本数据集（如nyala），推荐禁用triplet/circle，用ArcFace+IICL替代
                'metric_learning': {
                    'ce_loss': {'weight': 1.0, 'label_smoothing': 0.1},
                    'arcface_loss': {'weight': 1.0, 's': 30.0, 'm': 0.35},  # ArcFace: 适合长尾分布
                    'triplet_loss': {'weight': 0.0, 'margin': 0.3, 'mining_type': 'soft'},  # 禁用，IICL替代
                    'circle_loss': {'weight': 0.0, 'margin': 0.25, 'gamma': 256},           # 禁用，IICL替代
                },
                # 两阶段训练配置
                'phases': {
                    'phase1': {
                        'illumination_weight': 1.0,
                        'reid_weight': 0.3,
                        'illumination_lr': 1e-4,
                        'warmup_epochs': 5,
                    },
                    'phase2': {
                        'illumination_weight': 0.6,
                        'reid_weight': 1.0,
                        'illumination_lr': 5e-5,
                        'backbone_lr': 3.5e-4,
                    }
                },
                # 早停配置
                'early_stopping': {
                    'enabled': True,
                    'patience': 30,
                    'monitor': 'mAP',
                    'min_delta': 0.001,
                }
            },
            # 数据增强配置
            'data_augmentation': {
                'image_size': 256,
                'random_crop_scale': 1.125,  # 放大比例
                'random_horizontal_flip': 0.5,
                'color_jitter': {
                    'brightness': 0.2,
                    'contrast': 0.15,
                    'saturation': 0.15,
                    'hue': 0.03,
                },
                'random_erasing': {
                    'probability': 0.5,
                    'scale': [0.02, 0.25],
                    'ratio': [0.3, 3.3],
                },
            },
            # YOLO 配置
            'yolo': {
                'model_path': 'fea_data/yolov8m-seg.pt',
                'conf_threshold': 0.3,
            },
            # 评估配置
            'evaluation': {
                'eval_interval': 5,
                'flip_test': True,
            },
            # 检查点配置
            'checkpointing': {
                'save_interval': 10,
                'max_keep': 5,
            }
        }
        
        # 尝试加载外部配置文件
        config_files = [
            config_path,
            'app/core/illumination_config.yaml',
            'illumination_config.yaml',
        ]
        
        for cfg_file in config_files:
            if cfg_file and os.path.exists(cfg_file):
                try:
                    with open(cfg_file, 'r', encoding='utf-8') as f:
                        loaded_config = yaml.safe_load(f)
                    # 深度合并配置
                    self._deep_merge(default_config, loaded_config)
                    self.logger.info(f"已加载配置文件: {cfg_file}")
                    break
                except Exception as e:
                    self.logger.warning(f"加载配置文件失败 {cfg_file}: {e}")
        
        return default_config
    
    def _deep_merge(self, base: dict, update: dict):
        """深度合并字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _init_dataloader(self):
        """初始化数据加载器
        
        针对老虎/犬类 ReID 的数据增强策略：
        1. 输入尺寸 (256, 256) - 四足动物适合正方形
        2. Random Crop - 放大后随机裁剪，增加位置多样性
        3. Random Erasing - 模拟遮挡（草丛、树木等）
        4. 适度的颜色抖动 - 避免与光照模块冲突
        5. PK 采样 - 每 batch 保证有足够正样本对，提升 mAP
        """
        # 从配置读取参数
        train_cfg = self.config.get('training', {})
        aug_cfg = self.config.get('data_augmentation', {})
        
        # 图像尺寸：优先使用命令行参数，否则从配置读取
        if self.img_height != 256 or self.img_width != 256:
            # 用户通过命令行指定了尺寸
            img_h, img_w = self.img_height, self.img_width
            self.logger.info(f"使用命令行指定的图像尺寸: {img_h}x{img_w}")
        else:
            # 从配置文件读取
            img_h = train_cfg.get('image_height', train_cfg.get('image_size', 256))
            img_w = train_cfg.get('image_width', train_cfg.get('image_size', 256))
        
        # 保存图像尺寸供评估使用
        self.img_height = img_h
        self.img_width = img_w
        
        # 随机裁剪配置
        crop_cfg = aug_cfg.get('random_crop', {})
        if crop_cfg.get('enabled', True):
            crop_scale = crop_cfg.get('scale', [0.85, 1.0])[1] / crop_cfg.get('scale', [0.85, 1.0])[0]
            crop_scale = min(1.2, 1.0 / crop_cfg.get('scale', [0.85, 1.0])[0])  # 根据 scale 计算放大倍数
        else:
            crop_scale = 1.125
        crop_h = int(img_h * crop_scale)
        crop_w = int(img_w * crop_scale)
        
        color_cfg = aug_cfg.get('color_jitter', {})
        erase_cfg = aug_cfg.get('random_erasing', {})
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            # 放大后随机裁剪（支持非正方形）
            transforms.Resize((crop_h, crop_w)),
            transforms.RandomCrop((img_h, img_w)),
            transforms.RandomHorizontalFlip(p=aug_cfg.get('random_horizontal_flip', 0.5)),
            # 颜色抖动（幅度适中，避免与光照模块冲突）
            transforms.ColorJitter(
                brightness=color_cfg.get('brightness', 0.2),
                contrast=color_cfg.get('contrast', 0.15),
                saturation=color_cfg.get('saturation', 0.15),
                hue=color_cfg.get('hue', 0.03)
            ),
            transforms.ToTensor(),
            # 注意: 不做 ImageNet Normalize！
            # IPAID 模块期望 [0,1] 范围的 RGB 输入（见 illumination_module_v2.py 第 622 行）
            # 模型内部会在 IPAID 输出后自动做 Normalize（见 joint_model.py 第 476-479 行）
            # Random Erasing - 模拟遮挡
            transforms.RandomErasing(
                p=erase_cfg.get('probability', 0.5),
                scale=tuple(erase_cfg.get('scale', [0.02, 0.25])),
                ratio=tuple(erase_cfg.get('ratio', [0.3, 3.3])),
                value='random'
            )
        ])
        
        self.dataset = FullImageDataset(self.data_dir, transform=transform)
        self.num_classes = self.dataset.num_classes
        
        # 从配置读取 PK 采样器参数
        pk_cfg = self.config.get('training', {}).get('pk_sampler', {})
        use_pk_sampler = pk_cfg.get('enabled', True)
        num_workers = self.num_workers  # 使用命令行参数
        
        if use_pk_sampler and len(self.dataset.idx_to_samples) >= 4:
            # 优先使用命令行参数，否则使用配置文件
            k = self.k_size if self.k_size else pk_cfg.get('k', 4)
            
            if self.p_size:
                # 用户明确指定了 P
                p = self.p_size
            else:
                # 根据 batch_size 自动计算 P
                p = self.batch_size // k
            
            # 确保 P 至少为 2，否则 Triplet Loss 无法工作
            if p < 2:
                p = 2
                k = self.batch_size // p
                if k < 2:
                    k = 2
            
            # 确保 P 不超过实际身份数
            p = min(p, len(self.dataset.idx_to_samples))
            
            actual_batch_size = p * k
            
            self.pk_sampler = PKSampler(self.dataset, p=p, k=k)
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=actual_batch_size,
                sampler=self.pk_sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False
            )
            self.logger.info(f"[PK采样] P={p}, K={k}, 实际batch_size={actual_batch_size} (用户指定: {self.batch_size})")
        else:
            # 回退到普通采样
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True
            )
            self.logger.info("使用普通随机采样")
        
        self.logger.info(f"数据集: {len(self.dataset)} 样本, {self.num_classes} 类别")
    
    def _init_model(self):
        """初始化联合模型"""
        # 从配置读取IPAID开关和模型参数
        model_cfg = self.config.get('model', {})
        illum_cfg = model_cfg.get('illumination_module', {})
        local_cfg = model_cfg.get('local_extractor', {})
        use_ipaid = illum_cfg.get('enabled', True)
        
        # v2: 从配置读取 dropout 参数 (防过拟合关键)
        dropout = local_cfg.get('dropout', 0.0)
        
        self.model = JointReIDModel(
            num_classes=self.num_classes,
            backbone_name=self.backbone,
            num_stripes=self.num_stripes,
            pretrained_backbone=True,
            soft_mask_temperature=10.0,
            soft_mask_type='sigmoid',
            use_ipaid=use_ipaid,  # 传递IPAID开关
            dropout=dropout,  # v2: 传递 dropout 参数
        ).to(self.device)
        
        # 保存IPAID状态供后续使用
        self.use_ipaid = use_ipaid
        
        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"模型参数: {total_params/1e6:.2f}M")
        self.logger.info(f"IPAID光照模块: {'启用' if use_ipaid else '禁用'}")
    
    def _init_losses(self):
        """初始化损失函数（从配置读取详细参数）"""
        # 读取配置
        loss_params = self.config['illumination_module']['loss_params']
        train_cfg = self.config.get('training', {})
        metric_cfg = train_cfg.get('metric_learning', {})
        center_cfg = train_cfg.get('center_loss', {})
        
        # ========== IPAID 光照损失 ==========
        self.ipaid_loss = IPAIDLoss(
            lambda_recon=loss_params.get('lambda_recon', 1.0),
            lambda_smooth=loss_params.get('lambda_smooth', 0.15),
            lambda_edge=loss_params.get('lambda_edge', 0.08),
            lambda_sensitivity=loss_params.get('lambda_sensitivity', 0.02),
        ).to(self.device)
        
        # ========== 身份保持损失 ==========
        triplet_cfg = metric_cfg.get('triplet_loss', {})
        self.identity_preserving_loss = IdentityPreservingLoss(
            margin=triplet_cfg.get('margin', 0.3),
            mining='hard'
        ).to(self.device)
        self.identity_loss_weight = loss_params.get('lambda_identity', 0.1)
        
        # ========== ReID 度量学习损失 ==========
        # 交叉熵损失
        ce_cfg = metric_cfg.get('ce_loss', {})
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=ce_cfg.get('label_smoothing', 0.1)
        )
        self.ce_weight = ce_cfg.get('weight', 1.0)
        
        # Triplet Loss
        self.triplet_loss = TripletLoss(
            margin=triplet_cfg.get('margin', 0.3),
            mining_type=triplet_cfg.get('mining_type', 'soft')
        )
        self.triplet_weight = triplet_cfg.get('weight', 1.0)
        
        # Circle Loss - 优先使用命令行参数
        circle_cfg = metric_cfg.get('circle_loss', {})
        circle_gamma_value = self.circle_gamma if self.circle_gamma else circle_cfg.get('gamma', 256)
        self.circle_loss = CircleLoss(
            m=circle_cfg.get('margin', 0.25),
            gamma=circle_gamma_value
        )
        self.circle_weight = circle_cfg.get('weight', 0.5)
        
        # ========== ArcFace Loss（关键：适合长尾分布，不需要batch内正样本）==========
        arcface_cfg = metric_cfg.get('arcface_loss', {})
        self.arcface_weight = arcface_cfg.get('weight', 1.0)
        if self.arcface_weight > 0:
            # ArcFace 需要知道特征维度和类别数
            feat_dim = center_cfg.get('feat_dim', 256)  # 与 Center Loss 保持一致
            self.arcface_loss = ArcFaceLoss(
                in_features=feat_dim,
                out_features=self.num_classes,
                s=arcface_cfg.get('s', 30.0),
                m=arcface_cfg.get('m', 0.35)
            ).to(self.device)
        else:
            self.arcface_loss = None
        
        # ========== Center Loss（关键：提升 mAP）==========
        if center_cfg.get('enabled', True):
            self.center_loss = CenterLoss(
                num_classes=self.num_classes,
                feat_dim=center_cfg.get('feat_dim', 256),
            ).to(self.device)
            self.center_loss_weight = center_cfg.get('weight', 0.0005)
            self.center_lr_scale = center_cfg.get('lr_scale', 0.5)
        else:
            self.center_loss = None
            self.center_loss_weight = 0
        
        # ========== IICL: 光照不变对比学习损失（核心创新）==========
        iicl_cfg = train_cfg.get('iicl', {})
        # 命令行参数优先
        self.use_iicl = self.use_iicl_arg if hasattr(self, 'use_iicl_arg') else iicl_cfg.get('enabled', True)
        self.iicl_loss = IlluminationContrastiveLoss(
            temperature=iicl_cfg.get('temperature', 0.1),
            loss_type=iicl_cfg.get('loss_type', 'mse'),  # mse, cosine, or infonce
        )
        self.iicl_weight = self.iicl_weight_arg if hasattr(self, 'iicl_weight_arg') else iicl_cfg.get('weight', 0.5)
        self.iicl_num_variants = self.iicl_num_variants_arg if hasattr(self, 'iicl_num_variants_arg') else iicl_cfg.get('num_variants', 2)
        
        # 打印损失函数配置
        self.logger.info("=" * 50)
        self.logger.info("损失函数配置:")
        self.logger.info(f"  IPAID Loss: λ_recon={loss_params.get('lambda_recon', 1.0)}, "
                        f"λ_smooth={loss_params.get('lambda_smooth', 0.15)}, "
                        f"λ_edge={loss_params.get('lambda_edge', 0.08)}, "
                        f"λ_sens={loss_params.get('lambda_sensitivity', 0.02)}")
        self.logger.info(f"  Identity Preserving: weight={self.identity_loss_weight}")
        self.logger.info(f"  CE Loss: weight={self.ce_weight}, label_smooth={ce_cfg.get('label_smoothing', 0.1)}")
        self.logger.info(f"  Triplet Loss: weight={self.triplet_weight}, margin={triplet_cfg.get('margin', 0.3)}")
        self.logger.info(f"  Circle Loss: weight={self.circle_weight}, m={circle_cfg.get('margin', 0.25)}, γ={circle_gamma_value}")
        self.logger.info(f"  ArcFace Loss: weight={self.arcface_weight}, s={arcface_cfg.get('s', 30.0)}, m={arcface_cfg.get('m', 0.35)}")
        self.logger.info(f"  Center Loss: weight={self.center_loss_weight}, enabled={center_cfg.get('enabled', True)}")
        self.logger.info(f"  IICL Loss: weight={self.iicl_weight}, enabled={self.use_iicl}, variants={self.iicl_num_variants}")
        self.logger.info("=" * 50)
    
    def _init_yolo(self):
        """初始化YOLO检测器（已禁用，负面优化）"""
        # YOLO检测对于已裁剪好的ReID数据集是负面优化
        # 保留接口但默认禁用
        self.yolo_detector = None
        self.logger.info("YOLO检测器已禁用（对裁剪好的ReID数据集是负面优化）")

    def _maybe_resume(self):
        """如指定断点，则加载模型/优化器状态"""
        if not self.resume_checkpoint:
            return

        if not os.path.exists(self.resume_checkpoint):
            self.logger.warning(f"指定的断点不存在: {self.resume_checkpoint}，将从头开始训练")
            self.resume_checkpoint = None
            return

        self.logger.info(f"尝试从断点恢复: {self.resume_checkpoint}")
        checkpoint = torch.load(self.resume_checkpoint, map_location=self.device, weights_only=False)

        # 恢复模型参数
        model_state = checkpoint.get('model_state_dict')
        if model_state is None:
            self.logger.warning("断点不包含模型权重，忽略断点继续训练")
            self.resume_checkpoint = None
            return

        self.model.load_state_dict(model_state)

        self.resume_phase = checkpoint.get('phase', 1)
        # epoch 为零基索引，恢复时从下一轮开始
        self.resume_epoch = max(0, checkpoint.get('epoch', -1) + 1)
        self.best_acc = checkpoint.get('metrics', {}).get('accuracy', 0.0)
        self.resume_optimizer_state = checkpoint.get('optimizer_state_dict')
        self.resume_scheduler_state = checkpoint.get('scheduler_state_dict')

        self.logger.info(
            f"断点信息: phase={self.resume_phase}, 已完成 epoch={self.resume_epoch}, best_acc={self.best_acc:.2f}%"
        )
    
    def _create_scheduler(self, optimizer, total_epochs: int):
        """根据配置创建学习率调度器
        
        支持的调度器类型：
        - CosineAnnealingLR: 标准余弦退火
        - CosineAnnealingWarmRestarts: 带热重启的余弦退火
        - StepLR: 阶梯式衰减
        - MultiStepLR: 多阶梯式衰减
        """
        train_cfg = self.config.get('training', {})
        scheduler_cfg = train_cfg.get('scheduler', {})
        
        scheduler_type = scheduler_cfg.get('type', 'CosineAnnealingLR')
        eta_min = float(scheduler_cfg.get('eta_min', 1e-6))
        
        self.logger.info(f"  调度器类型: {scheduler_type}")
        
        if scheduler_type == 'CosineAnnealingWarmRestarts':
            # 带热重启的余弦退火 - 更适合长期训练
            T_0 = int(scheduler_cfg.get('T_0', 50))
            T_mult = int(scheduler_cfg.get('T_mult', 2))
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
            )
            self.logger.info(f"  CosineAnnealingWarmRestarts: T_0={T_0}, T_mult={T_mult}, eta_min={eta_min}")
        elif scheduler_type == 'StepLR':
            step_size = int(scheduler_cfg.get('step_size', 30))
            gamma = float(scheduler_cfg.get('gamma', 0.1))
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
            self.logger.info(f"  StepLR: step_size={step_size}, gamma={gamma}")
        elif scheduler_type == 'MultiStepLR':
            milestones = scheduler_cfg.get('milestones', [30, 60, 90])
            gamma = float(scheduler_cfg.get('gamma', 0.1))
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=gamma
            )
            self.logger.info(f"  MultiStepLR: milestones={milestones}, gamma={gamma}")
        else:
            # 默认: CosineAnnealingLR
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs, eta_min=eta_min
            )
            self.logger.info(f"  CosineAnnealingLR: T_max={total_epochs}, eta_min={eta_min}")
        
        return scheduler
    
    def _setup_optimizer_phase1(self):
        """Phase 1优化器设置：Backbone Warmup（任务驱动光照分解）
        
        Phase 1 核心目标：让Backbone适应ReID任务，产生有意义的梯度
        - 冻结光照模块（输出固定的R_retinex）
        - 只训练Backbone + LocalExtractor
        - 只用ReID损失（无光照重建损失）
        
        理由：
        1. 光照模块冻结，先让Backbone学会"什么特征对识别有用"
        2. Phase 2解冻光照模块时，Backbone能提供有意义的梯度信号
        3. 这是"任务驱动"的关键：光照模块的优化方向由ReID任务决定
        """
        # 关键：冻结光照模块，解冻Backbone
        self.model.freeze_illumination(True)
        self.model.freeze_backbone(False)
        
        # 训练Backbone和LocalExtractor
        backbone_params = list(self.model.backbone.parameters())
        extractor_params = list(self.model.local_extractor.parameters())
        center_params = list(self.center_loss.parameters())
        
        phase_config = self.config['training']['phases'].get('phase1', {})
        backbone_lr = float(phase_config.get('backbone_lr', self.learning_rate))
        
        # 从配置读取 weight_decay
        train_cfg = self.config.get('training', {})
        weight_decay = float(train_cfg.get('weight_decay', 5e-4))
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': extractor_params, 'lr': backbone_lr},
            {'params': center_params, 'lr': backbone_lr * 0.5},
        ], weight_decay=weight_decay)
        
        # 从配置读取调度器类型
        self.scheduler = self._create_scheduler(self.optimizer, self.phase1_epochs)
        
        # Phase 1: 只用ReID损失（Backbone warmup）
        self.loss_weights = {
            'illumination': 0.0,  # 不用光照损失
            'reid': 1.0
        }
        
        total_params = sum(p.numel() for p in backbone_params) + sum(p.numel() for p in extractor_params)
        self.logger.info(f"Phase 1 优化器设置完成 (Backbone Warmup - 任务驱动准备)")
        self.logger.info(f"  冻结: illumination 模块")
        self.logger.info(f"  训练: backbone + extractor ({total_params/1e6:.2f}M 参数)")
        self.logger.info(f"  backbone_lr: {backbone_lr}, weight_decay: {weight_decay}")
        self.logger.info(f"  损失权重: illum={self.loss_weights['illumination']}, reid={self.loss_weights['reid']}")
    
    def _setup_optimizer_phase2(self):
        """Phase 2优化器设置：任务驱动联合训练
        
        Phase 2 核心目标：ReID损失驱动光照模块优化
        - 解冻所有模块
        - ReID损失权重高，光照损失作为正则化
        - 光照模块的梯度主要来自L_reid
        """
        self.model.freeze_backbone(False)
        self.model.freeze_illumination(False)
        
        phase_config = self.config['training']['phases'].get('phase2', {})
        # 光照模块学习率略高，让它能被ReID梯度有效更新
        illum_lr = float(phase_config.get('illumination_lr', 1e-4))
        # Phase 2 backbone学习率应该从基础LR开始，不要太低
        backbone_lr = float(phase_config.get('backbone_lr', self.learning_rate * 0.5))  # 基础LR的一半
        
        # 分组参数（处理IPAID禁用的情况）
        param_groups = []
        
        # 光照模块参数（如果启用）
        if self.model.illumination is not None:
            illum_params = list(self.model.illumination.parameters())
            param_groups.append({'params': illum_params, 'lr': illum_lr})
        
        # Backbone参数
        backbone_params = list(self.model.backbone.parameters())
        param_groups.append({'params': backbone_params, 'lr': backbone_lr})
        
        # 局部特征提取器参数
        extractor_params = list(self.model.local_extractor.parameters())
        param_groups.append({'params': extractor_params, 'lr': backbone_lr})
        
        # Center Loss参数
        center_params = list(self.center_loss.parameters())
        param_groups.append({'params': center_params, 'lr': backbone_lr * 0.5})
        
        # 从配置读取 weight_decay
        train_cfg = self.config.get('training', {})
        weight_decay = float(train_cfg.get('weight_decay', 5e-4))
        
        self.optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
        
        # 从配置读取调度器类型
        self.scheduler = self._create_scheduler(self.optimizer, self.phase2_epochs)
        
        # 关键：ReID主导，光照损失只是正则化（防止R变成噪声）
        self.loss_weights = {
            'illumination': 0.2,  # 正则化作用
            'reid': 1.0           # 主导优化方向
        }
        
        self.logger.info(f"Phase 2 优化器设置完成 (任务驱动联合训练)")
        self.logger.info(f"  解冻: 所有模块")
        self.logger.info(f"  backbone_lr: {backbone_lr}, illum_lr: {illum_lr}, weight_decay: {weight_decay}")
        self.logger.info(f"  损失权重: illum={self.loss_weights['illumination']} (正则化), reid={self.loss_weights['reid']} (主导)")
        self.logger.info(f"  Center Loss 权重: {self.center_loss_weight}")
    
    def train_epoch(self, epoch: int, phase: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        illum_loss_sum = 0.0
        reid_loss_sum = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels, _) in enumerate(self.dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 将标准化后的图像反归一化到 [0, 1]，供光照模块和 YOLO 使用
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            images_denorm = images * std + mean
            images_denorm = torch.clamp(images_denorm, 0.0, 1.0)

            # YOLO检测获取边界框（detached）
            if self.yolo_detector is not None:
                boxes_list = self.yolo_detector.detect_batch(images_denorm)
            else:
                boxes_list = None
            
            # 前向传播（端到端可微）
            # 注意：给联合模型传入的是 [0,1] 归一化图像，
            # 与光照模块的预期输入范围一致
            
            # 判断是否使用 IICL（光照不变对比学习）
            if self.use_iicl and phase == 2:
                # Phase 2: 使用 IICL 增强训练
                output = self.model.forward_with_contrastive(
                    images_denorm, 
                    num_variants=self.iicl_num_variants
                )
                features_variants = output.get('features_variants', [])
            else:
                # Phase 1 或不使用 IICL
                output = self.model(images_denorm, boxes_list, return_illuminated=True)
                features_variants = []
            
            features = output['features']
            logits = output['logits']
            illuminated = output.get('illuminated')
            ipaid_details = output.get('ipaid_details')  # IPAID 中间结果
            
            # 计算损失
            # 1. IPAID 光照损失（重建 + 平滑 + 边缘保持 + 敏感度正则）
            if ipaid_details is not None:
                loss_illum, illum_loss_dict = self.ipaid_loss(
                    ipaid_details,
                    ipaid_module=self.model.illumination,
                )
                
                # 2. 身份保持损失（确保反射层保持身份判别性）
                loss_identity_preserve = self.identity_preserving_loss(features, labels)
                loss_illum = loss_illum + self.identity_loss_weight * loss_identity_preserve
            else:
                loss_illum = torch.tensor(0.0, device=self.device)
            
            # 3. ReID 度量学习损失（使用配置的权重）
            loss_ce = self.ce_loss(logits, labels)
            loss_triplet = self.triplet_loss(features, labels)
            loss_circle = self.circle_loss(features, labels)
            
            # 计算 ArcFace Loss（如果启用）
            if self.arcface_loss is not None and self.arcface_weight > 0:
                loss_arcface = self.arcface_loss(features, labels)
            else:
                loss_arcface = torch.tensor(0.0, device=self.device)
            
            # 计算 Center Loss（如果启用）
            if self.center_loss is not None:
                loss_center = self.center_loss(features, labels)
            else:
                loss_center = torch.tensor(0.0, device=self.device)
            
            # 加权组合 ReID 损失
            loss_reid = (
                self.ce_weight * loss_ce + 
                self.triplet_weight * loss_triplet + 
                self.circle_weight * loss_circle + 
                self.arcface_weight * loss_arcface +  # ArcFace
                self.center_loss_weight * loss_center
            )
            
            # 5. IICL 对比损失（光照不变对比学习 - 核心创新）
            if self.use_iicl and len(features_variants) > 0:
                loss_iicl = self.iicl_loss(features, features_variants)
            else:
                loss_iicl = torch.tensor(0.0, device=self.device)
            
            # 6. 联合损失
            loss = (
                self.loss_weights['illumination'] * loss_illum +
                self.loss_weights['reid'] * loss_reid +
                self.iicl_weight * loss_iicl  # IICL 损失
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 更新 Center Loss 的 centers（需单独更新，梯度缩放避免中心点剧烈变化）
            if self.center_loss is not None and self.center_loss_weight > 0:
                for param in self.center_loss.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1. / self.center_loss_weight)
            
            # 梯度裁剪（从配置读取）
            grad_clip = self.config.get('training', {}).get('gradient_clip', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            illum_loss_sum += loss_illum.item()
            reid_loss_sum += loss_reid.item()
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 20 == 0:
                iicl_info = f", iicl: {loss_iicl.item():.4f}" if self.use_iicl and len(features_variants) > 0 else ""
                self.logger.info(
                    f"Phase {phase} Epoch [{epoch+1}] "
                    f"Batch [{batch_idx+1}/{len(self.dataloader)}] "
                    f"Loss: {loss.item():.4f} (illum: {loss_illum.item():.4f}, reid: {loss_reid.item():.4f}{iicl_info}) "
                    f"Acc: {100.*correct/total:.2f}%"
                )
        
        num_batches = len(self.dataloader)
        return {
            'total_loss': total_loss / num_batches,
            'illum_loss': illum_loss_sum / num_batches,
            'reid_loss': reid_loss_sum / num_batches,
            'accuracy': 100. * correct / total
        }
    
    def train(self):
        """完整训练流程"""
        self.logger.info("=" * 70)
        self.logger.info("开始端到端联合训练")
        self.logger.info(f"Phase 1: {self.phase1_epochs} epochs (光照预训练)")
        self.logger.info(f"Phase 2: {self.phase2_epochs} epochs (联合微调)")
        self.logger.info("=" * 70)
        
        best_acc = self.best_acc

        # 确定各阶段的起始 epoch
        if self.resume_phase == 1:
            start_phase1_epoch = min(self.resume_epoch, self.phase1_epochs)
        elif self.resume_phase and self.resume_phase > 1:
            start_phase1_epoch = self.phase1_epochs
        else:
            start_phase1_epoch = 0
        
        # ==================== Phase 1: 光照预训练 ====================
        self.logger.info("\n" + "=" * 30 + " Phase 1: 光照预训练 " + "=" * 30)
        self._setup_optimizer_phase1()

        if self.resume_phase == 1 and self.resume_optimizer_state and self.resume_scheduler_state:
            self.optimizer.load_state_dict(self.resume_optimizer_state)
            self.scheduler.load_state_dict(self.resume_scheduler_state)
            if start_phase1_epoch >= self.phase1_epochs:
                self.logger.info("Phase 1 已完成，将直接进入 Phase 2")
            else:
                self.logger.info(f"Phase 1 将从第 {start_phase1_epoch + 1} 轮继续训练")
        
        for epoch in range(start_phase1_epoch, self.phase1_epochs):
            metrics = self.train_epoch(epoch, phase=1)
            self.scheduler.step()
            
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Phase 1 Epoch [{epoch+1}/{self.phase1_epochs}] | "
                f"Loss: {metrics['total_loss']:.4f} | "
                f"Illum: {metrics['illum_loss']:.4f} | "
                f"ReID: {metrics['reid_loss']:.4f} | "
                f"Acc: {metrics['accuracy']:.2f}% | LR: {lr:.6f}"
            )
            
            # 保存Phase 1最佳模型
            if metrics['accuracy'] > best_acc:
                best_acc = metrics['accuracy']
                self.save_checkpoint(epoch, metrics, phase=1, is_best=True)
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, metrics, phase=1, is_best=False)

        # Phase 1 相关断点只用于本阶段
        if self.resume_phase == 1:
            self.resume_phase = None
            self.resume_epoch = 0
            self.resume_optimizer_state = None
            self.resume_scheduler_state = None
        
        # ==================== Phase 2: 联合微调 ====================
        self.logger.info("\n" + "=" * 30 + " Phase 2: 联合微调 " + "=" * 30)
        self._setup_optimizer_phase2()

        load_phase2_state = False
        if self.resume_checkpoint and (self.resume_phase == 2 or (self.resume_phase is None and start_phase1_epoch >= self.phase1_epochs)):
            load_phase2_state = True

        if self.resume_phase == 2:
            start_phase2_epoch = min(self.resume_epoch, self.phase2_epochs)
        else:
            start_phase2_epoch = 0

        if load_phase2_state and self.resume_optimizer_state and self.resume_scheduler_state:
            self.optimizer.load_state_dict(self.resume_optimizer_state)
            self.scheduler.load_state_dict(self.resume_scheduler_state)
            if start_phase2_epoch >= self.phase2_epochs:
                self.logger.info("Phase 2 已完成，没有剩余 epoch 可训练")
            else:
                self.logger.info(f"Phase 2 将从第 {start_phase2_epoch + 1} 轮继续训练")
        
        for epoch in range(start_phase2_epoch, self.phase2_epochs):
            metrics = self.train_epoch(epoch, phase=2)
            self.scheduler.step()
            
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Phase 2 Epoch [{epoch+1}/{self.phase2_epochs}] | "
                f"Loss: {metrics['total_loss']:.4f} | "
                f"Illum: {metrics['illum_loss']:.4f} | "
                f"ReID: {metrics['reid_loss']:.4f} | "
                f"Acc: {metrics['accuracy']:.2f}% | LR: {lr:.6f}"
            )
            
            if metrics['accuracy'] > best_acc:
                best_acc = metrics['accuracy']
                self.save_checkpoint(epoch, metrics, phase=2, is_best=True)
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, metrics, phase=2, is_best=False)
            
            # 每 eval_interval 个 epoch 进行一次 ReID 评估
            if (epoch + 1) % self.eval_interval == 0 and self.query_dir and self.gallery_dir:
                eval_metrics = self.evaluate_reid()
                if eval_metrics:
                    # 根据 Rank-1 保存最佳模型
                    if eval_metrics['rank1'] > self.best_rank1:
                        self.best_rank1 = eval_metrics['rank1']
                        self.best_map = eval_metrics['mAP']
                        self.save_checkpoint(epoch, metrics, phase=2, is_best=True, suffix='_reid_best')
                        self.logger.info(f"新的最佳 ReID 模型! Rank-1: {self.best_rank1:.2f}%, mAP: {self.best_map:.2f}%")

        self.resume_optimizer_state = None
        self.resume_scheduler_state = None
        
        self.best_acc = best_acc
        self.logger.info("=" * 70)
        self.logger.info(f"训练完成! 最佳分类准确率: {best_acc:.2f}%")
        if self.best_rank1 > 0:
            self.logger.info(f"最佳 ReID 性能: Rank-1: {self.best_rank1:.2f}%, mAP: {self.best_map:.2f}%")
        self.logger.info("=" * 70)
    
    @torch.no_grad()
    def _evaluate_ipanda50_official(self, test_dir: str) -> Optional[Dict[str, float]]:
        """iPanda50 官方评估协议: All-vs-All 互检索
        
        每张图像作为 Query，在所有其他图像中检索
        排除自己后计算 Rank-1, mAP
        """
        from torchvision import transforms as T
        from torch.utils.data import Dataset, DataLoader
        from tqdm import tqdm
        
        class TestDataset(Dataset):
            def __init__(self, root_dir, transform):
                self.samples = []
                self.transform = transform
                for identity in sorted(os.listdir(root_dir)):
                    id_dir = os.path.join(root_dir, identity)
                    if not os.path.isdir(id_dir):
                        continue
                    for img_name in os.listdir(id_dir):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(id_dir, img_name)
                            self.samples.append((img_path, identity))
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                img_path, identity = self.samples[idx]
                img = cv2.imread(img_path)
                if img is None:
                    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.transform(img)
                return img, identity
        
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.img_height, self.img_width)),
            T.ToTensor(),
        ])
        
        test_dataset = TestDataset(test_dir, transform)
        import platform
        num_workers = 4 if platform.system() != 'Windows' else 0
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers)
        
        self.logger.info(f"iPanda50 官方评估: {len(test_dataset)} 张测试图像")
        
        # 提取特征
        all_feats, all_ids = [], []
        self.model.eval()
        for imgs, pids in test_loader:
            imgs = imgs.to(self.device)
            output = self.model(imgs, boxes_list=None)
            feat = output['features']
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            all_feats.append(feat.cpu())
            all_ids.extend(pids)
        
        all_feats = torch.cat(all_feats, dim=0).numpy()
        all_ids = np.array(all_ids)
        N = len(all_ids)
        
        # 计算距离矩阵
        feat_norm = np.sum(all_feats ** 2, axis=1, keepdims=True)
        distmat = feat_norm + feat_norm.T - 2 * np.dot(all_feats, all_feats.T)
        
        # 评估
        cmc = np.zeros(10, dtype=float)
        all_ap = []
        
        for i in range(N):
            query_id = all_ids[i]
            dist = distmat[i]
            order = np.argsort(dist)
            
            # 排除自己
            if order[0] == i:
                order = order[1:]
            else:
                self_idx = np.where(order == i)[0]
                if len(self_idx) > 0:
                    order = np.delete(order, self_idx[0])
            
            matches = (all_ids[order] == query_id).astype(np.int32)
            num_gt = matches.sum()
            if num_gt == 0:
                continue
            
            # CMC
            first_match = np.where(matches == 1)[0]
            if len(first_match) > 0 and first_match[0] < 10:
                cmc[first_match[0]:] += 1
            
            # AP
            tmp_cmc = matches.cumsum()
            precision = tmp_cmc * matches / (np.arange(len(matches)) + 1)
            ap = precision.sum() / num_gt
            all_ap.append(ap)
        
        if len(all_ap) == 0:
            self.logger.warning("iPanda50 评估失败：没有有效 query")
            return None
        
        cmc = cmc / len(all_ap)
        mAP = float(np.mean(all_ap))
        rank1 = cmc[0] * 100
        rank5 = cmc[4] * 100 if len(cmc) >= 5 else cmc[-1] * 100
        
        self.logger.info("===== iPanda50 Official Evaluation =====")
        self.logger.info(f"  Rank-1 : {rank1:.2f}%")
        self.logger.info(f"  Rank-5 : {rank5:.2f}%")
        self.logger.info(f"  mAP    : {mAP * 100:.2f}%")
        self.logger.info("=========================================")
        
        return {'rank1': rank1, 'rank5': rank5, 'mAP': mAP * 100}
    
    @torch.no_grad()
    def evaluate_reid(self) -> Optional[Dict[str, float]]:
        """在验证集上评估 ReID 性能
        
        支持两种评估协议:
        1. iPanda50 协议: 多 query 评估，每个 ID 多张 query，计算平均性能
        2. 标准 ReID 协议: 每个 ID 1 张 query，排除同摄像头图像
        
        自动检测数据集类型并选择合适的评估协议。
        """
        if not self.query_dir or not self.gallery_dir:
            return None
        
        # 检测是否是 iPanda50 数据集（通过路径判断）
        is_ipanda50 = 'ipanda50' in self.query_dir.lower() or 'ipanda50' in self.gallery_dir.lower()
        
        # iPanda50 官方协议: 使用 test 目录进行 All-vs-All 评估
        if is_ipanda50:
            # 尝试找到 test 目录
            ipanda_base = os.path.dirname(self.query_dir)
            test_dir = os.path.join(ipanda_base, "test")
            if os.path.exists(test_dir):
                self.logger.info(f"iPanda50 官方协议: 使用 test 目录 All-vs-All 评估")
                return self._evaluate_ipanda50_official(test_dir)
            else:
                self.logger.info(f"iPanda50: test 目录不存在，使用旧协议 (query/gallery)")
        
        if not os.path.exists(self.query_dir) or not os.path.exists(self.gallery_dir):
            self.logger.warning(f"评估目录不存在: query={self.query_dir}, gallery={self.gallery_dir}")
            return None
        
        self.logger.info("开始 ReID 评估...")
        if is_ipanda50:
            self.logger.info("检测到 iPanda50 数据集，使用多 Query 评估协议")
        self.model.eval()
        
        from torchvision import transforms as T
        from torch.utils.data import Dataset, DataLoader
        from PIL import Image
        
        # 从文件名提取摄像头 ID 的辅助函数
        def extract_camera_id(img_name: str) -> int:
            """
            从文件名提取摄像头 ID
            支持格式：
            - DukeMTMC/Market: {pid}_c{camid}_f{frame}.jpg (e.g., 0101_c1_f0070929.jpg)
            - ATRW: 无摄像头信息，返回 -1
            """
            import re
            # 尝试匹配 _c{camid}_ 模式
            match = re.search(r'_c(\d+)_', img_name)
            if match:
                return int(match.group(1))
            # 尝试匹配 _c{camid}s{seq} 模式 (Market-1501)
            match = re.search(r'_c(\d+)s', img_name)
            if match:
                return int(match.group(1))
            # 无摄像头信息
            return -1
        
        # 评估数据集（支持摄像头 ID 提取）
        class EvalDataset(Dataset):
            def __init__(self, root_dir, transform):
                self.samples = []  # (img_path, identity, camera_id)
                self.transform = transform
                
                for identity in sorted(os.listdir(root_dir)):
                    id_dir = os.path.join(root_dir, identity)
                    if not os.path.isdir(id_dir):
                        continue
                    for img_name in os.listdir(id_dir):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(id_dir, img_name)
                            cam_id = extract_camera_id(img_name)
                            self.samples.append((img_path, identity, cam_id))
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                img_path, identity, cam_id = self.samples[idx]
                img = cv2.imread(img_path)
                if img is None:
                    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.transform(img)
                return img, identity, cam_id
        
        # 评估时使用与训练相同的尺寸（不加 Normalize，模型内部会处理）
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.img_height, self.img_width)),
            T.ToTensor(),
            # 注意: 不做 Normalize，模型内部 forward 会自己归一化
            # 见 joint_model.py 第 476-479 行
        ])
        
        # 分别加载 query 和 gallery
        query_dataset = EvalDataset(self.query_dir, transform)
        gallery_dataset = EvalDataset(self.gallery_dir, transform)
        
        # Linux 可用多进程加速，Windows 设为 0
        import platform
        num_workers = 4 if platform.system() != 'Windows' else 0
        query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False, num_workers=num_workers)
        gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False, num_workers=num_workers)
        
        self.logger.info(f"Query: {len(query_dataset)} 张, Gallery: {len(gallery_dataset)} 张")
        
        # 提取特征（包含摄像头 ID）
        def extract_features(loader):
            feats, ids, cams = [], [], []
            with torch.no_grad():
                for imgs, pids, cam_ids in loader:
                    imgs = imgs.to(self.device)
                    output = self.model(imgs, boxes_list=None)
                    feat = output['features']
                    feat = torch.nn.functional.normalize(feat, p=2, dim=1)
                    feats.append(feat.cpu())
                    ids.extend(pids)
                    cams.extend(cam_ids.tolist() if isinstance(cam_ids, torch.Tensor) else cam_ids)
            return torch.cat(feats, dim=0).numpy(), ids, cams
        
        q_feats, q_ids, q_cams = extract_features(query_loader)
        g_feats, g_ids, g_cams = extract_features(gallery_loader)
        
        # 检查是否有摄像头信息
        has_camera_info = any(c != -1 for c in q_cams) and any(c != -1 for c in g_cams)
        
        # iPanda50 特殊处理：需要将同一 ID 的多个 query 合并统计
        # 同时需要把 query 也加入 gallery（官方协议：test 集内互相检索）
        if is_ipanda50:
            self.logger.info(f"iPanda50 评估: Query {len(q_ids)} 张, Gallery {len(g_ids)} 张")
            self.logger.info("使用官方协议：Query 图像也加入 Gallery 进行互检索")
            
            # 合并 query 和 gallery 特征（官方协议：test 集内所有图像互相检索）
            all_feats = np.concatenate([q_feats, g_feats], axis=0)
            all_ids = q_ids + g_ids
            all_cams = q_cams + g_cams
            
            # 计算 query vs all 距离矩阵
            q_norm = np.sum(q_feats ** 2, axis=1, keepdims=True)
            all_norm = np.sum(all_feats ** 2, axis=1, keepdims=True).T
            distmat = q_norm + all_norm - 2 * np.dot(q_feats, all_feats.T)
            
            all_ids_arr = np.array(all_ids)
            num_q = len(q_ids)
            
            # 评估
            cmc = np.zeros(10)
            all_ap = []
            
            for i in range(num_q):
                qid = q_ids[i]
                
                # 按距离排序
                order = np.argsort(distmat[i])
                
                # 找到所有同 ID 图像的索引
                same_id_index = np.argwhere(all_ids_arr == qid).flatten()
                
                # junk_index: 排除自己（距离为0）
                # 通过距离阈值排除自己：距离最小的那个通常是自己
                junk_index = np.array([order[0]], dtype=np.int64)  # 第一个匹配通常是自己
                
                # good_index: 同 ID 的其他图像
                good_index = np.setdiff1d(same_id_index, junk_index, assume_unique=False)
                
                if len(good_index) == 0:
                    continue
                
                # 从排序结果中移除 junk（自己）
                mask = np.isin(order, junk_index, invert=True)
                order = order[mask]
                
                # 在过滤后的 order 中查找匹配
                matches = np.isin(order, good_index)
                
                # CMC: 第一个正样本的位置
                first_match_positions = np.where(matches)[0]
                if len(first_match_positions) > 0:
                    first_match = first_match_positions[0]
                    if first_match < 10:
                        cmc[first_match:] += 1
                
                # AP
                num_rel = len(good_index)
                tmp_cmc = matches.astype(np.float32).cumsum()
                precision = tmp_cmc / (np.arange(len(matches)) + 1)
                ap = (precision * matches).sum() / num_rel
                all_ap.append(ap)
            
            if len(all_ap) == 0:
                self.logger.warning("评估失败：没有有效的 query")
                return None
            
            cmc = cmc / len(all_ap) * 100
            mAP = np.mean(all_ap) * 100
            
            self.logger.info("===== iPanda50 ReID Evaluation Results =====")
            self.logger.info(f"Query: {num_q}, Gallery (含Query): {len(all_ids)}")
            self.logger.info(f"Rank-1  : {cmc[0]:.2f}%")
            self.logger.info(f"Rank-5  : {cmc[4]:.2f}%")
            self.logger.info(f"Rank-10 : {cmc[9]:.2f}%")
            self.logger.info(f"mAP     : {mAP:.2f}%")
            self.logger.info("=============================================")
            
            self.model.train()
            return {
                'rank1': cmc[0],
                'rank5': cmc[4],
                'rank10': cmc[9],
                'mAP': mAP
            }
        
        # 标准评估协议（非 iPanda50）
        if has_camera_info:
            self.logger.info("检测到摄像头信息，使用标准 ReID 协议（排除同摄像头同ID图像）")
        else:
            self.logger.info("未检测到摄像头信息，使用简化协议（不排除同摄像头）")
        
        # 计算距离矩阵: query vs gallery
        q_norm = np.sum(q_feats ** 2, axis=1, keepdims=True)
        g_norm = np.sum(g_feats ** 2, axis=1, keepdims=True).T
        distmat = q_norm + g_norm - 2 * np.dot(q_feats, g_feats.T)
        
        g_ids_arr = np.array(g_ids)
        g_cams_arr = np.array(g_cams)
        num_q = len(q_ids)
        
        # 标准评估（支持摄像头排除）
        cmc = np.zeros(10)
        all_ap = []
        
        for i in range(num_q):
            qid = q_ids[i]
            qcam = q_cams[i]
            
            # 按距离排序
            order = np.argsort(distmat[i])
            
            # 标准 ReID 协议：定义 good_index 和 junk_index
            # good_index: 同 ID 不同摄像头的图像（真正的正样本）
            # junk_index: 同 ID 同摄像头的图像（需要排除）
            query_index = np.argwhere(g_ids_arr == qid).flatten()
            
            if has_camera_info and qcam != -1:
                camera_index = np.argwhere(g_cams_arr == qcam).flatten()
                # good_index: 同 ID 但不同摄像头
                good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
                # junk_index: 同 ID 且同摄像头（需排除）
                junk_index = np.intersect1d(query_index, camera_index)
            else:
                # 无摄像头信息时，所有同 ID 图像都是 good
                good_index = query_index
                junk_index = np.array([], dtype=np.int64)
            
            # 如果没有有效的正样本，跳过
            if len(good_index) == 0:
                continue
            
            # 从排序结果中移除 junk
            if len(junk_index) > 0:
                mask = np.in1d(order, junk_index, invert=True)
                order = order[mask]
            
            # 在过滤后的 order 中查找匹配
            matches = np.in1d(order, good_index)
            
            # CMC: 第一个正样本的位置
            first_match_positions = np.where(matches)[0]
            if len(first_match_positions) > 0:
                first_match = first_match_positions[0]
                if first_match < 10:
                    cmc[first_match:] += 1
            
            # AP (Average Precision) - 标准计算方式
            num_rel = len(good_index)
            tmp_cmc = matches.astype(np.float32).cumsum()
            precision = tmp_cmc / (np.arange(len(matches)) + 1)
            # 只在正样本位置计算 precision
            ap = (precision * matches).sum() / num_rel
            all_ap.append(ap)
        
        if len(all_ap) == 0:
            self.logger.warning("评估失败：没有有效的 query")
            return None
        
        cmc = cmc / len(all_ap) * 100
        mAP = np.mean(all_ap) * 100
        
        self.logger.info("===== ReID Evaluation Results =====")
        self.logger.info(f"Query: {num_q}, Gallery: {len(g_ids)}")
        self.logger.info(f"Rank-1  : {cmc[0]:.2f}%")
        self.logger.info(f"Rank-5  : {cmc[4]:.2f}%")
        self.logger.info(f"Rank-10 : {cmc[9]:.2f}%")
        self.logger.info(f"mAP     : {mAP:.2f}%")
        self.logger.info("===================================")
        
        self.model.train()
        
        return {
            'rank1': cmc[0],
            'rank5': cmc[4],
            'rank10': cmc[9],
            'mAP': mAP
        }
    
    def save_checkpoint(self, epoch: int, metrics: dict, phase: int, is_best: bool = False, suffix: str = ''):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'num_classes': self.num_classes,
            'metrics': metrics,
            'config': self.config,
            'best_rank1': self.best_rank1,
            'best_map': self.best_map
        }
        
        if is_best:
            path = os.path.join(self.output_dir, f'joint_best{suffix}.pth')
        else:
            path = os.path.join(self.output_dir, f'joint_phase{phase}_epoch{epoch+1}.pth')
        
        torch.save(checkpoint, path)
        self.logger.info(f"模型已保存: {path}")


# ============================================================================
#                           日志设置
# ============================================================================

def setup_logging(log_dir: str) -> logging.Logger:
    """配置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('JointTraining')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(os.path.join(log_dir, 'joint_training.log'), encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
#                           主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='端到端联合训练')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='原始完整图像目录 (结构: data_dir/identity_id/image.jpg)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='模型输出目录 (默认从yaml读取)')
    parser.add_argument('--config', type=str, default='./app/core/illumination_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--backbone', type=str, default=None,
                        choices=SUPPORTED_BACKBONES,
                        help='骨干网络类型 (默认从yaml读取)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小 (默认从yaml读取)')
    parser.add_argument('--phase1_epochs', type=int, default=None,
                        help='Phase 1 轮数 (默认从yaml读取)')
    parser.add_argument('--phase2_epochs', type=int, default=None,
                        help='Phase 2 轮数 (默认从yaml读取)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='基础学习率 (默认从yaml读取)')
    parser.add_argument('--num_stripes', type=int, default=None,
                        help='局部特征条纹数 (默认从yaml读取)')
    parser.add_argument('--device', type=str, default='auto',
                        help='训练设备')
    parser.add_argument('--resume', type=str, default=None,
                        help='断点路径（joint_phase*_epoch*.pth 或 joint_best.pth）')
    parser.add_argument('--query_dir', type=str, default=None,
                        help='评估用 query 目录 (默认从yaml读取)')
    parser.add_argument('--gallery_dir', type=str, default=None,
                        help='评估用 gallery 目录 (默认从yaml读取)')
    parser.add_argument('--eval_interval', type=int, default=None,
                        help='每隔多少 epoch 进行一次 ReID 评估 (默认从yaml读取)')
    parser.add_argument('--p_size', type=int, default=None,
                        help='PK采样: 每batch采样P个身份 (默认从yaml读取)')
    parser.add_argument('--k_size', type=int, default=None,
                        help='PK采样: 每身份采样K张图像 (默认从yaml读取)')
    parser.add_argument('--circle_gamma', type=int, default=None,
                        help='Circle Loss gamma参数 (默认从yaml读取)')
    parser.add_argument('--img_height', type=int, default=None,
                        help='输入图像高度 (默认从yaml读取)')
    parser.add_argument('--img_width', type=int, default=None,
                        help='输入图像宽度 (默认: 256，行人ReID用128，老虎用512)')
    
    # IICL (光照不变对比学习) 参数
    parser.add_argument('--use_iicl', action='store_true', default=True,
                        help='启用 IICL 光照不变对比学习 (默认: True)')
    parser.add_argument('--no_iicl', action='store_true',
                        help='禁用 IICL')
    parser.add_argument('--iicl_weight', type=float, default=0.5,
                        help='IICL 损失权重 (默认: 0.5)')
    parser.add_argument('--iicl_variants', type=int, default=2,
                        help='每张图生成的光照变体数量 (默认: 2)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载工作进程数 (默认: 4)')
    
    args = parser.parse_args()
    
    # 处理 IICL 开关
    if args.no_iicl:
        args.use_iicl = False
    
    # ========== 加载配置文件，命令行参数覆盖 ==========
    import yaml
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    
    # 从配置文件读取默认值
    training_cfg = config.get('training', {})
    model_cfg = config.get('model', {})
    pk_cfg = training_cfg.get('pk_sampler', {})
    phase1_cfg = training_cfg.get('phase1', {}) or config.get('training', {}).get('phases', {}).get('phase1', {})
    phase2_cfg = training_cfg.get('phase2', {}) or config.get('training', {}).get('phases', {}).get('phase2', {})
    
    # 合并参数：命令行 > 配置文件 > 硬编码默认值
    def get_param(cli_val, cfg_val, default):
        """命令行优先，其次配置文件，最后默认值"""
        if cli_val is not None:
            return cli_val
        if cfg_val is not None:
            return cfg_val
        return default
    
    # 合并所有参数
    output_dir = get_param(args.output_dir, training_cfg.get('output_dir'), './checkpoints/joint')
    backbone = get_param(args.backbone, model_cfg.get('backbone'), 'osnet_ain_x1_0')
    batch_size = get_param(args.batch_size, training_cfg.get('batch_size'), 32)
    phase1_epochs = get_param(args.phase1_epochs, phase1_cfg.get('epochs'), 10)
    phase2_epochs = get_param(args.phase2_epochs, phase2_cfg.get('epochs'), 100)
    learning_rate = get_param(args.learning_rate, training_cfg.get('optimizer', {}).get('lr'), 3.5e-4)
    num_stripes = get_param(args.num_stripes, model_cfg.get('local_extractor', {}).get('num_parts'), 6)
    query_dir = get_param(args.query_dir, training_cfg.get('query_dir'), None)
    gallery_dir = get_param(args.gallery_dir, training_cfg.get('gallery_dir'), None)
    eval_interval = get_param(args.eval_interval, training_cfg.get('eval_interval'), 5)
    p_size = get_param(args.p_size, pk_cfg.get('p'), None)
    k_size = get_param(args.k_size, pk_cfg.get('k'), 4)
    circle_gamma = get_param(args.circle_gamma, training_cfg.get('metric_learning', {}).get('circle_loss', {}).get('gamma'), 256)
    img_height = get_param(args.img_height, training_cfg.get('image_height'), 256)
    img_width = get_param(args.img_width, training_cfg.get('image_width'), 256)
    num_workers = get_param(args.num_workers, config.get('hardware', {}).get('num_workers'), 4)
    
    # 打印最终使用的参数
    print(f"\n{'='*60}")
    print(f"参数来源: 配置文件 = {args.config}")
    print(f"{'='*60}")
    print(f"  backbone: {backbone}")
    print(f"  batch_size: {batch_size}")
    print(f"  p_size: {p_size}, k_size: {k_size}")
    print(f"  phase1_epochs: {phase1_epochs}, phase2_epochs: {phase2_epochs}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  img_size: {img_height}x{img_width}")
    print(f"  num_workers: {num_workers}")
    print(f"{'='*60}\n")
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"[ERROR] 数据目录不存在: {args.data_dir}")
        sys.exit(1)
    
    logger = setup_logging(output_dir)
    
    trainer = JointTrainer(
        data_dir=args.data_dir,
        output_dir=output_dir,
        config_path=args.config,
        backbone=backbone,
        batch_size=batch_size,
        phase1_epochs=phase1_epochs,
        phase2_epochs=phase2_epochs,
        learning_rate=learning_rate,
        num_stripes=num_stripes,
        device=args.device,
        logger=logger,
        resume_checkpoint=args.resume,
        query_dir=query_dir,
        gallery_dir=gallery_dir,
        eval_interval=eval_interval,
        p_size=p_size,
        k_size=k_size,
        circle_gamma=circle_gamma,
        img_height=img_height,
        img_width=img_width,
        # IICL 参数
        use_iicl=args.use_iicl,
        iicl_weight=args.iicl_weight,
        iicl_num_variants=args.iicl_variants,
        # 数据加载参数
        num_workers=num_workers,
    )
    
    trainer.train()


if __name__ == '__main__':
    main()
