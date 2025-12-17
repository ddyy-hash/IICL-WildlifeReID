#!/usr/bin/env python3
"""联合 ReID 模型定义

包含：
- SoftMaskGenerator / SoftMaskGaussian：从检测框生成可微分软掩码
- LocalFeatureExtractor：条纹 + 全局 + 注意力 + BNNeck
- JointReIDModel：光照模块 + OSNet-AIN 骨干 + 软掩码 + 局部特征头
"""

from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchreid

# 使用 IPAID 模块替代 v1 掩码模块（TCSVT 级别的 Retinex 光照分解）
from .illumination_module_v2 import IPAIDModule, IlluminationVariantGenerator, IlluminationContrastiveLoss


# ============================================================================
#                           软掩码生成器
# ============================================================================

class SoftMaskGenerator(nn.Module):
    """可微分的软掩码生成器

    将检测框转换为可微分的注意力掩码，使得梯度可以从 ReID 损失
    流回光照模块（掩码本身是连续的 0~1）。

    数学原理：
        soft_mask = sigmoid(-signed_distance * temperature)
        - 框内: signed_distance < 0 → sigmoid(正数) → 接近 1
        - 框外: signed_distance > 0 → sigmoid(负数) → 接近 0
        - 边界: signed_distance = 0 → sigmoid(0) = 0.5
    """

    def __init__(self, temperature: float = 10.0, margin: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        boxes: torch.Tensor,
        feature_size: Tuple[int, int],
        image_size: Tuple[int, int],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        h, w = feature_size
        H, W = image_size
        device = device or (boxes.device if boxes is not None and boxes.numel() > 0 else torch.device("cpu"))

        # 创建归一化坐标网格 [0, 1]
        y_coords = torch.linspace(0, 1, h, device=device)
        x_coords = torch.linspace(0, 1, w, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        if boxes is None or boxes.numel() == 0:
            # 没有检测到目标，返回全 1 掩码（使用全图特征）
            return torch.ones(1, h, w, device=device)

        combined_mask = torch.zeros(h, w, device=device)

        # 归一化边界框坐标到 [0, 1]
        boxes_norm = boxes.clone().float()
        boxes_norm[:, [0, 2]] /= W  # x
        boxes_norm[:, [1, 3]] /= H  # y

        for box in boxes_norm:
            x1, y1, x2, y2 = box

            # 扩展边界框（避免截断目标边缘）
            bw, bh = x2 - x1, y2 - y1
            x1 = torch.clamp(x1 - self.margin * bw, 0, 1)
            y1 = torch.clamp(y1 - self.margin * bh, 0, 1)
            x2 = torch.clamp(x2 + self.margin * bw, 0, 1)
            y2 = torch.clamp(y2 + self.margin * bh, 0, 1)

            # 计算到边界框的距离（框外为正，框内为 0）
            dx = torch.maximum(x1 - xx, torch.zeros_like(xx))
            dx = torch.maximum(dx, xx - x2)

            dy = torch.maximum(y1 - yy, torch.zeros_like(yy))
            dy = torch.maximum(dy, yy - y2)

            distance = torch.sqrt(dx**2 + dy**2 + 1e-8)

            inside_x = (xx >= x1) & (xx <= x2)
            inside_y = (yy >= y1) & (yy <= y2)
            inside = inside_x & inside_y

            # 框内设为一个小的负值，使 sigmoid 后接近 1
            signed_distance = torch.where(inside, -0.1 * torch.ones_like(distance), distance)

            box_mask = torch.sigmoid(-signed_distance * self.temperature)
            combined_mask = torch.maximum(combined_mask, box_mask)

        return combined_mask.unsqueeze(0)  # (1, h, w)


class SoftMaskGaussian(nn.Module):
    """高斯软掩码生成器（以框中心为高斯中心）"""

    def __init__(self, sigma_ratio: float = 0.4) -> None:
        super().__init__()
        self.sigma_ratio = sigma_ratio

    def forward(
        self,
        boxes: torch.Tensor,
        feature_size: Tuple[int, int],
        image_size: Tuple[int, int],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        h, w = feature_size
        H, W = image_size
        device = device or (boxes.device if boxes is not None and boxes.numel() > 0 else torch.device("cpu"))

        y_coords = torch.linspace(0, H, h, device=device)
        x_coords = torch.linspace(0, W, w, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        if boxes is None or boxes.numel() == 0:
            return torch.ones(1, h, w, device=device)

        combined_mask = torch.zeros(h, w, device=device)

        for box in boxes:
            x1, y1, x2, y2 = box.float()

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = (x2 - x1), (y2 - y1)

            sigma_x = bw * self.sigma_ratio + 1e-8
            sigma_y = bh * self.sigma_ratio + 1e-8

            gaussian = torch.exp(
                -((xx - cx) ** 2 / (2 * sigma_x**2) + (yy - cy) ** 2 / (2 * sigma_y**2))
            )

            combined_mask = torch.maximum(combined_mask, gaussian)

        return combined_mask.unsqueeze(0)


# ============================================================================
#                           局部特征提取器
# ============================================================================

class LocalFeatureExtractor(nn.Module):
    """局部特征提取器（水平条纹划分 + 注意力融合 + 可形变条纹）
    
    新增 v2 功能：
    - dropout: 防过拟合的 Dropout 层 (对小类别数据集重要)
    """

    def __init__(
        self,
        backbone_dim: int = 512,
        hidden_dim: int = 256,
        num_stripes: int = 6,
        num_classes: int = 100,
        use_deformable_stripes: bool = False,
        max_offset_ratio: float = 0.2,
        dropout: float = 0.0,  # v2 新增: Dropout 概率
    ) -> None:
        super().__init__()
        self.num_stripes = num_stripes
        self.backbone_dim = backbone_dim
        self.hidden_dim = hidden_dim
        self.use_deformable_stripes = use_deformable_stripes
        self.max_offset_ratio = max_offset_ratio
        self.dropout_rate = dropout  # v2: 保存 dropout 率

        # 每个条纹的特征处理
        self.stripe_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(backbone_dim, hidden_dim, 1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_stripes)
            ]
        )

        # 全局池化分支
        self.global_conv = nn.Sequential(
            nn.Conv2d(backbone_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 注意力融合
        total_parts = num_stripes + 1  # 条纹 + 全局
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * total_parts, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, total_parts),
            nn.Softmax(dim=1),
        )

        # 最终融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * total_parts, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # v2 新增: Dropout 层 (防过拟合，对 iPanda50 等小类别数据集关键)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # BNNeck（用于度量学习）
        self.bn_neck = nn.BatchNorm1d(hidden_dim)
        self.bn_neck.bias.requires_grad_(False)

        # 分类器（可选，用于辅助训练）
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=False)

        # 可学习条纹偏移（Deformable Stripes）
        if self.use_deformable_stripes:
            # 每个条纹一个标量偏移，初始化为 0：初始时与均匀条纹一致
            self.stripe_offsets = nn.Parameter(torch.zeros(num_stripes))

    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """从骨干特征图提取局部+全局融合特征"""
        B, C, H, W = feature_map.shape

        part_features = []

        # 条纹特征
        for i in range(self.num_stripes):
            stripe_h = H // self.num_stripes

            if self.use_deformable_stripes:
                # 以当前条纹中心为基准，加入可学习偏移
                base_center = (i + 0.5) * stripe_h
                offset = torch.tanh(self.stripe_offsets[i]) * self.max_offset_ratio * H
                center = base_center + offset

                start_h = center - stripe_h / 2.0
                end_h = center + stripe_h / 2.0

                # 限制在合法范围，并确保至少包含 1 个像素行
                start_h = int(torch.clamp(start_h, 0, max(H - 1, 0)).item())
                end_h = int(torch.clamp(end_h, start_h + 1, H).item())
            else:
                start_h = i * stripe_h
                end_h = (i + 1) * stripe_h if i < self.num_stripes - 1 else H

            stripe = feature_map[:, :, start_h:end_h, :]

            stripe_feat = self.stripe_convs[i](stripe)
            stripe_feat = F.adaptive_avg_pool2d(stripe_feat, 1).flatten(1)
            part_features.append(stripe_feat)

        # 全局特征
        global_feat = self.global_conv(feature_map)
        global_feat = F.adaptive_avg_pool2d(global_feat, 1).flatten(1)
        part_features.append(global_feat)

        # 拼接
        concat_features = torch.cat(part_features, dim=1)

        # 注意力加权
        attention_weights = self.attention(concat_features)

        weighted_features = []
        for i, feat in enumerate(part_features):
            weighted = feat * attention_weights[:, i : i + 1]
            weighted_features.append(weighted)

        # 融合
        fused = torch.cat(weighted_features, dim=1)
        features = self.fusion(fused)

        # v2: 应用 Dropout (训练时随机丢弃，防止过拟合)
        features = self.dropout(features)

        # BNNeck
        features_bn = self.bn_neck(features)

        # 分类
        logits = self.classifier(features_bn)

        return features, logits


# ============================================================================
#                           联合 ReID 模型
# ============================================================================

# Backbone 维度映射
BACKBONE_DIM_MAP = {
    # OSNet 系列
    "osnet_x1_0": 512,
    "osnet_x0_75": 512,
    "osnet_x0_5": 512,
    "osnet_x0_25": 512,
    "osnet_ain_x1_0": 512,
    "osnet_ain_x0_75": 512,
    "osnet_ain_x0_5": 512,
    "osnet_ain_x0_25": 512,
    "osnet_ibn_x1_0": 512,
    # ResNet 系列
    "resnet50": 2048,
    "resnet50_fc512": 512,
    "resnet101": 2048,
    "resnet152": 2048,
    # ResNet-IBN 系列
    "resnet50_ibn_a": 2048,
    "resnet50_ibn_b": 2048,
    "resnet101_ibn_a": 2048,
    # 其他
    "densenet121": 1024,
    "mobilenetv2_x1_0": 1280,
    "mobilenetv2_x1_4": 1792,
    "shufflenet": 960,
    "squeezenet1_0": 512,
    "squeezenet1_1": 512,
}

# 支持的 backbone 列表
SUPPORTED_BACKBONES = list(BACKBONE_DIM_MAP.keys())


def get_backbone_dim(backbone_name: str) -> int:
    """获取 backbone 输出维度"""
    if backbone_name in BACKBONE_DIM_MAP:
        return BACKBONE_DIM_MAP[backbone_name]
    # 默认猜测
    if "resnet" in backbone_name.lower():
        return 2048
    return 512


class JointReIDModel(nn.Module):
    """端到端联合 ReID 模型

    整合光照不变性模块、骨干网络、软掩码注意力、局部特征提取器，
    实现完全可微的端到端训练和推理。
    
    支持的 backbone:
    - OSNet 系列: osnet_x1_0, osnet_ain_x1_0, osnet_ibn_x1_0 等
    - ResNet 系列: resnet50, resnet101, resnet152
    - ResNet-IBN 系列: resnet50_ibn_a, resnet50_ibn_b, resnet101_ibn_a
    - 其他: densenet121, mobilenetv2_x1_0 等
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "osnet_ain_x1_0",
        num_stripes: int = 6,
        hidden_dim: int = 256,
        pretrained_backbone: bool = True,
        soft_mask_temperature: float = 10.0,
        soft_mask_type: str = "sigmoid",  # "sigmoid" or "gaussian"
        use_ipaid: bool = True,  # 是否使用IPAID光照模块
        dropout: float = 0.0,  # v2 新增: 特征 Dropout 概率 (防过拟合)
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.use_ipaid = use_ipaid  # 保存IPAID开关状态
        self.dropout_rate = dropout  # v2: 保存 dropout 率

        # 1. 光照不变性模块（IPAID：基于 Retinex 的物理光照分解）
        if use_ipaid:
            self.illumination = IPAIDModule(
                base_channels=32,
                num_scales=3,
                refine_iterations=1,
                use_sensitivity=True,
                use_refinement=True,
            )
            
            # 1.5 IICL: 光照变体生成器（用于对比学习）
            self.variant_generator = IlluminationVariantGenerator(
                num_variants=2,
                gamma_range=(0.6, 1.4),
                scale_range=(0.7, 1.3),
                local_noise_std=0.1,
            )
        else:
            self.illumination = None
            self.variant_generator = None

        # 2. 骨干网络
        self.backbone = torchreid.models.build_model(
            name=backbone_name,
            num_classes=num_classes,
            loss="softmax",
            pretrained=pretrained_backbone,
        )
        self.backbone_dim = get_backbone_dim(backbone_name)

        # 3. 软掩码生成器
        if soft_mask_type == "gaussian":
            self.soft_mask_generator = SoftMaskGaussian(sigma_ratio=0.4)
        else:
            self.soft_mask_generator = SoftMaskGenerator(
                temperature=soft_mask_temperature,
                margin=0.1,
            )

        # 4. 局部特征提取器
        self.local_extractor = LocalFeatureExtractor(
            backbone_dim=self.backbone_dim,
            hidden_dim=hidden_dim,
            num_stripes=num_stripes,
            num_classes=num_classes,
            use_deformable_stripes=True,
            max_offset_ratio=0.2,
            dropout=dropout,  # v2: 传递 dropout 参数
        )

        print("[INFO] 联合ReID模型初始化完成")
        print(f"  骨干网络: {backbone_name}")
        print(f"  骨干维度: {self.backbone_dim}")
        print(f"  条纹数: {num_stripes}")
        print(f"  类别数: {num_classes}")
        print(f"  软掩码类型: {soft_mask_type}")
        print(f"  IPAID光照模块: {'启用' if use_ipaid else '禁用'}")

    def freeze_backbone(self, freeze: bool = True) -> None:
        """冻结或解冻骨干网络参数"""
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def freeze_illumination(self, freeze: bool = True) -> None:
        """冻结或解冻光照模块参数"""
        if self.illumination is not None:
            for p in self.illumination.parameters():
                p.requires_grad = not freeze

    def extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取骨干网络特征图（支持多种 backbone）"""
        name = self.backbone_name.lower()
        
        if "osnet" in name:
            # OSNet 系列结构
            x = self.backbone.conv1(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.conv2(x)
            x = self.backbone.conv3(x)
            x = self.backbone.conv4(x)
            x = self.backbone.conv5(x)
        elif "resnet" in name or "densenet" in name:
            # ResNet / ResNet-IBN / DenseNet 结构
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
        elif "mobilenet" in name:
            # MobileNetV2 结构
            x = self.backbone.features(x)
        elif "shufflenet" in name:
            # ShuffleNet 结构
            x = self.backbone.conv1(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.stage2(x)
            x = self.backbone.stage3(x)
            x = self.backbone.stage4(x)
            x = self.backbone.conv5(x)
        elif "squeezenet" in name:
            # SqueezeNet 结构
            x = self.backbone.features(x)
        else:
            # 尝试通用方法：调用 backbone 的 featuremaps 方法
            if hasattr(self.backbone, 'featuremaps'):
                x = self.backbone.featuremaps(x)
            else:
                raise NotImplementedError(f"不支持的 backbone: {self.backbone_name}")
        return x

    def forward(
        self,
        images: torch.Tensor,
        boxes_list: Optional[List[torch.Tensor]] = None,
        return_illuminated: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """端到端前向传播（简化版：无前景掩码）"""
        B, _, H, W = images.shape
        device = images.device

        # 1. 光照归一化（IPAID：Retinex 分解，直接输出R_retinex）
        if self.use_ipaid and self.illumination is not None:
            ipaid_details = self.illumination.forward_with_details(images)
            illuminated = ipaid_details['reflectance']  # 反射层作为光照归一化后的图像
            illumination_map = ipaid_details['illumination']  # 光照图（用于可视化和损失）
        else:
            # 禁用IPAID时，直接使用原始图像
            ipaid_details = None
            illuminated = images
            illumination_map = None

        # 2. ImageNet 归一化（OSNet 骨干网络期望 ImageNet 归一化的输入）
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        illuminated_normalized = (illuminated - mean) / std

        # 3. 骨干网络特征提取
        feature_map = self.extract_backbone_features(illuminated_normalized)
        _, _, fh, fw = feature_map.shape

        # 4. 软掩码加权（可选，用于检测框场景）
        soft_masks: Optional[List[torch.Tensor]]
        if boxes_list is not None:
            weighted_features = []
            soft_masks = []

            for i in range(B):
                feat = feature_map[i : i + 1]  # (1, C, fh, fw)
                boxes = boxes_list[i]

                soft_mask = self.soft_mask_generator(boxes, (fh, fw), (H, W), device)  # (1, fh, fw)
                soft_masks.append(soft_mask)

                mask_expanded = soft_mask.unsqueeze(0)  # (1, 1, fh, fw)
                weighted_feat = feat * mask_expanded  # (1, C, fh, fw)
                weighted_features.append(weighted_feat)

            feature_map = torch.cat(weighted_features, dim=0)
        else:
            soft_masks = None

        # 5. 局部特征提取
        features, logits = self.local_extractor(feature_map)

        output: Dict[str, torch.Tensor] = {
            "features": features,
            "logits": logits,
            "ipaid_details": ipaid_details,  # IPAID 中间结果（用于 IPAIDLoss）
            "illumination_map": illumination_map,  # 光照图
        }

        if return_illuminated:
            output["illuminated"] = illuminated

        if soft_masks is not None:
            output["soft_masks"] = soft_masks

        return output

    def forward_with_contrastive(
        self,
        images: torch.Tensor,
        num_variants: int = 2,
    ) -> Dict[str, torch.Tensor]:
        """
        带光照对比学习的前向传播 (IICL) - 显存优化版本
        
        核心流程：
        1. Retinex分解原图 → R, L
        2. 变换L生成光照变体 → L'_1, L'_2, ...
        3. 重建变体图像 → I'_1 = R × L'_1, ...
        4. 对原图提取特征（有梯度）
        5. 对变体提取特征（无梯度，节省显存）
        6. 计算对比损失（拉近原图和变体的特征）
        
        显存优化：变体特征提取使用 torch.no_grad()
        对比损失只需要拉近特征距离，变体侧不需要梯度
        
        Args:
            images: 输入图像 [B, 3, H, W]
            num_variants: 每张图生成的光照变体数量
        
        Returns:
            output: 包含原图特征、变体特征、IPAID细节等
        """
        B, _, H, W = images.shape
        device = images.device
        
        # 如果IPAID被禁用，退回到普通forward
        if not self.use_ipaid or self.illumination is None:
            output = self.forward(images)
            output['features_variants'] = []  # 无变体
            return output
        
        # 1. IPAID光照分解
        ipaid_details = self.illumination.forward_with_details(images)
        reflectance = ipaid_details['reflectance']  # R
        illumination = ipaid_details['illumination']  # L
        
        # 2. 生成光照变体（detach，不需要通过变体反传梯度）
        with torch.no_grad():
            variants = self.variant_generator(
                reflectance=reflectance.detach(),
                illumination=illumination.detach(),
                num_variants=num_variants,
            )
        
        # 3. ImageNet归一化
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        
        # 原图特征（有梯度）
        reflectance_normalized = (reflectance - mean) / std
        feature_map_orig = self.extract_backbone_features(reflectance_normalized)
        features_orig, logits_orig = self.local_extractor(feature_map_orig)
        
        # 变体特征（无梯度，节省显存）
        features_variants = []
        with torch.no_grad():
            for variant in variants:
                variant_normalized = (variant - mean) / std
                feature_map_var = self.extract_backbone_features(variant_normalized)
                features_var, _ = self.local_extractor(feature_map_var)
                features_variants.append(features_var.detach())
        
        # 4. 返回结果
        output = {
            "features": features_orig,
            "logits": logits_orig,
            "features_variants": features_variants,  # IICL核心：变体特征列表（detached）
            "variants": variants,                    # 变体图像（用于可视化）
            "ipaid_details": ipaid_details,
            "illumination_map": illumination,
            "illuminated": reflectance,
        }
        
        return output
