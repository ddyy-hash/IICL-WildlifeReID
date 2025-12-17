#!/usr/bin/env python3
"""
Identity-Preserving Adaptive Illumination Decomposition Module (IPAID)

TCSVT级别的光照不变性模块，核心创新点：
1. 基于Retinex的物理光照分解 (I = R × L)
2. 多尺度光照估计 (全局 + 局部)
3. 自适应校正强度 (敏感度图引导)
4. 身份保持约束 (反射层保持判别性)
5. 即插即用设计 (适配任意backbone)

作者: [Ding yu]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import logging


# ============================================================================
#                           基础组件
# ============================================================================

class ConvBNReLU(nn.Module):
    """标准卷积块: Conv + BN + ReLU"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, 
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积: 减少参数量"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, 
                                    padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(self.bn(x))


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


# ============================================================================
#                           多尺度光照估计器
# ============================================================================

class MultiScaleIlluminationEstimator(nn.Module):
    """
    多尺度光照估计器
    
    处理不同尺度的光照变化：
    - 全局尺度：整体曝光度
    - 中尺度：大面积阴影
    - 局部尺度：小区域高光/阴影
    """
    
    def __init__(self, in_channels: int = 1, base_channels: int = 32, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        
        # 共享特征提取器
        self.shared_encoder = nn.Sequential(
            ConvBNReLU(in_channels, base_channels, 3, 1, 1),
            ConvBNReLU(base_channels, base_channels, 3, 1, 1),
        )
        
        # 各尺度的光照估计分支
        self.scale_branches = nn.ModuleList()
        for i in range(num_scales):
            branch = nn.Sequential(
                DepthwiseSeparableConv(base_channels, base_channels, 3, 1, 1),
                DepthwiseSeparableConv(base_channels, base_channels, 3, 1, 1),
                nn.Conv2d(base_channels, 1, 1, bias=True),
            )
            self.scale_branches.append(branch)
        
        # 可学习的尺度融合权重
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
        # 通道注意力用于加权
        self.channel_attention = ChannelAttention(base_channels)
    
    def forward(self, luminance: torch.Tensor) -> torch.Tensor:
        """
        Args:
            luminance: 亮度图 [B, 1, H, W]
        
        Returns:
            illumination: 光照图 [B, 1, H, W]，范围 [0.1, 10]
        """
        B, _, H, W = luminance.shape
        
        # 共享特征
        features = self.shared_encoder(luminance)
        features = features * self.channel_attention(features)
        
        # 多尺度光照估计
        illuminations = []
        
        for i, branch in enumerate(self.scale_branches):
            # 计算当前尺度的下采样因子
            scale_factor = 2 ** (self.num_scales - 1 - i)  # 4, 2, 1
            
            if scale_factor > 1:
                # 下采样 → 估计 → 上采样
                feat_down = F.interpolate(features, scale_factor=1/scale_factor, 
                                         mode='bilinear', align_corners=False)
                illum_down = branch(feat_down)
                illum = F.interpolate(illum_down, size=(H, W), 
                                     mode='bilinear', align_corners=False)
            else:
                illum = branch(features)
            
            illuminations.append(illum)
        
        # 加权融合
        weights = F.softmax(self.scale_weights, dim=0)
        illumination = sum(w * illum for w, illum in zip(weights, illuminations))
        
        # 映射到合理范围 [0.1, 10]，使用 softplus 确保正值
        illumination = F.softplus(illumination) + 0.1
        illumination = torch.clamp(illumination, 0.1, 10.0)
        
        return illumination


# ============================================================================
#                           敏感度估计器（自适应校正强度）
# ============================================================================

class SensitivityEstimator(nn.Module):
    """
    光照敏感度估计器
    
    学习每个像素的"光照敏感度"：
    - 高敏感度区域（如条纹）：需要强光照校正
    - 低敏感度区域（如背景）：弱校正或不校正
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        
        # 边缘检测分支（识别纹理区域）
        self.edge_branch = nn.Sequential(
            ConvBNReLU(in_channels, base_channels // 2, 3, 1, 1),
            ConvBNReLU(base_channels // 2, base_channels // 2, 3, 1, 1),
        )
        
        # 内容分支
        self.content_branch = nn.Sequential(
            ConvBNReLU(in_channels, base_channels, 3, 1, 1),
            DepthwiseSeparableConv(base_channels, base_channels, 3, 1, 1),
        )
        
        # 融合 + 敏感度输出
        self.fusion = nn.Sequential(
            ConvBNReLU(base_channels + base_channels // 2, base_channels, 3, 1, 1),
            DepthwiseSeparableConv(base_channels, base_channels // 2, 3, 1, 1),
            nn.Conv2d(base_channels // 2, 1, 1, bias=True),
            nn.Sigmoid()  # 输出 [0, 1]
        )
        
        # 空间注意力
        self.spatial_attention = SpatialAttention(kernel_size=7)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB图像 [B, 3, H, W]
        
        Returns:
            sensitivity: 敏感度图 [B, 1, H, W]，范围 [0, 1]
        """
        # 边缘特征
        edge_feat = self.edge_branch(x)
        
        # 内容特征
        content_feat = self.content_branch(x)
        content_feat = content_feat * self.spatial_attention(content_feat)
        
        # 融合
        fused = torch.cat([edge_feat, content_feat], dim=1)
        sensitivity = self.fusion(fused)
        
        # 降低敏感度上限，避免过度校正（调试发现S均值0.6-0.8太高）
        sensitivity = 0.1 + 0.5 * sensitivity  # 映射到 [0.1, 0.6]
        
        return sensitivity


# ============================================================================
#                           身份感知增强器（Identity-Aware Enhancer）
# ============================================================================

class IdentityAwareEnhancer(nn.Module):
    """
    身份感知增强器
    
    核心目标：
    - 增强有助于区分不同个体的特征（如独特花纹、斑点）
    - 抑制对识别无帮助的区域（如共同背景）
    - 与ReID损失联合训练，自动学习"什么特征对识别有用"
    
    设计思路：
    1. 基于R_retinex（颜色正确）作为输入
    2. 预测"身份显著性图"：哪些区域对区分个体重要
    3. 在显著区域增强对比度/细节，非显著区域保持或轻微平滑
    4. 通过triplet/softmax损失反向传播，学习最优增强策略
    
    数学形式：
    R_out = R_retinex + saliency × enhancement_residual
    - saliency: 身份显著性图 [0,1]
    - enhancement_residual: 增强残差（细节强化）
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        
        # === 显著性预测分支 ===
        # 预测哪些区域对身份识别重要
        self.saliency_encoder = nn.Sequential(
            ConvBNReLU(in_channels * 2, base_channels, 3, 1, 1),  # 原图 + R_retinex
            ConvBNReLU(base_channels, base_channels, 3, 1, 1),
            ConvBNReLU(base_channels, base_channels * 2, 3, 2, 1),  # 下采样
            ConvBNReLU(base_channels * 2, base_channels * 2, 3, 1, 1),
        )
        
        self.saliency_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(base_channels * 2, base_channels, 3, 1, 1),
            nn.Conv2d(base_channels, 1, 1),
            nn.Sigmoid()  # 输出 [0, 1] 的显著性图
        )
        
        # === 增强残差预测分支 ===
        # 预测细节增强的方向和幅度
        self.enhance_encoder = nn.Sequential(
            ConvBNReLU(in_channels + 1, base_channels, 3, 1, 1),  # R_retinex + 显著性
            ConvBNReLU(base_channels, base_channels, 3, 1, 1),
            DepthwiseSeparableConv(base_channels, base_channels, 3, 1, 1),
        )
        
        self.enhance_decoder = nn.Sequential(
            ConvBNReLU(base_channels, base_channels, 3, 1, 1),
            nn.Conv2d(base_channels, 3, 3, 1, 1),
            nn.Tanh()  # 输出 [-1, 1]，后面会缩放
        )
        
        # 可学习的增强强度参数
        self.enhance_strength = nn.Parameter(torch.tensor(0.1))  # 初始较小
        
        # 多尺度特征提取（捕捉不同粒度的身份特征）
        self.multiscale_pool = nn.ModuleList([
            nn.AdaptiveAvgPool2d(output_size) for output_size in [(8, 8), (16, 16), (32, 32)]
        ])
    
    def forward(self, x: torch.Tensor, R_retinex: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        对 R_retinex 进行身份感知增强
        
        Args:
            x: 原始图像 [B, 3, H, W]
            R_retinex: Retinex分解的反射层 [B, 3, H, W]（颜色准确）
        
        Returns:
            R_enhanced: 增强后的反射层 [B, 3, H, W]
            aux_outputs: 辅助输出（显著性图等，用于可视化和损失计算）
        """
        B, C, H, W = R_retinex.shape
        
        # 1. 预测身份显著性图
        saliency_input = torch.cat([x, R_retinex], dim=1)
        saliency_feat = self.saliency_encoder(saliency_input)
        saliency_map = self.saliency_decoder(saliency_feat)  # [B, 1, H, W]
        
        # 2. 提取多尺度上下文（帮助理解全局身份特征）
        context_features = []
        for pool in self.multiscale_pool:
            pooled = pool(saliency_feat)
            upsampled = F.interpolate(pooled, size=(H // 2, W // 2), mode='bilinear', align_corners=False)
            context_features.append(upsampled)
        
        # 3. 预测增强残差
        enhance_input = torch.cat([R_retinex, saliency_map], dim=1)
        enhance_feat = self.enhance_encoder(enhance_input)
        
        # 增强残差：[-strength, +strength] 范围
        strength = torch.clamp(self.enhance_strength, 0.05, 0.3)
        enhance_residual = self.enhance_decoder(enhance_feat) * strength
        
        # 4. 应用身份感知增强
        # 只在显著区域应用增强，非显著区域保持原样
        R_enhanced = R_retinex + saliency_map * enhance_residual
        
        # 5. 确保输出范围
        R_enhanced = torch.clamp(R_enhanced, 0.0, 1.0)
        
        # 辅助输出
        aux_outputs = {
            'saliency_map': saliency_map,          # 身份显著性图
            'enhance_residual': enhance_residual,  # 增强残差
            'enhance_strength': strength,          # 当前增强强度
        }
        
        return R_enhanced, aux_outputs


class LocalContrastEnhancer(IdentityAwareEnhancer):
    """
    [兼容别名] 局部对比度增强器 -> 身份感知增强器
    保留用于旧代码兼容
    """
    def forward(self, x: torch.Tensor, R_retinex: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        R_enhanced, aux = super().forward(x, R_retinex)
        # 返回显著性图作为"增益图"以保持接口兼容
        return R_enhanced, aux['saliency_map']


# ============================================================================
#                           反射层估计器（边缘感知）- 已弃用，保留兼容
# ============================================================================

class ReflectanceEstimator(nn.Module):
    """
    [已弃用] 原始反射层估计器
    保留用于加载旧checkpoint的兼容性
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            ConvBNReLU(in_channels, base_channels, 3, 1, 1),
            ConvBNReLU(base_channels, base_channels, 3, 1, 1),
            ConvBNReLU(base_channels, base_channels * 2, 3, 2, 1),
            ConvBNReLU(base_channels * 2, base_channels * 2, 3, 1, 1),
        )
        
        self.edge_detector = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1),
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(base_channels * 2 + 1, base_channels, 3, 1, 1),
            ConvBNReLU(base_channels, base_channels, 3, 1, 1),
            nn.Conv2d(base_channels, 3, 3, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge = torch.abs(self.edge_detector(x))
        edge_down = F.interpolate(edge, scale_factor=0.5, mode='bilinear', align_corners=False)
        feat = self.encoder(x)
        feat = torch.cat([feat, edge_down], dim=1)
        reflectance = self.decoder(feat)
        return reflectance


# ============================================================================
#                           一致性精炼模块
# ============================================================================

class ConsistencyRefiner(nn.Module):
    """
    一致性精炼模块
    
    确保 R × L ≈ I 的物理一致性，并通过残差学习精炼反射层
    """
    
    def __init__(self, base_channels: int = 32, num_iterations: int = 1):
        super().__init__()
        self.num_iterations = num_iterations
        
        # 残差修正网络
        self.refine_net = nn.Sequential(
            ConvBNReLU(7, base_channels, 3, 1, 1),  # 输入: I(3) + R(3) + L(1) = 7
            DepthwiseSeparableConv(base_channels, base_channels, 3, 1, 1),
            DepthwiseSeparableConv(base_channels, base_channels // 2, 3, 1, 1),
            nn.Conv2d(base_channels // 2, 3, 3, 1, 1),
            nn.Tanh()  # 残差范围 [-1, 1]
        )
        
        # 残差权重（可学习）
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, reflectance: torch.Tensor, illumination: torch.Tensor, 
                original: torch.Tensor) -> torch.Tensor:
        """
        Args:
            reflectance: 初始反射层 [B, 3, H, W]
            illumination: 光照图 [B, 1, H, W]
            original: 原始图像 [B, 3, H, W]
        
        Returns:
            refined_reflectance: 精炼后的反射层 [B, 3, H, W]
        """
        R = reflectance
        L = illumination
        I = original
        
        for _ in range(self.num_iterations):
            # 计算重建残差
            I_recon = R * L
            residual_input = torch.cat([I, R, L], dim=1)
            
            # 预测修正量
            correction = self.refine_net(residual_input)
            
            # 应用残差修正
            R = R + self.residual_weight * correction
            R = torch.clamp(R, 0.01, 0.99)
        
        return R


# ============================================================================
#                           主模块：IPAID
# ============================================================================

class IPAIDModule(nn.Module):
    """
    Identity-Preserving Adaptive Illumination Decomposition Module (IPAID)
    
    即插即用的光照不变性模块，可插入任意ReID backbone之前
    
    核心特点：
    1. 物理启发的Retinex分解
    2. 多尺度光照估计
    3. 自适应校正强度（敏感度图）
    4. 边缘感知的反射层估计
    5. 一致性精炼
    
    用法：
        ipaid = IPAIDModule()
        reflectance = ipaid(image)  # [B,3,H,W] → [B,3,H,W]
        features = backbone(reflectance)
    """
    
    def __init__(
        self,
        base_channels: int = 32,
        num_scales: int = 3,
        refine_iterations: int = 1,
        use_sensitivity: bool = True,
        use_refinement: bool = True,
    ):
        super().__init__()
        
        self.use_sensitivity = use_sensitivity
        self.use_refinement = use_refinement
        
        # 1. 多尺度光照估计器
        self.illumination_estimator = MultiScaleIlluminationEstimator(
            in_channels=1, 
            base_channels=base_channels,
            num_scales=num_scales
        )
        
        # 2. 敏感度估计器（可选）
        if use_sensitivity:
            self.sensitivity_estimator = SensitivityEstimator(
                in_channels=3,
                base_channels=base_channels
            )
        
        # 3. 反射层估计器（保留用于旧checkpoint兼容）
        self.reflectance_estimator = ReflectanceEstimator(
            in_channels=3,
            base_channels=base_channels * 2
        )
        
        # 4. 身份感知增强器（核心模块：增强有利于识别的特征）
        self.identity_enhancer = IdentityAwareEnhancer(
            in_channels=3,
            base_channels=base_channels,
        )
        # 兼容旧接口
        self.texture_enhancer = self.identity_enhancer
        
        # 5. 一致性精炼（可选）
        if use_refinement:
            self.refiner = ConsistencyRefiner(
                base_channels=base_channels,
                num_iterations=refine_iterations
            )
        
        # 缓存中间结果（用于损失计算和可视化）
        self.cached_illumination = None
        self.cached_sensitivity = None
        self.cached_reflectance_init = None
        self.cached_color_risk = None  # 颜色失真风险图
        self.cached_saliency_map = None  # 身份显著性图
        self.cached_gain_map = None  # 兼容旧接口
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def rgb_to_luminance(self, x: torch.Tensor) -> torch.Tensor:
        """RGB → 亮度通道"""
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    
    def rgb_to_chromaticity(self, x: torch.Tensor) -> torch.Tensor:
        """
        RGB → 色度坐标 (颜色角度保护)
        返回归一化的颜色方向向量，与亮度无关
        """
        eps = 1e-6
        norm = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + eps)
        return x / norm  # [B, 3, H, W] 单位向量
    
    def apply_illumination_in_lab(self, x: torch.Tensor, L_factor: torch.Tensor) -> torch.Tensor:
        """
        在LAB空间应用光照校正（只修改L通道，保持ab色度不变）
        这是颜色保护的核心方法
        
        Args:
            x: RGB图像 [B, 3, H, W], 范围[0,1]
            L_factor: 光照校正因子 [B, 1, H, W], >1表示提亮, <1表示变暗
        
        Returns:
            校正后的RGB图像 [B, 3, H, W]
        """
        # RGB -> 近似LAB的亮度和色度分离
        # 使用简化的线性变换避免复杂的非线性LAB转换
        
        eps = 1e-6
        
        # 计算亮度 (使用标准系数)
        L_orig = self.rgb_to_luminance(x)  # [B, 1, H, W]
        
        # 计算目标亮度
        L_target = L_orig * L_factor
        L_target = torch.clamp(L_target, 0.01, 0.99)
        
        # 计算亮度变化比例
        scale = L_target / (L_orig + eps)
        
        # 对RGB应用相同的缩放 (保持颜色比例)
        x_corrected = x * scale
        
        # 关键：色度保护 - 确保颜色方向不变
        chrom_orig = self.rgb_to_chromaticity(x)
        chrom_corrected = self.rgb_to_chromaticity(x_corrected + eps)
        
        # 计算颜色偏移量
        color_drift = torch.sum((chrom_corrected - chrom_orig) ** 2, dim=1, keepdim=True)
        
        # 自适应混合：颜色偏移大时，减少校正强度
        color_protection = torch.exp(-color_drift * 50)  # 偏移越大，保护越强
        
        # 混合原图和校正图
        x_out = color_protection * x_corrected + (1 - color_protection) * x * scale.mean(dim=[2,3], keepdim=True)
        
        return torch.clamp(x_out, 0.01, 0.99)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 直接输出R_retinex
        
        Args:
            x: RGB图像 [B, 3, H, W]，范围 [0, 1]
        
        Returns:
            reflectance: 光照归一化的反射层 [B, 3, H, W]，范围 [0, 1]
        """
        # 输入检查
        if x.min() < -0.1 or x.max() > 1.1:
            logging.warning(f"输入范围异常 [{x.min():.3f}, {x.max():.3f}]，已裁剪")
            x = torch.clamp(x, 0.0, 1.0)
        
        # 1. 提取亮度通道
        luminance = self.rgb_to_luminance(x)  # [B, 1, H, W]

        # 2. 多尺度光照估计
        illumination = self.illumination_estimator(luminance)  # [B, 1, H, W]

        # 3. 敏感度估计（自适应校正强度）
        if self.use_sensitivity:
            sensitivity = self.sensitivity_estimator(x)  # [B, 1, H, W]
            self.cached_sensitivity = sensitivity
            # 自适应光照校正：L_adaptive = 1 + (L - 1) * S
            illumination_adaptive = 1.0 + (illumination - 1.0) * sensitivity
        else:
            illumination_adaptive = illumination
            self.cached_sensitivity = None

        # 4. 光照图约束与平滑
        illumination_adaptive = torch.clamp(illumination_adaptive, 0.2, 3.0)
        illumination_adaptive = F.avg_pool2d(illumination_adaptive, kernel_size=5, stride=1, padding=2)
        self.cached_illumination = illumination_adaptive

        # ====================================================================
        # 5. Retinex分解：R = I / L（核心，直接输出）
        # ====================================================================
        eps = 1e-4
        
        # 计算光照校正因子
        L_correction_factor = 1.0 / illumination_adaptive  # L>1时变暗，L<1时提亮
        L_correction_factor = torch.clamp(L_correction_factor, 0.33, 3.0)
        
        # 在LAB空间应用光照校正（保持颜色）
        reflectance_retinex = self.apply_illumination_in_lab(x, L_correction_factor)
        
        # 颜色安全检查
        chrom_orig = self.rgb_to_chromaticity(x)
        chrom_retinex = self.rgb_to_chromaticity(reflectance_retinex + eps)
        color_angle_diff = 1.0 - torch.sum(chrom_orig * chrom_retinex, dim=1, keepdim=True)
        color_risk = torch.clamp(color_angle_diff * 10, 0.0, 1.0)
        self.cached_color_risk = color_risk
        
        # 颜色失真时混合原图保护
        reflectance = (1 - color_risk) * reflectance_retinex + color_risk * x
        
        # 缓存用于损失计算
        self.cached_reflectance_init = reflectance

        # 6. 一致性精炼（可选）
        if self.use_refinement:
            reflectance = self.refiner(reflectance, illumination_adaptive, x)

        # 确保输出范围
        reflectance = torch.clamp(reflectance, 0.01, 0.99)

        return reflectance
    
    def forward_with_details(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        带详细输出的前向传播（用于可视化和损失计算）
        
        Args:
            x: RGB图像 [B, 3, H, W]
        """
        reflectance = self.forward(x)
        
        details = {
            'reflectance': reflectance,
            'illumination': self.cached_illumination,
            'sensitivity': self.cached_sensitivity,
            'reflectance_init': self.cached_reflectance_init,
            'original': x,
            'color_risk': self.cached_color_risk,
        }
        
        return details
    
    def get_reconstruction(self) -> Optional[torch.Tensor]:
        """获取重建图像 R × L"""
        if self.cached_illumination is None or self.cached_reflectance_init is None:
            return None
        return self.cached_reflectance_init * self.cached_illumination


# ============================================================================
#                           损失函数
# ============================================================================

class IPAIDLoss(nn.Module):
    """
    IPAID模块的联合损失函数
    
    包含：
    1. 重建损失: ||R × L - I||
    2. 光照平滑损失: 光照图应平滑
    3. 反射一致性损失: 边缘处反射层应保持
    4. 敏感度正则化: 防止极端值
    """
    
    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_smooth: float = 0.1,
        lambda_edge: float = 0.05,
        lambda_sensitivity: float = 0.01,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_smooth = lambda_smooth
        self.lambda_edge = lambda_edge
        self.lambda_sensitivity = lambda_sensitivity
        
        # Sobel算子用于边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def compute_gradient(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算图像梯度"""
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        return grad_x, grad_y
    
    def forward(
        self,
        details: Dict[str, torch.Tensor],
        ipaid_module: Optional[IPAIDModule] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算损失
        
        Args:
            details: forward_with_details的输出
            ipaid_module: IPAID模块实例（可选，用于额外正则化）
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        R = details['reflectance']
        L = details['illumination']
        I = details['original']
        S = details.get('sensitivity')
        
        device = R.device
        
        # 1. 重建损失
        I_recon = R * L
        loss_recon = F.l1_loss(I_recon, I)
        
        # 2. 光照平滑损失（边缘感知）
        grad_L_x, grad_L_y = self.compute_gradient(L)
        grad_I_x, grad_I_y = self.compute_gradient(I)
        
        # 在图像边缘处允许光照不连续
        weight_x = torch.exp(-torch.abs(grad_I_x) * 10)
        weight_y = torch.exp(-torch.abs(grad_I_y) * 10)
        
        loss_smooth = torch.mean(weight_x * grad_L_x ** 2) + \
                      torch.mean(weight_y * grad_L_y ** 2)
        
        # 3. 反射层边缘保持损失
        grad_R_x, grad_R_y = self.compute_gradient(R)
        loss_edge = -torch.mean(torch.abs(grad_R_x) + torch.abs(grad_R_y))  # 鼓励边缘
        loss_edge = torch.clamp(loss_edge, -1.0, 0.0)  # 限制范围
        
        # 4. 敏感度正则化
        if S is not None:
            # 鼓励敏感度图稀疏（只在重要区域高）
            loss_sensitivity = torch.mean(S)
        else:
            loss_sensitivity = torch.tensor(0.0, device=device)
        
        # 5. 颜色一致性损失（新增）
        # 确保校正前后的颜色角度保持一致
        eps = 1e-6
        chrom_orig = I / (torch.sqrt(torch.sum(I ** 2, dim=1, keepdim=True)) + eps)
        chrom_refl = R / (torch.sqrt(torch.sum(R ** 2, dim=1, keepdim=True)) + eps)
        # 余弦相似度应接近1
        color_consistency = 1.0 - torch.sum(chrom_orig * chrom_refl, dim=1, keepdim=True)
        loss_color = torch.mean(color_consistency)
        
        # 总损失 (增加颜色一致性权重)
        lambda_color = 0.1  # 颜色保护权重
        total_loss = (
            self.lambda_recon * loss_recon +
            self.lambda_smooth * loss_smooth +
            self.lambda_edge * loss_edge +
            self.lambda_sensitivity * loss_sensitivity +
            lambda_color * loss_color
        )
        
        loss_dict = {
            'loss_recon': loss_recon,
            'loss_smooth': loss_smooth,
            'loss_edge': loss_edge,
            'loss_sensitivity': loss_sensitivity,
            'loss_color': loss_color,
            'total': total_loss,
        }
        
        return total_loss, loss_dict


# ============================================================================
#                           身份保持损失（与ReID联合训练）
# ============================================================================

class IdentityPreservingLoss(nn.Module):
    """
    身份保持损失
    
    确保光照归一化后的反射层保持身份判别性：
    - 同ID的反射层特征应相似
    - 不同ID的反射层特征应不同
    """
    
    def __init__(self, margin: float = 0.3, mining: str = 'hard'):
        super().__init__()
        self.margin = margin
        self.mining = mining
    
    def forward(
        self,
        reflectance_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算身份保持损失（基于Triplet思想）
        
        Args:
            reflectance_features: 反射层的特征 [B, C]（来自backbone）
            labels: 身份标签 [B]
        
        Returns:
            loss: 身份保持损失
        """
        B = reflectance_features.shape[0]
        
        # L2归一化
        features = F.normalize(reflectance_features, p=2, dim=1)
        
        # 计算距离矩阵
        dist_mat = torch.cdist(features, features, p=2)  # [B, B]
        
        # 构建正负样本mask
        labels = labels.view(-1, 1)
        is_same_id = (labels == labels.T).float()  # [B, B]
        
        # Hard mining
        loss = torch.tensor(0.0, device=features.device)
        valid_triplets = 0
        
        for i in range(B):
            # 正样本（同ID）
            pos_mask = is_same_id[i].bool()
            pos_mask[i] = False  # 排除自己
            
            # 负样本（不同ID）
            neg_mask = ~is_same_id[i].bool()
            
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            
            # Hard positive: 最远的正样本
            pos_dists = dist_mat[i][pos_mask]
            hardest_pos = pos_dists.max()
            
            # Hard negative: 最近的负样本
            neg_dists = dist_mat[i][neg_mask]
            hardest_neg = neg_dists.min()
            
            # Triplet loss
            triplet_loss = F.relu(hardest_pos - hardest_neg + self.margin)
            loss = loss + triplet_loss
            valid_triplets += 1
        
        if valid_triplets > 0:
            loss = loss / valid_triplets
        
        return loss


# ============================================================================
#                           工具函数
# ============================================================================

# ============================================================================
#                    IICL: Illumination-Invariant Contrastive Learning
# ============================================================================

class IlluminationVariantGenerator(nn.Module):
    """
    光照变体生成器 (Illumination Variant Generator)
    
    核心创新：利用Retinex分解自动生成"同一目标不同光照"的正样本对
    
    原理：
        原图 I = R × L (Retinex分解)
        变体 I' = R × L' (保持R不变，只变换L)
        
    L的变换方式：
        1. Gamma变换: L' = L^γ (γ<1变亮, γ>1变暗)
        2. 缩放变换: L' = α * L (α<1变亮, α>1变暗)
        3. 局部扰动: L' = L + noise (模拟局部光照变化)
    
    关键优势：
        - 保持反射层R不变 → 花纹/斑点等身份特征完整保留
        - 自动生成配对数据 → 无需真实的光照配对采集
        - 物理合理 → 比随机颜色抖动更科学
    """
    
    def __init__(
        self,
        num_variants: int = 2,
        gamma_range: Tuple[float, float] = (0.6, 1.4),
        scale_range: Tuple[float, float] = (0.7, 1.3),
        local_noise_std: float = 0.1,
    ):
        super().__init__()
        self.num_variants = num_variants
        self.gamma_range = gamma_range
        self.scale_range = scale_range
        self.local_noise_std = local_noise_std
    
    def forward(
        self,
        reflectance: torch.Tensor,
        illumination: torch.Tensor,
        num_variants: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        生成光照变体
        
        Args:
            reflectance: Retinex分解的反射层 R [B, 3, H, W]
            illumination: Retinex分解的光照图 L [B, 1, H, W]
            num_variants: 生成变体数量（默认self.num_variants）
        
        Returns:
            variants: 光照变体图像列表 [I'_1, I'_2, ...]，每个 [B, 3, H, W]
        """
        n_var = num_variants or self.num_variants
        B, _, H, W = reflectance.shape
        device = reflectance.device
        
        variants = []
        
        for i in range(n_var):
            # 随机选择变换方式
            transform_type = torch.randint(0, 3, (1,)).item()
            
            if transform_type == 0:
                # Gamma变换
                gamma = torch.empty(B, 1, 1, 1, device=device).uniform_(
                    self.gamma_range[0], self.gamma_range[1]
                )
                L_transformed = torch.pow(illumination + 1e-6, gamma)
                
            elif transform_type == 1:
                # 缩放变换
                scale = torch.empty(B, 1, 1, 1, device=device).uniform_(
                    self.scale_range[0], self.scale_range[1]
                )
                L_transformed = illumination * scale
                
            else:
                # 局部扰动（空间变化的光照调整）
                noise = torch.randn(B, 1, H // 8, W // 8, device=device) * self.local_noise_std
                noise = F.interpolate(noise, size=(H, W), mode='bilinear', align_corners=False)
                L_transformed = illumination * (1 + noise)
            
            # 约束光照范围
            L_transformed = torch.clamp(L_transformed, 0.2, 3.0)
            
            # 重新合成图像: I' = R × L'
            variant = reflectance * L_transformed
            variant = torch.clamp(variant, 0.01, 0.99)
            
            variants.append(variant)
        
        return variants


class IlluminationContrastiveLoss(nn.Module):
    """
    光照不变对比学习损失 (Illumination-Invariant Contrastive Loss)
    
    核心思想：同一图像的不同光照变体应该有相似的特征表示
    
    损失形式：
        L_IICL = Σ ||f(I) - f(I')||² / N
        
    其中：
        - f(·) 是特征提取器
        - I 是原图
        - I' 是光照变体
        
    与SimCLR的区别：
        - SimCLR用随机颜色抖动 → 可能改变花纹颜色
        - 我们用Retinex变换L → 保持R（花纹）不变
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        loss_type: str = "mse",  # "mse" or "cosine" or "infonce"
    ):
        super().__init__()
        self.temperature = temperature
        self.loss_type = loss_type
    
    def forward(
        self,
        features_orig: torch.Tensor,
        features_variants: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            features_orig: 原图特征 [B, D]
            features_variants: 变体特征列表 [[B, D], [B, D], ...]
        
        Returns:
            loss: 对比损失标量
        """
        if len(features_variants) == 0:
            return torch.tensor(0.0, device=features_orig.device)
        
        if self.loss_type == "mse":
            # MSE损失：特征应该相近
            loss = 0.0
            for feat_var in features_variants:
                loss = loss + F.mse_loss(features_orig, feat_var)
            loss = loss / len(features_variants)
            
        elif self.loss_type == "cosine":
            # 余弦相似度损失：特征方向应该相同
            loss = 0.0
            for feat_var in features_variants:
                cos_sim = F.cosine_similarity(features_orig, feat_var, dim=1)
                loss = loss + (1 - cos_sim).mean()  # 越相似损失越小
            loss = loss / len(features_variants)
            
        elif self.loss_type == "infonce":
            # InfoNCE损失：对比学习标准损失
            # 正样本：同一图像的光照变体
            # 负样本：batch内其他图像
            loss = self._infonce_loss(features_orig, features_variants)
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def _infonce_loss(
        self,
        features_orig: torch.Tensor,
        features_variants: List[torch.Tensor],
    ) -> torch.Tensor:
        """InfoNCE对比损失"""
        B, D = features_orig.shape
        device = features_orig.device
        
        # 合并所有变体特征
        all_variants = torch.stack(features_variants, dim=1)  # [B, num_var, D]
        
        # 归一化
        features_orig_norm = F.normalize(features_orig, dim=1)  # [B, D]
        all_variants_norm = F.normalize(all_variants, dim=2)    # [B, num_var, D]
        
        loss = 0.0
        
        for i in range(B):
            # 当前样本的原图特征
            anchor = features_orig_norm[i]  # [D]
            
            # 正样本：自己的光照变体
            positives = all_variants_norm[i]  # [num_var, D]
            pos_sim = torch.sum(anchor.unsqueeze(0) * positives, dim=1) / self.temperature  # [num_var]
            
            # 负样本：batch内其他图像的原图特征
            neg_mask = torch.ones(B, dtype=torch.bool, device=device)
            neg_mask[i] = False
            negatives = features_orig_norm[neg_mask]  # [B-1, D]
            neg_sim = torch.sum(anchor.unsqueeze(0) * negatives, dim=1) / self.temperature  # [B-1]
            
            # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            for pos_s in pos_sim:
                logits = torch.cat([pos_s.unsqueeze(0), neg_sim])  # [1 + B-1]
                labels = torch.zeros(1, dtype=torch.long, device=device)  # 正样本在位置0
                loss = loss + F.cross_entropy(logits.unsqueeze(0), labels)
        
        loss = loss / (B * len(features_variants))
        return loss


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_ipaid_module():
    """测试IPAID模块"""
    print("=" * 60)
    print("测试 IPAID 模块")
    print("=" * 60)
    
    # 创建模块
    ipaid = IPAIDModule(
        base_channels=32,
        num_scales=3,
        refine_iterations=1,
        use_sensitivity=True,
        use_refinement=True,
    )
    
    # 统计参数
    params = count_parameters(ipaid)
    print(f"模块参数量: {params / 1e6:.2f}M")
    
    # 测试前向传播
    x = torch.rand(4, 3, 256, 256)
    
    with torch.no_grad():
        # 简单前向
        reflectance = ipaid(x)
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {reflectance.shape}")
        print(f"输出范围: [{reflectance.min():.3f}, {reflectance.max():.3f}]")
        
        # 带详细输出
        details = ipaid.forward_with_details(x)
        print(f"光照图形状: {details['illumination'].shape}")
        print(f"光照图范围: [{details['illumination'].min():.3f}, {details['illumination'].max():.3f}]")
        
        if details['sensitivity'] is not None:
            print(f"敏感度图形状: {details['sensitivity'].shape}")
            print(f"敏感度范围: [{details['sensitivity'].min():.3f}, {details['sensitivity'].max():.3f}]")
    
    # 测试损失
    loss_fn = IPAIDLoss()
    loss, loss_dict = loss_fn(details)
    print(f"\n损失值:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")
    
    print("\n✓ IPAID模块测试通过!")
    return ipaid


if __name__ == "__main__":
    test_ipaid_module()
