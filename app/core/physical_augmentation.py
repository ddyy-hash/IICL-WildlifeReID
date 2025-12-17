#!/usr/bin/env python3
"""
物理仿真增强模块
实现局部阴影、强光/过曝、色温漂移、逆光/剪影、随机遮挡、运动模糊等物理仿真效果
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict, Any
import random
from PIL import Image, ImageEnhance, ImageFilter


class PhysicalAugmentation:
    """物理仿真增强类"""
    
    def __init__(self, seed: Optional[int] = None):
        """
        初始化物理仿真增强
        
        Args:
            seed: 随机种子
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # 增强强度参数
        self.shadow_intensity_range = (0.3, 0.8)  # 阴影强度范围
        self.overexposure_threshold_range = (0.7, 0.9)  # 过曝阈值范围
        self.color_temp_shift_range = (-1000, 1000)  # 色温漂移范围（K）
        self.motion_blur_kernel_range = (5, 15)  # 运动模糊核大小范围
        self.occlusion_ratio_range = (0.1, 0.3)  # 遮挡比例范围
        
    def apply_local_shadow(
        self, 
        image: torch.Tensor, 
        shadow_intensity: Optional[float] = None,
        shadow_region: Optional[Tuple[int, int, int, int]] = None
    ) -> torch.Tensor:
        """
        应用局部阴影
        
        Args:
            image: 输入图像 [C, H, W] 或 [B, C, H, W]
            shadow_intensity: 阴影强度 (0-1)，值越大阴影越暗
            shadow_region: 阴影区域 (x, y, w, h)，如果为None则随机生成
            
        Returns:
            添加阴影后的图像
        """
        if shadow_intensity is None:
            shadow_intensity = random.uniform(*self.shadow_intensity_range)
        
        # 确保输入是4D张量 [B, C, H, W]
        is_single_image = image.dim() == 3
        if is_single_image:
            image = image.unsqueeze(0)
        
        B, C, H, W = image.shape
        
        # 生成阴影区域
        if shadow_region is None:
            # 随机生成阴影区域（覆盖20%-50%的图像）
            region_w = random.randint(int(W * 0.2), int(W * 0.5))
            region_h = random.randint(int(H * 0.2), int(H * 0.5))
            region_x = random.randint(0, max(0, W - region_w))
            region_y = random.randint(0, max(0, H - region_h))
        else:
            region_x, region_y, region_w, region_h = shadow_region
        
        # 创建阴影掩码
        shadow_mask = torch.ones_like(image)
        shadow_mask[:, :, region_y:region_y+region_h, region_x:region_x+region_w] = 1.0 - shadow_intensity
        
        # 应用阴影
        shadowed_image = image * shadow_mask
        
        if is_single_image:
            shadowed_image = shadowed_image.squeeze(0)
        
        return shadowed_image
    
    def apply_overexposure(
        self, 
        image: torch.Tensor, 
        threshold: Optional[float] = None,
        overexposure_strength: Optional[float] = None
    ) -> torch.Tensor:
        """
        应用强光/过曝效果
        
        Args:
            image: 输入图像 [C, H, W] 或 [B, C, H, W]
            threshold: 过曝阈值 (0-1)，高于此值的像素会被过曝
            overexposure_strength: 过曝强度 (0-1)
            
        Returns:
            添加过曝效果的图像
        """
        if threshold is None:
            threshold = random.uniform(*self.overexposure_threshold_range)
        
        if overexposure_strength is None:
            overexposure_strength = random.uniform(0.5, 1.0)
        
        # 确保输入是4D张量 [B, C, H, W]
        is_single_image = image.dim() == 3
        if is_single_image:
            image = image.unsqueeze(0)
        
        # 创建过曝掩码（高于阈值的区域）
        overexposure_mask = (image > threshold).float()
        
        # 应用过曝效果
        overexposed_image = image.clone()
        overexposed_image = overexposed_image + overexposure_mask * overexposure_strength
        overexposed_image = torch.clamp(overexposed_image, 0.0, 1.0)
        
        if is_single_image:
            overexposed_image = overexposed_image.squeeze(0)
        
        return overexposed_image
    
    def apply_color_temperature_shift(
        self, 
        image: torch.Tensor, 
        temperature_shift: Optional[float] = None
    ) -> torch.Tensor:
        """
        应用色温漂移
        
        Args:
            image: 输入图像 [C, H, W] 或 [B, C, H, W]
            temperature_shift: 色温漂移值（正数：变暖，负数：变冷）
            
        Returns:
            色温漂移后的图像
        """
        if temperature_shift is None:
            temperature_shift = random.uniform(*self.color_temp_shift_range)
        
        # 确保输入是4D张量 [B, C, H, W]
        is_single_image = image.dim() == 3
        if is_single_image:
            image = image.unsqueeze(0)
        
        B, C, H, W = image.shape
        
        # 色温漂移矩阵（简化版）
        # 暖色温：增加红色，减少蓝色
        # 冷色温：增加蓝色，减少红色
        if temperature_shift > 0:  # 变暖
            # 红色通道增强
            red_factor = 1.0 + temperature_shift / 2000.0
            blue_factor = 1.0 - temperature_shift / 4000.0
        else:  # 变冷
            # 蓝色通道增强
            red_factor = 1.0 + temperature_shift / 4000.0
            blue_factor = 1.0 - temperature_shift / 2000.0
        
        # 应用色温漂移
        shifted_image = image.clone()
        shifted_image[:, 0, :, :] = shifted_image[:, 0, :, :] * red_factor   # R通道
        shifted_image[:, 2, :, :] = shifted_image[:, 2, :, :] * blue_factor  # B通道
        
        shifted_image = torch.clamp(shifted_image, 0.0, 1.0)
        
        if is_single_image:
            shifted_image = shifted_image.squeeze(0)
        
        return shifted_image
    
    def apply_backlight_silhouette(
        self, 
        image: torch.Tensor, 
        silhouette_strength: Optional[float] = None
    ) -> torch.Tensor:
        """
        应用逆光/剪影效果
        
        Args:
            image: 输入图像 [C, H, W] 或 [B, C, H, W]
            silhouette_strength: 剪影强度 (0-1)，值越大剪影效果越明显
            
        Returns:
            添加剪影效果的图像
        """
        if silhouette_strength is None:
            silhouette_strength = random.uniform(0.5, 0.9)
        
        # 确保输入是4D张量 [B, C, H, W]
        is_single_image = image.dim() == 3
        if is_single_image:
            image = image.unsqueeze(0)
        
        B, C, H, W = image.shape
        
        # 创建逆光效果（边缘亮，中心暗）
        y_coords = torch.arange(H).float().to(image.device) / H
        x_coords = torch.arange(W).float().to(image.device) / W
        
        # 创建径向渐变掩码
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        center_y, center_x = 0.5, 0.5
        distance_from_center = torch.sqrt((yy - center_y)**2 + (xx - center_x)**2)
        distance_from_center = distance_from_center / distance_from_center.max()
        
        # 逆光掩码（边缘亮，中心暗）
        backlight_mask = 1.0 - distance_from_center * silhouette_strength
        
        # 扩展维度以匹配图像
        backlight_mask = backlight_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        backlight_mask = backlight_mask.expand(B, C, H, W)
        
        # 应用逆光效果
        backlit_image = image * backlight_mask
        
        if is_single_image:
            backlit_image = backlit_image.squeeze(0)
        
        return backlit_image
    
    def apply_motion_blur(
        self, 
        image: torch.Tensor, 
        kernel_size: Optional[int] = None,
        angle: Optional[float] = None
    ) -> torch.Tensor:
        """
        应用运动模糊
        
        Args:
            image: 输入图像 [C, H, W] 或 [B, C, H, W]
            kernel_size: 模糊核大小（奇数）
            angle: 运动角度（度）
            
        Returns:
            添加运动模糊的图像
        """
        if kernel_size is None:
            kernel_size = random.randint(*self.motion_blur_kernel_range)
        
        if kernel_size % 2 == 0:
            kernel_size += 1  # 确保为奇数
        
        if angle is None:
            angle = random.uniform(0, 360)
        
        # 确保输入是4D张量 [B, C, H, W]
        is_single_image = image.dim() == 3
        if is_single_image:
            image = image.unsqueeze(0)
        
        B, C, H, W = image.shape
        
        # 创建运动模糊核
        kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
        
        # 计算运动方向
        angle_rad = np.deg2rad(angle)
        center = kernel_size // 2
        
        # 在中心线上设置值
        for i in range(kernel_size):
            x = i - center
            y = int(np.tan(angle_rad) * x)
            y_idx = center + y
            if 0 <= y_idx < kernel_size:
                kernel[y_idx, i] = 1.0
        
        # 归一化
        kernel = kernel / kernel.sum()
        
        # 扩展为卷积核 [out_channels, in_channels/groups, H, W]
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(C, 1, kernel_size, kernel_size).to(image.device)
        
        # 应用运动模糊
        blurred_image = F.conv2d(image, kernel, padding=kernel_size//2, groups=C)
        
        if is_single_image:
            blurred_image = blurred_image.squeeze(0)
        
        return blurred_image
    
    def apply_random_occlusion(
        self, 
        image: torch.Tensor, 
        occlusion_ratio: Optional[float] = None,
        occlusion_region: Optional[Tuple[int, int, int, int]] = None
    ) -> torch.Tensor:
        """
        应用随机遮挡
        
        Args:
            image: 输入图像 [C, H, W] 或 [B, C, H, W]
            occlusion_ratio: 遮挡区域占图像的比例
            occlusion_region: 遮挡区域 (x, y, w, h)，如果为None则随机生成
            
        Returns:
            添加遮挡后的图像
        """
        if occlusion_ratio is None:
            occlusion_ratio = random.uniform(*self.occlusion_ratio_range)
        
        # 确保输入是4D张量 [B, C, H, W]
        is_single_image = image.dim() == 3
        if is_single_image:
            image = image.unsqueeze(0)
        
        B, C, H, W = image.shape
        
        # 计算遮挡区域大小
        region_area = int(H * W * occlusion_ratio)
        region_w = int(np.sqrt(region_area * W / H))
        region_h = int(region_area / region_w)
        
        # 生成遮挡区域
        if occlusion_region is None:
            region_x = random.randint(0, max(0, W - region_w))
            region_y = random.randint(0, max(0, H - region_h))
        else:
            region_x, region_y, region_w, region_h = occlusion_region
        
        # 应用遮挡（将区域置为0）
        occluded_image = image.clone()
        occluded_image[:, :, region_y:region_y+region_h, region_x:region_x+region_w] = 0.0
        
        if is_single_image:
            occluded_image = occluded_image.squeeze(0)
        
        return occluded_image
    
    def apply_comprehensive_augmentation(
        self, 
        image: torch.Tensor, 
        augmentation_probs: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        应用综合物理仿真增强
        
        Args:
            image: 输入图像 [C, H, W] 或 [B, C, H, W]
            augmentation_probs: 各种增强的概率配置
            
        Returns:
            应用增强后的图像
        """
        if augmentation_probs is None:
            augmentation_probs = {
                'shadow': 0.3,
                'overexposure': 0.2,
                'color_temp_shift': 0.3,
                'backlight': 0.2,
                'motion_blur': 0.1,
                'occlusion': 0.2
            }
        
        augmented_image = image.clone()
        
        # 局部阴影
        if random.random() < augmentation_probs.get('shadow', 0.3):
            augmented_image = self.apply_local_shadow(augmented_image)
        
        # 强光/过曝
        if random.random() < augmentation_probs.get('overexposure', 0.2):
            augmented_image = self.apply_overexposure(augmented_image)
        
        # 色温漂移
        if random.random() < augmentation_probs.get('color_temp_shift', 0.3):
            augmented_image = self.apply_color_temperature_shift(augmented_image)
        
        # 逆光/剪影
        if random.random() < augmentation_probs.get('backlight', 0.2):
            augmented_image = self.apply_backlight_silhouette(augmented_image)
        
        # 运动模糊
        if random.random() < augmentation_probs.get('motion_blur', 0.1):
            augmented_image = self.apply_motion_blur(augmented_image)
        
        # 随机遮挡
        if random.random() < augmentation_probs.get('occlusion', 0.2):
            augmented_image = self.apply_random_occlusion(augmented_image)
        
        return augmented_image


# 测试代码
if __name__ == "__main__":
    print("🧪 测试物理仿真增强模块")
    
    # 创建增强器
    augmentor = PhysicalAugmentation(seed=42)
    
    # 创建测试图像
    test_image = torch.rand(3, 256, 256)  # 随机图像
    
    print("测试各种增强效果：")
    
    # 测试局部阴影
    shadow_image = augmentor.apply_local_shadow(test_image, shadow_intensity=0.6)
    print(f"[OK] 局部阴影: {shadow_image.shape}, 范围 [{shadow_image.min():.3f}, {shadow_image.max():.3f}]")
    
    # 测试过曝
    overexposed_image = augmentor.apply_overexposure(test_image, threshold=0.7)
    print(f"[OK] 强光过曝: {overexposed_image.shape}, 范围 [{overexposed_image.min():.3f}, {overexposed_image.max():.3f}]")
    
    # 测试色温漂移
    warm_image = augmentor.apply_color_temperature_shift(test_image, temperature_shift=500)
    cold_image = augmentor.apply_color_temperature_shift(test_image, temperature_shift=-500)
    print(f"[OK] 色温漂移（暖）: {warm_image.shape}")
    print(f"[OK] 色温漂移（冷）: {cold_image.shape}")
    
    # 测试逆光
    backlit_image = augmentor.apply_backlight_silhouette(test_image, silhouette_strength=0.7)
    print(f"[OK] 逆光剪影: {backlit_image.shape}, 范围 [{backlit_image.min():.3f}, {backlit_image.max():.3f}]")
    
    # 测试运动模糊
    blurred_image = augmentor.apply_motion_blur(test_image, kernel_size=7, angle=45)
    print(f"[OK] 运动模糊: {blurred_image.shape}")
    
    # 测试遮挡
    occluded_image = augmentor.apply_random_occlusion(test_image, occlusion_ratio=0.2)
    print(f"[OK] 随机遮挡: {occluded_image.shape}, 范围 [{occluded_image.min():.3f}, {occluded_image.max():.3f}]")
    
    # 测试综合增强
    comprehensive_image = augmentor.apply_comprehensive_augmentation(test_image)
    print(f"[OK] 综合增强: {comprehensive_image.shape}, 范围 [{comprehensive_image.min():.3f}, {comprehensive_image.max():.3f}]")
    
    # 批量测试
    batch_image = torch.rand(4, 3, 256, 256)
    batch_augmented = augmentor.apply_comprehensive_augmentation(batch_image)
    print(f"[OK] 批量增强: {batch_augmented.shape}")
    
    print("[OK] 物理仿真增强模块测试完成")