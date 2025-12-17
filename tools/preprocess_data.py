#!/usr/bin/env python3
"""
数据预处理脚本

流程：
1. 读取原始图像
2. 使用预训练光照模块进行光照归一化
3. 使用YOLO+EfficientSAM进行分割
4. 保存处理后的图像

用途：为ReID训练准备分割后的数据集
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.dog_reid_system import IlluminationInvariantModule
from app.core.yolo_segment import yolo_seg


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(
        self,
        illumination_checkpoint: str = None,
        device: str = 'auto'
    ):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"[INFO] 使用设备: {self.device}")
        
        # 加载光照模块（可选）
        self.illumination = None
        if illumination_checkpoint and os.path.exists(illumination_checkpoint):
            self.illumination = IlluminationInvariantModule().to(self.device)
            checkpoint = torch.load(illumination_checkpoint, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.illumination.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.illumination.load_state_dict(checkpoint)
            self.illumination.eval()
            print(f"[INFO] 光照模块已加载: {illumination_checkpoint}")
        else:
            print("[INFO] 未加载光照模块，跳过光照归一化步骤")
        
        # 加载分割模块
        self.segmentor = yolo_seg()
        print("[INFO] 分割模块初始化完成")
    
    def normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """光照归一化"""
        if self.illumination is None:
            return image
        
        # 转换为tensor
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            normalized, _ = self.illumination(img_tensor)
        
        # 转回numpy
        normalized = normalized.squeeze(0).permute(1, 2, 0).cpu().numpy()
        normalized = (normalized * 255).clip(0, 255).astype(np.uint8)
        
        return normalized
    
    def segment_image(self, image: np.ndarray) -> np.ndarray:
        """分割图像"""
        result = self.segmentor.get_single_mask_from_cvimage(image)
        return result
    
    def process_dataset(
        self,
        input_dir: str,
        output_dir: str,
        apply_illumination: bool = True,
        extensions: tuple = ('.jpg', '.jpeg', '.png')
    ):
        """
        处理整个数据集
        
        Args:
            input_dir: 输入目录 (结构: input_dir/identity_id/image.jpg)
            output_dir: 输出目录 (保持相同结构)
            apply_illumination: 是否应用光照归一化
            extensions: 支持的图像扩展名
        """
        # 收集所有图像
        image_paths = []
        for identity_dir in os.listdir(input_dir):
            identity_path = os.path.join(input_dir, identity_dir)
            if not os.path.isdir(identity_path):
                continue
            
            for img_file in os.listdir(identity_path):
                if img_file.lower().endswith(extensions):
                    img_path = os.path.join(identity_path, img_file)
                    image_paths.append((img_path, identity_dir, img_file))
        
        print(f"[INFO] 找到 {len(image_paths)} 张图像待处理")
        
        # 统计
        success_count = 0
        fail_count = 0
        
        # 处理每张图像
        for img_path, identity_dir, img_file in tqdm(image_paths, desc="处理进度"):
            try:
                # 读取图像
                image = cv2.imread(img_path)
                if image is None:
                    fail_count += 1
                    continue
                
                # RGB转换（光照模块需要RGB）
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 光照归一化（可选）
                if apply_illumination and self.illumination is not None:
                    image_rgb = self.normalize_illumination(image_rgb)
                
                # 转回BGR用于分割
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                # 分割
                segmented = self.segment_image(image_bgr)
                
                if segmented is None:
                    # 分割失败，跳过或保存原图
                    fail_count += 1
                    continue
                
                # 创建输出目录
                out_identity_dir = os.path.join(output_dir, identity_dir)
                os.makedirs(out_identity_dir, exist_ok=True)
                
                # 保存
                out_path = os.path.join(out_identity_dir, img_file)
                cv2.imwrite(out_path, segmented)
                success_count += 1
                
            except Exception as e:
                print(f"[WARN] 处理失败 {img_path}: {e}")
                fail_count += 1
        
        print(f"\n[DONE] 处理完成!")
        print(f"  成功: {success_count}")
        print(f"  失败: {fail_count}")
        print(f"  输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='数据预处理：光照归一化 + 分割')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='原始图像目录 (结构: input_dir/identity_id/image.jpg)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--illumination_checkpoint', type=str, default=None,
                        help='光照模块权重路径（可选，如不提供则跳过光照归一化）')
    parser.add_argument('--no_illumination', action='store_true',
                        help='跳过光照归一化，只做分割')
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备')
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(
        illumination_checkpoint=args.illumination_checkpoint if not args.no_illumination else None,
        device=args.device
    )
    
    preprocessor.process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        apply_illumination=not args.no_illumination
    )


if __name__ == '__main__':
    main()
