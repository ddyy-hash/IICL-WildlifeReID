#!/usr/bin/env python3
"""
开放集识别机制
实现动态阈值策略、质量拒识、相似度拒识和不确定性量化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import math


@dataclass
class OpenSetRecognitionResult:
    """开放集识别结果"""
    is_known: bool  # 是否属于已知类别
    predicted_id: Optional[int]  # 预测的类别ID（如果是已知类别）
    confidence: float  # 置信度
    uncertainty: float  # 不确定性
    rejection_reason: Optional[str]  # 拒识原因（如果被拒识）
    similarity_score: float  # 相似度分数
    quality_score: float  # 质量分数


class QualityAssessor:
    """质量评估器"""
    
    def __init__(self):
        # 质量评估阈值
        self.sharpness_threshold = 15.0
        self.contrast_threshold = 10.0
        self.brightness_min = 0.1
        self.brightness_max = 0.9
        
    def assess_quality(self, image: torch.Tensor) -> Dict[str, float]:
        """
        评估图像质量
        
        Args:
            image: 输入图像 [C, H, W] 或 [B, C, H, W]
            
        Returns:
            质量评估结果
        """
        # 确保输入是4D张量 [B, C, H, W]
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        B, C, H, W = image.shape
        
        # 转换为灰度图
        if C == 3:
            gray_image = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        else:
            gray_image = image[:, 0, :, :]
        
        # 计算清晰度（使用拉普拉斯算子）
        laplacian = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32).to(image.device)
        laplacian = laplacian.expand(1, 1, 3, 3)
        
        sharpness_map = F.conv2d(gray_image.unsqueeze(1), laplacian, padding=1)
        sharpness = torch.mean(torch.abs(sharpness_map), dim=[1, 2, 3])
        
        # 计算对比度
        contrast = torch.std(gray_image, dim=[1, 2])
        
        # 计算亮度
        brightness = torch.mean(gray_image, dim=[1, 2])
        
        # 计算质量分数
        sharpness_score = torch.clamp(sharpness / self.sharpness_threshold, 0, 1)
        contrast_score = torch.clamp(contrast / self.contrast_threshold, 0, 1)
        brightness_score = torch.clamp(
            (brightness - self.brightness_min) / (self.brightness_max - self.brightness_min), 0, 1
        )
        
        # 综合质量分数
        quality_score = (sharpness_score + contrast_score + brightness_score) / 3.0
        
        return {
            'sharpness': sharpness.mean().item(),
            'contrast': contrast.mean().item(),
            'brightness': brightness.mean().item(),
            'sharpness_score': sharpness_score.mean().item(),
            'contrast_score': contrast_score.mean().item(),
            'brightness_score': brightness_score.mean().item(),
            'quality_score': quality_score.mean().item()
        }


class DynamicThresholdStrategy:
    """
    动态阈值策略（改进版：基于特征分布的阈值标定）
    
    改进点：
    1. 阈值基于已知样本的特征分布来标定，而非推理结果
    2. 使用验证集预标定，避免标签泄露
    3. 支持基于统计的自适应调整（仅使用已知样本的分布）
    """
    
    def __init__(self, initial_threshold: float = 0.7, adaptation_rate: float = 0.01,
                 percentile: float = 95.0):
        """
        初始化动态阈值策略
        
        Args:
            initial_threshold: 初始阈值
            adaptation_rate: 阈值适应率
            percentile: 用于标定阈值的百分位数（基于已知样本内距离分布）
        """
        self.initial_threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.current_threshold = initial_threshold
        self.percentile = percentile
        
        # 【改进】仅存储已知样本的类内距离，用于阈值标定
        self.intra_class_distances = []
        
        # 统计信息（仅用于监控，不用于阈值更新）
        self.accepted_count = 0
        self.rejected_count = 0
        self.query_scores = []  # 仅记录查询分数，不区分已知/未知
        
        # 标定状态
        self.is_calibrated = False
        self.calibrated_threshold = initial_threshold
        
    def calibrate_from_gallery(self, gallery_features: np.ndarray, gallery_labels: np.ndarray):
        """
        【新增】使用已知样本库（gallery）标定阈值
        
        基于已知样本的类内距离分布来确定阈值，避免标签泄露
        
        Args:
            gallery_features: 已知样本特征 [N, D]
            gallery_labels: 已知样本标签 [N]
        """
        if len(gallery_features) < 10:
            logging.warning("样本数量不足，无法标定阈值")
            return
        
        # 计算类内距离
        intra_distances = []
        unique_labels = np.unique(gallery_labels)
        
        for label in unique_labels:
            # 获取同类样本
            mask = gallery_labels == label
            class_features = gallery_features[mask]
            
            if len(class_features) < 2:
                continue
            
            # 计算类内成对距离
            for i in range(len(class_features)):
                for j in range(i + 1, len(class_features)):
                    dist = np.linalg.norm(class_features[i] - class_features[j])
                    intra_distances.append(dist)
        
        if len(intra_distances) > 0:
            self.intra_class_distances = intra_distances
            # 使用百分位数作为阈值（类内距离的上界）
            self.calibrated_threshold = np.percentile(intra_distances, self.percentile)
            self.current_threshold = self.calibrated_threshold
            self.is_calibrated = True
            logging.info(f"[OK] 阈值标定完成: {self.calibrated_threshold:.4f} "
                        f"(基于 {len(intra_distances)} 个类内距离对)")
    
    def update_threshold(self, similarity_scores: List[float], is_known: List[bool] = None):
        """
        更新阈值（改进版：仅基于查询分数分布的统计特性）
        
        Args:
            similarity_scores: 相似度分数列表
            is_known: 【废弃】不再使用此参数，保留仅为兼容性
        """
        # 记录查询分数
        self.query_scores.extend(similarity_scores)
        
        # 限制列表长度
        max_stats = 1000
        self.query_scores = self.query_scores[-max_stats:]
        
        # 【改进】如果已标定，仅做轻微的自适应调整
        if self.is_calibrated and len(self.query_scores) > 50:
            # 使用查询分数的分布来微调（不依赖标签）
            # 假设：大部分查询应该是已知类，使用中位数作为参考
            median_score = np.median(self.query_scores[-50:])
            
            # 仅在中位数显著偏离标定阈值时微调
            if abs(median_score - self.calibrated_threshold) > 0.1:
                # 保守调整，向标定阈值靠近
                self.current_threshold = (
                    self.current_threshold * (1 - self.adaptation_rate * 0.1) + 
                    self.calibrated_threshold * self.adaptation_rate * 0.1
                )
        
        # 限制阈值范围
        self.current_threshold = np.clip(self.current_threshold, 0.3, 0.95)
    
    def get_threshold(self) -> float:
        """获取当前阈值"""
        return self.current_threshold
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'current_threshold': self.current_threshold,
            'calibrated_threshold': self.calibrated_threshold,
            'is_calibrated': self.is_calibrated,
            'accepted_count': self.accepted_count,
            'rejected_count': self.rejected_count,
            'mean_query_score': np.mean(self.query_scores) if self.query_scores else 0.0,
            'std_query_score': np.std(self.query_scores) if self.query_scores else 0.0,
            'num_intra_distances': len(self.intra_class_distances)
        }


class UncertaintyQuantifier:
    """不确定性量化器"""
    
    def __init__(self, num_mc_samples: int = 10):
        """
        初始化不确定性量化器
        
        Args:
            num_mc_samples: Monte Carlo采样次数
        """
        self.num_mc_samples = num_mc_samples
        
    def quantify_uncertainty(
        self, 
        features: torch.Tensor, 
        feature_database: Any, 
        k: int = 5
    ) -> Dict[str, float]:
        """
        量化识别不确定性
        
        Args:
            features: 查询特征 [D] 或 [B, D]
            feature_database: 特征数据库
            k: 考虑的最近邻数量
            
        Returns:
            不确定性指标
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        # 在特征数据库中搜索
        similar_ids, distances = feature_database.search(features.cpu().numpy(), k=k)
        
        uncertainties = []
        
        for i, (neighbor_ids, dists) in enumerate(zip(similar_ids, distances)):
            if len(neighbor_ids) < 2:
                uncertainties.append(1.0)  # 最大不确定性
                continue
            
            # 获取邻居的类别
            neighbor_classes = []
            for neighbor_id in neighbor_ids:
                result = feature_database.get_feature_by_id(neighbor_id)
                if result is not None:
                    _, metadata = result
                    neighbor_classes.append(metadata.get('dog_id', -1))
            
            if len(set(neighbor_classes)) <= 1:
                # 所有邻居都是同一类别，不确定性低
                uncertainty = 0.1
            else:
                # 计算类别分布的熵
                class_counts = {}
                for cls in neighbor_classes:
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                
                # 计算熵
                total = sum(class_counts.values())
                entropy = 0.0
                for count in class_counts.values():
                    p = count / total
                    entropy -= p * math.log(p + 1e-10)
                
                # 归一化熵
                max_entropy = math.log(len(class_counts) + 1e-10)
                uncertainty = entropy / max_entropy if max_entropy > 0 else 0.0
            
            uncertainties.append(uncertainty)
        
        return {
            'uncertainty': np.mean(uncertainties),
            'uncertainty_std': np.std(uncertainties),
            'max_uncertainty': max(uncertainties),
            'min_uncertainty': min(uncertainties)
        }


class OpenSetRecognizer:
    """开放集识别器"""
    
    def __init__(
        self,
        feature_database: Any,
        initial_threshold: float = 0.7,
        quality_threshold: float = 0.5,
        uncertainty_threshold: float = 0.6,
        adaptation_rate: float = 0.01
    ):
        """
        初始化开放集识别器
        
        Args:
            feature_database: 特征数据库
            initial_threshold: 初始相似度阈值
            quality_threshold: 质量阈值
            uncertainty_threshold: 不确定性阈值
            adaptation_rate: 阈值适应率
        """
        self.feature_database = feature_database
        self.quality_threshold = quality_threshold
        self.uncertainty_threshold = uncertainty_threshold
        
        # 质量评估器
        self.quality_assessor = QualityAssessor()
        
        # 动态阈值策略
        self.threshold_strategy = DynamicThresholdStrategy(
            initial_threshold=initial_threshold,
            adaptation_rate=adaptation_rate
        )
        
        # 不确定性量化器
        self.uncertainty_quantifier = UncertaintyQuantifier(num_mc_samples=10)
        
        # 统计信息
        self.recognition_stats = {
            'total_queries': 0,
            'accepted_known': 0,
            'rejected_unknown': 0,
            'rejected_quality': 0,
            'rejected_uncertainty': 0
        }
        
        logging.info(f"[OK] 开放集识别器初始化完成")
        logging.info(f"   初始阈值: {initial_threshold}")
        logging.info(f"   质量阈值: {quality_threshold}")
        logging.info(f"   不确定性阈值: {uncertainty_threshold}")
    
    def recognize(
        self,
        query_features: torch.Tensor,
        query_image: Optional[torch.Tensor] = None,
        k: int = 5,
        return_details: bool = False
    ) -> OpenSetRecognitionResult:
        """
        执行开放集识别
        
        Args:
            query_features: 查询特征 [D] 或 [1, D]
            query_image: 查询图像（用于质量评估）
            k: 最近邻数量
            return_details: 是否返回详细信息
            
        Returns:
            开放集识别结果
        """
        self.recognition_stats['total_queries'] += 1
        
        # 确保特征是一维的
        if query_features.dim() > 1:
            query_features = query_features.squeeze()
        
        # 1. 质量评估
        if query_image is not None:
            quality_result = self.quality_assessor.assess_quality(query_image)
            quality_score = quality_result['quality_score']
            
            if quality_score < self.quality_threshold:
                self.recognition_stats['rejected_quality'] += 1
                return OpenSetRecognitionResult(
                    is_known=False,
                    predicted_id=None,
                    confidence=0.0,
                    uncertainty=1.0,
                    rejection_reason=f"图像质量过低 (score: {quality_score:.3f})",
                    similarity_score=0.0,
                    quality_score=quality_score
                )
        else:
            quality_score = 1.0
        
        # 2. 在特征数据库中搜索
        similar_ids, distances = self.feature_database.search(
            query_features.cpu().numpy().reshape(1, -1), 
            k=k
        )
        
        if not similar_ids[0]:
            # 没有找到相似特征
            self.recognition_stats['rejected_unknown'] += 1
            return OpenSetRecognitionResult(
                is_known=False,
                predicted_id=None,
                confidence=0.0,
                uncertainty=1.0,
                rejection_reason="未找到相似特征",
                similarity_score=0.0,
                quality_score=quality_score
            )
        
        # 3. 计算相似度分数（将距离转换为相似度）
        similarity_scores = [1.0 / (1.0 + dist) for dist in distances[0]]
        max_similarity = max(similarity_scores)
        
        # 4. 获取最相似特征的类别
        best_match_id = similar_ids[0][0]
        best_match_result = self.feature_database.get_feature_by_id(best_match_id)
        
        if best_match_result is None:
            self.recognition_stats['rejected_unknown'] += 1
            return OpenSetRecognitionResult(
                is_known=False,
                predicted_id=None,
                confidence=0.0,
                uncertainty=1.0,
                rejection_reason="无法获取匹配特征",
                similarity_score=max_similarity,
                quality_score=quality_score
            )
        
        _, metadata = best_match_result
        predicted_id = metadata.get('dog_id', -1)
        
        # 5. 量化不确定性
        uncertainty_result = self.uncertainty_quantifier.quantify_uncertainty(
            query_features, 
            self.feature_database, 
            k=k
        )
        uncertainty = uncertainty_result['uncertainty']
        
        # 6. 获取动态阈值
        current_threshold = self.threshold_strategy.get_threshold()
        
        # 7. 决策
        is_known = True
        rejection_reason = None
        
        # 检查相似度阈值
        if max_similarity < current_threshold:
            is_known = False
            rejection_reason = f"相似度低于阈值 ({max_similarity:.3f} < {current_threshold:.3f})"
        
        # 检查不确定性阈值
        elif uncertainty > self.uncertainty_threshold:
            is_known = False
            rejection_reason = f"不确定性过高 ({uncertainty:.3f} > {self.uncertainty_threshold:.3f})"
        
        # 更新统计
        if is_known:
            self.recognition_stats['accepted_known'] += 1
        else:
            self.recognition_stats['rejected_unknown'] += 1
        
        # 8. 更新动态阈值
        self.threshold_strategy.update_threshold([max_similarity], [is_known])
        
        # 9. 计算置信度
        if is_known:
            confidence = max_similarity * (1.0 - uncertainty) * quality_score
        else:
            confidence = 0.0
        
        return OpenSetRecognitionResult(
            is_known=is_known,
            predicted_id=predicted_id if is_known else None,
            confidence=confidence,
            uncertainty=uncertainty,
            rejection_reason=rejection_reason,
            similarity_score=max_similarity,
            quality_score=quality_score
        )
    
    def batch_recognize(
        self,
        query_features: torch.Tensor,
        query_images: Optional[torch.Tensor] = None,
        k: int = 5
    ) -> List[OpenSetRecognitionResult]:
        """
        批量开放集识别
        
        Args:
            query_features: 查询特征 [B, D]
            query_images: 查询图像 [B, C, H, W]
            k: 最近邻数量
            
        Returns:
            识别结果列表
        """
        results = []
        
        for i in range(query_features.shape[0]):
            feature = query_features[i]
            image = query_images[i] if query_images is not None else None
            
            result = self.recognize(feature, image, k=k)
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取识别统计信息"""
        threshold_stats = self.threshold_strategy.get_stats()
        
        return {
            **self.recognition_stats,
            'threshold_stats': threshold_stats,
            'acceptance_rate': (
                self.recognition_stats['accepted_known'] / 
                max(1, self.recognition_stats['total_queries'])
            ),
            'rejection_rate': (
                (self.recognition_stats['rejected_unknown'] + 
                 self.recognition_stats['rejected_quality']) / 
                max(1, self.recognition_stats['total_queries'])
            )
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.recognition_stats = {
            'total_queries': 0,
            'accepted_known': 0,
            'rejected_unknown': 0,
            'rejected_quality': 0,
            'rejected_uncertainty': 0
        }
        self.threshold_strategy = DynamicThresholdStrategy(
            initial_threshold=self.threshold_strategy.initial_threshold,
            adaptation_rate=self.threshold_strategy.adaptation_rate
        )


# 测试代码
if __name__ == "__main__":
    print("🧪 测试开放集识别机制")
    
    # 创建模拟特征数据库
    from app.core.feature_database import FeatureDatabase
    
    feature_db = FeatureDatabase(feature_dim=128, index_type="Flat")
    
    # 添加已知类别的特征
    known_features = torch.randn(50, 128)
    known_metadata = [{'dog_id': i // 10} for i in range(50)]  # 5个类别，每个10个样本
    
    feature_db.add_features(known_features.numpy(), known_metadata)
    
    print(f"[OK] 特征数据库创建完成，特征数: {feature_db.total_features}")
    
    # 创建开放集识别器
    recognizer = OpenSetRecognizer(
        feature_database=feature_db,
        initial_threshold=0.6,
        quality_threshold=0.3,
        uncertainty_threshold=0.7,
        adaptation_rate=0.01
    )
    
    # 测试已知类别识别
    print("\n测试已知类别识别:")
    known_query = known_features[0]  # 使用已知特征
    result = recognizer.recognize(known_query, k=5)
    
    print(f"  是否已知: {result.is_known}")
    print(f"  预测ID: {result.predicted_id}")
    print(f"  置信度: {result.confidence:.3f}")
    print(f"  不确定性: {result.uncertainty:.3f}")
    print(f"  相似度: {result.similarity_score:.3f}")
    
    # 测试未知类别识别
    print("\n测试未知类别识别:")
    unknown_query = torch.randn(128)  # 随机特征，模拟未知类别
    result = recognizer.recognize(unknown_query, k=5)
    
    print(f"  是否已知: {result.is_known}")
    print(f"  拒识原因: {result.rejection_reason}")
    print(f"  置信度: {result.confidence:.3f}")
    print(f"  不确定性: {result.uncertainty:.3f}")
    
    # 测试质量拒识
    print("\n测试质量拒识:")
    # 创建低质量图像（随机噪声）
    low_quality_image = torch.rand(3, 64, 64) * 0.1
    result = recognizer.recognize(known_query, query_image=low_quality_image)
    
    print(f"  是否已知: {result.is_known}")
    print(f"  拒识原因: {result.rejection_reason}")
    print(f"  质量分数: {result.quality_score:.3f}")
    
    # 批量测试
    print("\n批量测试:")
    batch_features = torch.cat([known_features[:3], torch.randn(2, 128)], dim=0)
    batch_results = recognizer.batch_recognize(batch_features)
    
    for i, result in enumerate(batch_results):
        print(f"  样本 {i}: 已知={result.is_known}, ID={result.predicted_id}, 置信度={result.confidence:.3f}")
    
    # 统计信息
    print("\n统计信息:")
    stats = recognizer.get_stats()
    print(f"  总查询数: {stats['total_queries']}")
    print(f"  接受已知: {stats['accepted_known']}")
    print(f"  拒识未知: {stats['rejected_unknown']}")
    print(f"  拒识质量: {stats['rejected_quality']}")
    print(f"  接受率: {stats['acceptance_rate']:.3f}")
    print(f"  拒识率: {stats['rejection_rate']:.3f}")
    
    print("🎉 开放集识别机制测试完成")
