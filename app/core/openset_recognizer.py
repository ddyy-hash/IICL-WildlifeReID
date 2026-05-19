#!/usr/bin/env python3

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
    is_known: bool
    predicted_id: Optional[int]
    confidence: float
    uncertainty: float
    rejection_reason: Optional[str]
    similarity_score: float
    quality_score: float


class QualityAssessor:
    
    def __init__(self):
        self.sharpness_threshold = 15.0
        self.contrast_threshold = 10.0
        self.brightness_min = 0.1
        self.brightness_max = 0.9
        
    def assess_quality(self, image: torch.Tensor) -> Dict[str, float]:
        """
        
        Args:
            
        Returns:
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        B, C, H, W = image.shape
        
        if C == 3:
            gray_image = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        else:
            gray_image = image[:, 0, :, :]
        
        laplacian = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32).to(image.device)
        laplacian = laplacian.expand(1, 1, 3, 3)
        
        sharpness_map = F.conv2d(gray_image.unsqueeze(1), laplacian, padding=1)
        sharpness = torch.mean(torch.abs(sharpness_map), dim=[1, 2, 3])
        
        contrast = torch.std(gray_image, dim=[1, 2])
        
        brightness = torch.mean(gray_image, dim=[1, 2])
        
        sharpness_score = torch.clamp(sharpness / self.sharpness_threshold, 0, 1)
        contrast_score = torch.clamp(contrast / self.contrast_threshold, 0, 1)
        brightness_score = torch.clamp(
            (brightness - self.brightness_min) / (self.brightness_max - self.brightness_min), 0, 1
        )
        
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
    
    def __init__(self, initial_threshold: float = 0.7, adaptation_rate: float = 0.01,
                 percentile: float = 95.0):
        """
        
        Args:
        """
        self.initial_threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.current_threshold = initial_threshold
        self.percentile = percentile
        
        self.intra_class_distances = []
        
        self.accepted_count = 0
        self.rejected_count = 0
        self.query_scores = []
        
        self.is_calibrated = False
        self.calibrated_threshold = initial_threshold
        
    def calibrate_from_gallery(self, gallery_features: np.ndarray, gallery_labels: np.ndarray):
        """
        
        
        Args:
        """
        if len(gallery_features) < 10:
            logging.warning("Not enough samples to calibrate the threshold")
            return
        
        intra_distances = []
        unique_labels = np.unique(gallery_labels)
        
        for label in unique_labels:
            mask = gallery_labels == label
            class_features = gallery_features[mask]
            
            if len(class_features) < 2:
                continue
            
            for i in range(len(class_features)):
                for j in range(i + 1, len(class_features)):
                    dist = np.linalg.norm(class_features[i] - class_features[j])
                    intra_distances.append(dist)
        
        if len(intra_distances) > 0:
            self.intra_class_distances = intra_distances
            self.calibrated_threshold = np.percentile(intra_distances, self.percentile)
            self.current_threshold = self.calibrated_threshold
            self.is_calibrated = True
            logging.info(f"[OK] Threshold calibration complete: {self.calibrated_threshold:.4f} "
                        f"(based on {len(intra_distances)} intra-class distance pairs)")
    
    def update_threshold(self, similarity_scores: List[float], is_known: List[bool] = None):
        """
        
        Args:
        """
        self.query_scores.extend(similarity_scores)
        
        max_stats = 1000
        self.query_scores = self.query_scores[-max_stats:]
        
        if self.is_calibrated and len(self.query_scores) > 50:
            median_score = np.median(self.query_scores[-50:])
            
            if abs(median_score - self.calibrated_threshold) > 0.1:
                self.current_threshold = (
                    self.current_threshold * (1 - self.adaptation_rate * 0.1) + 
                    self.calibrated_threshold * self.adaptation_rate * 0.1
                )
        
        self.current_threshold = np.clip(self.current_threshold, 0.3, 0.95)
    
    def get_threshold(self) -> float:
        return self.current_threshold
    
    def get_stats(self) -> Dict[str, Any]:
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
    
    def __init__(self, num_mc_samples: int = 10):
        """
        
        Args:
        """
        self.num_mc_samples = num_mc_samples
        
    def quantify_uncertainty(
        self, 
        features: torch.Tensor, 
        feature_database: Any, 
        k: int = 5
    ) -> Dict[str, float]:
        """
        
        Args:
            
        Returns:
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        similar_ids, distances = feature_database.search(features.cpu().numpy(), k=k)
        
        uncertainties = []
        
        for i, (neighbor_ids, dists) in enumerate(zip(similar_ids, distances)):
            if len(neighbor_ids) < 2:
                uncertainties.append(1.0)
                continue
            
            neighbor_classes = []
            for neighbor_id in neighbor_ids:
                result = feature_database.get_feature_by_id(neighbor_id)
                if result is not None:
                    _, metadata = result
                    neighbor_classes.append(metadata.get('dog_id', -1))
            
            if len(set(neighbor_classes)) <= 1:
                uncertainty = 0.1
            else:
                class_counts = {}
                for cls in neighbor_classes:
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                
                total = sum(class_counts.values())
                entropy = 0.0
                for count in class_counts.values():
                    p = count / total
                    entropy -= p * math.log(p + 1e-10)
                
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
    
    def __init__(
        self,
        feature_database: Any,
        initial_threshold: float = 0.7,
        quality_threshold: float = 0.5,
        uncertainty_threshold: float = 0.6,
        adaptation_rate: float = 0.01
    ):
        """
        
        Args:
        """
        self.feature_database = feature_database
        self.quality_threshold = quality_threshold
        self.uncertainty_threshold = uncertainty_threshold
        
        self.quality_assessor = QualityAssessor()
        
        self.threshold_strategy = DynamicThresholdStrategy(
            initial_threshold=initial_threshold,
            adaptation_rate=adaptation_rate
        )
        
        self.uncertainty_quantifier = UncertaintyQuantifier(num_mc_samples=10)
        
        self.recognition_stats = {
            'total_queries': 0,
            'accepted_known': 0,
            'rejected_unknown': 0,
            'rejected_quality': 0,
            'rejected_uncertainty': 0
        }
        
        logging.info("[OK] Open-set recognizer initialized")
        logging.info(f"   Initial threshold: {initial_threshold}")
        logging.info(f"   Quality threshold: {quality_threshold}")
        logging.info(f"   Uncertainty threshold: {uncertainty_threshold}")
    
    def recognize(
        self,
        query_features: torch.Tensor,
        query_image: Optional[torch.Tensor] = None,
        k: int = 5,
        return_details: bool = False
    ) -> OpenSetRecognitionResult:
        """
        
        Args:
            
        Returns:
        """
        self.recognition_stats['total_queries'] += 1
        
        if query_features.dim() > 1:
            query_features = query_features.squeeze()
        
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
                    rejection_reason=f"Image quality is too low (score: {quality_score:.3f})",
                    similarity_score=0.0,
                    quality_score=quality_score
                )
        else:
            quality_score = 1.0
        
        similar_ids, distances = self.feature_database.search(
            query_features.cpu().numpy().reshape(1, -1), 
            k=k
        )
        
        if not similar_ids[0]:
            self.recognition_stats['rejected_unknown'] += 1
            return OpenSetRecognitionResult(
                is_known=False,
                predicted_id=None,
                confidence=0.0,
                uncertainty=1.0,
                rejection_reason="No similar features found",
                similarity_score=0.0,
                quality_score=quality_score
            )
        
        similarity_scores = [1.0 / (1.0 + dist) for dist in distances[0]]
        max_similarity = max(similarity_scores)
        
        best_match_id = similar_ids[0][0]
        best_match_result = self.feature_database.get_feature_by_id(best_match_id)
        
        if best_match_result is None:
            self.recognition_stats['rejected_unknown'] += 1
            return OpenSetRecognitionResult(
                is_known=False,
                predicted_id=None,
                confidence=0.0,
                uncertainty=1.0,
                rejection_reason="Unable to retrieve matching features",
                similarity_score=max_similarity,
                quality_score=quality_score
            )
        
        _, metadata = best_match_result
        predicted_id = metadata.get('dog_id', -1)
        
        uncertainty_result = self.uncertainty_quantifier.quantify_uncertainty(
            query_features, 
            self.feature_database, 
            k=k
        )
        uncertainty = uncertainty_result['uncertainty']
        
        current_threshold = self.threshold_strategy.get_threshold()
        
        is_known = True
        rejection_reason = None
        
        if max_similarity < current_threshold:
            is_known = False
            rejection_reason = f"Similarity is below the threshold ({max_similarity:.3f} < {current_threshold:.3f})"
        
        elif uncertainty > self.uncertainty_threshold:
            is_known = False
            rejection_reason = f"Uncertainty is too high ({uncertainty:.3f} > {self.uncertainty_threshold:.3f})"
        
        if is_known:
            self.recognition_stats['accepted_known'] += 1
        else:
            self.recognition_stats['rejected_unknown'] += 1
        
        self.threshold_strategy.update_threshold([max_similarity], [is_known])
        
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
        
        Args:
            
        Returns:
        """
        results = []
        
        for i in range(query_features.shape[0]):
            feature = query_features[i]
            image = query_images[i] if query_images is not None else None
            
            result = self.recognize(feature, image, k=k)
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
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


if __name__ == "__main__":
    print("Testing the open-set recognition pipeline")
    
    from app.core.feature_database import FeatureDatabase
    
    feature_db = FeatureDatabase(feature_dim=128, index_type="Flat")
    
    known_features = torch.randn(50, 128)
    known_metadata = [{'dog_id': i // 10} for i in range(50)]
    
    feature_db.add_features(known_features.numpy(), known_metadata)
    
    print(f"[OK] Feature database created; feature count: {feature_db.total_features}")
    
    recognizer = OpenSetRecognizer(
        feature_database=feature_db,
        initial_threshold=0.6,
        quality_threshold=0.3,
        uncertainty_threshold=0.7,
        adaptation_rate=0.01
    )
    
    print("\nTesting known-class recognition:")
    known_query = known_features[0]
    result = recognizer.recognize(known_query, k=5)
    
    print(f"  Is known: {result.is_known}")
    print(f"  Predicted ID: {result.predicted_id}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Uncertainty: {result.uncertainty:.3f}")
    print(f"  Similarity: {result.similarity_score:.3f}")
    
    print("\nTesting unknown-class rejection:")
    unknown_query = torch.randn(128)
    result = recognizer.recognize(unknown_query, k=5)
    
    print(f"  Is known: {result.is_known}")
    print(f"  Rejection reason: {result.rejection_reason}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Uncertainty: {result.uncertainty:.3f}")
    
    print("\nTesting quality-based rejection:")
    low_quality_image = torch.rand(3, 64, 64) * 0.1
    result = recognizer.recognize(known_query, query_image=low_quality_image)
    
    print(f"  Is known: {result.is_known}")
    print(f"  Rejection reason: {result.rejection_reason}")
    print(f"  Quality score: {result.quality_score:.3f}")
    
    print("\nBatch test:")
    batch_features = torch.cat([known_features[:3], torch.randn(2, 128)], dim=0)
    batch_results = recognizer.batch_recognize(batch_features)
    
    for i, result in enumerate(batch_results):
        print(f"  Sample {i}: known={result.is_known}, ID={result.predicted_id}, confidence={result.confidence:.3f}")
    
    print("\nStatistics:")
    stats = recognizer.get_stats()
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Accepted known: {stats['accepted_known']}")
    print(f"  Rejected unknown: {stats['rejected_unknown']}")
    print(f"  Rejected for quality: {stats['rejected_quality']}")
    print(f"  Acceptance rate: {stats['acceptance_rate']:.3f}")
    print(f"  Rejection rate: {stats['rejection_rate']:.3f}")
    
    print("Open-set recognition smoke test complete")
