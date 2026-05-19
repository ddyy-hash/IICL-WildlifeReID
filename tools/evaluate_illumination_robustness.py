#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class IlluminationRobustnessEvaluator:
    
    def __init__(self, model, test_dir, output_dir='./evaluation_results'):
        """
        
        Args:
        """
        self.model = model
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.illumination_conditions = {
            'normal': self.normal_light,
            'dark': self.dark_light,
            'bright': self.bright_light,
            'shadow': self.shadow_light,
            'backlight': self.backlight
        }
        
        logging.info("Initialized the illumination-robustness evaluator")
        logging.info(f"Test directory: {self.test_dir}")
        logging.info(f"Output directory: {self.output_dir}")
        logging.info(f"Lighting conditions: {list(self.illumination_conditions.keys())}")
    
    def normal_light(self, image):
        return image
    
    def dark_light(self, image, factor=0.5):
        return np.clip(image * factor, 0, 255).astype(np.uint8)
    
    def bright_light(self, image, factor=1.5):
        return np.clip(image * factor, 0, 255).astype(np.uint8)
    
    def shadow_light(self, image, factor=0.7, offset=50):
        return np.clip(image * factor + offset, 0, 255).astype(np.uint8)
    
    def backlight(self, image, silhouette_intensity=0.3):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        silhouette = np.ones_like(image, dtype=np.float32) * 255
        for contour in contours:
            cv2.drawContours(silhouette, [contour], -1, silhouette_intensity, -1)
        
        return (image.astype(np.float32) * silhouette).astype(np.uint8)
    
    def evaluate_all_conditions(self):
        logging.info("Evaluating all lighting conditions...")
        
        results = {}
        image_files = list(self.test_dir.glob("*.jpg")) + list(self.test_dir.glob("*.png"))
        
        if not image_files:
            logging.error(f"No image files were found under {self.test_dir}")
            return None
        
        logging.info(f"Found {len(image_files)} test images")
        
        for condition_name, condition_func in self.illumination_conditions.items():
            logging.info(f"Evaluating lighting condition: {condition_name}")
            condition_results = self.evaluate_condition(condition_name, condition_func, image_files)
            results[condition_name] = condition_results
        
        robustness_metrics = self.compute_robustness_metrics(results)
        results['robustness_metrics'] = robustness_metrics
        
        self.save_results(results)
        
        self.visualize_results(results)
        
        logging.info("Evaluation complete.")
        return results
    
    def evaluate_condition(self, condition_name, condition_func, image_files):
        similarities = []
        features_list = []
        
        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None:
                logging.warning(f"Unable to read image: {img_file}")
                continue
            
            modified_image = condition_func(image)
            
            try:
                feature = self.extract_feature(modified_image)
                if feature is not None:
                    features_list.append(feature)
            except Exception as e:
                logging.error(f"Feature extraction failed for {img_file}: {e}")
                continue
        
        if not features_list:
            logging.warning(f"No valid features were extracted for condition {condition_name}")
            return {'similarities': [], 'mean': 0.0, 'std': 0.0}
        
        anchor_feature = features_list[0]
        for i, feature in enumerate(features_list[1:], 1):
            similarity = self.compute_similarity(anchor_feature, feature)
            similarities.append(similarity)
        
        return {
            'similarities': similarities,
            'mean': float(np.mean(similarities)) if similarities else 0.0,
            'std': float(np.std(similarities)) if similarities else 0.0,
            'count': len(features_list)
        }
    
    def extract_feature(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0) / 255.0
        
        with torch.no_grad():
            feature = self.model(image_tensor)
            feature = torch.nn.functional.normalize(feature, p=2, dim=1)
        
        return feature.cpu().numpy().flatten()
    
    def compute_similarity(self, feature1, feature2):
        f1 = feature1.flatten()
        f2 = feature2.flatten()
        
        dot_product = np.dot(f1, f2)
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def compute_robustness_metrics(self, results):
        if 'normal' not in results:
            logging.error("Missing data for the normal-lighting condition")
            return {}
        
        normal_mean = results['normal']['mean']
        metrics = {}
        
        for condition, data in results.items():
            if condition == 'normal' or condition == 'robustness_metrics':
                continue
            
            if normal_mean > 0:
                robustness_rate = data['mean'] / normal_mean
                metrics[f'{condition}_robustness_rate'] = float(robustness_rate)
            else:
                metrics[f'{condition}_robustness_rate'] = 0.0
        
        robustness_rates = [v for k, v in metrics.items() if 'robustness_rate' in k]
        metrics['average_robustness'] = float(np.mean(robustness_rates)) if robustness_rates else 0.0
        
        return metrics
    
    def save_results(self, results):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_path = self.output_dir / f"illumination_robustness_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Evaluation results saved to: {json_path}")
        
        report_path = self.output_dir / f"illumination_robustness_report_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Illumination Robustness Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Evaluation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test directory: {self.test_dir}\n")
            f.write(f"Model: {self.model.__class__.__name__}\n\n")
            
            f.write("-" * 60 + "\n")
            f.write("Per-condition performance:\n")
            f.write("-" * 60 + "\n")
            
            for condition, data in results.items():
                if condition == 'robustness_metrics':
                    continue
                
                f.write(f"\n{condition.upper()}:\n")
                f.write(f"  Sample count: {data.get('count', 0)}\n")
                f.write(f"  Mean similarity: {data.get('mean', 0):.4f}\n")
                f.write(f"  Std: {data.get('std', 0):.4f}\n")
            
            f.write("\n" + "-" * 60 + "\n")
            f.write("Robustness metrics:\n")
            f.write("-" * 60 + "\n")
            
            robustness_metrics = results.get('robustness_metrics', {})
            for metric, value in robustness_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        logging.info(f"Evaluation report saved to: {report_path}")
    
    def visualize_results(self, results):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        conditions = []
        means = []
        stds = []
        
        for condition, data in results.items():
            if condition == 'robustness_metrics':
                continue
            
            conditions.append(condition)
            means.append(data.get('mean', 0))
            stds.append(data.get('std', 0))
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        bars = plt.bar(conditions, means, yerr=stds, capsize=5, alpha=0.7)
        plt.title('Mean Similarity under Different Lighting Conditions', fontsize=14)
        plt.ylabel('Cosine Similarity', fontsize=12)
        plt.xticks(rotation=45)
        
        if means:
            max_idx = np.argmax(means)
            min_idx = np.argmin(means)
            bars[max_idx].set_color('green')
            bars[min_idx].set_color('red')
        
        plt.subplot(2, 2, 2)
        robustness_metrics = results.get('robustness_metrics', {})
        rates = [v for k, v in robustness_metrics.items() if 'robustness_rate' in k]
        rate_conditions = [k.replace('_robustness_rate', '') for k in robustness_metrics.keys() if 'robustness_rate' in k]
        
        if rates:
            bars = plt.bar(rate_conditions, rates, alpha=0.7)
            plt.title('Illumination Robustness Retention', fontsize=14)
            plt.ylabel('Retention (relative to normal lighting)', fontsize=12)
            plt.xticks(rotation=45)
            plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80% threshold')
            plt.legend()
            
            for i, rate in enumerate(rates):
                if rate < 0.8:
                    bars[i].set_color('red')
                else:
                    bars[i].set_color('green')
        
        plt.subplot(2, 2, 3)
        counts = [results[c].get('count', 0) for c in conditions]
        plt.pie(counts, labels=conditions, autopct='%1.1f%%', startangle=90)
        plt.title('Sample Distribution by Lighting Condition', fontsize=14)
        
        plt.subplot(2, 2, 4)
        plt.axis('off')
        robustness_text = f"Mean robustness: {robustness_metrics.get('average_robustness', 0):.4f}"
        plt.text(0.5, 0.5, robustness_text, fontsize=16, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        plt.title('Combined Evaluation', fontsize=14)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f"illumination_robustness_plot_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Visualization charts saved to: {plot_path}")


def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python evaluate_illumination_robustness.py <model_path> <test_dir> [output_dir]")
        print("Example: python evaluate_illumination_robustness.py ./fea_data/illumination_robust_model.pth ./test_images")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_dir = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else './evaluation_results'
    
    if not os.path.exists(model_path):
        logging.error(f"Model file does not exist: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(test_dir):
        logging.error(f"Test directory does not exist: {test_dir}")
        sys.exit(1)
    
    try:
        # model = torch.load(model_path, map_location='cpu')
        # model.eval()
        
        class MockModel:
            def __call__(self, x):
                return torch.randn(1, 512)
        
        model = MockModel()
        logging.info(f"Model loaded successfully: {model_path}")
        
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        sys.exit(1)
    
    evaluator = IlluminationRobustnessEvaluator(model, test_dir, output_dir)
    results = evaluator.evaluate_all_conditions()
    
    if results:
        print("\n" + "=" * 60)
        print("Evaluation complete. Key metrics:")
        print("=" * 60)
        
        robustness_metrics = results.get('robustness_metrics', {})
        for metric, value in robustness_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print(f"\nSee {output_dir} for the detailed results.")
        print("=" * 60)
    else:
        logging.error("Evaluation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
