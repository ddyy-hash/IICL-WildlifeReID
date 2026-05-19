#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import torch
from torchvision import transforms
import cv2
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.core.joint_model import JointReIDModel


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])


def tensor_to_uint8_img(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu().clamp(0.0, 1.0)
    arr = t.numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * 255.0).round().astype(np.uint8)
    return arr


def compute_brightness(img: np.ndarray) -> float:
    if img.ndim == 3:
        gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    else:
        gray = img
    return float(gray.mean())


def simulate_lighting_conditions(img_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    results = {}
    
    results["Original"] = img_tensor
    
    for factor in [0.3, 0.5]:
        results[f"Dark_{factor}x"] = (img_tensor * factor).clamp(0, 1)
    
    for factor in [1.5, 2.0]:
        results[f"Bright_{factor}x"] = (img_tensor * factor).clamp(0, 1)
    
    warm = img_tensor.clone()
    warm[0] = (warm[0] * 1.3).clamp(0, 1)  # R+
    warm[2] = (warm[2] * 0.7).clamp(0, 1)  # B-
    results["Warm"] = warm
    
    cold = img_tensor.clone()
    cold[0] = (cold[0] * 0.75).clamp(0, 1)  # R-
    cold[2] = (cold[2] * 1.25).clamp(0, 1)  # B+
    results["Cold"] = cold
    
    low_contrast = img_tensor * 0.5 + 0.25
    results["LowContrast"] = low_contrast.clamp(0, 1)
    
    high_contrast = (img_tensor - 0.5) * 1.5 + 0.5
    results["HighContrast"] = high_contrast.clamp(0, 1)
    
    return results


@torch.no_grad()
def validate_illumination_module(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str,
    device: torch.device,
    max_samples: int = 20,
):
    
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = checkpoint.get("num_classes", 100)
    
    model = JointReIDModel(
        num_classes=num_classes,
        num_stripes=6,
        pretrained_backbone=False,
        soft_mask_temperature=10.0,
        soft_mask_type="sigmoid",
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    transform = build_transform()
    
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    all_images = []
    
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if any(f.lower().endswith(e) for e in exts):
                all_images.append(os.path.join(root, f))
                if len(all_images) >= max_samples:
                    break
        if len(all_images) >= max_samples:
            break
    
    print(f"[INFO] Selected {len(all_images)} images for validation")
    
    all_brightness_before = []
    all_brightness_after = []
    all_feature_distances = []
    
    detailed_results = []
    
    for img_idx, img_path in enumerate(all_images):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img_rgb)
        
        lighting_conditions = simulate_lighting_conditions(img_tensor)
        
        condition_brightness_before = []
        condition_brightness_after = []
        condition_features = []
        
        for cond_name, cond_tensor in lighting_conditions.items():
            cond_batch = cond_tensor.unsqueeze(0).to(device)
            output = model(cond_batch, boxes_list=None, return_illuminated=True)
            
            illum_tensor = output.get("illuminated", None)
            features = output.get("features", None)
            
            if illum_tensor is not None:
                orig_img = tensor_to_uint8_img(cond_tensor)
                illum_img = tensor_to_uint8_img(illum_tensor[0])
                
                b_before = compute_brightness(orig_img)
                b_after = compute_brightness(illum_img)
                
                condition_brightness_before.append(b_before)
                condition_brightness_after.append(b_after)
            
            if features is not None:
                feat = torch.nn.functional.normalize(features[0], p=2, dim=0)
                condition_features.append(feat.cpu())
        
        if len(condition_brightness_before) > 1:
            var_before = np.var(condition_brightness_before)
            var_after = np.var(condition_brightness_after)
            
            all_brightness_before.extend(condition_brightness_before)
            all_brightness_after.extend(condition_brightness_after)
            
            detailed_results.append({
                "image": os.path.basename(img_path),
                "brightness_var_before": var_before,
                "brightness_var_after": var_after,
                "brightness_values_before": condition_brightness_before,
                "brightness_values_after": condition_brightness_after,
            })
        
        if len(condition_features) > 1:
            orig_feat = condition_features[0]
            distances = []
            for feat in condition_features[1:]:
                dist = 1 - torch.dot(orig_feat, feat).item()
                distances.append(dist)
            all_feature_distances.extend(distances)
            detailed_results[-1]["feature_distances"] = distances
        
        if (img_idx + 1) % 5 == 0:
            print(f"[INFO] Processed {img_idx + 1}/{len(all_images)} images")
    
    print("\n" + "=" * 60)
    print("           Illumination Module Validation Report")
    print("=" * 60)
    
    print("\n[1. Brightness Consistency Analysis]")
    print("-" * 40)
    
    if detailed_results:
        vars_before = [r["brightness_var_before"] for r in detailed_results]
        vars_after = [r["brightness_var_after"] for r in detailed_results]
        
        mean_var_before = np.mean(vars_before)
        mean_var_after = np.mean(vars_after)
        
        print(f"Brightness variance across lighting conditions (before normalization): {mean_var_before:.2f}")
        print(f"Brightness variance across lighting conditions (after normalization): {mean_var_after:.2f}")
        
        if mean_var_before > 0:
            reduction = (1 - mean_var_after / mean_var_before) * 100
            print(f"Variance reduction ratio: {reduction:.1f}%")
            
            if reduction > 50:
                print("PASS: the illumination module substantially reduces brightness variance.")
            elif reduction > 20:
                print("WARN: the illumination module helps, but the gain is modest.")
            else:
                print("FAIL: the illumination module effect is weak.")
    
    print("\n[2. Feature Stability Analysis]")
    print("-" * 40)
    
    if all_feature_distances:
        mean_dist = np.mean(all_feature_distances)
        std_dist = np.std(all_feature_distances)
        max_dist = np.max(all_feature_distances)
        
        print("Feature cosine distance across lighting conditions:")
        print(f"  Mean: {mean_dist:.4f}")
        print(f"  Std: {std_dist:.4f}")
        print(f"  Max: {max_dist:.4f}")
        
        if mean_dist < 0.1:
            print("PASS: features are very stable under illumination changes.")
        elif mean_dist < 0.2:
            print("WARN: features are fairly stable, but lighting still has some effect.")
        else:
            print("FAIL: features are not stable enough; the illumination module needs refinement.")
    
    print("\n[3. Exporting visualization charts]")
    print("-" * 40)
    
    if all_brightness_before and all_brightness_after:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].hist(all_brightness_before, bins=30, alpha=0.7, label='Before Normalization', color='red')
        axes[0].hist(all_brightness_after, bins=30, alpha=0.7, label='After Normalization', color='green')
        axes[0].set_xlabel('Brightness')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Brightness Distribution')
        axes[0].legend()
        
        if detailed_results:
            vars_before = [r["brightness_var_before"] for r in detailed_results]
            vars_after = [r["brightness_var_after"] for r in detailed_results]
            
            x = np.arange(len(vars_before))
            width = 0.35
            axes[1].bar(x - width/2, vars_before, width, label='Before', color='red', alpha=0.7)
            axes[1].bar(x + width/2, vars_after, width, label='After', color='green', alpha=0.7)
            axes[1].set_xlabel('Sample Index')
            axes[1].set_ylabel('Brightness Variance')
            axes[1].set_title('Brightness Variance per Sample')
            axes[1].legend()
        
        if all_feature_distances:
            axes[2].hist(all_feature_distances, bins=20, color='blue', alpha=0.7)
            axes[2].axvline(x=np.mean(all_feature_distances), color='red', linestyle='--', label=f'Mean: {np.mean(all_feature_distances):.4f}')
            axes[2].set_xlabel('Feature Cosine Distance')
            axes[2].set_ylabel('Count')
            axes[2].set_title('Feature Distance Distribution\n(Lower is Better)')
            axes[2].legend()
        
        plt.tight_layout()
        chart_path = os.path.join(output_dir, "illumination_validation_chart.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Chart saved to: {chart_path}")
    
    report_path = os.path.join(output_dir, "illumination_validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("        Illumination Module Validation Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("[Experiment Setup]\n")
        f.write(f"  Model: {os.path.basename(checkpoint_path)}\n")
        f.write(f"  Samples: {len(all_images)}\n")
        f.write(f"  Lighting conditions: {len(simulate_lighting_conditions(torch.zeros(3, 256, 256)))}\n\n")
        
        if detailed_results:
            vars_before = [r["brightness_var_before"] for r in detailed_results]
            vars_after = [r["brightness_var_after"] for r in detailed_results]
            
            f.write("[Brightness Consistency]\n")
            f.write(f"  Mean variance before normalization: {np.mean(vars_before):.2f}\n")
            f.write(f"  Mean variance after normalization: {np.mean(vars_after):.2f}\n")
            if np.mean(vars_before) > 0:
                f.write(f"  Variance reduction ratio: {(1 - np.mean(vars_after) / np.mean(vars_before)) * 100:.1f}%\n\n")
        
        if all_feature_distances:
            f.write("[Feature Stability]\n")
            f.write(f"  Mean feature distance: {np.mean(all_feature_distances):.4f}\n")
            f.write(f"  Std feature distance: {np.std(all_feature_distances):.4f}\n")
            f.write(f"  Max feature distance: {np.max(all_feature_distances):.4f}\n\n")
        
        f.write("[Conclusion]\n")
        if detailed_results and all_feature_distances:
            reduction = (1 - np.mean(vars_after) / np.mean(vars_before)) * 100 if np.mean(vars_before) > 0 else 0
            mean_dist = np.mean(all_feature_distances)
            
            if reduction > 50 and mean_dist < 0.1:
                f.write("  PASS: the illumination module is highly effective; brightness variance drops and features stay stable.\n")
            elif reduction > 30 or mean_dist < 0.15:
                f.write("  WARN: the illumination module helps, but there is still room for improvement.\n")
            else:
                f.write("  FAIL: the illumination module effect is limited and needs further improvement.\n")
    
    print(f"Detailed report saved to: {report_path}")
    print("\n" + "=" * 60)
    print("Validation complete. Please inspect the charts and report in the output directory.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate illumination-module effectiveness")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model path")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--max_samples", type=int, default=20, help="Maximum number of samples")
    
    args = parser.parse_args()
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"[INFO] Using device: {device}")
    
    validate_illumination_module(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=device,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
