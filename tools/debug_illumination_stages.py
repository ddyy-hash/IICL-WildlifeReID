#!/usr/bin/env python3
"""Debug intermediate stages of the illumination module on single images or datasets.

The script exports stage-by-stage panels, compact comparison strips, and a text
summary of simple heuristics for potential failure modes.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.core.joint_model import JointReIDModel
from app.core.illumination_module_v2 import IPAIDModule


class IlluminationStageDebugger:
    """Utility for inspecting IPAID intermediate tensors and exported figures."""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = self._load_model(checkpoint_path)
        self.ipaid = self.model.illumination
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
    def _load_model(self, checkpoint_path: str) -> JointReIDModel:
        """Load a JointReIDModel checkpoint for debugging."""
        print(f"Loading model: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        num_classes = 107
        backbone_name = 'osnet_x1_0'
        
        for key in state_dict.keys():
            if 'classifier' in key and 'weight' in key:
                num_classes = state_dict[key].shape[0]
                print(f"  Inferred number of classes: {num_classes}")
                break
        
        model = JointReIDModel(
            num_classes=num_classes,
            backbone_name=backbone_name,
            num_stripes=6,
            hidden_dim=256,
            pretrained_backbone=False,
        )
        
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        
        return model
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess a single image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return tensor
    
    def debug_forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the illumination module and collect intermediate tensors."""
        ipaid = self.ipaid
        results = {}
        
        with torch.no_grad():
            results['original'] = x.clone()
            
            luminance = ipaid.rgb_to_luminance(x)
            results['luminance_Y'] = luminance.clone()
            
            illum_estimator = ipaid.illumination_estimator
            B, _, H, W = luminance.shape
            
            features = illum_estimator.shared_encoder(luminance)
            features_attended = features * illum_estimator.channel_attention(features)
            results['illum_features'] = features_attended.mean(dim=1, keepdim=True).clone()
            
            illuminations = []
            for i, branch in enumerate(illum_estimator.scale_branches):
                scale_factor = 2 ** (illum_estimator.num_scales - 1 - i)
                
                if scale_factor > 1:
                    feat_scaled = F.adaptive_avg_pool2d(features_attended, 
                                                        (H // scale_factor, W // scale_factor))
                else:
                    feat_scaled = features_attended
                
                illum_raw = branch(feat_scaled)
                illum_raw = torch.sigmoid(illum_raw) * 2.0 + 0.1
                
                if scale_factor > 1:
                    illum_raw = F.interpolate(illum_raw, size=(H, W), 
                                              mode='bilinear', align_corners=False)
                
                illuminations.append(illum_raw)
                results[f'illum_scale_{i}'] = illum_raw.clone()
            
            weights = F.softmax(illum_estimator.scale_weights, dim=0)
            illumination = sum(w * ill for w, ill in zip(weights, illuminations))
            results['illum_raw'] = illumination.clone()
            results['scale_weights'] = weights.clone()
            
            if ipaid.use_sensitivity:
                sensitivity = ipaid.sensitivity_estimator(x)
                results['sensitivity'] = sensitivity.clone()
                
                illumination_adaptive = 1.0 + (illumination - 1.0) * sensitivity
                results['illum_adaptive_before_clamp'] = illumination_adaptive.clone()
                
                alpha_residual = 0.5 * sensitivity
                results['alpha_residual'] = alpha_residual.clone()
            else:
                illumination_adaptive = illumination
                sensitivity = None
                alpha_residual = torch.full_like(illumination, 0.3)
            
            illumination_clamped = torch.clamp(illumination_adaptive, 0.2, 3.0)
            results['illum_after_clamp'] = illumination_clamped.clone()
            
            illumination_smooth = F.avg_pool2d(illumination_clamped, kernel_size=5, 
                                               stride=1, padding=2)
            results['illum_final'] = illumination_smooth.clone()
            
            eps = 1e-4
            
            L_correction_factor = 1.0 / illumination_smooth
            L_correction_factor = torch.clamp(L_correction_factor, 0.33, 3.0)
            results['L_correction_factor'] = L_correction_factor.clone()
            
            if hasattr(ipaid, 'apply_safe_illumination_correction'):
                reflectance_retinex = ipaid.apply_safe_illumination_correction(x, L_correction_factor)
            else:
                luminance_safe = torch.clamp(luminance, eps, 1.0)
                color_ratio = x / luminance_safe
                color_ratio = torch.clamp(color_ratio, 0.0, 5.0)
                log_Y = torch.log(luminance_safe)
                log_L = torch.log(illumination_smooth)
                log_R = log_Y - log_L
                luminance_retinex = torch.exp(log_R)
                luminance_retinex = torch.clamp(luminance_retinex, 0.0, 1.0)
                reflectance_retinex = luminance_retinex * color_ratio
                reflectance_retinex = torch.clamp(reflectance_retinex, 0.0, 1.0)
            
            results['reflectance_retinex'] = reflectance_retinex.clone()
            
            if hasattr(ipaid, 'rgb_to_chromaticity'):
                chrom_orig = ipaid.rgb_to_chromaticity(x)
                chrom_retinex = ipaid.rgb_to_chromaticity(reflectance_retinex + eps)
                color_angle_diff = 1.0 - torch.sum(chrom_orig * chrom_retinex, dim=1, keepdim=True)
                color_risk = torch.clamp(color_angle_diff * 10, 0.0, 1.0)
                results['color_risk'] = color_risk.clone()
                
                reflectance_retinex_safe = (1 - color_risk) * reflectance_retinex + color_risk * x
                results['reflectance_retinex_safe'] = reflectance_retinex_safe.clone()
            else:
                color_risk = torch.zeros_like(luminance)
                reflectance_retinex_safe = reflectance_retinex
            
            if hasattr(ipaid, 'identity_enhancer'):
                reflectance_enhanced, aux_outputs = ipaid.identity_enhancer(x, reflectance_retinex_safe)
                results['reflectance_enhanced'] = reflectance_enhanced.clone()
                results['saliency_map'] = aux_outputs['saliency_map'].clone()
                results['enhance_residual'] = aux_outputs['enhance_residual'].clone()
                results['gain_map'] = aux_outputs['saliency_map'].clone()
                
                reflectance_fused = reflectance_enhanced
            elif hasattr(ipaid, 'texture_enhancer'):
                reflectance_enhanced, gain_map = ipaid.texture_enhancer(x, reflectance_retinex_safe)
                results['reflectance_enhanced'] = reflectance_enhanced.clone()
                results['gain_map'] = gain_map.clone()
                reflectance_fused = reflectance_enhanced
            else:
                reflectance_direct = ipaid.reflectance_estimator(x)
                results['reflectance_direct'] = reflectance_direct.clone()
                
                fusion_alpha = 0.5 * (1 - color_risk) + 0.7 * color_risk
                reflectance_fused = fusion_alpha * reflectance_direct + (1 - fusion_alpha) * reflectance_retinex_safe
            
            results['reflectance_fused'] = reflectance_fused.clone()
            
            alpha_effective = alpha_residual * (1 - color_risk * 0.3)
            results['alpha_effective'] = alpha_effective.clone()
            
            reflectance_before_refine = x + alpha_effective * (reflectance_fused - x)
            results['reflectance_before_refine'] = reflectance_before_refine.clone()
            
            if ipaid.use_refinement:
                reflectance_refined = ipaid.refiner(reflectance_before_refine, 
                                                     illumination_smooth, x)
            else:
                reflectance_refined = reflectance_before_refine
            results['reflectance_after_refine'] = reflectance_refined.clone()
            
            reflectance_final = torch.clamp(reflectance_refined, 0.01, 0.99)
            results['final_output'] = reflectance_final.clone()
            
            diff = torch.abs(reflectance_final - x)
            results['difference'] = diff.clone()
            
        return results
    
    def visualize_stages(self, results: Dict[str, torch.Tensor], 
                        output_path: str, title: str = ""):
        """"""
        
        def tensor_to_img(t: torch.Tensor, is_gray: bool = False) -> np.ndarray:
            """"""
            t = t[0].detach().cpu().clamp(0, 1)
            if is_gray or t.shape[0] == 1:
                return t[0].numpy()
            else:
                return t.permute(1, 2, 0).numpy()
        
        def apply_colormap(gray: np.ndarray, cmap: str = 'jet') -> np.ndarray:
            """"""
            gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
            cmap_fn = plt.get_cmap(cmap)
            return cmap_fn(gray_norm)[:, :, :3]
        
        fig = plt.figure(figsize=(28, 24))
        fig.suptitle(f'IPAID Module Debug (v3 - Identity-Aware Enhancement)\n{title}', fontsize=14, fontweight='bold')
        
        rows, cols = 7, 4
        
        plot_configs = [
            ('original', 'Original Image', False, 'viridis'),
            ('luminance_Y', 'Luminance Y', True, 'gray'),
            ('illum_features', 'Illumination Features (mean)', True, 'hot'),
            ('illum_raw', 'Raw Illumination (multi-scale)', True, 'hot'),
            
            ('illum_scale_0', 'Illum Scale 0 (global)', True, 'hot'),
            ('illum_scale_1', 'Illum Scale 1 (mid)', True, 'hot'),
            ('illum_scale_2', 'Illum Scale 2 (local)', True, 'hot'),
            ('sensitivity', 'Sensitivity Map S', True, 'RdYlGn'),
            
            ('illum_adaptive_before_clamp', 'L_adaptive (before clamp)', True, 'hot'),
            ('illum_after_clamp', 'L_adaptive (after clamp)', True, 'hot'),
            ('illum_final', 'L_final (smoothed)', True, 'hot'),
            ('L_correction_factor', 'Correction Factor 1/L', True, 'coolwarm'),
            
            ('reflectance_retinex', 'R_retinex (physics)', False, 'viridis'),
            ('color_risk', 'Color Risk Map', True, 'Reds'),
            ('reflectance_retinex_safe', 'R_retinex_safe', False, 'viridis'),
            ('saliency_map', 'Identity Saliency Map', True, 'hot'),
            
            ('enhance_residual', 'Enhance Residual', False, 'viridis'),
            ('gain_map', 'Saliency (as Gain)', True, 'RdYlGn'),
            ('reflectance_enhanced', 'R_enhanced', False, 'viridis'),
            ('reflectance_fused', 'R_fused (final reflectance)', False, 'viridis'),
            
            ('alpha_residual', 'Alpha Residual', True, 'Blues'),
            ('alpha_effective', 'Alpha Effective', True, 'Blues'),
            ('reflectance_before_refine', 'Before Refinement', False, 'viridis'),
            ('reflectance_after_refine', 'After Refinement', False, 'viridis'),
            
            ('final_output', 'Final Output', False, 'viridis'),
            ('original', 'Original (compare)', False, 'viridis'),
            ('difference', 'Difference |Out-In|', True, 'hot'),
            ('difference', 'Difference (amplified x3)', True, 'hot'),
        ]
        
        for idx, (key, label, is_gray, cmap) in enumerate(plot_configs):
            ax = fig.add_subplot(rows, cols, idx + 1)
            
            if key not in results:
                ax.text(0.5, 0.5, f'{key}\nNot Available', ha='center', va='center')
                ax.set_title(label)
                ax.axis('off')
                continue
            
            data = results[key]
            
            if key == 'enhance_residual':
                img = tensor_to_img(data, is_gray=False)
                img = 0.5 + img * 5
                img = np.clip(img, 0, 1)
                ax.imshow(img)
                ax.set_title(f'{label}\n(0.5 + res*5)', fontsize=9)
            elif is_gray:
                img = tensor_to_img(data, is_gray=True)
                
                if label == 'Difference (amplified x3)':
                    img = img * 3
                    img = np.clip(img, 0, 1)
                
                if cmap in ['hot', 'jet', 'coolwarm', 'Reds', 'Blues', 'RdYlGn']:
                    img_show = apply_colormap(img, cmap)
                else:
                    img_show = img
                    cmap = 'gray'
                
                im = ax.imshow(img_show)
                vmin, vmax = img.min(), img.max()
                ax.set_title(f'{label}\n[{vmin:.3f}, {vmax:.3f}]', fontsize=9)
            else:
                img = tensor_to_img(data, is_gray=False)
                ax.imshow(img)
                ax.set_title(label, fontsize=9)
            
            ax.axis('off')
        
        if 'scale_weights' in results:
            weights = results['scale_weights'].cpu().numpy()
            weight_text = f"Scale Weights: [{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]"
            fig.text(0.5, 0.02, weight_text, ha='center', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        info_parts = []
        if 'saliency_map' in results:
            sal = results['saliency_map'][0, 0].detach().cpu().numpy()
            info_parts.append(f"Saliency: [{sal.min():.3f}, {sal.max():.3f}], mean={sal.mean():.3f}")
        if 'enhance_residual' in results:
            res = results['enhance_residual'][0].detach().cpu().numpy()
            info_parts.append(f"Residual: [{res.min():.3f}, {res.max():.3f}]")
        if info_parts:
            fig.text(0.5, 0.01, " | ".join(info_parts), ha='center', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved debug panel: {output_path}")
    
    def create_comparison_strip(self, results: Dict[str, torch.Tensor], 
                                output_path: str):
        """"""
        
        def tensor_to_img(t: torch.Tensor) -> np.ndarray:
            t = t[0].detach().cpu().clamp(0, 1)
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        key_stages = [
            ('original', 'Input'),
            ('reflectance_retinex', 'R_retinex'),
            ('saliency_map', 'Identity Saliency'),
            ('enhance_residual', 'Enhance Residual'),
            ('reflectance_enhanced', 'R_enhanced'),
            ('final_output', 'Output'),
        ]
        
        images = []
        for key, label in key_stages:
            if key in results:
                if key == 'saliency_map':
                    t = results[key][0, 0].detach().cpu().numpy()
                    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-8)
                    cmap = plt.get_cmap('hot')
                    img = (cmap(t_norm)[:, :, :3] * 255).astype(np.uint8)
                elif key == 'enhance_residual':
                    t = results[key][0].detach().cpu()
                    t = 0.5 + t * 5
                    t = t.clamp(0, 1)
                    if t.shape[0] == 1:
                        t = t.repeat(3, 1, 1)
                    img = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                else:
                    img = tensor_to_img(results[key])
                
                img_labeled = cv2.copyMakeBorder(img, 30, 0, 0, 0, 
                                                  cv2.BORDER_CONSTANT, value=(255,255,255))
                cv2.putText(img_labeled, label, (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                images.append(img_labeled)
        
        if images:
            strip = np.concatenate(images, axis=1)
            cv2.imwrite(output_path, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
            print(f"Saved comparison strip: {output_path}")
    
    def analyze_statistics(self, results: Dict[str, torch.Tensor]) -> str:
        """Build a short textual summary of the collected stage statistics."""
        lines = ["=" * 60, "Stage Statistics Analysis (v3 - identity-aware enhancement)", "=" * 60]
        
        key_stats = [
            ('luminance_Y', 'Luminance Y'),
            ('illum_raw', 'Raw illumination L'),
            ('sensitivity', 'Sensitivity S'),
            ('illum_final', 'Final illumination L_final'),
            ('L_correction_factor', 'Correction factor 1/L'),
            ('reflectance_retinex', 'R_retinex'),
            ('color_risk', 'Color risk'),
            ('saliency_map', 'Identity saliency'),
            ('enhance_residual', 'Enhancement residual'),
            ('reflectance_enhanced', 'R_enhanced'),
            ('alpha_effective', 'Effective alpha'),
            ('reflectance_fused', 'R_fused'),
            ('final_output', 'Final output'),
            ('difference', 'Input-output difference'),
        ]
        
        for key, name in key_stats:
            if key in results:
                t = results[key]
                lines.append(f"{name:20s}: min={t.min().item():.4f}, "
                           f"max={t.max().item():.4f}, "
                           f"mean={t.mean().item():.4f}, "
                           f"std={t.std().item():.4f}")
        
        lines.append("\n" + "=" * 60)
        lines.append("Potential issue checks")
        lines.append("=" * 60)
        
        issues = []
        
        if 'illum_final' in results:
            L = results['illum_final']
            if L.max() - L.min() < 0.1:
                issues.append("WARNING: illumination range is very small; the estimator may be under-trained.")
            if (L > 2.8).float().mean() > 0.3 or (L < 0.25).float().mean() > 0.3:
                issues.append("WARNING: illumination values frequently hit clamp boundaries; consider widening the range.")
        
        if 'sensitivity' in results:
            S = results['sensitivity']
            if S.mean() < 0.2:
                issues.append("WARNING: sensitivity is globally low; correction strength may be insufficient.")
            if S.mean() > 0.55:
                issues.append("WARNING: sensitivity is globally high; the model may over-correct.")
        
        if 'saliency_map' in results:
            sal = results['saliency_map']
            lines.append(f"OK: identity-saliency range [{sal.min().item():.3f}, {sal.max().item():.3f}], mean={sal.mean().item():.3f}")
            if sal.max() - sal.min() < 0.1:
                issues.append("WARNING: saliency variation is too small; identity-aware enhancement may not be active yet.")
        
        if 'enhance_residual' in results:
            res = results['enhance_residual']
            lines.append(f"OK: enhancement-residual range [{res.min().item():.3f}, {res.max().item():.3f}]")
        
        if 'color_risk' in results:
            risk = results['color_risk']
            if risk.mean() > 0.3:
                issues.append(f"WARNING: color risk is relatively high (mean={risk.mean().item():.3f}); Retinex may introduce color drift.")
        
        if 'difference' in results:
            diff = results['difference']
            if diff.mean() < 0.01:
                issues.append("WARNING: output is almost identical to input; the module effect is weak.")
            if diff.mean() > 0.3:
                issues.append("WARNING: output differs too much from input; the model may be over-processing.")
        
        if issues:
            for issue in issues:
                lines.append(issue)
        else:
            lines.append("OK: no obvious issues detected.")
        
        return "\n".join(lines)
    
    def process_single_image(self, image_path: str, output_dir: str):
        """Process and export debug artifacts for a single image."""
        os.makedirs(output_dir, exist_ok=True)
        
        x = self.load_image(image_path)
        
        results = self.debug_forward(x)
        
        base_name = Path(image_path).stem
        
        self.visualize_stages(
            results, 
            os.path.join(output_dir, f"{base_name}_stages.jpg"),
            title=image_path
        )
        
        self.create_comparison_strip(
            results,
            os.path.join(output_dir, f"{base_name}_strip.jpg")
        )
        
        stats = self.analyze_statistics(results)
        print(stats)
        
        with open(os.path.join(output_dir, f"{base_name}_stats.txt"), 'w', 
                  encoding='utf-8') as f:
            f.write(stats)
        
        return results
    
    def process_dataset(self, data_dir: str, output_dir: str, num_samples: int = 10):
        """Sample a dataset directory and export debug artifacts for multiple images."""
        os.makedirs(output_dir, exist_ok=True)
        
        image_paths = []
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, f))
        
        if len(image_paths) > num_samples:
            indices = np.linspace(0, len(image_paths) - 1, num_samples, dtype=int)
            image_paths = [image_paths[i] for i in indices]
        
        print(f"Processing {len(image_paths)} images...")
        
        all_stats = []
        for i, img_path in enumerate(image_paths):
            print(f"\n[{i+1}/{len(image_paths)}] {img_path}")
            try:
                results = self.process_single_image(img_path, output_dir)
                all_stats.append(self.analyze_statistics(results))
            except Exception as e:
                print(f"Processing failed: {e}")
        
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            for i, (path, stats) in enumerate(zip(image_paths, all_stats)):
                f.write(f"\n{'='*80}\n")
                f.write(f"Image {i+1}: {path}\n")
                f.write(stats)
                f.write("\n")
        
        print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Debug intermediate stages of the illumination module')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument('--image', type=str, default=None,
                        help='Single-image path')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs/debug_stages',
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of images to sample from the dataset')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.image is None and args.data_dir is None:
        parser.error("You must specify either --image or --data_dir.")
    
    debugger = IlluminationStageDebugger(args.checkpoint, args.device)
    
    if args.image:
        debugger.process_single_image(args.image, args.output_dir)
    else:
        debugger.process_dataset(args.data_dir, args.output_dir, args.num_samples)
    
    print("\nDebug run complete.")


if __name__ == "__main__":
    main()
