#!/usr/bin/env python3
"""Visualize illumination-normalization behavior for a trained JointReIDModel.

The script exports original-vs-normalized comparisons, optional IPAID
decomposition views, synthetic lighting-condition boards, and a few compact
consistency statistics over a directory of identities.
"""

import os
import sys
import argparse
from typing import List, Dict, Optional

import numpy as np
import torch
from torchvision import transforms
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.core.joint_model import JointReIDModel, SUPPORTED_BACKBONES, get_backbone_dim


def build_transform() -> transforms.Compose:
    """Match the evaluation transform: resize and convert to tensor in [0, 1]."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])


def tensor_to_uint8_img(t: torch.Tensor) -> np.ndarray:
    """Convert a [C, H, W] tensor in [0, 1] to an RGB uint8 image."""
    t = t.detach().cpu().clamp(0.0, 1.0)
    arr = t.numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))  # C,H,W -> H,W,C
    arr = (arr * 255.0).round().astype(np.uint8)
    return arr


def gray_to_heatmap(gray: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Convert a grayscale map into an RGB heatmap."""
    if gray.dtype != np.uint8:
        gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
        gray = (gray_norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(gray, colormap)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def visualize_ipaid_details(
    orig_img: np.ndarray,
    illum_img: np.ndarray,
    ipaid_details: Optional[Dict[str, torch.Tensor]],
    output_path: str,
) -> None:
    """Render a compact IPAID decomposition panel for a single sample."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('IPAID Illumination Module Decomposition', fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(illum_img)
    axes[0, 1].set_title('After IPAID Normalization')
    axes[0, 1].axis('off')
    
    diff = np.abs(orig_img.astype(np.float32) - illum_img.astype(np.float32))
    diff_norm = (diff / (diff.max() + 1e-8) * 255).astype(np.uint8)
    axes[0, 2].imshow(diff_norm)
    axes[0, 2].set_title('Difference (|Orig - Norm|)')
    axes[0, 2].axis('off')
    
    diff_gray = cv2.cvtColor(diff_norm, cv2.COLOR_RGB2GRAY)
    diff_heatmap = gray_to_heatmap(diff_gray)
    axes[0, 3].imshow(diff_heatmap)
    axes[0, 3].set_title('Correction Intensity Heatmap')
    axes[0, 3].axis('off')
    
    if ipaid_details is not None:
        if 'Y' in ipaid_details:
            Y = ipaid_details['Y'][0, 0].detach().cpu().numpy()
            axes[1, 0].imshow(Y, cmap='gray')
            axes[1, 0].set_title(f'Luminance Y\n[{Y.min():.2f}, {Y.max():.2f}]')
        else:
            axes[1, 0].text(0.5, 0.5, 'Y not available', ha='center', va='center')
        axes[1, 0].axis('off')
        
        if 'L' in ipaid_details:
            L = ipaid_details['L'][0, 0].detach().cpu().numpy()
            axes[1, 1].imshow(L, cmap='hot')
            axes[1, 1].set_title(f'Illumination Map L\n[{L.min():.2f}, {L.max():.2f}]')
        else:
            axes[1, 1].text(0.5, 0.5, 'L not available', ha='center', va='center')
        axes[1, 1].axis('off')
        
        if 'R' in ipaid_details:
            R = ipaid_details['R'][0, 0].detach().cpu().numpy()
            axes[1, 2].imshow(R, cmap='gray')
            axes[1, 2].set_title(f'Reflectance R = Y/L\n[{R.min():.2f}, {R.max():.2f}]')
        else:
            axes[1, 2].text(0.5, 0.5, 'R not available', ha='center', va='center')
        axes[1, 2].axis('off')
        
        if 'alpha' in ipaid_details:
            alpha = ipaid_details['alpha'][0, 0].detach().cpu().numpy()
            im = axes[1, 3].imshow(alpha, cmap='RdYlGn', vmin=0, vmax=0.5)
            axes[1, 3].set_title(f'Residual Gate α\n[{alpha.min():.3f}, {alpha.max():.3f}]')
            plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)
        else:
            axes[1, 3].text(0.5, 0.5, 'α not available', ha='center', va='center')
        axes[1, 3].axis('off')
    else:
        for i in range(4):
            axes[1, i].text(0.5, 0.5, 'IPAID details\nnot available', 
                           ha='center', va='center', fontsize=12)
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_histogram(
    orig_img: np.ndarray,
    illum_img: np.ndarray,
    output_path: str,
) -> None:
    """"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Brightness Histogram Comparison', fontsize=12, fontweight='bold')
    
    def get_luminance(img):
        return 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    
    orig_lum = get_luminance(orig_img.astype(np.float32))
    illum_lum = get_luminance(illum_img.astype(np.float32))
    
    axes[0].hist(orig_lum.ravel(), bins=50, color='blue', alpha=0.7, density=True)
    axes[0].set_title(f'Original\nMean: {orig_lum.mean():.1f}, Std: {orig_lum.std():.1f}')
    axes[0].set_xlabel('Luminance')
    axes[0].set_ylabel('Density')
    axes[0].set_xlim([0, 255])
    
    axes[1].hist(illum_lum.ravel(), bins=50, color='green', alpha=0.7, density=True)
    axes[1].set_title(f'After IPAID\nMean: {illum_lum.mean():.1f}, Std: {illum_lum.std():.1f}')
    axes[1].set_xlabel('Luminance')
    axes[1].set_ylabel('Density')
    axes[1].set_xlim([0, 255])
    
    axes[2].hist(orig_lum.ravel(), bins=50, color='blue', alpha=0.5, density=True, label='Original')
    axes[2].hist(illum_lum.ravel(), bins=50, color='green', alpha=0.5, density=True, label='IPAID')
    axes[2].set_title('Comparison')
    axes[2].set_xlabel('Luminance')
    axes[2].set_ylabel('Density')
    axes[2].set_xlim([0, 255])
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def compute_brightness_mean(img: np.ndarray) -> float:
    """Compute mean image brightness from RGB input."""
    if img.ndim == 3 and img.shape[2] == 3:
        r = img[..., 0].astype(np.float32)
        g = img[..., 1].astype(np.float32)
        b = img[..., 2].astype(np.float32)
        y = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        y = img.astype(np.float32)
    return float(y.mean())


def simulate_lighting_conditions(img_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Synthesize a small set of simple lighting variants for one image tensor."""
    results = {"original": img_tensor}
    
    dark = (img_tensor * 0.4).clamp(0, 1)
    results["dark"] = dark
    
    bright = (img_tensor * 1.6).clamp(0, 1)
    results["bright"] = bright
    
    low_contrast = img_tensor * 0.5 + 0.25
    results["low_contrast"] = low_contrast.clamp(0, 1)
    
    warm = img_tensor.clone()
    warm[0] = (warm[0] * 1.2).clamp(0, 1)  # R
    warm[2] = (warm[2] * 0.8).clamp(0, 1)  # B
    results["warm"] = warm
    
    cold = img_tensor.clone()
    cold[0] = (cold[0] * 0.85).clamp(0, 1)  # R
    cold[2] = (cold[2] * 1.15).clamp(0, 1)  # B
    results["cold"] = cold
    
    return results


def add_text_to_image(img: np.ndarray, text: str, position: str = "top") -> np.ndarray:
    """Overlay a short label on an RGB image."""
    img = img.copy()
    h, w = img.shape[:2]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x = (w - text_w) // 2
    if position == "top":
        y = text_h + 5
    else:
        y = h - 5
    
    cv2.rectangle(img, (x - 2, y - text_h - 2), (x + text_w + 2, y + baseline + 2), (0, 0, 0), -1)
    
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    
    return img


@torch.no_grad()
def visualize_illumination(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str,
    device: torch.device,
    max_ids: int = 0,
    max_imgs_per_id: int = 0,
    show_details: bool = False,
    backbone: str = "osnet_ain_x1_0",
    num_stripes: int = 0,
):
    """Run visualization and summary-statistic export for a directory of identities."""

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    os.makedirs(output_dir, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = checkpoint.get("num_classes", 100)
    
    saved_backbone = checkpoint.get("backbone", backbone)
    print(f"[INFO] Using backbone: {saved_backbone}")
    
    if num_stripes > 0:
        saved_num_stripes = num_stripes
    else:
        saved_config = checkpoint.get("config", {})
        local_cfg = saved_config.get("model", {}).get("local_extractor", {})
        saved_num_stripes = local_cfg.get("num_parts", 6)
        if saved_num_stripes == 6:
            train_cfg = saved_config.get("training", {})
            saved_num_stripes = train_cfg.get("num_stripes", 6)
    print(f"[INFO] Using num_stripes: {saved_num_stripes}")
    
    saved_config = checkpoint.get("config", {})
    model_cfg = saved_config.get("model", {})
    illum_cfg = model_cfg.get("illumination_module", {})
    use_ipaid = illum_cfg.get("enabled", True)
    print(f"[INFO] IPAID module: {'enabled' if use_ipaid else 'disabled'}")

    model = JointReIDModel(
        num_classes=num_classes,
        backbone_name=saved_backbone,
        num_stripes=saved_num_stripes,
        pretrained_backbone=False,
        soft_mask_temperature=10.0,
        soft_mask_type="sigmoid",
        use_ipaid=use_ipaid,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print("[INFO] Model loaded successfully.")
    if show_details:
        print("[INFO] Detailed IPAID decomposition views and histograms will be exported.")

    transform = build_transform()

    id_dirs: List[str] = [
        d for d in sorted(os.listdir(data_dir))
        if os.path.isdir(os.path.join(data_dir, d))
    ]

    if max_ids > 0:
        id_dirs = id_dirs[:max_ids]

    print(f"[INFO] Found {len(id_dirs)} identities for visualization and statistics.")

    exts = (".jpg", ".jpeg", ".png", ".bmp")

    brightness_stats: Dict[str, Dict[str, List[float]]] = {}
    feature_stats: Dict[str, List[torch.Tensor]] = {}

    for idx, identity in enumerate(id_dirs, start=1):
        id_path = os.path.join(data_dir, identity)
        img_files = [
            f for f in sorted(os.listdir(id_path))
            if any(f.lower().endswith(e) for e in exts)
        ]
        if not img_files:
            continue
        if max_imgs_per_id > 0:
            img_files = img_files[:max_imgs_per_id]

        brightness_stats[identity] = {"orig": [], "illum": []}
        feature_stats[identity] = []

        vis_saved = False
        out_path = ""

        for j, img_name in enumerate(img_files):
            img_path = os.path.join(id_path, img_name)

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                img_bgr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            img_tensor = transform(img_rgb)  # (3,H,W), [0,1]
            img_batch = img_tensor.unsqueeze(0).to(device)

            output = model(img_batch, boxes_list=None, return_illuminated=True)
            illuminated = output.get("illuminated", None)

            if illuminated is None:
                if j == 0:
                    print(f"[WARN] Model did not return an illuminated image; skipping ID {identity}.")
                continue

            orig_img = tensor_to_uint8_img(img_tensor)
            illum_img = tensor_to_uint8_img(illuminated[0])

            b_orig = compute_brightness_mean(orig_img)
            b_illum = compute_brightness_mean(illum_img)
            brightness_stats[identity]["orig"].append(b_orig)
            brightness_stats[identity]["illum"].append(b_illum)

            feats = output.get("features", None)
            if feats is not None:
                feature_stats[identity].append(feats[0].detach().cpu())

            if not vis_saved and j == 0:
                h = min(orig_img.shape[0], illum_img.shape[0])
                orig_resized = cv2.resize(orig_img, (orig_img.shape[1], h))
                illum_resized = cv2.resize(illum_img, (illum_img.shape[1], h))
                
                orig_labeled = add_text_to_image(orig_resized, "Original")
                illum_labeled = add_text_to_image(illum_resized, "Illumination Normalized")
                
                vis_rgb = np.concatenate([orig_labeled, illum_labeled], axis=1)

                vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
                out_name = f"{identity}_compare.jpg"
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, vis_bgr)
                
                lighting_conditions = simulate_lighting_conditions(img_tensor)
                multi_row_orig = []
                multi_row_illum = []
                
                for cond_name, cond_tensor in lighting_conditions.items():
                    cond_batch = cond_tensor.unsqueeze(0).to(device)
                    cond_output = model(cond_batch, boxes_list=None, return_illuminated=True)
                    cond_illum = cond_output.get("illuminated", None)
                    
                    if cond_illum is not None:
                        cond_orig_img = tensor_to_uint8_img(cond_tensor)
                        cond_illum_img = tensor_to_uint8_img(cond_illum[0])
                        
                        cond_orig_labeled = add_text_to_image(cond_orig_img, cond_name)
                        cond_illum_labeled = add_text_to_image(cond_illum_img, f"{cond_name} -> norm")
                        
                        multi_row_orig.append(cond_orig_labeled)
                        multi_row_illum.append(cond_illum_labeled)
                
                if multi_row_orig and multi_row_illum:
                    row1 = np.concatenate(multi_row_orig, axis=1)
                    row2 = np.concatenate(multi_row_illum, axis=1)
                    multi_vis = np.concatenate([row1, row2], axis=0)
                    
                    multi_bgr = cv2.cvtColor(multi_vis, cv2.COLOR_RGB2BGR)
                    multi_out_path = os.path.join(output_dir, f"{identity}_multi.jpg")
                    cv2.imwrite(multi_out_path, multi_bgr)
                
                if show_details:
                    ipaid_details = output.get("ipaid_details", None)
                    details_path = os.path.join(output_dir, f"{identity}_details.jpg")
                    visualize_ipaid_details(orig_img, illum_img, ipaid_details, details_path)
                    
                    hist_path = os.path.join(output_dir, f"{identity}_histogram.jpg")
                    visualize_histogram(orig_img, illum_img, hist_path)
                
                vis_saved = True

        if idx % 50 == 0 or idx == len(id_dirs):
            msg = f"[INFO] Processed {idx}/{len(id_dirs)} identities"
            if vis_saved:
                msg += f"; latest saved comparison: {out_path}"
            print(msg)

    print(f"[INFO] Visualization complete. Outputs saved under: {output_dir}")

    print("[INFO] Building the summary grid...")
    grid_images = []
    max_grid = min(16, len(id_dirs))
    
    for identity in id_dirs[:max_grid]:
        compare_path = os.path.join(output_dir, f"{identity}_compare.jpg")
        if os.path.exists(compare_path):
            img = cv2.imread(compare_path)
            if img is not None:
                img_resized = cv2.resize(img, (400, 200))
                grid_images.append(img_resized)
    
    if grid_images:
        cols = 4
        rows = (len(grid_images) + cols - 1) // cols
        
        while len(grid_images) < rows * cols:
            grid_images.append(np.zeros_like(grid_images[0]))
        
        grid_rows = []
        for r in range(rows):
            row_imgs = grid_images[r * cols:(r + 1) * cols]
            grid_rows.append(np.concatenate(row_imgs, axis=1))
        
        summary_grid = np.concatenate(grid_rows, axis=0)
        summary_path = os.path.join(output_dir, "summary_grid.jpg")
        cv2.imwrite(summary_path, summary_grid)
        print(f"[INFO] Summary grid saved to: {summary_path}")

    id_brightness_var_orig = []
    id_brightness_var_illum = []
    id_intra_feat_dist = []

    for identity in id_dirs:
        b_orig_list = brightness_stats.get(identity, {}).get("orig", [])
        b_illum_list = brightness_stats.get(identity, {}).get("illum", [])

        if len(b_orig_list) >= 2 and len(b_illum_list) >= 2:
            var_orig = float(np.var(b_orig_list))
            var_illum = float(np.var(b_illum_list))
            id_brightness_var_orig.append(var_orig)
            id_brightness_var_illum.append(var_illum)

        feats_list = feature_stats.get(identity, [])
        if len(feats_list) >= 2:
            feats_tensor = torch.stack(feats_list, dim=0)  # (N, D)
            feats_tensor = torch.nn.functional.normalize(feats_tensor, p=2, dim=1)
            sim_mat = feats_tensor @ feats_tensor.t()  # (N,N)
            n = sim_mat.shape[0]
            iu = torch.triu_indices(n, n, offset=1)
            sims = sim_mat[iu[0], iu[1]]
            dists = 1.0 - sims
            id_intra_feat_dist.append(float(dists.mean().item()))

    if id_brightness_var_orig and id_brightness_var_illum:
        mean_var_orig = float(np.mean(id_brightness_var_orig))
        mean_var_illum = float(np.mean(id_brightness_var_illum))
        print("\n===== Brightness Consistency Statistics (per ID) =====")
        print(f"Original-image brightness variance (mean over IDs): {mean_var_orig:.4f}")
        print(f"Normalized-image brightness variance (mean over IDs): {mean_var_illum:.4f}")
        if mean_var_orig > 1e-6:
            print(f"Variance compression ratio: {mean_var_illum/mean_var_orig:.4f} (lower is better)")

    if id_intra_feat_dist:
        mean_intra_dist = float(np.mean(id_intra_feat_dist))
        print("\n===== Feature Stability Statistics (per ID) =====")
        print(f"Mean intra-ID cosine distance: {mean_intra_dist:.4f} (lower is better)")

    print("=================================\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize illumination-normalization behavior")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to joint_best.pth")
    parser.add_argument("--data_dir", type=str, required=True, help="Image root directory organized by identity")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for visualizations")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto / cpu / cuda")
    parser.add_argument("--max_ids", type=int, default=0, help="Maximum number of identities to visualize; 0 means all")
    parser.add_argument(
        "--max_imgs_per_id",
        type=int,
        default=0,
        help="Maximum number of source images per identity; 0 means all",
    )
    parser.add_argument(
        "--show_details",
        action="store_true",
        help="Also export detailed IPAID decomposition panels and histograms",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="osnet_ain_x1_0",
        choices=SUPPORTED_BACKBONES,
        help="Backbone type when it is not stored in the checkpoint",
    )
    parser.add_argument(
        "--num_stripes",
        type=int,
        default=0,
        help="Number of stripes; 0 reads it from the checkpoint config",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] Using device: {device}")

    visualize_illumination(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=device,
        max_ids=args.max_ids,
        max_imgs_per_id=args.max_imgs_per_id,
        show_details=args.show_details,
        backbone=args.backbone,
        num_stripes=args.num_stripes,
    )


if __name__ == "__main__":
    main()
