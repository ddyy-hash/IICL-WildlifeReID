#!/usr/bin/env python3
"""\
可视化光照归一效果脚本

功能：
    - 加载已训练好的 JointReIDModel checkpoint
    - 从指定数据目录中按 ID（子文件夹名）遍历，每个 ID 选取一张图像
    - 对图像做光照归一（通过模型的 illumination 模块），生成归一化后的图
    - 可视化 IPAID 模块详细信息：光照图 L、反射率 R、修正因子、残差门控等
    - 生成多种不同光照条件下的对比图（亮/暗/正常/偏色）
    - 生成亮度直方图对比

目录结构约定（与训练/评估一致）：
    data_dir/ID_1/img1.jpg
            /ID_1/img2.jpg
            /ID_2/img3.jpg
    ...

使用示例（ATRW query 集）：
    python tools/visualize_illumination_effect.py \
        --checkpoint checkpoints/joint_atrw_ipaid/joint_best.pth \
        --data_dir data/processed/atrw/query \
        --output_dir outputs/illum_vis_atrw \
        --device cuda \
        --max_ids 20 \
        --show_details

运行结果：
    outputs/illum_vis_atrw/0001_compare.jpg         # 原图 vs 光照归一图
    outputs/illum_vis_atrw/0001_details.jpg         # IPAID 详细分解图
    outputs/illum_vis_atrw/0001_histogram.jpg       # 亮度直方图对比
    outputs/illum_vis_atrw/0001_multi.jpg           # 多种光照条件对比
    outputs/illum_vis_atrw/summary_grid.jpg         # 总览大图
    ... 
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
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt

# 确保可以从项目根目录导入 app.core
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.core.joint_model import JointReIDModel, SUPPORTED_BACKBONES, get_backbone_dim


def build_transform() -> transforms.Compose:
    """与评估阶段保持一致：Resize + ToTensor（不做标准化，保持 [0,1]）"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])


def tensor_to_uint8_img(t: torch.Tensor) -> np.ndarray:
    """将 [C, H, W] 的张量转换为 RGB uint8 图像 (H, W, 3)"""
    t = t.detach().cpu().clamp(0.0, 1.0)
    arr = t.numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))  # C,H,W -> H,W,C
    arr = (arr * 255.0).round().astype(np.uint8)
    return arr


def gray_to_heatmap(gray: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """将灰度图转换为热力图"""
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
    """可视化 IPAID 模块的详细分解
    
    生成一个包含以下内容的大图：
    - 原图
    - 光照归一化后的图
    - 亮度通道 Y
    - 估计的光照图 L
    - 反射率 R = Y / L
    - 颜色比率保持
    - 残差门控权重 alpha
    - 修正差异图
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('IPAID Illumination Module Decomposition', fontsize=14, fontweight='bold')
    
    # 原图
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 光照归一化后
    axes[0, 1].imshow(illum_img)
    axes[0, 1].set_title('After IPAID Normalization')
    axes[0, 1].axis('off')
    
    # 差异图
    diff = np.abs(orig_img.astype(np.float32) - illum_img.astype(np.float32))
    diff_norm = (diff / (diff.max() + 1e-8) * 255).astype(np.uint8)
    axes[0, 2].imshow(diff_norm)
    axes[0, 2].set_title('Difference (|Orig - Norm|)')
    axes[0, 2].axis('off')
    
    # 差异热力图
    diff_gray = cv2.cvtColor(diff_norm, cv2.COLOR_RGB2GRAY)
    diff_heatmap = gray_to_heatmap(diff_gray)
    axes[0, 3].imshow(diff_heatmap)
    axes[0, 3].set_title('Correction Intensity Heatmap')
    axes[0, 3].axis('off')
    
    if ipaid_details is not None:
        # 亮度通道 Y
        if 'Y' in ipaid_details:
            Y = ipaid_details['Y'][0, 0].detach().cpu().numpy()
            axes[1, 0].imshow(Y, cmap='gray')
            axes[1, 0].set_title(f'Luminance Y\n[{Y.min():.2f}, {Y.max():.2f}]')
        else:
            axes[1, 0].text(0.5, 0.5, 'Y not available', ha='center', va='center')
        axes[1, 0].axis('off')
        
        # 光照图 L
        if 'L' in ipaid_details:
            L = ipaid_details['L'][0, 0].detach().cpu().numpy()
            axes[1, 1].imshow(L, cmap='hot')
            axes[1, 1].set_title(f'Illumination Map L\n[{L.min():.2f}, {L.max():.2f}]')
        else:
            axes[1, 1].text(0.5, 0.5, 'L not available', ha='center', va='center')
        axes[1, 1].axis('off')
        
        # 反射率 R
        if 'R' in ipaid_details:
            R = ipaid_details['R'][0, 0].detach().cpu().numpy()
            axes[1, 2].imshow(R, cmap='gray')
            axes[1, 2].set_title(f'Reflectance R = Y/L\n[{R.min():.2f}, {R.max():.2f}]')
        else:
            axes[1, 2].text(0.5, 0.5, 'R not available', ha='center', va='center')
        axes[1, 2].axis('off')
        
        # 残差门控权重 alpha
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
    """生成亮度直方图对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Brightness Histogram Comparison', fontsize=12, fontweight='bold')
    
    # 计算亮度
    def get_luminance(img):
        return 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    
    orig_lum = get_luminance(orig_img.astype(np.float32))
    illum_lum = get_luminance(illum_img.astype(np.float32))
    
    # 原图直方图
    axes[0].hist(orig_lum.ravel(), bins=50, color='blue', alpha=0.7, density=True)
    axes[0].set_title(f'Original\nMean: {orig_lum.mean():.1f}, Std: {orig_lum.std():.1f}')
    axes[0].set_xlabel('Luminance')
    axes[0].set_ylabel('Density')
    axes[0].set_xlim([0, 255])
    
    # 归一化后直方图
    axes[1].hist(illum_lum.ravel(), bins=50, color='green', alpha=0.7, density=True)
    axes[1].set_title(f'After IPAID\nMean: {illum_lum.mean():.1f}, Std: {illum_lum.std():.1f}')
    axes[1].set_xlabel('Luminance')
    axes[1].set_ylabel('Density')
    axes[1].set_xlim([0, 255])
    
    # 对比直方图
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
    """计算单张 RGB 图像的平均亮度（灰度均值）"""
    if img.ndim == 3 and img.shape[2] == 3:
        # 转灰度：Y = 0.299 R + 0.587 G + 0.114 B
        r = img[..., 0].astype(np.float32)
        g = img[..., 1].astype(np.float32)
        b = img[..., 2].astype(np.float32)
        y = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        y = img.astype(np.float32)
    return float(y.mean())


def simulate_lighting_conditions(img_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """模拟不同光照条件
    
    Args:
        img_tensor: (C, H, W) 的图像张量，范围 [0, 1]
    
    Returns:
        包含不同光照条件图像的字典
    """
    results = {"original": img_tensor}
    
    # 暗光条件（降低亮度）
    dark = (img_tensor * 0.4).clamp(0, 1)
    results["dark"] = dark
    
    # 过曝条件（提高亮度）
    bright = (img_tensor * 1.6).clamp(0, 1)
    results["bright"] = bright
    
    # 低对比度
    low_contrast = img_tensor * 0.5 + 0.25
    results["low_contrast"] = low_contrast.clamp(0, 1)
    
    # 偏色（模拟钨丝灯）
    warm = img_tensor.clone()
    warm[0] = (warm[0] * 1.2).clamp(0, 1)  # R
    warm[2] = (warm[2] * 0.8).clamp(0, 1)  # B
    results["warm"] = warm
    
    # 偏冷色（模拟阴天）
    cold = img_tensor.clone()
    cold[0] = (cold[0] * 0.85).clamp(0, 1)  # R
    cold[2] = (cold[2] * 1.15).clamp(0, 1)  # B
    results["cold"] = cold
    
    return results


def add_text_to_image(img: np.ndarray, text: str, position: str = "top") -> np.ndarray:
    """在图像上添加文字标签
    
    Args:
        img: RGB uint8 图像
        text: 要添加的文字
        position: "top" 或 "bottom"
    """
    img = img.copy()
    h, w = img.shape[:2]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    # 计算文字大小
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # 计算位置
    x = (w - text_w) // 2
    if position == "top":
        y = text_h + 5
    else:
        y = h - 5
    
    # 添加背景框
    cv2.rectangle(img, (x - 2, y - text_h - 2), (x + text_w + 2, y + baseline + 2), (0, 0, 0), -1)
    
    # 添加文字
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
    """为每个 ID 生成一张"原图 vs 光照归一图"的对比图

    同时统计：
        - 每个 ID 内原图亮度方差 vs 光照归一图亮度方差
        - 每个 ID 内特征的平均余弦距离（同 ID 稳定性）
    
    新增功能：
        - show_details=True 时生成 IPAID 详细分解图和直方图
        - 支持多 backbone
        - num_stripes=0 时自动从checkpoint读取
    """

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

    os.makedirs(output_dir, exist_ok=True)

    # 加载 checkpoint (兼容 PyTorch 2.6+)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = checkpoint.get("num_classes", 100)
    
    # 从 checkpoint 读取 backbone（如果有的话）
    saved_backbone = checkpoint.get("backbone", backbone)
    print(f"[INFO] 使用 backbone: {saved_backbone}")
    
    # 从 checkpoint config 读取 num_stripes（如果命令行未指定）
    if num_stripes > 0:
        # 命令行指定优先
        saved_num_stripes = num_stripes
    else:
        # 从 checkpoint config 读取
        saved_config = checkpoint.get("config", {})
        local_cfg = saved_config.get("model", {}).get("local_extractor", {})
        saved_num_stripes = local_cfg.get("num_parts", 6)  # 默认6
        # 也尝试从 training 配置读取
        if saved_num_stripes == 6:
            train_cfg = saved_config.get("training", {})
            saved_num_stripes = train_cfg.get("num_stripes", 6)
    print(f"[INFO] 使用 num_stripes: {saved_num_stripes}")
    
    # 从 checkpoint config 读取 use_ipaid（如果有的话）
    saved_config = checkpoint.get("config", {})
    model_cfg = saved_config.get("model", {})
    illum_cfg = model_cfg.get("illumination_module", {})
    use_ipaid = illum_cfg.get("enabled", True)
    print(f"[INFO] IPAID 模块: {'启用' if use_ipaid else '禁用'}")

    # 初始化模型（支持多 backbone）
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
    
    print(f"[INFO] 模型加载成功")
    if show_details:
        print(f"[INFO] 将生成 IPAID 详细分解图和直方图")

    transform = build_transform()

    # 收集所有 ID（子目录）
    id_dirs: List[str] = [
        d for d in sorted(os.listdir(data_dir))
        if os.path.isdir(os.path.join(data_dir, d))
    ]

    if max_ids > 0:
        id_dirs = id_dirs[:max_ids]

    print(f"[INFO] 共发现 {len(id_dirs)} 个身份，将为每个身份生成一张对比图并做统计分析")

    exts = (".jpg", ".jpeg", ".png", ".bmp")

    # 统计容器
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

            # 读取原图（BGR -> RGB）
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                # 容错：读图失败时使用噪声图
                img_bgr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # 变换到 [0,1] 张量
            img_tensor = transform(img_rgb)  # (3,H,W), [0,1]
            img_batch = img_tensor.unsqueeze(0).to(device)

            # 通过模型做光照归一
            # 不关心 YOLO 软掩码，这里只看 illumination 效果，boxes_list=None 即可
            output = model(img_batch, boxes_list=None, return_illuminated=True)
            illuminated = output.get("illuminated", None)

            if illuminated is None:
                # 理论上不会发生，如果发生了就直接跳过
                if j == 0:
                    print(f"[WARN] 模型未返回 illuminated 图像，跳过 ID {identity}")
                continue

            # 转回可视化用的 RGB uint8 图像
            orig_img = tensor_to_uint8_img(img_tensor)
            illum_img = tensor_to_uint8_img(illuminated[0])

            # 亮度统计
            b_orig = compute_brightness_mean(orig_img)
            b_illum = compute_brightness_mean(illum_img)
            brightness_stats[identity]["orig"].append(b_orig)
            brightness_stats[identity]["illum"].append(b_illum)

            # 特征统计（使用模型输出特征）
            feats = output.get("features", None)
            if feats is not None:
                feature_stats[identity].append(feats[0].detach().cpu())

            # 仅用该 ID 的第一张图生成可视化对比
            if not vis_saved and j == 0:
                # 横向拼接：左边原图，右边光照归一图
                h = min(orig_img.shape[0], illum_img.shape[0])
                orig_resized = cv2.resize(orig_img, (orig_img.shape[1], h))
                illum_resized = cv2.resize(illum_img, (illum_img.shape[1], h))
                
                # 添加标签
                orig_labeled = add_text_to_image(orig_resized, "Original")
                illum_labeled = add_text_to_image(illum_resized, "Illumination Normalized")
                
                vis_rgb = np.concatenate([orig_labeled, illum_labeled], axis=1)

                # 保存为 BGR
                vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
                out_name = f"{identity}_compare.jpg"
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, vis_bgr)
                
                # ===== 生成多光照条件对比图 =====
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
                        
                        # 添加标签
                        cond_orig_labeled = add_text_to_image(cond_orig_img, cond_name)
                        cond_illum_labeled = add_text_to_image(cond_illum_img, f"{cond_name} -> norm")
                        
                        multi_row_orig.append(cond_orig_labeled)
                        multi_row_illum.append(cond_illum_labeled)
                
                if multi_row_orig and multi_row_illum:
                    # 拼接多光照对比图
                    row1 = np.concatenate(multi_row_orig, axis=1)
                    row2 = np.concatenate(multi_row_illum, axis=1)
                    multi_vis = np.concatenate([row1, row2], axis=0)
                    
                    multi_bgr = cv2.cvtColor(multi_vis, cv2.COLOR_RGB2BGR)
                    multi_out_path = os.path.join(output_dir, f"{identity}_multi.jpg")
                    cv2.imwrite(multi_out_path, multi_bgr)
                
                # ===== 生成 IPAID 详细分解图（如果启用）=====
                if show_details:
                    ipaid_details = output.get("ipaid_details", None)
                    details_path = os.path.join(output_dir, f"{identity}_details.jpg")
                    visualize_ipaid_details(orig_img, illum_img, ipaid_details, details_path)
                    
                    # 生成直方图对比
                    hist_path = os.path.join(output_dir, f"{identity}_histogram.jpg")
                    visualize_histogram(orig_img, illum_img, hist_path)
                
                vis_saved = True

        if idx % 50 == 0 or idx == len(id_dirs):
            msg = f"[INFO] 已处理 {idx}/{len(id_dirs)} 个身份"
            if vis_saved:
                msg += f"，最近保存对比图: {out_path}"
            print(msg)

    print(f"[INFO] 可视化完成，对比图保存在: {output_dir}")

    # ===== 生成总览大图（Grid） =====
    print("[INFO] 生成总览大图...")
    grid_images = []
    max_grid = min(16, len(id_dirs))  # 最多显示 16 个
    
    for identity in id_dirs[:max_grid]:
        compare_path = os.path.join(output_dir, f"{identity}_compare.jpg")
        if os.path.exists(compare_path):
            img = cv2.imread(compare_path)
            if img is not None:
                # 缩小到合适大小
                img_resized = cv2.resize(img, (400, 200))
                grid_images.append(img_resized)
    
    if grid_images:
        # 排列成 4xN 的网格
        cols = 4
        rows = (len(grid_images) + cols - 1) // cols
        
        # 填充到完整网格
        while len(grid_images) < rows * cols:
            grid_images.append(np.zeros_like(grid_images[0]))
        
        grid_rows = []
        for r in range(rows):
            row_imgs = grid_images[r * cols:(r + 1) * cols]
            grid_rows.append(np.concatenate(row_imgs, axis=1))
        
        summary_grid = np.concatenate(grid_rows, axis=0)
        summary_path = os.path.join(output_dir, "summary_grid.jpg")
        cv2.imwrite(summary_path, summary_grid)
        print(f"[INFO] 总览大图已保存: {summary_path}")

    # ==================== 统计分析 ====================
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
            # L2 归一化
            feats_tensor = torch.nn.functional.normalize(feats_tensor, p=2, dim=1)
            # 余弦相似度矩阵
            sim_mat = feats_tensor @ feats_tensor.t()  # (N,N)
            # 只取上三角（不含对角线）
            n = sim_mat.shape[0]
            iu = torch.triu_indices(n, n, offset=1)
            sims = sim_mat[iu[0], iu[1]]
            # 余弦距离 = 1 - 相似度
            dists = 1.0 - sims
            id_intra_feat_dist.append(float(dists.mean().item()))

    if id_brightness_var_orig and id_brightness_var_illum:
        mean_var_orig = float(np.mean(id_brightness_var_orig))
        mean_var_illum = float(np.mean(id_brightness_var_illum))
        print("\n===== 光照一致性统计（按 ID） =====")
        print(f"原图亮度方差（ID 内平均）: {mean_var_orig:.4f}")
        print(f"归一图亮度方差（ID 内平均）: {mean_var_illum:.4f}")
        if mean_var_orig > 1e-6:
            print(f"方差压缩比例: {mean_var_illum/mean_var_orig:.4f} (越小越好)")

    if id_intra_feat_dist:
        mean_intra_dist = float(np.mean(id_intra_feat_dist))
        print("\n===== 特征稳定性统计（按 ID） =====")
        print(f"同 ID 特征平均余弦距离: {mean_intra_dist:.4f} (越小越好)")

    print("=================================\n")


def main():
    parser = argparse.ArgumentParser(description="可视化光照归一效果（增强版）")
    parser.add_argument("--checkpoint", type=str, required=True, help="joint_best.pth 的路径")
    parser.add_argument("--data_dir", type=str, required=True, help="按 ID 组织的图像根目录")
    parser.add_argument("--output_dir", type=str, required=True, help="对比图输出目录")
    parser.add_argument("--device", type=str, default="auto", help="设备: auto / cpu / cuda")
    parser.add_argument("--max_ids", type=int, default=0, help="最多可视化多少个 ID，0 表示全部")
    parser.add_argument(
        "--max_imgs_per_id",
        type=int,
        default=0,
        help="每个 ID 最多使用多少张原图做统计与可视化，0 表示全部",
    )
    parser.add_argument(
        "--show_details",
        action="store_true",
        help="生成 IPAID 详细分解图（光照图L、反射率R、残差门控α等）和直方图",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="osnet_ain_x1_0",
        choices=SUPPORTED_BACKBONES,
        help="骨干网络类型（如果 checkpoint 中未保存）",
    )
    parser.add_argument(
        "--num_stripes",
        type=int,
        default=0,
        help="条纹数量（0表示自动从checkpoint读取，Duke行人用4，老虎用6）",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] 使用设备: {device}")

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

