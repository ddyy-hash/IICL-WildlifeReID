#!/usr/bin/env python3
"""Visualize detail-preserving color correction for the trained IPAID model."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.core.evaluation import ReIDDataset
from app.core.joint_model import JointReIDModel


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            state_dict = checkpoint
        if isinstance(state_dict, dict):
            return state_dict
    raise ValueError("Checkpoint does not contain model_state_dict/state_dict.")


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _to_rgb_u8(img_chw: torch.Tensor) -> np.ndarray:
    arr = img_chw.detach().cpu().clamp(0.0, 1.0).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """RGB [0,1] -> HSV [H:0-360, S:0-1, V:0-1]"""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    maxc = torch.max(rgb, dim=1)[0]
    minc = torch.min(rgb, dim=1)[0]
    delta = maxc - minc

    # V
    v = maxc

    # S
    s = torch.where(maxc > 1e-8, delta / maxc, torch.zeros_like(maxc))

    # H
    h = torch.zeros_like(maxc)
    mask = delta > 1e-8

    r_mask = mask & (maxc == r)
    g_mask = mask & (maxc == g)
    b_mask = mask & (maxc == b)

    h[r_mask] = 60.0 * (((g - b) / delta) % 6)[r_mask]
    h[g_mask] = 60.0 * (((b - r) / delta) + 2)[g_mask]
    h[b_mask] = 60.0 * (((r - g) / delta) + 4)[b_mask]

    h = h % 360.0

    return torch.stack([h, s, v], dim=1)


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """HSV -> RGB [0,1]"""
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    h = h / 60.0
    c = v * s
    x = c * (1 - torch.abs(h % 2 - 1))
    m = v - c

    h_i = h.long() % 6

    rgb = torch.zeros_like(hsv)

    for i in range(6):
        mask = (h_i == i)
        if i == 0:
            rgb[:, 0][mask] = c[mask]
            rgb[:, 1][mask] = x[mask]
        elif i == 1:
            rgb[:, 0][mask] = x[mask]
            rgb[:, 1][mask] = c[mask]
        elif i == 2:
            rgb[:, 1][mask] = c[mask]
            rgb[:, 2][mask] = x[mask]
        elif i == 3:
            rgb[:, 1][mask] = x[mask]
            rgb[:, 2][mask] = c[mask]
        elif i == 4:
            rgb[:, 0][mask] = x[mask]
            rgb[:, 2][mask] = c[mask]
        elif i == 5:
            rgb[:, 0][mask] = c[mask]
            rgb[:, 2][mask] = x[mask]

    rgb = rgb + m.unsqueeze(1)
    return rgb.clamp(0, 1)


def auto_white_balance(img_corrected: torch.Tensor, img_original: torch.Tensor) -> torch.Tensor:
    hsv_corr = rgb_to_hsv(img_corrected)
    hsv_orig = rgb_to_hsv(img_original)

    hsv_balanced = hsv_corr.clone()
    hsv_balanced[:, 0] = hsv_orig[:, 0]
    hsv_balanced[:, 1] = hsv_orig[:, 1] * 0.85 + hsv_corr[:, 1] * 0.15

    return hsv_to_rgb(hsv_balanced)


def extract_high_freq(img: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    sigma = kernel_size / 6.0
    kernel = torch.zeros(1, 1, kernel_size, kernel_size, device=img.device)
    center = kernel_size // 2

    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            kernel[0, 0, i, j] = torch.exp(torch.tensor(-(x**2 + y**2) / (2 * sigma**2)))

    kernel = kernel / kernel.sum()

    low_freq = torch.cat([
        F.conv2d(img[:, i:i+1], kernel, padding=kernel_size//2)
        for i in range(img.shape[1])
    ], dim=1)

    high_freq = img - low_freq
    return high_freq


def detail_enhancement(img_corrected: torch.Tensor, img_original: torch.Tensor,
                       strength: float = 0.6) -> torch.Tensor:
    high_freq_orig = extract_high_freq(img_original)

    enhanced = img_corrected + strength * high_freq_orig

    return enhanced.clamp(0, 1)


def build_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    device: torch.device,
    fallback_backbone: str,
    fallback_num_classes: int,
) -> JointReIDModel:
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    illum_cfg_model = model_cfg.get("illumination_module", {}) if isinstance(model_cfg, dict) else {}
    illum_cfg_top = cfg.get("illumination_module", ) if isinstance(cfg, dict) else {}
    local_cfg = model_cfg.get("local_extractor", {}) if isinstance(model_cfg, dict) else {}

    num_classes = int(checkpoint.get("num_classes", fallback_num_classes))
    backbone = model_cfg.get("backbone", fallback_backbone)
    num_stripes = int(local_cfg.get("num_parts", 6))
    dropout = float(local_cfg.get("dropout", 0.0))

    if "enabled" in illum_cfg_model:
        use_ipaid = bool(illum_cfg_model.get("enabled", True))
    else:
        module_type = str(illum_cfg_top.get("module_type", "IPAIDModule")).lower()
        use_ipaid = module_type not in {"none", "disabled", "null"}

    ipaid_params = illum_cfg_model.get("module_params")
    if not ipaid_params:
        ipaid_params = illum_cfg_top.get("module_params", {})

    model = JointReIDModel(
        num_classes=num_classes,
        backbone_name=backbone,
        num_stripes=num_stripes,
        pretrained_backbone=False,
        soft_mask_temperature=10.0,
        soft_mask_type="sigmoid",
        use_ipaid=use_ipaid,
        dropout=dropout,
        ipaid_params=ipaid_params,
    ).to(device)

    state_dict = _extract_state_dict(checkpoint)
    load_ret = model.load_state_dict(state_dict, strict=False)
    missing = getattr(load_ret, "missing_keys", [])
    unexpected = getattr(load_ret, "unexpected_keys", [])
    if missing:
        print(f"[WARN] missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}")
    return model


@torch.no_grad()
def save_color_correction_comparison_v2(
    model: JointReIDModel,
    dataset: ReIDDataset,
    device: torch.device,
    output_path: str,
    num_examples: int,
    detail_strength: float = 0.6,
) -> None:
    if len(dataset) == 0:
        print("[WARN] Dataset is empty; skipping visualization.")
        return

    n = min(num_examples, len(dataset))
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.5 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    prev_flag = getattr(model, "use_ipaid", None)
    if prev_flag is not None:
        model.use_ipaid = True
    model.eval()

    for i in range(n):
        img_t, pid, _, path = dataset[i]
        inp = img_t.unsqueeze(0).to(device)

        original = _to_rgb_u8(img_t)

        out = model(inp, boxes_list=None, return_illuminated=True)
        illuminated_raw = out.get("illuminated", inp).squeeze(0)
        illuminated_raw_u8 = _to_rgb_u8(illuminated_raw)

        illuminated_awb = auto_white_balance(
            illuminated_raw.unsqueeze(0),
            img_t.unsqueeze(0).to(device)
        ).squeeze(0)
        illuminated_awb_u8 = _to_rgb_u8(illuminated_awb)

        illuminated_detail = detail_enhancement(
            illuminated_awb.unsqueeze(0),
            img_t.unsqueeze(0).to(device),
            strength=detail_strength
        ).squeeze(0)
        illuminated_detail_u8 = _to_rgb_u8(illuminated_detail)

        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f"Original (ID:{pid})", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(illuminated_raw_u8)
        axes[i, 1].set_title("IPAID Raw", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(illuminated_awb_u8)
        axes[i, 2].set_title("IPAID + AWB", fontsize=10)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(illuminated_detail_u8)
        axes[i, 3].set_title(f"IPAID + AWB + Detail (α={detail_strength})", fontsize=10)
        axes[i, 3].axis("off")

    if prev_flag is not None:
        model.use_ipaid = prev_flag

    fig.suptitle("Color Correction with Detail Preservation", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize detail-preserving color correction")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint path")
    parser.add_argument("--input_dir", type=str, required=True, help="input image directory")
    parser.add_argument("--output_dir", type=str, default="outputs/color_correction_v2", help="output directory")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument("--backbone", type=str, default="osnet_ain_x1_0", help="fallback backbone")
    parser.add_argument("--num_classes", type=int, default=107, help="fallback num classes")
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_width", type=int, default=512)
    parser.add_argument("--num_examples", type=int, default=8, help="number of examples to visualize")
    parser.add_argument("--detail_strength", type=float, default=0.6, help="detail enhancement strength (0-1)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"input_dir not found: {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    print("[1/3] Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        checkpoint = {"state_dict": checkpoint}
    model = build_model_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
        fallback_backbone=args.backbone,
        fallback_num_classes=args.num_classes,
    )

    print("[2/3] Loading dataset...")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
    ])
    dataset = ReIDDataset(root=args.input_dir, transform=transform)
    print(f"  Found {len(dataset)} images")

    print("[3/3] Generating comparison visualization...")
    output_path = os.path.join(args.output_dir, "color_correction_comparison_v2.png")
    save_color_correction_comparison_v2(
        model=model,
        dataset=dataset,
        device=device,
        output_path=output_path,
        num_examples=args.num_examples,
        detail_strength=args.detail_strength,
    )

    print("\nDone!")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
