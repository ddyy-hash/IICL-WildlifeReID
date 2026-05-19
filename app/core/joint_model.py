#!/usr/bin/env python3
"""Joint ReID model with IPAID correction and stripe-based local feature extraction."""

from contextlib import nullcontext
import math
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchreid
from torch.utils.checkpoint import checkpoint

from .illumination_module_v2 import IPAIDModule, DataDrivenVariantGenerator


# ============================================================================
#                           Soft Mask Helpers
# ============================================================================

class SoftMaskGenerator(nn.Module):
    """Generate a differentiable box-aware mask on the backbone feature map.

    The mask is defined by a signed distance field around each box and then
    squashed by a sigmoid. Pixels inside the box approach 1, pixels outside
    decay smoothly toward 0, and border pixels stay near 0.5.
    """

    def __init__(self, temperature: float = 10.0, margin: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        boxes: torch.Tensor,
        feature_size: Tuple[int, int],
        image_size: Tuple[int, int],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        h, w = feature_size
        H, W = image_size
        device = device or (boxes.device if boxes is not None and boxes.numel() > 0 else torch.device("cpu"))

        # Build a normalized sampling grid in [0, 1].
        y_coords = torch.linspace(0, 1, h, device=device)
        x_coords = torch.linspace(0, 1, w, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        if boxes is None or boxes.numel() == 0:
            # When no boxes are provided, fall back to an all-ones mask.
            return torch.ones(1, h, w, device=device)

        combined_mask = torch.zeros(h, w, device=device)

        # Normalize box coordinates to [0, 1].
        boxes_norm = boxes.clone().float()
        boxes_norm[:, [0, 2]] /= W  # x
        boxes_norm[:, [1, 3]] /= H  # y

        for box in boxes_norm:
            x1, y1, x2, y2 = box

            # Expand the box slightly to avoid overly sharp boundaries.
            bw, bh = x2 - x1, y2 - y1
            x1 = torch.clamp(x1 - self.margin * bw, 0, 1)
            y1 = torch.clamp(y1 - self.margin * bh, 0, 1)
            x2 = torch.clamp(x2 + self.margin * bw, 0, 1)
            y2 = torch.clamp(y2 + self.margin * bh, 0, 1)

            # Distance to the expanded box boundary; zero inside the box.
            dx = torch.maximum(x1 - xx, torch.zeros_like(xx))
            dx = torch.maximum(dx, xx - x2)

            dy = torch.maximum(y1 - yy, torch.zeros_like(yy))
            dy = torch.maximum(dy, yy - y2)

            distance = torch.sqrt(dx**2 + dy**2 + 1e-8)

            inside_x = (xx >= x1) & (xx <= x2)
            inside_y = (yy >= y1) & (yy <= y2)
            inside = inside_x & inside_y

            # Use a negative signed distance inside the box so the sigmoid
            # produces values close to one in the interior.
            signed_distance = torch.where(inside, -0.1 * torch.ones_like(distance), distance)

            box_mask = torch.sigmoid(-signed_distance * self.temperature)
            combined_mask = torch.maximum(combined_mask, box_mask)

        return combined_mask.unsqueeze(0)  # (1, h, w)


class SoftMaskGaussian(nn.Module):
    """Gaussian soft-mask generator centered on the detected box."""

    def __init__(self, sigma_ratio: float = 0.4) -> None:
        super().__init__()
        self.sigma_ratio = sigma_ratio

    def forward(
        self,
        boxes: torch.Tensor,
        feature_size: Tuple[int, int],
        image_size: Tuple[int, int],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        h, w = feature_size
        H, W = image_size
        device = device or (boxes.device if boxes is not None and boxes.numel() > 0 else torch.device("cpu"))

        y_coords = torch.linspace(0, H, h, device=device)
        x_coords = torch.linspace(0, W, w, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        if boxes is None or boxes.numel() == 0:
            return torch.ones(1, h, w, device=device)

        combined_mask = torch.zeros(h, w, device=device)

        for box in boxes:
            x1, y1, x2, y2 = box.float()

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = (x2 - x1), (y2 - y1)

            sigma_x = bw * self.sigma_ratio + 1e-8
            sigma_y = bh * self.sigma_ratio + 1e-8

            gaussian = torch.exp(
                -((xx - cx) ** 2 / (2 * sigma_x**2) + (yy - cy) ** 2 / (2 * sigma_y**2))
            )

            combined_mask = torch.maximum(combined_mask, gaussian)

        return combined_mask.unsqueeze(0)


# ============================================================================
#                           Local Feature Extractor
# ============================================================================

class LocalFeatureExtractor(nn.Module):
    """Extract stripe-level and global features, then fuse them for ReID.

    This module applies part-specific 1x1 convolutions, a global branch,
    attention-based weighting, and an optional dropout layer before BNNeck.
    """

    def __init__(
        self,
        backbone_dim: int = 512,
        hidden_dim: int = 256,
        num_stripes: int = 6,
        num_classes: int = 100,
        use_deformable_stripes: bool = False,
        max_offset_ratio: float = 0.2,
        dropout: float = 0.0,  # Optional dropout before BNNeck.
    ) -> None:
        super().__init__()
        self.num_stripes = num_stripes
        self.backbone_dim = backbone_dim
        self.hidden_dim = hidden_dim
        self.use_deformable_stripes = use_deformable_stripes
        self.max_offset_ratio = max_offset_ratio
        self.dropout_rate = dropout

        # Per-stripe projection layers.
        self.stripe_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(backbone_dim, hidden_dim, 1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_stripes)
            ]
        )

        # Global feature projection branch.
        self.global_conv = nn.Sequential(
            nn.Conv2d(backbone_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Attention weights across stripe branches plus the global branch.
        total_parts = num_stripes + 1  # Stripe branches + one global branch.
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * total_parts, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, total_parts),
            nn.Softmax(dim=1),
        )

        # Final fusion block.
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * total_parts, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Optional dropout added for stronger regularization.
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Standard BNNeck used by many ReID models.
        self.bn_neck = nn.BatchNorm1d(hidden_dim)
        self.bn_neck.bias.requires_grad_(False)

        # Linear classifier for identity supervision.
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=False)

        # Learn stripe offsets when deformable stripes are enabled.
        if self.use_deformable_stripes:
            # Each offset is later squashed by tanh and scaled by
            # max_offset_ratio, so a zero initialization is safe.
            self.stripe_offsets = nn.Parameter(torch.zeros(num_stripes))

    def forward(
        self,
        feature_map: torch.Tensor,
        return_parts: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract local stripe features and the global pooled feature.

        When ``return_parts`` is enabled, the method also returns the stripe
        descriptors used before attention fusion. This keeps the training API
        backward-compatible while exposing pattern-aware local descriptors for
        retrieval-time global-local matching.
        """
        B, C, H, W = feature_map.shape

        part_features = []

        # Extract stripe features.
        for i in range(self.num_stripes):
            stripe_h = H // self.num_stripes

            if self.use_deformable_stripes:
                # Shift the stripe center while keeping the stripe size fixed.
                base_center = (i + 0.5) * stripe_h
                offset = torch.tanh(self.stripe_offsets[i]) * self.max_offset_ratio * H
                center = base_center + offset

                start_h = center - stripe_h / 2.0
                end_h = center + stripe_h / 2.0

                # Clamp the sampled range to valid feature-map coordinates.
                start_h = int(torch.clamp(start_h, 0, max(H - 1, 0)).item())
                end_h = int(torch.clamp(end_h, start_h + 1, H).item())
            else:
                start_h = i * stripe_h
                end_h = (i + 1) * stripe_h if i < self.num_stripes - 1 else H

            stripe = feature_map[:, :, start_h:end_h, :]

            stripe_feat = self.stripe_convs[i](stripe)
            stripe_feat = F.adaptive_avg_pool2d(stripe_feat, 1).flatten(1)
            part_features.append(stripe_feat)

        # Extract the global pooled feature.
        global_feat = self.global_conv(feature_map)
        global_feat = F.adaptive_avg_pool2d(global_feat, 1).flatten(1)
        part_features.append(global_feat)

        # Concatenate all part descriptors.
        concat_features = torch.cat(part_features, dim=1)

        # Predict attention weights across parts.
        attention_weights = self.attention(concat_features)

        weighted_features = []
        for i, feat in enumerate(part_features):
            weighted = feat * attention_weights[:, i : i + 1]
            weighted_features.append(weighted)

        # Fuse the weighted part features.
        fused = torch.cat(weighted_features, dim=1)
        features = self.fusion(fused)

        # Apply dropout before BNNeck if requested.
        features = self.dropout(features)

        # BNNeck
        features_bn = self.bn_neck(features)

        # Classification logits are computed from the BNNeck output.
        logits = self.classifier(features_bn)

        if return_parts:
            part_details = {
                "stripe_features": torch.stack(part_features[:-1], dim=1),
                "global_part_feature": part_features[-1],
                "part_features": torch.stack(part_features, dim=1),
                "part_attention": attention_weights,
            }
            return features, logits, part_details

        return features, logits


class PlainGlobalExtractor(nn.Module):
    """Minimal global ReID head: GAP -> optional dropout -> BNNeck -> Linear."""

    def __init__(
        self,
        backbone_dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone_dim = backbone_dim
        self.dropout_rate = dropout
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.bn_neck = nn.BatchNorm1d(backbone_dim)
        self.bn_neck.bias.requires_grad_(False)
        self.classifier = nn.Linear(backbone_dim, num_classes, bias=False)

    def forward(
        self,
        feature_map: torch.Tensor,
        return_parts: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.pool(feature_map).flatten(1)
        features_bn = self.bn_neck(self.dropout(features))
        logits = self.classifier(features_bn)
        if return_parts:
            return features, logits, {
                "stripe_features": None,
                "global_part_feature": features,
                "part_features": features.unsqueeze(1),
                "part_attention": None,
            }
        return features, logits


# ============================================================================
#                           ReID Backbone Metadata
# ============================================================================

# Backbone output widths.
BACKBONE_DIM_MAP = {
    # OSNet family
    "osnet_x1_0": 512,
    "osnet_x0_75": 512,
    "osnet_x0_5": 512,
    "osnet_x0_25": 512,
    "osnet_ain_x1_0": 512,
    "osnet_ain_x0_75": 512,
    "osnet_ain_x0_5": 512,
    "osnet_ain_x0_25": 512,
    "osnet_ibn_x1_0": 512,
    # ResNet family
    "resnet50": 2048,
    "resnet50_fc512": 512,
    "resnet101": 2048,
    "resnet152": 2048,
    # ResNet-IBN family
    "resnet50_ibn_a": 2048,
    "resnet50_ibn_b": 2048,
    "resnet101_ibn_a": 2048,
    # Other supported backbones
    "densenet121": 1024,
    "mobilenetv2_x1_0": 1280,
    "mobilenetv2_x1_4": 1792,
    "shufflenet": 960,
    "squeezenet1_0": 512,
    "squeezenet1_1": 512,
}

# Backbone widths used for the intermediate FGID guidance feature.
# OSNet uses conv3 outputs, while ResNet-style models use layer2 outputs.
BACKBONE_MID_DIM_MAP = {
    "osnet_x1_0": 384, "osnet_x0_75": 288, "osnet_x0_5": 192, "osnet_x0_25": 96,
    "osnet_ain_x1_0": 384, "osnet_ain_x0_75": 288, "osnet_ain_x0_5": 192, "osnet_ain_x0_25": 96,
    "osnet_ibn_x1_0": 384,
    "resnet50": 512, "resnet50_fc512": 512, "resnet101": 512, "resnet152": 512,
    "resnet50_ibn_a": 512, "resnet50_ibn_b": 512, "resnet101_ibn_a": 512,
    "densenet121": 512, "mobilenetv2_x1_0": 32, "mobilenetv2_x1_4": 48,
    "shufflenet": 240, "squeezenet1_0": 256, "squeezenet1_1": 256,
}

# Public list of supported backbone names.
SUPPORTED_BACKBONES = list(BACKBONE_DIM_MAP.keys())


def get_backbone_dim(backbone_name: str) -> int:
    """Get the output width of the selected backbone."""
    if backbone_name in BACKBONE_DIM_MAP:
        return BACKBONE_DIM_MAP[backbone_name]
    if "resnet" in backbone_name.lower():
        return 2048
    return 512


def get_backbone_mid_dim(backbone_name: str) -> int:
    """Get the intermediate backbone width used by FGID guidance."""
    if backbone_name in BACKBONE_MID_DIM_MAP:
        return BACKBONE_MID_DIM_MAP[backbone_name]
    if "resnet" in backbone_name.lower():
        return 512
    return 384


class TaskAdaptiveFeatureFusion(nn.Module):
    """Fuse corrected features as a trust-bounded residual over raw features."""

    def __init__(
        self,
        channels: int,
        hidden_dim: int = 128,
        init_corrected_bias: float = 2.0,
        aux_dim: int = 0,
        max_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.aux_dim = int(aux_dim)
        self.max_residual_scale = min(max(float(max_residual_scale), 0.0), 1.0)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 3 + self.aux_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels),
        )
        nn.init.constant_(self.mlp[-1].bias, init_corrected_bias)

    def forward(
        self,
        raw_feature_map: torch.Tensor,
        corrected_feature_map: torch.Tensor,
        aux_stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raw_desc = self.pool(raw_feature_map).flatten(1)
        corrected_desc = self.pool(corrected_feature_map).flatten(1)
        diff_desc = torch.abs(corrected_desc - raw_desc)
        fusion_inputs = [raw_desc, corrected_desc, diff_desc]
        if self.aux_dim > 0:
            if aux_stats is None:
                aux_stats = raw_desc.new_zeros(raw_desc.size(0), self.aux_dim)
            fusion_inputs.append(aux_stats)
        residual_gate = self.max_residual_scale * torch.sigmoid(
            self.mlp(torch.cat(fusion_inputs, dim=1))
        )
        residual_gate = residual_gate.unsqueeze(-1).unsqueeze(-1)
        return raw_feature_map + residual_gate * (corrected_feature_map - raw_feature_map)


class StripeAwareBranchAttentionFusion(nn.Module):
    """Fuse multiple feature-map branches with stripe-wise competitive attention."""

    def __init__(
        self,
        channels: int,
        num_stripes: int = 6,
        hidden_dim: int = 128,
        num_branches: int = 3,
        aux_dim: int = 0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_stripes = max(int(num_stripes), 1)
        self.num_branches = max(int(num_branches), 2)
        self.aux_dim = int(aux_dim)
        self.temperature = max(float(temperature), 1e-6)
        self.branch_mlp = nn.Sequential(
            nn.Linear(channels + self.aux_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def _pool_stripes(self, feature_map: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(feature_map, (self.num_stripes, 1))
        return pooled.squeeze(-1).permute(0, 2, 1).contiguous()

    def _stripe_ranges(self, height: int) -> List[Tuple[int, int]]:
        stripe_h = max(height // self.num_stripes, 1)
        ranges: List[Tuple[int, int]] = []
        for idx in range(self.num_stripes):
            start = idx * stripe_h
            end = (idx + 1) * stripe_h if idx < self.num_stripes - 1 else height
            ranges.append((start, max(end, start + 1)))
        return ranges

    def forward(
        self,
        branch_feature_maps: List[torch.Tensor],
        aux_stats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(branch_feature_maps) != self.num_branches:
            raise ValueError(
                f"Expected {self.num_branches} branch feature maps, got {len(branch_feature_maps)}"
            )

        batch_size, channels, height, width = branch_feature_maps[0].shape
        branch_desc = torch.stack(
            [self._pool_stripes(feature_map) for feature_map in branch_feature_maps],
            dim=2,
        )
        scorer_input = branch_desc
        if self.aux_dim > 0:
            if aux_stats is None:
                aux_stats = branch_desc.new_zeros(batch_size, self.aux_dim)
            aux_stats = aux_stats.unsqueeze(1).unsqueeze(2).expand(
                -1,
                self.num_stripes,
                self.num_branches,
                -1,
            )
            scorer_input = torch.cat([branch_desc, aux_stats], dim=-1)

        scores = self.branch_mlp(
            scorer_input.reshape(batch_size * self.num_stripes * self.num_branches, -1)
        )
        scores = scores.view(batch_size, self.num_stripes, self.num_branches)
        branch_attention_weights = F.softmax(scores / self.temperature, dim=-1)

        stripe_outputs = []
        for stripe_idx, (start, end) in enumerate(self._stripe_ranges(height)):
            stripe_fused = branch_feature_maps[0].new_zeros(batch_size, channels, end - start, width)
            for branch_idx, feature_map in enumerate(branch_feature_maps):
                alpha = branch_attention_weights[:, stripe_idx, branch_idx].view(batch_size, 1, 1, 1)
                stripe_fused = stripe_fused + alpha * feature_map[:, :, start:end, :]
            stripe_outputs.append(stripe_fused)

        fused_feature_map = torch.cat(stripe_outputs, dim=2)
        return fused_feature_map, branch_attention_weights


class FeatureProjectionHead(nn.Module):
    """Project identity features into a compact auxiliary embedding."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class JointReIDModel(nn.Module):
    """Joint wildlife ReID model with illumination correction and local stripes.

    The model combines an optional IPAID illumination module, a configurable
    ReID backbone, soft spatial masking, and a stripe-based local feature
    extractor. When enabled, a task-adaptive fusion module mixes raw-image and
    corrected-image backbone features.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "osnet_ain_x1_0",
        num_stripes: int = 6,
        hidden_dim: int = 256,
        pretrained_backbone: bool = True,
        soft_mask_temperature: float = 10.0,
        soft_mask_type: str = "sigmoid",  # "sigmoid" or "gaussian"
        use_ipaid: bool = True,  # Enable the illumination module.
        dropout: float = 0.0,  # Dropout before BNNeck in the local extractor.
        use_backbone_checkpointing: bool = True,
        ipaid_params: dict = None,  # Additional IPAID and fusion settings.
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.use_ipaid = use_ipaid
        self.dropout_rate = dropout
        self.use_backbone_checkpointing = use_backbone_checkpointing

        # 1. Configure optional IPAID / FGID components.
        _p = dict(ipaid_params or {})
        feature_fusion_cfg = _p.pop("_feature_fusion", _p.pop("feature_fusion", {})) or {}
        branch_attention_cfg = _p.pop("_branch_attention_fusion", _p.pop("branch_attention_fusion", {})) or {}
        nuisance_head_cfg = _p.pop("_nuisance_head", {}) or {}
        reid_head_cfg = _p.pop("_reid_head", {}) or {}
        backbone_erasing_cfg = _p.pop("_backbone_random_erasing", {}) or {}
        self.feature_fusion = None
        self.feature_fusion_enabled = bool(feature_fusion_cfg.get("enabled", False))
        self.feature_fusion_include_stats = bool(feature_fusion_cfg.get("include_illum_stats", True))
        self.feature_fusion_aux_dim = 5 if self.feature_fusion_enabled and self.feature_fusion_include_stats else 0
        self.branch_attention_fusion = None
        self.branch_attention_fusion_enabled = bool(branch_attention_cfg.get("enabled", False))
        self.branch_attention_include_stats = bool(branch_attention_cfg.get("include_illum_stats", True))
        self.branch_attention_aux_dim = (
            5 if self.branch_attention_fusion_enabled and self.branch_attention_include_stats else 0
        )
        self.branch_attention_num_branches = max(int(branch_attention_cfg.get("num_branches", 3)), 2)
        self.nuisance_head_enabled = bool(nuisance_head_cfg.get("enabled", False))
        self.nuisance_dim = int(nuisance_head_cfg.get("nuisance_dim", 64))
        self.photometric_dim = int(nuisance_head_cfg.get("photometric_dim", 4))
        self.reid_head_type = str(reid_head_cfg.get("type", "local_stripe")).lower()
        self.nuisance_projection = None
        self.photometric_regressor = None
        self.enable_coarse_task_grad = bool(_p.get("enable_coarse_task_grad", True))
        self.coarse_guidance_mode = str(_p.get("coarse_guidance_mode", "safe"))
        self.num_grad_variants = int(_p.get("num_grad_variants", 1))
        self.backbone_random_erasing_enabled = bool(backbone_erasing_cfg.get("enabled", False))
        self.backbone_random_erasing_prob = float(
            backbone_erasing_cfg.get("probability", backbone_erasing_cfg.get("p", 0.5))
        )
        scale_values = backbone_erasing_cfg.get("scale", [0.02, 0.25])
        ratio_values = backbone_erasing_cfg.get("ratio", [0.3, 3.3])
        self.backbone_random_erasing_scale = (float(scale_values[0]), float(scale_values[1]))
        self.backbone_random_erasing_ratio = (float(ratio_values[0]), float(ratio_values[1]))
        self.backbone_random_erasing_value = backbone_erasing_cfg.get("value", "random")
        self.backbone_random_erasing_max_attempts = int(backbone_erasing_cfg.get("max_attempts", 10))
        self._freeze_backbone = False
        self._freeze_illumination = False
        self._freeze_local_extractor = False
        self._freeze_feature_fusion = False

        if use_ipaid:
            backbone_mid_ch = get_backbone_mid_dim(backbone_name)
            self.illumination = IPAIDModule(
                base_channels=_p.get('base_channels', 32),
                num_scales=_p.get('num_scales', 3),
                refine_iterations=_p.get('refine_iterations', 2),
                use_sensitivity=_p.get('use_sensitivity', True),
                use_refinement=_p.get('use_refinement', True),
                backbone_mid_channels=_p.get('backbone_mid_channels', backbone_mid_ch),
                use_feature_guided=_p.get('use_feature_guided', True),
                use_color_illumination=_p.get('use_color_illumination', True),
                color_illumination_mode=_p.get('color_illumination_mode', 'local_rgb'),
                safe_color_enabled=_p.get('safe_color_enabled', True),
                max_color_shift=_p.get('max_color_shift', 0.08),
                safe_gain_min=_p.get('safe_gain_min', 0.5),
                safe_gain_max=_p.get('safe_gain_max', 1.6),
                color_risk_scale=_p.get('color_risk_scale', 3.0),
                max_color_risk=_p.get('max_color_risk', 0.55),
                clamp_input_range=_p.get('clamp_input_range', False),
                wb_max_shift=_p.get('wb_max_shift', 0.12),
                enable_task_aware_rollback=_p.get('enable_task_aware_rollback', True),
                rollback_hidden_dim=_p.get('rollback_hidden_dim', 64),
                rollback_min_alpha=_p.get('rollback_min_alpha', 0.05),
                rollback_max_alpha=_p.get('rollback_max_alpha', 0.98),
                rollback_granularity=_p.get('rollback_granularity', 'global'),
                rollback_num_stripes=_p.get('rollback_num_stripes', num_stripes),
                coarse_guidance_mode=_p.get('coarse_guidance_mode', 'safe'),
            )

            # IICL uses photometric variants generated from reflectance and illumination.
            self.variant_generator = DataDrivenVariantGenerator(
                num_variants=2,
                mix_ratio=0.7,
                gamma_range=(0.6, 1.4),
            )
        else:
            self.illumination = None
            self.variant_generator = None

        # 2. Build the ReID backbone and optional feature-fusion module.
        self.backbone = torchreid.models.build_model(
            name=backbone_name,
            num_classes=num_classes,
            loss="softmax",
            pretrained=pretrained_backbone,
        )
        self.backbone_dim = get_backbone_dim(backbone_name)
        if self.feature_fusion_enabled:
            self.feature_fusion = TaskAdaptiveFeatureFusion(
                channels=self.backbone_dim,
                hidden_dim=int(feature_fusion_cfg.get("hidden_dim", 128)),
                init_corrected_bias=float(feature_fusion_cfg.get("init_corrected_bias", 2.0)),
                aux_dim=self.feature_fusion_aux_dim,
                max_residual_scale=float(feature_fusion_cfg.get("max_residual_scale", 1.0)),
            )
        if self.branch_attention_fusion_enabled:
            self.branch_attention_fusion = StripeAwareBranchAttentionFusion(
                channels=self.backbone_dim,
                num_stripes=num_stripes,
                hidden_dim=int(branch_attention_cfg.get("hidden_dim", 128)),
                num_branches=self.branch_attention_num_branches,
                aux_dim=self.branch_attention_aux_dim,
                temperature=float(branch_attention_cfg.get("temperature", 1.0)),
            )

        # 3. Choose the soft-mask implementation.
        if soft_mask_type == "gaussian":
            self.soft_mask_generator = SoftMaskGaussian(sigma_ratio=0.4)
        else:
            self.soft_mask_generator = SoftMaskGenerator(
                temperature=soft_mask_temperature,
                margin=0.1,
            )

        # 4. Build the ReID head.
        if self.reid_head_type == "plain_global":
            self.local_extractor = PlainGlobalExtractor(
                backbone_dim=self.backbone_dim,
                num_classes=num_classes,
                dropout=dropout,
            )
        else:
            self.local_extractor = LocalFeatureExtractor(
                backbone_dim=self.backbone_dim,
                hidden_dim=hidden_dim,
                num_stripes=num_stripes,
                num_classes=num_classes,
                use_deformable_stripes=True,
                max_offset_ratio=0.2,
                dropout=dropout,
            )
        if self.nuisance_head_enabled:
            nuisance_hidden_dim = int(nuisance_head_cfg.get("hidden_dim", max(hidden_dim, 64)))
            self.nuisance_projection = FeatureProjectionHead(
                input_dim=self.backbone_dim if self.reid_head_type == "plain_global" else hidden_dim,
                hidden_dim=nuisance_hidden_dim,
                output_dim=self.nuisance_dim,
            )
            self.photometric_regressor = nn.Linear(self.nuisance_dim, self.photometric_dim)

        # Some historical trainer flags still exist, but this model variant
        # intentionally omits adversarial and uncertainty-weighted heads.
        # Adversarial and uncertainty-weighting modules are intentionally disabled in
        # this model variant. Trainer-side compatibility flags may still exist.

        print("[INFO] JointReIDModel initialized")
        print(f"  backbone: {backbone_name}")
        print(f"  backbone_dim: {self.backbone_dim}")
        print(f"  num_stripes: {num_stripes}")
        print(f"  reid_head: {self.reid_head_type}")
        print(f"  num_classes: {num_classes}")
        print(f"  soft_mask_type: {soft_mask_type}")
        print(f"  use_ipaid: {use_ipaid}")
        print(f"  nuisance_head: {self.nuisance_head_enabled}")
        if use_ipaid:
            print(f"  use_feature_guided: {_p.get('use_feature_guided', True)}")
            print(f"  use_color_illumination: {_p.get('use_color_illumination', True)}")
            print(f"  color_illumination_mode: {_p.get('color_illumination_mode', 'local_rgb')}")

    def freeze_backbone(self, freeze: bool = True) -> None:
        self._freeze_backbone = freeze
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        if freeze:
            self.backbone.eval()

    def freeze_illumination(self, freeze: bool = True) -> None:
        self._freeze_illumination = freeze
        if self.illumination is not None:
            for p in self.illumination.parameters():
                p.requires_grad = not freeze
            if freeze:
                self.illumination.eval()

    def freeze_local_extractor(self, freeze: bool = True) -> None:
        self._freeze_local_extractor = freeze
        for p in self.local_extractor.parameters():
            p.requires_grad = not freeze
        if freeze:
            self.local_extractor.eval()

    def freeze_feature_fusion(self, freeze: bool = True) -> None:
        self._freeze_feature_fusion = freeze
        for module in (self.feature_fusion, self.branch_attention_fusion):
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = not freeze
                if freeze:
                    module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._freeze_backbone:
            self.backbone.eval()
        if mode and self._freeze_illumination and self.illumination is not None:
            self.illumination.eval()
        if mode and self._freeze_local_extractor:
            self.local_extractor.eval()
        if mode and self._freeze_feature_fusion:
            if self.feature_fusion is not None:
                self.feature_fusion.eval()
            if self.branch_attention_fusion is not None:
                self.branch_attention_fusion.eval()
        return self

    def _imagenet_normalize(self, images: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        return (images - mean) / std

    def _sample_backbone_random_erasing_region(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> Optional[Tuple[int, int, int, int]]:
        area = height * width
        ratio_min = max(self.backbone_random_erasing_ratio[0], 1e-6)
        ratio_max = max(self.backbone_random_erasing_ratio[1], ratio_min)
        log_ratio_min = math.log(ratio_min)
        log_ratio_max = math.log(ratio_max)

        for _ in range(max(self.backbone_random_erasing_max_attempts, 1)):
            target_area = float(
                area * torch.empty(1, device=device).uniform_(
                    self.backbone_random_erasing_scale[0],
                    self.backbone_random_erasing_scale[1],
                ).item()
            )
            aspect_ratio = math.exp(
                float(torch.empty(1, device=device).uniform_(log_ratio_min, log_ratio_max).item())
            )

            erase_h = int(round(math.sqrt(target_area * aspect_ratio)))
            erase_w = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < erase_h <= height and 0 < erase_w <= width:
                top = int(torch.randint(0, height - erase_h + 1, (1,), device=device).item())
                left = int(torch.randint(0, width - erase_w + 1, (1,), device=device).item())
                return top, left, erase_h, erase_w

        return None

    def _apply_backbone_random_erasing(self, normalized: torch.Tensor) -> torch.Tensor:
        if (
            not self.training
            or not self.backbone_random_erasing_enabled
            or self.backbone_random_erasing_prob <= 0.0
        ):
            return normalized

        erased = normalized.clone()
        batch_size, channels, height, width = erased.shape
        device = erased.device

        for idx in range(batch_size):
            if float(torch.rand(1, device=device).item()) > self.backbone_random_erasing_prob:
                continue

            region = self._sample_backbone_random_erasing_region(height, width, device)
            if region is None:
                continue

            top, left, erase_h, erase_w = region
            if self.backbone_random_erasing_value == "random":
                fill = torch.randn(
                    (channels, erase_h, erase_w),
                    device=device,
                    dtype=erased.dtype,
                )
            else:
                fill_value = float(self.backbone_random_erasing_value)
                fill = torch.full(
                    (channels, erase_h, erase_w),
                    fill_value,
                    device=device,
                    dtype=erased.dtype,
                )
            erased[idx, :, top:top + erase_h, left:left + erase_w] = fill

        return erased

    def _prepare_backbone_input(self, images: torch.Tensor) -> torch.Tensor:
        images = torch.clamp(images, 0.0, 1.0)
        normalized = self._imagenet_normalize(images)
        normalized = self._apply_backbone_random_erasing(normalized)
        return normalized

    def _autocast_disabled_context(self, tensor: torch.Tensor):
        device_type = tensor.device.type
        if device_type in {"cuda", "cpu"}:
            return torch.amp.autocast(device_type=device_type, enabled=False)
        return nullcontext()

    def _backbone_trainable(self) -> bool:
        return any(p.requires_grad for p in self.backbone.parameters())

    def _illumination_trainable(self) -> bool:
        if self.illumination is None:
            return False
        return any(p.requires_grad for p in self.illumination.parameters())

    def _extract_raw_feature_map_for_fusion(self, images: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.feature_fusion_enabled and not self.branch_attention_fusion_enabled:
            return None

        normalized = self._prepare_backbone_input(images)
        # Keep the raw branch on-graph so fusion can learn whether preserving
        # original appearance cues improves identity discrimination.
        return self.extract_backbone_features(normalized)

    def _maybe_fuse_feature_maps(
        self,
        raw_feature_map: Optional[torch.Tensor],
        corrected_feature_map: torch.Tensor,
        aux_stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if raw_feature_map is None or self.feature_fusion is None:
            return corrected_feature_map
        return self.feature_fusion(raw_feature_map, corrected_feature_map, aux_stats=aux_stats)

    def _build_identity_protection_map(
        self,
        reference_feature_map: Optional[torch.Tensor],
        image_size: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        if reference_feature_map is None:
            return None

        activation = reference_feature_map.detach().float().pow(2).mean(dim=1, keepdim=True)
        flat = activation.flatten(1)
        min_value = flat.min(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
        max_value = flat.max(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
        normalized = (activation - min_value) / (max_value - min_value + 1e-6)
        return F.interpolate(
            normalized,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)

    def _maybe_fuse_branch_feature_maps(
        self,
        branch_feature_maps: List[Optional[torch.Tensor]],
        aux_stats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        valid_branch_feature_maps = [feature_map for feature_map in branch_feature_maps if feature_map is not None]
        if not valid_branch_feature_maps:
            raise ValueError("Expected at least one feature map for branch fusion")

        if not self.branch_attention_fusion_enabled or self.branch_attention_fusion is None:
            fallback_feature_map = valid_branch_feature_maps[-1]
            if len(valid_branch_feature_maps) >= 2 and self.feature_fusion is not None:
                fallback_feature_map = self._maybe_fuse_feature_maps(
                    valid_branch_feature_maps[0],
                    fallback_feature_map,
                    aux_stats=aux_stats,
                )
            return fallback_feature_map, None

        if len(valid_branch_feature_maps) != self.branch_attention_num_branches:
            fallback_feature_map = valid_branch_feature_maps[-1]
            if len(valid_branch_feature_maps) >= 2 and self.feature_fusion is not None:
                fallback_feature_map = self._maybe_fuse_feature_maps(
                    valid_branch_feature_maps[0],
                    fallback_feature_map,
                    aux_stats=aux_stats,
                )
            return fallback_feature_map, None

        fused_feature_map, branch_attention_weights = self.branch_attention_fusion(
            valid_branch_feature_maps,
            aux_stats=aux_stats if self.branch_attention_include_stats else None,
        )
        return fused_feature_map, branch_attention_weights

    def _build_feature_fusion_stats(
        self,
        ipaid_details: Optional[Dict[str, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        use_stats = (
            (self.feature_fusion_enabled and self.feature_fusion_include_stats)
            or (self.branch_attention_fusion_enabled and self.branch_attention_include_stats)
        )
        if not use_stats or ipaid_details is None:
            return None

        illumination = ipaid_details.get("effective_illumination", ipaid_details.get("illumination"))
        color_risk = ipaid_details.get("color_risk")
        rollback_alpha = ipaid_details.get("rollback_alpha")
        correction_gap = ipaid_details.get("correction_gap")
        if illumination is None:
            return None

        stats = [
            illumination.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1),
            illumination.flatten(1).std(dim=1, unbiased=False).unsqueeze(1),
        ]
        if color_risk is None:
            stats.append(illumination.new_zeros(illumination.size(0), 1))
        else:
            stats.append(color_risk.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1))
        if rollback_alpha is None:
            stats.append(illumination.new_zeros(illumination.size(0), 1))
        else:
            stats.append(rollback_alpha.view(rollback_alpha.size(0), -1).mean(dim=1, keepdim=True))
        if correction_gap is None:
            stats.append(illumination.new_zeros(illumination.size(0), 1))
        else:
            stats.append(correction_gap.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1))
        return torch.cat(stats, dim=1)

    def extract_early_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the early backbone feature map used by FGID guidance.

        OSNet returns features up to ``conv3``. ResNet-style backbones return
        features up to ``layer2``.
        """
        name = self.backbone_name.lower()

        if "osnet" in name:
            x = self.backbone.conv1(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.conv2(x)
            x = self.backbone.conv3(x)
        elif "resnet" in name or "densenet" in name:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
        elif "mobilenet" in name:
            # Stop at the intermediate MobileNetV2 feature stage.
            for i, layer in enumerate(self.backbone.features):
                x = layer(x)
                if i == 6:
                    break
        elif "shufflenet" in name:
            x = self.backbone.conv1(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.stage2(x)
        else:
            if hasattr(self.backbone, 'featuremaps'):
                x = self.backbone.featuremaps(x)
            else:
                raise NotImplementedError(f"Unsupported backbone for early features: {self.backbone_name}")
        return x

    def _compute_illumination_guidance(
        self,
        images: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Compute coarse illumination outputs and early guidance features."""
        if self.illumination is None:
            raise RuntimeError("Illumination module is required for guidance computation")

        use_guidance_grad = self.enable_coarse_task_grad and self._illumination_trainable()
        if use_guidance_grad:
            coarse_out = self.illumination.forward_coarse(images)
            guidance = self.illumination.compute_coarse_guidance_reflectance(
                images,
                coarse_out,
                mode=self.coarse_guidance_mode,
            )
            if self._backbone_trainable():
                feat_mid = self.extract_early_backbone_features(self._prepare_backbone_input(guidance))
            else:
                with torch.no_grad():
                    feat_mid = self.extract_early_backbone_features(self._prepare_backbone_input(guidance))
            return coarse_out, feat_mid

        with torch.no_grad():
            coarse_out = self.illumination.forward_coarse(images)
            guidance = self.illumination.compute_coarse_guidance_reflectance(
                images,
                coarse_out,
                mode=self.coarse_guidance_mode,
            )
            feat_mid = self.extract_early_backbone_features(self._prepare_backbone_input(guidance))
        return coarse_out, feat_mid

    def _extract_backbone_features_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Extract full backbone feature maps without checkpoint wrapping."""
        name = self.backbone_name.lower()
        
        if "osnet" in name:
            # Full OSNet feature path.
            x = self.backbone.conv1(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.conv2(x)
            x = self.backbone.conv3(x)
            x = self.backbone.conv4(x)
            x = self.backbone.conv5(x)
        elif "resnet" in name or "densenet" in name:
            # Full feature path for ResNet-style backbones.
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
        elif "mobilenet" in name:
            # Full MobileNetV2 feature stack.
            x = self.backbone.features(x)
        elif "shufflenet" in name:
            # Full ShuffleNet feature stack.
            x = self.backbone.conv1(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.stage2(x)
            x = self.backbone.stage3(x)
            x = self.backbone.stage4(x)
            x = self.backbone.conv5(x)
        elif "squeezenet" in name:
            # Full SqueezeNet feature stack.
            x = self.backbone.features(x)
        else:
            # Fall back to a generic ``featuremaps`` API when available.
            if hasattr(self.backbone, 'featuremaps'):
                x = self.backbone.featuremaps(x)
            else:
                raise NotImplementedError(f"Unsupported backbone: {self.backbone_name}")
        return x

    def extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract full backbone features with optional activation checkpointing."""
        # Non-reentrant checkpointing still works when the tensor input itself
        # does not require gradients, which is the common case for image inputs.
        if self.use_backbone_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(self._extract_backbone_features_impl, x, use_reentrant=False)
        return self._extract_backbone_features_impl(x)

    def forward(
        self,
        images: torch.Tensor,
        boxes_list: Optional[List[torch.Tensor]] = None,
        return_illuminated: bool = False,
        return_local_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run the standard joint forward path.

        The pipeline first computes illumination guidance, then refines the
        corrected image, extracts backbone features, optionally applies soft
        spatial masks, and finally produces ReID features and logits.
        """
        B, _, H, W = images.shape
        device = images.device
        raw_feature_map = None
        fusion_stats = None
        branch_attention_weights = None

        if self.use_ipaid and self.illumination is not None:
            with self._autocast_disabled_context(images):
                illum_images = images.float()
                coarse_out, feat_mid = self._compute_illumination_guidance(illum_images)
                identity_protection_map = self._build_identity_protection_map(
                    feat_mid,
                    image_size=(H, W),
                )
                ipaid_details = self.illumination.forward_refine(
                    illum_images,
                    coarse_out,
                    feat_mid,
                    identity_protection_map=identity_protection_map,
                )
            illuminated = ipaid_details['reflectance']
            illumination_map = ipaid_details['illumination']
            fusion_stats = self._build_feature_fusion_stats(ipaid_details)
            raw_feature_map = self._extract_raw_feature_map_for_fusion(images)
        else:
            ipaid_details = None
            illuminated = images
            illumination_map = None

        if ipaid_details is not None and self.branch_attention_fusion_enabled:
            reflectance_base = ipaid_details.get("reflectance_base", illuminated)
            reflectance_adapted = ipaid_details.get("reflectance", illuminated)
            base_feature_map = self.extract_backbone_features(
                self._prepare_backbone_input(reflectance_base)
            )
            adapted_feature_map = self.extract_backbone_features(
                self._prepare_backbone_input(reflectance_adapted)
            )
            feature_map, branch_attention_weights = self._maybe_fuse_branch_feature_maps(
                [raw_feature_map, base_feature_map, adapted_feature_map],
                aux_stats=fusion_stats,
            )
            ipaid_details["branch_attention_weights"] = branch_attention_weights
        else:
            illuminated_normalized = self._prepare_backbone_input(illuminated)
            feature_map = self.extract_backbone_features(illuminated_normalized)
            feature_map = self._maybe_fuse_feature_maps(raw_feature_map, feature_map, aux_stats=fusion_stats)
        _, _, fh, fw = feature_map.shape

        # Apply soft box masks on the feature map when detections are available.
        soft_masks: Optional[List[torch.Tensor]]
        if boxes_list is not None:
            weighted_features = []
            soft_masks = []
            for i in range(B):
                feat = feature_map[i : i + 1]
                boxes = boxes_list[i]
                soft_mask = self.soft_mask_generator(boxes, (fh, fw), (H, W), device)
                soft_masks.append(soft_mask)
                weighted_feat = feat * soft_mask.unsqueeze(0)
                weighted_features.append(weighted_feat)
            feature_map = torch.cat(weighted_features, dim=0)
        else:
            soft_masks = None

        # Extract ReID descriptors and classification logits.
        local_details = None
        if return_local_features:
            local_output = self.local_extractor(feature_map, return_parts=True)
            features, logits = local_output[0], local_output[1]
            local_details = local_output[2] if len(local_output) > 2 else None
        else:
            features, logits = self.local_extractor(feature_map)
        nuisance_features = None
        photometric_prediction = None
        if self.nuisance_head_enabled and self.nuisance_projection is not None:
            nuisance_features = self.nuisance_projection(features)
            photometric_prediction = self.photometric_regressor(nuisance_features)

        # This model variant only returns ReID outputs and IPAID details.
        output: Dict[str, torch.Tensor] = {
            "features": features,
            "logits": logits,
            "nuisance_features": nuisance_features,
            "photometric_prediction": photometric_prediction,
            "ipaid_details": ipaid_details,
            "illumination_map": illumination_map,
        }
        if local_details is not None:
            output["local_features"] = local_details.get("stripe_features")
            output["part_features"] = local_details.get("part_features")
            output["part_attention"] = local_details.get("part_attention")
            output["global_part_feature"] = local_details.get("global_part_feature")

        if return_illuminated:
            output["illuminated"] = illuminated
        if soft_masks is not None:
            output["soft_masks"] = soft_masks

        return output

    def forward_illumination_only(
        self,
        images: torch.Tensor,
        return_illuminated: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run only the illumination branch when no ReID supervision is active.

        This path is used by phase-2 illumination-only optimization when both
        the ReID loss and identity-preserving feature losses are disabled. It
        avoids building the much larger backbone/local-extractor graph.
        """
        if self.use_ipaid and self.illumination is not None:
            with self._autocast_disabled_context(images):
                illum_images = images.float()
                coarse_out, feat_mid = self._compute_illumination_guidance(illum_images)
                identity_protection_map = self._build_identity_protection_map(
                    feat_mid,
                    image_size=(images.shape[2], images.shape[3]),
                )
                ipaid_details = self.illumination.forward_refine(
                    illum_images,
                    coarse_out,
                    feat_mid,
                    identity_protection_map=identity_protection_map,
                )
            illuminated = ipaid_details["reflectance"]
            illumination_map = ipaid_details["illumination"]
        else:
            ipaid_details = None
            illuminated = images
            illumination_map = None

        output: Dict[str, torch.Tensor] = {
            "features": None,
            "logits": None,
            "nuisance_features": None,
            "photometric_prediction": None,
            "ipaid_details": ipaid_details,
            "illumination_map": illumination_map,
        }
        if return_illuminated:
            output["illuminated"] = illuminated
        return output

    def forward_raw_reference(
        self,
        images: torch.Tensor,
        detach: bool = True,
        return_local_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Extract raw-image descriptors for geometry-preserving supervision."""
        if detach:
            backbone_was_training = self.backbone.training
            extractor_was_training = self.local_extractor.training
            self.backbone.eval()
            self.local_extractor.eval()
            try:
                with torch.no_grad():
                    with self._autocast_disabled_context(images):
                        raw_normalized = self._prepare_backbone_input(images)
                        feature_map = self.extract_backbone_features(raw_normalized)
                        if return_local_features:
                            local_output = self.local_extractor(feature_map, return_parts=True)
                            features, logits = local_output[0], local_output[1]
                            local_details = local_output[2] if len(local_output) > 2 else None
                        else:
                            features, logits = self.local_extractor(feature_map)
                            local_details = None
            finally:
                self.backbone.train(backbone_was_training)
                self.local_extractor.train(extractor_was_training)
            output = {
                "features": features.detach(),
                "logits": logits.detach(),
            }
            if local_details is not None:
                output["local_features"] = local_details.get("stripe_features")
                output["part_features"] = local_details.get("part_features")
                output["part_attention"] = local_details.get("part_attention")
                output["global_part_feature"] = local_details.get("global_part_feature")
            return output

        with self._autocast_disabled_context(images):
            raw_normalized = self._prepare_backbone_input(images)
            feature_map = self.extract_backbone_features(raw_normalized)
            if return_local_features:
                local_output = self.local_extractor(feature_map, return_parts=True)
                features, logits = local_output[0], local_output[1]
                local_details = local_output[2] if len(local_output) > 2 else None
            else:
                features, logits = self.local_extractor(feature_map)
                local_details = None
        output = {
            "features": features,
            "logits": logits,
        }
        if local_details is not None:
            output["local_features"] = local_details.get("stripe_features")
            output["part_features"] = local_details.get("part_features")
            output["part_attention"] = local_details.get("part_attention")
            output["global_part_feature"] = local_details.get("global_part_feature")
        return output

    def forward_with_consistency_variants(
        self,
        images: torch.Tensor,
        num_variants: int = 2,
        return_local_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run the IICL forward path with illumination-aware variants.

        The method computes the corrected reflectance image, samples synthetic
        illumination variants, and extracts features for the original corrected
        branch and each variant branch for consistency supervision.
        """
        B, _, H, W = images.shape
        device = images.device

        if not self.use_ipaid or self.illumination is None:
            output = self.forward(images, return_local_features=return_local_features)
            output['features_variants'] = []
            return output

        raw_feature_map = None
        fusion_stats = None
        branch_attention_weights = None

        with self._autocast_disabled_context(images):
            illum_images = images.float()
            coarse_out, feat_mid = self._compute_illumination_guidance(illum_images)
            identity_protection_map = self._build_identity_protection_map(
                feat_mid,
                image_size=(H, W),
            )
            ipaid_details = self.illumination.forward_refine(
                illum_images,
                coarse_out,
                feat_mid,
                identity_protection_map=identity_protection_map,
            )
            reflectance = ipaid_details['reflectance']
            illumination = ipaid_details.get('effective_illumination', ipaid_details['illumination'])
            variants = self.variant_generator(
                reflectance=reflectance,
                illumination=illumination,
                num_variants=num_variants,
            )
        fusion_stats = self._build_feature_fusion_stats(ipaid_details)
        raw_feature_map = self._extract_raw_feature_map_for_fusion(images)

        # Features for the corrected reference branch.
        if self.branch_attention_fusion_enabled:
            reflectance_base = ipaid_details.get("reflectance_base", reflectance)
            base_feature_map = self.extract_backbone_features(
                self._prepare_backbone_input(reflectance_base)
            )
            adapted_feature_map = self.extract_backbone_features(
                self._prepare_backbone_input(reflectance)
            )
            feature_map_orig, branch_attention_weights = self._maybe_fuse_branch_feature_maps(
                [raw_feature_map, base_feature_map, adapted_feature_map],
                aux_stats=fusion_stats,
            )
            ipaid_details["branch_attention_weights"] = branch_attention_weights
        else:
            reflectance_normalized = self._prepare_backbone_input(reflectance)
            feature_map_orig = self.extract_backbone_features(reflectance_normalized)
            feature_map_orig = self._maybe_fuse_feature_maps(raw_feature_map, feature_map_orig, aux_stats=fusion_stats)
        local_details = None
        if return_local_features:
            local_output = self.local_extractor(feature_map_orig, return_parts=True)
            features_orig, logits_orig = local_output[0], local_output[1]
            local_details = local_output[2] if len(local_output) > 2 else None
        else:
            features_orig, logits_orig = self.local_extractor(feature_map_orig)
        nuisance_features = None
        photometric_prediction = None
        if self.nuisance_head_enabled and self.nuisance_projection is not None:
            nuisance_features = self.nuisance_projection(features_orig)
            photometric_prediction = self.photometric_regressor(nuisance_features)

        # Only a subset of variant branches keep gradients to control memory.
        num_grad_variants = getattr(self, 'num_grad_variants', 1)
        features_variants = []
        for i, variant in enumerate(variants):
            variant_normalized = self._prepare_backbone_input(variant)
            variant_raw_feature_map = self._extract_raw_feature_map_for_fusion(variant)
            if i < num_grad_variants:
                feature_map_var = self.extract_backbone_features(variant_normalized)
                feature_map_var = self._maybe_fuse_feature_maps(variant_raw_feature_map, feature_map_var, aux_stats=fusion_stats)
                features_var, _ = self.local_extractor(feature_map_var)
                features_variants.append(features_var)
            else:
                with torch.no_grad():
                    feature_map_var = self.extract_backbone_features(variant_normalized)
                    detached_variant_raw = variant_raw_feature_map.detach() if variant_raw_feature_map is not None else None
                    feature_map_var = self._maybe_fuse_feature_maps(detached_variant_raw, feature_map_var, aux_stats=fusion_stats)
                    features_var, _ = self.local_extractor(feature_map_var)
                    features_variants.append(features_var.detach())

        # This model variant only returns ReID outputs and IPAID details.
        output = {
            "features": features_orig,
            "logits": logits_orig,
            "nuisance_features": nuisance_features,
            "photometric_prediction": photometric_prediction,
            "features_variants": features_variants,
            "variants": variants,
            "ipaid_details": ipaid_details,
            "illumination_map": ipaid_details['illumination'],
            "illuminated": reflectance,
        }
        if local_details is not None:
            output["local_features"] = local_details.get("stripe_features")
            output["part_features"] = local_details.get("part_features")
            output["part_attention"] = local_details.get("part_attention")
            output["global_part_feature"] = local_details.get("global_part_feature")

        return output

