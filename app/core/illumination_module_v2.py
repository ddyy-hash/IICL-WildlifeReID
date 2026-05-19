#!/usr/bin/env python3
"""
Identity-Preserving Adaptive Illumination Decomposition (IPAID).

This module implements a task-oriented illumination correction pipeline for ReID:
1. Estimate spatial illumination at multiple scales.
2. Modulate correction strength with a learned sensitivity map.
3. Optionally refine the illumination map with backbone mid-level features.
4. Apply a safe Retinex-like correction with color-drift constraints.
5. Feed corrected reflectance into the downstream ReID model.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List


def _stripe_ranges(height: int, num_stripes: int) -> List[Tuple[int, int]]:
    """Split an image height into contiguous stripe ranges."""
    num_stripes = max(int(num_stripes), 1)
    stripe_h = max(height // num_stripes, 1)
    ranges: List[Tuple[int, int]] = []
    for idx in range(num_stripes):
        start = idx * stripe_h
        end = (idx + 1) * stripe_h if idx < num_stripes - 1 else height
        ranges.append((start, max(end, start + 1)))
    return ranges


def _stripe_mean_map(x: torch.Tensor, num_stripes: int) -> torch.Tensor:
    """Average a single-channel map inside each horizontal stripe."""
    if x.dim() != 4 or x.size(1) != 1:
        raise ValueError(f"Expected [B, 1, H, W] tensor, got {tuple(x.shape)}")
    pooled = F.adaptive_avg_pool2d(x, (max(int(num_stripes), 1), 1))
    return pooled.squeeze(-1).squeeze(1)


def _stripe_std_map(x: torch.Tensor, num_stripes: int) -> torch.Tensor:
    """Compute per-stripe standard deviation for a single-channel map."""
    mean = F.adaptive_avg_pool2d(x, (max(int(num_stripes), 1), 1))
    sq_mean = F.adaptive_avg_pool2d(x * x, (max(int(num_stripes), 1), 1))
    var = torch.clamp(sq_mean - mean * mean, min=0.0)
    return torch.sqrt(var + 1e-6).squeeze(-1).squeeze(1)


def _stripe_feature_pool(x: torch.Tensor, num_stripes: int) -> torch.Tensor:
    """Pool a feature map into one descriptor per horizontal stripe."""
    pooled = F.adaptive_avg_pool2d(x, (max(int(num_stripes), 1), 1))
    return pooled.squeeze(-1).permute(0, 2, 1).contiguous()


# ============================================================================
# Building Blocks
# ============================================================================

class ConvBNReLU(nn.Module):
    """Standard convolution block with batch normalization and ReLU."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, 
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, 
                                    padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(self.bn(x))


class ChannelAttention(nn.Module):
    """Channel attention block."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention block."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


def soft_min(a: torch.Tensor, b: torch.Tensor, tau: float = 10.0) -> torch.Tensor:
    """Differentiable approximation of min() using LogSumExp."""
    return -1.0 / tau * torch.logsumexp(
        torch.stack([-tau * a, -tau * b], dim=0), dim=0
    )


# ============================================================================
# ============================================================================

class MultiScaleIlluminationEstimator(nn.Module):
    
    def __init__(self, in_channels: int = 1, base_channels: int = 32, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        
        self.shared_encoder = nn.Sequential(
            ConvBNReLU(in_channels, base_channels, 3, 1, 1),
            ConvBNReLU(base_channels, base_channels, 3, 1, 1),
        )
        
        self.scale_branches = nn.ModuleList()
        for i in range(num_scales):
            branch = nn.Sequential(
                DepthwiseSeparableConv(base_channels, base_channels, 3, 1, 1),
                DepthwiseSeparableConv(base_channels, base_channels, 3, 1, 1),
                nn.Conv2d(base_channels, 1, 1, bias=True),
            )
            self.scale_branches.append(branch)
        
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
        self.channel_attention = ChannelAttention(base_channels)
    
    def forward(self, luminance: torch.Tensor) -> torch.Tensor:
        """
        Args:
        
        Returns:
        """
        B, _, H, W = luminance.shape
        
        features = self.shared_encoder(luminance)
        features = features * self.channel_attention(features)
        
        illuminations = []
        
        for i, branch in enumerate(self.scale_branches):
            scale_factor = 2 ** (self.num_scales - 1 - i)  # 4, 2, 1
            
            if scale_factor > 1:
                feat_down = F.interpolate(features, scale_factor=1/scale_factor, 
                                         mode='bilinear', align_corners=False)
                illum_down = branch(feat_down)
                illum = F.interpolate(illum_down, size=(H, W), 
                                     mode='bilinear', align_corners=False)
            else:
                illum = branch(features)
            
            illuminations.append(illum)
        
        weights = F.softmax(self.scale_weights, dim=0)
        illumination = sum(w * illum for w, illum in zip(weights, illuminations))
        
        illumination = F.softplus(illumination) + 0.1
        illumination = torch.clamp(illumination, 0.1, 10.0)
        
        return illumination


# ============================================================================
# ============================================================================

class SensitivityEstimator(nn.Module):

    def __init__(self, in_channels: int = 3, base_channels: int = 32, species_type: str = 'stripe'):
        super().__init__()

        self.species_type = species_type

        self.edge_branch = nn.Sequential(
            ConvBNReLU(in_channels, base_channels // 2, 3, 1, 1),
            ConvBNReLU(base_channels // 2, base_channels // 2, 3, 1, 1),
        )

        self.content_branch = nn.Sequential(
            ConvBNReLU(in_channels, base_channels, 3, 1, 1),
            DepthwiseSeparableConv(base_channels, base_channels, 3, 1, 1),
        )

        self.fusion = nn.Sequential(
            ConvBNReLU(base_channels + base_channels // 2, base_channels, 3, 1, 1),
            DepthwiseSeparableConv(base_channels, base_channels // 2, 3, 1, 1),
            nn.Conv2d(base_channels // 2, 1, 1, bias=True),
            nn.Sigmoid()
        )

        self.spatial_attention = SpatialAttention(kernel_size=7)

        if species_type == 'spot':
            self.range_low = nn.Parameter(torch.tensor(-2.0))
            self.range_high = nn.Parameter(torch.tensor(-0.5))
        else:
            self.range_low = nn.Parameter(torch.tensor(0.1))
            self.range_high = nn.Parameter(torch.tensor(0.8))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        
        Returns:
        """
        edge_feat = self.edge_branch(x)
        
        content_feat = self.content_branch(x)
        content_feat = content_feat * self.spatial_attention(content_feat)
        
        fused = torch.cat([edge_feat, content_feat], dim=1)
        sensitivity = self.fusion(fused)
        
        low_val = torch.sigmoid(self.range_low) * 0.3           # [0, 0.3]
        high_val = 0.3 + (1.0 - 0.3) * torch.sigmoid(self.range_high)  # [0.3, 1.0]
        sensitivity = low_val + (high_val - low_val) * sensitivity
        
        return sensitivity


# ============================================================================
#   FGID components: GRL, FeatureGuidedRefinement, LocalRGBIlluminationHead,
#   IlluminationConditionClassifier, auto_label_illumination,
#   HomoscedasticUncertaintyWeighting
# ============================================================================

class GradientReversalFunction(torch.autograd.Function):
    """"""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Gradient reversal layer used for adversarial training."""
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_


class FeatureGuidedRefinement(nn.Module):
    def __init__(self, illum_channels: int = 1, feat_channels: int = 384,
                 hidden_dim: int = 32):
        super().__init__()
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feat_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.illum_proj = nn.Sequential(
            nn.Conv2d(illum_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.query = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.key = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.value = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.attn_scale = hidden_dim ** -0.5
        self.output_conv = nn.Sequential(
            DepthwiseSeparableConv(hidden_dim, hidden_dim, 3, 1, 1),
            nn.Conv2d(hidden_dim, 1, 1, bias=True),
            nn.Tanh(),
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, L_coarse: torch.Tensor,
                feat_mid: torch.Tensor) -> torch.Tensor:
        """
        Args:
        Returns:
            L_refined: [B, 1, H, W]
        """
        feat_up = F.interpolate(feat_mid, size=L_coarse.shape[2:],
                                mode='bilinear', align_corners=False)
        f_proj = self.feat_proj(feat_up)
        l_proj = self.illum_proj(L_coarse)

        Q = self.query(l_proj)
        K = self.key(f_proj)
        V = self.value(f_proj)

        attn = (Q * K).sum(dim=1, keepdim=True) * self.attn_scale
        attn = torch.sigmoid(attn)
        attended = V * attn

        correction = self.output_conv(attended)
        alpha = torch.sigmoid(self.residual_weight)
        L_refined = L_coarse + alpha * correction
        L_refined = F.softplus(L_refined) + 0.1
        return L_refined


class LocalRGBIlluminationHead(nn.Module):
    """Predict a local 3-channel illumination field from gray illumination and RGB input."""
    def __init__(self, base_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            DepthwiseSeparableConv(4, base_channels, 3, 1, 1),
            DepthwiseSeparableConv(base_channels, base_channels, 3, 1, 1),
            nn.Conv2d(base_channels, 3, 1, bias=True),
            nn.Tanh(),
        )
        self.delta_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, L_gray: torch.Tensor,
                rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            L_gray: [B, 1, H, W]
            rgb: [B, 3, H, W]
        Returns:
            effective_illumination: [B, 3, H, W]
        """
        rgb_down = F.interpolate(rgb, size=L_gray.shape[2:],
                                 mode='bilinear', align_corners=False)
        inp = torch.cat([L_gray, rgb_down], dim=1)
        delta = self.net(inp) * torch.sigmoid(self.delta_scale)
        effective_illumination = L_gray * (1.0 + delta)
        return torch.clamp(effective_illumination, 0.1, 10.0)


class GlobalWhiteBalanceHead(nn.Module):
    """Predict a weak global white-balance gain from global image statistics."""

    def __init__(self, hidden_dim: int = 32, max_log_shift: float = 0.12):
        super().__init__()
        self.max_log_shift = max(float(max_log_shift), 0.0)
        self.net = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        L_gray: torch.Tensor,
        rgb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            L_gray: [B, 1, H, W]
            rgb: [B, 3, H, W]
        Returns:
            gain: [B, 3, 1, 1] with geometric mean 1
            log_gain: [B, 3, 1, 1]
        """
        rgb_mean = rgb.mean(dim=(2, 3))
        rgb_std = rgb.flatten(2).std(dim=2, unbiased=False)
        lum_mean = L_gray.mean(dim=(2, 3))
        lum_std = L_gray.flatten(2).std(dim=2, unbiased=False)
        stats = torch.cat([rgb_mean, rgb_std, lum_mean, lum_std], dim=1)

        centered = self.net(stats)
        centered = centered - centered.mean(dim=1, keepdim=True)
        log_gain = torch.tanh(centered) * self.max_log_shift
        gain = torch.exp(log_gain).view(rgb.size(0), 3, 1, 1)
        log_gain = log_gain.view(rgb.size(0), 3, 1, 1)
        return gain, log_gain


class IlluminationConditionClassifier(nn.Module):
    def __init__(self, feat_dim: int = 256, num_conditions: int = 4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_conditions),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


@torch.no_grad()
def auto_label_illumination(images: torch.Tensor) -> torch.Tensor:
    """

    Args:
        images: [B, 3, H, W] in [0, 1]
    Returns:
        labels: [B] long, 0=dark, 1=normal, 2=bright, 3=uneven
    """
    B = images.size(0)
    device = images.device
    lum = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
    mean_lum = lum.view(B, -1).mean(dim=1)

    lum_4d = lum.unsqueeze(1)
    patch_means = F.adaptive_avg_pool2d(lum_4d, 8).view(B, -1)
    unevenness = patch_means.std(dim=1)

    labels = torch.ones(B, dtype=torch.long, device=device)  # default: normal
    labels[mean_lum < 0.3] = 0   # dark
    labels[mean_lum > 0.7] = 2   # bright
    labels[unevenness > 0.2] = 3  # uneven (overrides)
    return labels


class HomoscedasticUncertaintyWeighting(nn.Module):
    def __init__(self, num_tasks: int = 4):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: List[torch.Tensor]
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = torch.tensor(0.0, device=losses[0].device)
        info = {}
        task_names = ['reid', 'illum', 'iicl', 'adv']
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
            name = task_names[i] if i < len(task_names) else f'task_{i}'
            info[f'w_{name}'] = precision.item()
        return total, info


# ============================================================================
# ============================================================================

class ConsistencyRefiner(nn.Module):
    
    def __init__(self, base_channels: int = 32, num_iterations: int = 1):
        super().__init__()
        self.num_iterations = num_iterations
        
        self.refine_net = nn.Sequential(
            ConvBNReLU(7, base_channels, 3, 1, 1),
            DepthwiseSeparableConv(base_channels, base_channels, 3, 1, 1),
            DepthwiseSeparableConv(base_channels, base_channels // 2, 3, 1, 1),
            nn.Conv2d(base_channels // 2, 3, 3, 1, 1),
            nn.Tanh()
        )
        
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, reflectance: torch.Tensor, illumination: torch.Tensor, 
                original: torch.Tensor) -> torch.Tensor:
        """
        Args:
        
        Returns:
        """
        R = reflectance
        L = illumination
        I = original
        
        for _ in range(self.num_iterations):
            residual_input = torch.cat([I, R, L], dim=1)
            
            correction = self.refine_net(residual_input)
            
            R = R + self.residual_weight * correction
            R = torch.clamp(R, 0.01, 0.99)
        
        return R


# ============================================================================
# ============================================================================

class AdaptiveColorTolerance(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(8)
        self.net = nn.Sequential(
            nn.Linear(4 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, illumination: torch.Tensor) -> torch.Tensor:
        """
        Args:
        Returns:
        """
        combined = torch.cat([x, illumination], dim=1)  # [B, 4, H, W]
        pooled = self.pool(combined)                     # [B, 4, 8, 8]
        flat = pooled.flatten(1)                         # [B, 256]
        raw = self.net(flat)                             # [B, 1]

        return 0.02 + 0.48 * torch.sigmoid(raw)


class TaskAwareRollbackGate(nn.Module):
    """Predict a per-image or per-stripe rollback weight from correction statistics."""

    def __init__(
        self,
        feat_channels: int = 384,
        hidden_dim: int = 64,
        min_alpha: float = 0.05,
        max_alpha: float = 0.98,
        granularity: str = "global",
        num_stripes: int = 6,
    ):
        super().__init__()
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.granularity = str(granularity).strip().lower()
        if self.granularity not in {"global", "stripe"}:
            self.granularity = "global"
        self.num_stripes = max(int(num_stripes), 1)
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_channels, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.stat_proj = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.fusion[-1].bias, 1.2)

    def forward(
        self,
        original: torch.Tensor,
        corrected: torch.Tensor,
        illumination: torch.Tensor,
        color_risk: torch.Tensor,
        lambda_color: torch.Tensor,
        correction_gap: torch.Tensor,
        feat_mid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = original.shape[0]
        device = original.device

        lum = 0.299 * original[:, 0:1] + 0.587 * original[:, 1:2] + 0.114 * original[:, 2:3]
        if self.granularity == "stripe":
            illum_scalar = illumination.mean(dim=1, keepdim=True)
            illum_mean = _stripe_mean_map(illum_scalar, self.num_stripes)
            illum_std = _stripe_std_map(illum_scalar, self.num_stripes)
            color_mean = _stripe_mean_map(color_risk, self.num_stripes)
            corr_mean = _stripe_mean_map(correction_gap, self.num_stripes)
            lum_std = _stripe_std_map(lum, self.num_stripes)
            lambda_color = lambda_color.view(B, 1).expand(-1, self.num_stripes)

            stat_vec = torch.stack(
                [illum_mean, illum_std, color_mean, lambda_color, corr_mean, lum_std],
                dim=-1,
            )
            stat_desc = self.stat_proj(stat_vec.view(B * self.num_stripes, -1))

            if feat_mid is not None:
                feat_vec = _stripe_feature_pool(torch.abs(feat_mid), self.num_stripes)
                feat_desc = self.feat_proj(feat_vec.view(B * self.num_stripes, -1))
            else:
                feat_desc = torch.zeros_like(stat_desc, device=device)

            alpha = torch.sigmoid(self.fusion(torch.cat([feat_desc, stat_desc], dim=1)))
            alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * alpha
            return alpha.view(B, self.num_stripes, 1, 1)

        illum_mean = illumination.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        illum_std = illumination.flatten(1).std(dim=1, unbiased=False).unsqueeze(1)
        color_mean = color_risk.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        corr_mean = correction_gap.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        lum_std = lum.flatten(1).std(dim=1, unbiased=False).unsqueeze(1)
        lambda_color = lambda_color.view(B, 1)
        stat_vec = torch.cat(
            [illum_mean, illum_std, color_mean, lambda_color, corr_mean, lum_std], dim=1
        )
        stat_desc = self.stat_proj(stat_vec)

        if feat_mid is not None:
            feat_vec = F.adaptive_avg_pool2d(torch.abs(feat_mid), 1).flatten(1)
            feat_desc = self.feat_proj(feat_vec)
        else:
            feat_desc = torch.zeros(B, stat_desc.shape[1], device=device, dtype=stat_desc.dtype)

        alpha = torch.sigmoid(self.fusion(torch.cat([feat_desc, stat_desc], dim=1)))
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * alpha
        return alpha.view(B, 1, 1, 1)


class ModelAwareReflectanceResidual(nn.Module):
    """Learn a model-aware residual on top of the Retinex base reflectance."""

    def __init__(
        self,
        feat_channels: int = 384,
        hidden_dim: int = 64,
        residual_scale: float = 0.15,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.image_proj = nn.Sequential(
            ConvBNReLU(7, self.hidden_dim, 3, 1, 1),
            DepthwiseSeparableConv(self.hidden_dim, self.hidden_dim, 3, 1, 1),
        )
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feat_channels, self.hidden_dim, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            DepthwiseSeparableConv(self.hidden_dim * 2, self.hidden_dim, 3, 1, 1),
            ConvBNReLU(self.hidden_dim, self.hidden_dim, 3, 1, 1),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.delta_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 3, 1, bias=True),
            nn.Tanh(),
        )

    def forward(
        self,
        original: torch.Tensor,
        reflectance: torch.Tensor,
        illumination: torch.Tensor,
        feat_mid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if illumination.shape[1] != 1:
            illumination = illumination.mean(dim=1, keepdim=True)

        if feat_mid is not None:
            feat_up = F.interpolate(
                feat_mid,
                size=reflectance.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            feat_ctx = self.feat_proj(feat_up)
        else:
            feat_ctx = reflectance.new_zeros(
                reflectance.size(0),
                self.hidden_dim,
                reflectance.size(2),
                reflectance.size(3),
            )

        image_ctx = self.image_proj(torch.cat([original, reflectance, illumination], dim=1))
        fused = self.fusion(torch.cat([image_ctx, feat_ctx], dim=1))
        gate = self.gate_head(fused)
        delta = self.delta_head(fused)
        reflectance_att = torch.clamp(
            reflectance + self.residual_scale * gate * delta,
            0.01,
            0.99,
        )
        return reflectance_att, gate, delta


# ============================================================================
# ============================================================================

class IPAIDModule(nn.Module):
    """
    Identity-Preserving Adaptive Illumination Decomposition Module (IPAID)

    Actual pipeline:
      Stage 1 (`forward_coarse`): estimate 1-channel spatial illumination and sensitivity.
      Stage 2 (`forward_refine`): optionally refine illumination with backbone features,
      expand it to an effective illumination field, and apply safe Retinex-like correction.

    The module outputs a corrected reflectance image for ReID, not a physically exact
    Retinex decomposition.
    """

    def __init__(
        self,
        base_channels: int = 32,
        num_scales: int = 3,
        refine_iterations: int = 1,
        use_sensitivity: bool = True,
        use_refinement: bool = True,
        backbone_mid_channels: int = 384,
        use_feature_guided: bool = True,
        use_color_illumination: bool = True,
        color_illumination_mode: str = 'local_rgb',
        species_type: str = 'stripe',
        safe_color_enabled: bool = True,
        max_color_shift: float = 0.08,
        safe_gain_min: float = 0.5,
        safe_gain_max: float = 1.6,
        color_risk_scale: float = 3.0,
        max_color_risk: float = 0.55,
        clamp_input_range: bool = False,
        wb_max_shift: float = 0.12,
        enable_task_aware_rollback: bool = True,
        rollback_hidden_dim: int = 64,
        rollback_min_alpha: float = 0.05,
        rollback_max_alpha: float = 0.98,
        rollback_granularity: str = "global",
        rollback_num_stripes: int = 6,
        use_model_aware_residual: bool = False,
        model_residual_hidden_dim: int = 64,
        model_residual_scale: float = 0.15,
        coarse_guidance_mode: str = "safe",
    ):
        super().__init__()

        self.use_sensitivity = use_sensitivity
        self.use_refinement = use_refinement
        self.use_feature_guided = use_feature_guided
        self.species_type = species_type
        self.clamp_input_range = bool(clamp_input_range)
        self.safe_color_enabled = bool(safe_color_enabled)
        self.color_illumination_mode = str(color_illumination_mode).strip().lower()
        self.max_color_shift = max(float(max_color_shift), 0.0)
        self.safe_gain_min = float(safe_gain_min)
        self.safe_gain_max = float(safe_gain_max)
        if self.safe_gain_min > self.safe_gain_max:
            self.safe_gain_min, self.safe_gain_max = self.safe_gain_max, self.safe_gain_min
        self.color_risk_scale = max(float(color_risk_scale), 0.0)
        self.max_color_risk = float(min(max(max_color_risk, 0.0), 1.0))
        self.enable_task_aware_rollback = bool(enable_task_aware_rollback)
        self.rollback_granularity = str(rollback_granularity).strip().lower()
        if self.rollback_granularity not in {"global", "stripe"}:
            self.rollback_granularity = "global"
        self.rollback_num_stripes = max(int(rollback_num_stripes), 1)
        self.use_model_aware_residual = bool(use_model_aware_residual)
        self.coarse_guidance_mode = str(coarse_guidance_mode).strip().lower()

        if species_type == 'spot':
            self.use_color_illumination = False
            self.color_illumination_mode = 'none'
            print(f"[IPAID] Species type '{species_type}': Color illumination DISABLED")
        else:
            self.use_color_illumination = bool(use_color_illumination)
            if not self.use_color_illumination:
                self.color_illumination_mode = 'none'
            elif self.color_illumination_mode not in {'local_rgb', 'global_white_balance'}:
                self.color_illumination_mode = 'global_white_balance'

        self.illumination_estimator = MultiScaleIlluminationEstimator(
            in_channels=1,
            base_channels=base_channels,
            num_scales=num_scales
        )

        if use_sensitivity:
            self.sensitivity_estimator = SensitivityEstimator(
                in_channels=3,
                base_channels=base_channels,
                species_type=species_type
            )

        if use_refinement:
            self.refiner = ConsistencyRefiner(
                base_channels=base_channels,
                num_iterations=refine_iterations
            )

        self.color_tolerance_net = AdaptiveColorTolerance()

        self.illumination_smooth = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)

        if use_feature_guided:
            self.feature_guided = FeatureGuidedRefinement(
                illum_channels=1,
                feat_channels=backbone_mid_channels,
            )

        # 7. Optional color-aware illumination expansion.
        self.color_head = None
        if self.use_color_illumination:
            if self.color_illumination_mode == 'local_rgb':
                self.color_head = LocalRGBIlluminationHead()
            else:
                self.color_head = GlobalWhiteBalanceHead(
                    hidden_dim=max(base_channels, 16),
                    max_log_shift=wb_max_shift,
                )

        self.rollback_gate = TaskAwareRollbackGate(
            feat_channels=backbone_mid_channels,
            hidden_dim=rollback_hidden_dim,
            min_alpha=rollback_min_alpha,
            max_alpha=rollback_max_alpha,
            granularity=self.rollback_granularity,
            num_stripes=self.rollback_num_stripes,
        )
        self.model_aware_residual = None
        if self.use_model_aware_residual:
            self.model_aware_residual = ModelAwareReflectanceResidual(
                feat_channels=backbone_mid_channels,
                hidden_dim=model_residual_hidden_dim,
                residual_scale=model_residual_scale,
            )

        self.cached_illumination = None
        self.cached_reconstruction_illumination = None
        self.cached_sensitivity = None
        self.cached_reflectance_init = None
        self.cached_reflectance_base = None
        self.cached_reflectance_att = None
        self.cached_color_risk = None
        self.cached_lambda_color = None
        self.cached_wb_gain = None
        self.cached_rollback_alpha = None
        self.cached_rollback_alpha_map = None
        self.cached_correction_gap = None
        self.cached_model_residual_gate = None
        self.cached_model_residual_delta = None

        self._init_weights()
        with torch.no_grad():
            self.illumination_smooth.weight.fill_(1.0 / 25.0)

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def rgb_to_luminance(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def rgb_to_chromaticity(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        eps = 1e-6
        norm = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + eps)
        return x / norm

    def _prepare_refine_input(self, x: torch.Tensor) -> torch.Tensor:
        """Optionally clamp the refinement branch into a physical RGB range."""
        if self.clamp_input_range:
            return torch.clamp(x, 0.0, 1.0)
        return x

    def _expand_illumination(self, illumination: torch.Tensor, channels: int = 3) -> torch.Tensor:
        if illumination.shape[1] == channels:
            return illumination
        return illumination.expand(-1, channels, -1, -1)

    def _sanitize_white_balance(self, wb_gain: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        if self.safe_color_enabled:
            wb_gain = torch.clamp(wb_gain, self.safe_gain_min, self.safe_gain_max)
            gray_gain = wb_gain.mean(dim=1, keepdim=True)
            delta = torch.clamp(wb_gain - gray_gain, -self.max_color_shift, self.max_color_shift)
            wb_gain = gray_gain + delta
        else:
            wb_gain = torch.clamp(wb_gain, 0.8, 1.25)

        geom_mean = torch.exp(torch.mean(torch.log(wb_gain + eps), dim=1, keepdim=True))
        wb_gain = wb_gain / geom_mean
        return torch.clamp(wb_gain, 0.8, 1.25)

    def compose_effective_illumination(
        self,
        spatial_illumination: torch.Tensor,
        wb_gain: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Combine spatial illumination with optional white-balance gain into a 3-channel field."""
        eps = 1e-6
        effective = self._expand_illumination(spatial_illumination)
        if wb_gain is not None:
            effective = effective / (wb_gain + eps)
        return torch.clamp(effective, 0.1, 10.0)

    def _expand_rollback_alpha(
        self,
        rollback_alpha: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Expand rollback weights into an image-space alpha map."""
        if rollback_alpha.dim() != 4:
            raise ValueError(f"Expected rollback alpha [B, C, 1, 1], got {tuple(rollback_alpha.shape)}")
        if rollback_alpha.size(1) == 1:
            return rollback_alpha.expand(-1, -1, height, width)

        alpha_map = rollback_alpha.new_empty(rollback_alpha.size(0), 1, height, width)
        for stripe_idx, (start, end) in enumerate(_stripe_ranges(height, rollback_alpha.size(1))):
            alpha_map[:, :, start:end, :] = rollback_alpha[:, stripe_idx : stripe_idx + 1]
        return alpha_map

    def compute_coarse_guidance_reflectance(
        self,
        x: torch.Tensor,
        coarse_out: Dict[str, torch.Tensor],
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Build a safer coarse guidance image for early backbone features."""
        mode = (mode or self.coarse_guidance_mode).strip().lower()
        x_phys = self._prepare_refine_input(x)
        L_coarse = coarse_out["L_coarse"]

        if mode == "divide":
            reflectance = x_phys / (L_coarse + 1e-4)
            return torch.clamp(reflectance, 0.01, 0.99)

        reflectance = self.apply_safe_illumination_correction(
            x_phys,
            L_coarse,
            apply_color_protection=True,
        )
        return torch.clamp(reflectance, 0.01, 0.99)

    def apply_safe_illumination_correction(
        self,
        x: torch.Tensor,
        effective_illumination: torch.Tensor,
        apply_color_protection: bool = True,
        return_stats: bool = False,
    ) -> torch.Tensor:
        """
        Safe task-oriented illumination correction.

        The implementation applies inverse illumination scaling with `soft_min`,
        then mixes back toward the original RGB direction when chromaticity drift
        becomes too large. This is a constrained Retinex-like correction, not a
        full Lab-space decomposition.

        Args:
            x: RGB image [B, 3, H, W], expected in [0, 1].
            effective_illumination: illumination field [B, 1|3, H, W].

        Returns:
            Corrected reflectance-like RGB image [B, 3, H, W].
        """
        eps = 1e-6
        effective_illumination = self._expand_illumination(effective_illumination)
        desired_scale = 1.0 / (effective_illumination + eps)
        if self.safe_color_enabled:
            desired_scale = torch.clamp(desired_scale, self.safe_gain_min, self.safe_gain_max)
            if effective_illumination.shape[1] == 3:
                gray_scale = desired_scale.mean(dim=1, keepdim=True)
                channel_delta = torch.clamp(
                    desired_scale - gray_scale,
                    -self.max_color_shift,
                    self.max_color_shift,
                )
                desired_scale = gray_scale + channel_delta
        else:
            desired_scale = torch.clamp(desired_scale, 0.33, 3.0)

        safe_scale = 0.99 / (x + eps)  # [B, 3, H, W]

        scale = soft_min(desired_scale, safe_scale, tau=10.0)
        scale = torch.clamp(scale, min=0.01)

        reflectance_raw = x * scale

        chrom_orig = self.rgb_to_chromaticity(x)
        chrom_corrected = self.rgb_to_chromaticity(reflectance_raw + eps)

        color_diff = torch.sum((chrom_orig - chrom_corrected) ** 2, dim=1, keepdim=True)

        if self.species_type == 'spot':
            protection = torch.exp(-color_diff * 20)
        else:
            protection = torch.exp(-color_diff * 10)

        if apply_color_protection:
            reflectance = protection * reflectance_raw + (1 - protection) * x
        else:
            reflectance = reflectance_raw

        if return_stats:
            return reflectance, {
                "color_diff": color_diff,
                "protection": protection,
                "scale": scale,
                "desired_scale": desired_scale,
                "correction_gap": torch.mean(torch.abs(reflectance_raw - x), dim=1, keepdim=True),
            }

        return reflectance
    
    def forward_coarse(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Stage 1: Coarse illumination estimation (pixel-level, no backbone dependency)

        Args:
            x: RGB image [B, 3, H, W], range [0, 1]
        Returns:
            dict: L_coarse, sensitivity, lambda_color, illumination_adaptive
        """
        # Clamp input to valid range without warning (normalized inputs may exceed [0,1])
        x = torch.clamp(x, 0.0, 1.0)

        luminance = self.rgb_to_luminance(x)
        illumination = self.illumination_estimator(luminance)

        if self.use_sensitivity:
            sensitivity = self.sensitivity_estimator(x)
            illumination_adaptive = 1.0 + (illumination - 1.0) * sensitivity
        else:
            sensitivity = None
            illumination_adaptive = illumination

        illumination_adaptive = torch.clamp(illumination_adaptive, 0.2, 3.0)
        illumination_adaptive = self.illumination_smooth(illumination_adaptive)

        lambda_color = self.color_tolerance_net(x, illumination_adaptive)

        return {
            'L_coarse': illumination_adaptive,
            'sensitivity': sensitivity,
            'lambda_color': lambda_color,
        }

    def forward_refine(
        self,
        x: torch.Tensor,
        coarse_out: Dict[str, torch.Tensor],
        feat_mid: Optional[torch.Tensor] = None,
        identity_protection_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 2: feature-guided illumination refinement and safe image correction.

        Args:
            x: original RGB image [B, 3, H, W]
            coarse_out: output from `forward_coarse`
            feat_mid: optional backbone mid-level feature map [B, C_mid, h, w]
        Returns:
            Detail dictionary used by the ReID branch and IPAID losses.
        """
        L = coarse_out['L_coarse']
        sensitivity = coarse_out['sensitivity']
        lambda_color = coarse_out['lambda_color']
        x_phys = self._prepare_refine_input(x)
        eps = 1e-4
        wb_gain = None
        wb_log_gain = None

        if self.use_feature_guided and feat_mid is not None:
            L = self.feature_guided(L, feat_mid)

        # Expand scalar illumination into an effective correction field when enabled.
        if self.use_color_illumination and self.color_head is not None:
            if self.color_illumination_mode == 'local_rgb':
                effective_illumination = self.color_head(L, x_phys)  # [B, 3, H, W]
            else:
                wb_gain, wb_log_gain = self.color_head(L, x_phys)
                wb_gain = self._sanitize_white_balance(wb_gain)
                effective_illumination = self.compose_effective_illumination(L, wb_gain=wb_gain)
        else:
            effective_illumination = L  # [B, 1, H, W]

        self.cached_illumination = L
        self.cached_reconstruction_illumination = effective_illumination
        self.cached_sensitivity = sensitivity
        self.cached_lambda_color = lambda_color
        self.cached_wb_gain = wb_gain

        # Safe Retinex-like correction.
        use_color_protection = not self.enable_task_aware_rollback
        reflectance_retinex, correction_stats = self.apply_safe_illumination_correction(
            x_phys,
            effective_illumination,
            apply_color_protection=use_color_protection,
            return_stats=True,
        )
        reflectance_base = reflectance_retinex
        reflectance_att = reflectance_base
        illum_for_residual = effective_illumination
        if illum_for_residual.shape[1] != 1:
            illum_for_residual = illum_for_residual.mean(dim=1, keepdim=True)
        model_residual_gate = reflectance_base.new_zeros(
            reflectance_base.size(0),
            1,
            reflectance_base.size(2),
            reflectance_base.size(3),
        )
        model_residual_delta = torch.zeros_like(reflectance_base)
        if self.use_model_aware_residual and self.model_aware_residual is not None:
            reflectance_att, model_residual_gate, model_residual_delta = self.model_aware_residual(
                original=x_phys,
                reflectance=reflectance_base,
                illumination=illum_for_residual,
                feat_mid=feat_mid,
            )

        if identity_protection_map is not None:
            identity_protection_map = identity_protection_map.detach().to(
                device=x_phys.device,
                dtype=x_phys.dtype,
            )
            if identity_protection_map.shape[1] != 1:
                identity_protection_map = identity_protection_map.mean(dim=1, keepdim=True)
            if identity_protection_map.shape[-2:] != x_phys.shape[-2:]:
                identity_protection_map = F.interpolate(
                    identity_protection_map,
                    size=x_phys.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            identity_protection_map = identity_protection_map.clamp(0.0, 1.0)
            reflectance_att = x_phys + (1.0 - identity_protection_map) * (reflectance_att - x_phys)

        chrom_orig = self.rgb_to_chromaticity(x_phys)
        chrom_retinex = self.rgb_to_chromaticity(reflectance_att + eps)
        color_angle_diff = 1.0 - torch.sum(chrom_orig * chrom_retinex, dim=1, keepdim=True)
        color_risk = torch.clamp(
            color_angle_diff * self.color_risk_scale,
            0.0,
            self.max_color_risk,
        )
        self.cached_color_risk = color_risk
        correction_gap = torch.mean(torch.abs(reflectance_att - x_phys), dim=1, keepdim=True)
        self.cached_correction_gap = correction_gap

        if self.enable_task_aware_rollback:
            rollback_alpha = self.rollback_gate(
                original=x_phys,
                corrected=reflectance_att,
                illumination=effective_illumination,
                color_risk=color_risk,
                lambda_color=lambda_color,
                correction_gap=correction_gap,
                feat_mid=feat_mid,
            )
            rollback_alpha_map = self._expand_rollback_alpha(
                rollback_alpha,
                height=x_phys.shape[2],
                width=x_phys.shape[3],
            )
            reflectance = rollback_alpha_map * reflectance_att + (1.0 - rollback_alpha_map) * x_phys
        else:
            rollback_alpha = 1.0 - color_risk.mean(dim=(1, 2, 3), keepdim=True)
            rollback_alpha_map = rollback_alpha.expand(-1, -1, x_phys.shape[2], x_phys.shape[3])
            reflectance = (1 - color_risk) * reflectance_att + color_risk * x_phys

        self.cached_rollback_alpha = rollback_alpha
        self.cached_rollback_alpha_map = rollback_alpha_map
        self.cached_reflectance_init = reflectance
        self.cached_reflectance_base = reflectance_base
        self.cached_reflectance_att = reflectance_att
        self.cached_model_residual_gate = model_residual_gate
        self.cached_model_residual_delta = model_residual_delta

        if self.use_refinement:
            reflectance = self.refiner(reflectance, L, x_phys)

        if identity_protection_map is not None:
            reflectance = x_phys + (1.0 - identity_protection_map) * (reflectance - x_phys)

        reflectance = torch.clamp(reflectance, 0.01, 0.99)

        return {
            'reflectance': reflectance,
            'reflectance_base': reflectance_base,
            'reflectance_att': reflectance_att,
            'illumination': L,
            'effective_illumination': effective_illumination,
            'reconstruction_illumination': effective_illumination,
            'sensitivity': sensitivity,
            'reflectance_init': self.cached_reflectance_init,
            'original': x_phys,
            'color_risk': color_risk,
            'rollback_alpha': rollback_alpha,
            'rollback_alpha_map': rollback_alpha_map,
            'correction_gap': correction_gap,
            'lambda_color': lambda_color,
            'wb_gain': wb_gain,
            'wb_log_gain': wb_log_gain,
            'model_residual_gate': model_residual_gate,
            'model_residual_delta': model_residual_delta,
            'identity_protection_map': identity_protection_map,
        }

    def forward(self, x: torch.Tensor, feat_mid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """

        Args:
        Returns:
            reflectance [B, 3, H, W]
        """
        coarse_out = self.forward_coarse(x)
        details = self.forward_refine(x, coarse_out, feat_mid)
        return details['reflectance']

    def forward_with_details(self, x: torch.Tensor,
                             feat_mid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """"""
        coarse_out = self.forward_coarse(x)
        return self.forward_refine(x, coarse_out, feat_mid)

    def get_reconstruction(self) -> Optional[torch.Tensor]:
        """"""
        if self.cached_reconstruction_illumination is None or self.cached_reflectance_init is None:
            return None
        return self.cached_reflectance_init * self.cached_reconstruction_illumination


# ============================================================================
# ============================================================================

class IPAIDLoss(nn.Module):
    
    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_smooth: float = 0.1,
        lambda_edge: float = 0.05,
        lambda_structure: Optional[float] = None,
        lambda_sensitivity: float = 0.01,
        lambda_lab_chroma: float = 0.1,
        lambda_high_freq: float = 0.05,
        lambda_log_chroma: float = 0.0,
        chroma_mode: str = "dual",
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_smooth = lambda_smooth
        self.lambda_edge = lambda_edge
        self.lambda_structure = lambda_edge if lambda_structure is None else float(lambda_structure)
        self.lambda_sensitivity = lambda_sensitivity
        self.lambda_lab_chroma = lambda_lab_chroma
        self.lambda_high_freq = lambda_high_freq
        self.lambda_log_chroma = lambda_log_chroma
        self.chroma_mode = str(chroma_mode).strip().lower()
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def compute_gradient(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        return grad_x, grad_y

    def compute_structure_loss(self, corrected: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """Preserve structural gradients instead of blindly amplifying all edges."""
        grad_corr_x, grad_corr_y = self.compute_gradient(corrected)
        grad_orig_x, grad_orig_y = self.compute_gradient(original)
        weight_x = 1.0 - torch.exp(-torch.abs(grad_orig_x) * 5.0)
        weight_y = 1.0 - torch.exp(-torch.abs(grad_orig_y) * 5.0)
        loss_x = torch.mean(weight_x * torch.abs(grad_corr_x - grad_orig_x))
        loss_y = torch.mean(weight_y * torch.abs(grad_corr_y - grad_orig_y))
        return loss_x + loss_y


    def compute_lab_chroma_loss(self, corrected: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """"""
        def rgb_to_lab(rgb):
            rgb = torch.clamp(rgb, 0.0, 1.0)
            r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            r = torch.where(r > 0.04045, ((r + 0.055) / 1.055) ** 2.4, r / 12.92)
            g = torch.where(g > 0.04045, ((g + 0.055) / 1.055) ** 2.4, g / 12.92)
            b = torch.where(b > 0.04045, ((b + 0.055) / 1.055) ** 2.4, b / 12.92)
            x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
            y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
            z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
            x, y, z = x / 0.95047, y / 1.0, z / 1.08883
            def f(t):
                delta = 6.0 / 29.0
                return torch.where(t > delta**3, t**(1/3), t / (3 * delta**2) + 4/29)
            fx, fy, fz = f(x), f(y), f(z)
            L = 116 * fy - 16
            a = 500 * (fx - fy)
            b = 200 * (fy - fz)
            return torch.stack([L, a, b], dim=1)
        lab_corr = rgb_to_lab(corrected)
        lab_orig = rgb_to_lab(original)
        loss_a = F.mse_loss(lab_corr[:, 1] / 255.0, lab_orig[:, 1] / 255.0)
        loss_b = F.mse_loss(lab_corr[:, 2] / 255.0, lab_orig[:, 2] / 255.0)
        return loss_a + loss_b

    def compute_high_freq_loss(self, corrected: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """"""
        kernel_size = 5
        sigma = kernel_size / 6.0
        kernel = torch.zeros(1, 1, kernel_size, kernel_size, device=corrected.device)
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[0, 0, i, j] = torch.exp(torch.tensor(-(x**2 + y**2) / (2 * sigma**2), device=corrected.device))
        kernel = kernel / kernel.sum()
        def extract_high_freq(img):
            padding = kernel_size // 2
            low_freq = torch.cat([F.conv2d(img[:, i:i+1], kernel, padding=padding) for i in range(img.shape[1])], dim=1)
            return img - low_freq
        return F.l1_loss(extract_high_freq(corrected), extract_high_freq(original))

    def compute_log_chroma_loss(self, corrected: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """Log-chromaticity consistency to suppress color drift while allowing luminance changes."""
        eps = 1e-6

        def log_chroma(img: torch.Tensor) -> torch.Tensor:
            img = torch.clamp(img, min=eps)
            log_img = torch.log(img)
            rg = log_img[:, 0:1] - log_img[:, 1:2]
            bg = log_img[:, 2:3] - log_img[:, 1:2]
            return torch.cat([rg, bg], dim=1)

        return F.smooth_l1_loss(log_chroma(corrected), log_chroma(original))

    def forward(
        self,
        details: Dict[str, torch.Tensor],
        ipaid_module: Optional[IPAIDModule] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        
        Args:
        
        Returns:
        """
        R = details['reflectance']
        L = details['illumination']
        L_recon = details.get('reconstruction_illumination', details.get('effective_illumination', L))
        I = details['original']
        S = details.get('sensitivity')
        
        device = R.device
        
        I_recon = R * L_recon
        loss_recon = F.l1_loss(I_recon, I)
        
        grad_L_x, grad_L_y = self.compute_gradient(L)
        grad_I_x, grad_I_y = self.compute_gradient(I)
        
        weight_x = torch.exp(-torch.abs(grad_I_x) * 10)
        weight_y = torch.exp(-torch.abs(grad_I_y) * 10)
        
        loss_smooth = torch.mean(weight_x * grad_L_x ** 2) + \
                      torch.mean(weight_y * grad_L_y ** 2)
        
        # 3. Structural consistency on reflectance gradients.
        loss_structure = self.compute_structure_loss(R, I)
        
        if S is not None:
            loss_sensitivity = torch.mean(S)
        else:
            loss_sensitivity = torch.tensor(0.0, device=device)
        
        eps = 1e-6
        chrom_orig = I / (torch.sqrt(torch.sum(I ** 2, dim=1, keepdim=True)) + eps)
        chrom_refl = R / (torch.sqrt(torch.sum(R ** 2, dim=1, keepdim=True)) + eps)
        color_consistency = 1.0 - torch.sum(chrom_orig * chrom_refl, dim=1, keepdim=True)

        lambda_color_per_image = details.get('lambda_color', None)
        if lambda_color_per_image is not None:
            loss_color_per_image = color_consistency.mean(dim=[1, 2, 3])  # [B]
            lambda_color_squeezed = lambda_color_per_image.squeeze(1)     # [B]
            loss_color = torch.mean(lambda_color_squeezed * loss_color_per_image)
            lambda_color_mean = lambda_color_squeezed.mean().item()
        else:
            loss_color = torch.mean(color_consistency) * 0.1
            lambda_color_mean = 0.1

        # 6. Lab/log chroma consistency.
        if self.chroma_mode in {"dual", "lab"}:
            loss_lab_chroma = self.compute_lab_chroma_loss(R, I)
        else:
            loss_lab_chroma = torch.tensor(0.0, device=device)

        loss_high_freq = self.compute_high_freq_loss(R, I)

        # 8. Log-chromaticity consistency
        if self.chroma_mode in {"dual", "log", "log_chroma"}:
            loss_log_chroma = self.compute_log_chroma_loss(R, I)
        else:
            loss_log_chroma = torch.tensor(0.0, device=device)

        total_loss = (
            self.lambda_recon * loss_recon +
            self.lambda_smooth * loss_smooth +
            self.lambda_structure * loss_structure +
            self.lambda_sensitivity * loss_sensitivity +
            loss_color
         +
            self.lambda_lab_chroma * loss_lab_chroma +
            self.lambda_high_freq * loss_high_freq +
            self.lambda_log_chroma * loss_log_chroma
        )

        loss_dict = {
            'loss_recon': loss_recon,
            'loss_smooth': loss_smooth,
            'loss_edge': loss_structure,
            'loss_structure': loss_structure,
            'loss_sensitivity': loss_sensitivity,
            'loss_color': loss_color,
            'lambda_color_mean': lambda_color_mean,
            'loss_lab_chroma': loss_lab_chroma.item() if isinstance(loss_lab_chroma, torch.Tensor) else 0.0,
            'loss_high_freq': loss_high_freq.item() if isinstance(loss_high_freq, torch.Tensor) else 0.0,
            'loss_log_chroma': loss_log_chroma.item() if isinstance(loss_log_chroma, torch.Tensor) else 0.0,
            'total': total_loss,
        }

        return total_loss, loss_dict


# ============================================================================
# ============================================================================

class IdentityPreservingLoss(nn.Module):
    
    def __init__(self, margin: float = 0.3, mining: str = 'hard'):
        super().__init__()
        self.margin = margin
        self.mining = mining
    
    def forward(
        self,
        reflectance_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        
        Args:
        
        Returns:
        """
        B = reflectance_features.shape[0]
        
        features = F.normalize(reflectance_features, p=2, dim=1)
        
        dist_mat = torch.cdist(features, features, p=2)  # [B, B]
        
        labels = labels.view(-1, 1)
        is_same_id = (labels == labels.T).float()  # [B, B]
        
        # Hard mining
        loss = torch.tensor(0.0, device=features.device)
        valid_triplets = 0
        
        for i in range(B):
            pos_mask = is_same_id[i].bool()
            pos_mask[i] = False
            
            neg_mask = ~is_same_id[i].bool()
            
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            
            pos_dists = dist_mat[i][pos_mask]
            hardest_pos = pos_dists.max()
            
            neg_dists = dist_mat[i][neg_mask]
            hardest_neg = neg_dists.min()
            
            # Triplet loss
            triplet_loss = F.relu(hardest_pos - hardest_neg + self.margin)
            loss = loss + triplet_loss
            valid_triplets += 1
        
        if valid_triplets > 0:
            loss = loss / valid_triplets
        
        return loss


class TeacherAnchorLoss(nn.Module):
    """Anchor adapted descriptors to raw-reference teacher descriptors."""

    def __init__(self, metric: str = "cosine", eps: float = 1e-12):
        super().__init__()
        self.metric = metric
        self.eps = eps

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        if student_features.numel() == 0 or teacher_features.numel() == 0:
            return student_features.new_tensor(0.0)

        if self.metric == "cosine":
            student = F.normalize(student_features.float(), p=2, dim=1, eps=self.eps)
            teacher = F.normalize(teacher_features.float(), p=2, dim=1, eps=self.eps)
            return (1.0 - (student * teacher).sum(dim=1)).mean()
        if self.metric == "mse":
            return F.mse_loss(student_features, teacher_features)
        raise ValueError(f"Unsupported teacher-anchor metric: {self.metric}")


class TeacherPrototypeAnchorLoss(nn.Module):
    """Anchor adapted descriptors to same-identity teacher class prototypes."""

    def __init__(self, metric: str = "cosine", eps: float = 1e-12):
        super().__init__()
        self.metric = metric
        self.eps = eps

    def _build_prototypes(
        self,
        teacher_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        prototypes = []
        for idx in range(teacher_features.size(0)):
            same_mask = labels == labels[idx]
            prototype = teacher_features[same_mask].mean(dim=0, keepdim=True)
            prototypes.append(prototype)
        return torch.cat(prototypes, dim=0)

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if student_features.numel() == 0 or teacher_features.numel() == 0:
            return student_features.new_tensor(0.0)

        labels = labels.view(-1)
        if labels.numel() != teacher_features.size(0):
            raise ValueError(
                f"Label count {labels.numel()} does not match feature batch {teacher_features.size(0)}"
            )

        teacher_proto = self._build_prototypes(teacher_features.float(), labels)
        if self.metric == "cosine":
            student = F.normalize(student_features.float(), p=2, dim=1, eps=self.eps)
            teacher_proto = F.normalize(teacher_proto, p=2, dim=1, eps=self.eps)
            return (1.0 - (student * teacher_proto).sum(dim=1)).mean()
        if self.metric == "mse":
            return F.mse_loss(student_features.float(), teacher_proto)
        raise ValueError(f"Unsupported teacher-prototype metric: {self.metric}")


class RelativeClassStructureLoss(nn.Module):
    """Preserve teacher same-class relative geometry without absolute center anchoring."""

    def __init__(
        self,
        metric: str = "cosine",
        radial_weight: float = 0.5,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.metric = str(metric).lower()
        self.radial_weight = float(radial_weight)
        self.eps = float(eps)

    def _pairwise_structure(self, centered_features: torch.Tensor) -> torch.Tensor:
        if self.metric == "cosine":
            centered_features = F.normalize(centered_features.float(), p=2, dim=1, eps=self.eps)
            return centered_features @ centered_features.t()
        if self.metric == "mse":
            return torch.cdist(centered_features.float(), centered_features.float(), p=2)
        raise ValueError(f"Unsupported relative class structure metric: {self.metric}")

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if student_features.size(0) <= 1 or teacher_features.size(0) <= 1:
            return student_features.new_tensor(0.0)

        labels = labels.view(-1)
        sample_losses = []
        for label in labels.unique():
            class_mask = labels == label
            if int(class_mask.sum().item()) <= 1:
                continue

            student_cls = student_features[class_mask].float()
            teacher_cls = teacher_features[class_mask].float()
            student_centered = student_cls - student_cls.mean(dim=0, keepdim=True)
            teacher_centered = teacher_cls - teacher_cls.mean(dim=0, keepdim=True)

            student_structure = self._pairwise_structure(student_centered)
            teacher_structure = self._pairwise_structure(teacher_centered)
            if student_structure.size(0) > 1:
                mask = ~torch.eye(
                    student_structure.size(0),
                    dtype=torch.bool,
                    device=student_structure.device,
                )
                structure_loss = F.mse_loss(
                    student_structure[mask],
                    teacher_structure[mask],
                )
            else:
                structure_loss = student_features.new_tensor(0.0)

            student_radius = torch.norm(student_centered, dim=1, p=2)
            teacher_radius = torch.norm(teacher_centered, dim=1, p=2)
            radius_loss = F.smooth_l1_loss(student_radius, teacher_radius)
            sample_losses.append(structure_loss + self.radial_weight * radius_loss)

        if not sample_losses:
            return student_features.new_tensor(0.0)
        return torch.stack(sample_losses).mean()


class GeometryPreservingLoss(nn.Module):
    """Preserve the raw-branch pairwise geometry after illumination adaptation."""

    def __init__(
        self,
        metric: str = "cosine",
        loss_type: str = "mse",
        ignore_diagonal: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.metric = metric
        self.loss_type = loss_type
        self.ignore_diagonal = ignore_diagonal
        self.eps = eps

    def _pairwise_geometry(self, features: torch.Tensor) -> torch.Tensor:
        if self.metric == "cosine":
            normalized = F.normalize(features.float(), p=2, dim=1, eps=self.eps)
            return normalized @ normalized.t()
        if self.metric == "l2":
            return torch.cdist(features.float(), features.float(), p=2)
        raise ValueError(f"Unsupported geometry metric: {self.metric}")

    def _cross_geometry(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.metric == "cosine":
            student = F.normalize(student_features.float(), p=2, dim=1, eps=self.eps)
            teacher = F.normalize(teacher_features.float(), p=2, dim=1, eps=self.eps)
            return student @ teacher.t()
        if self.metric == "l2":
            return torch.cdist(student_features.float(), teacher_features.float(), p=2)
        raise ValueError(f"Unsupported geometry metric: {self.metric}")

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        if student_features.size(0) <= 1 or teacher_features.size(0) <= 1:
            return student_features.new_tensor(0.0)

        teacher_geom = self._pairwise_geometry(teacher_features)
        student_teacher_geom = self._cross_geometry(student_features, teacher_features)

        if self.ignore_diagonal:
            mask = ~torch.eye(teacher_geom.size(0), dtype=torch.bool, device=teacher_geom.device)
            student_teacher_geom = student_teacher_geom[mask]
            teacher_geom = teacher_geom[mask]

        if self.loss_type == "mse":
            return F.mse_loss(student_teacher_geom, teacher_geom)
        if self.loss_type == "l1":
            return F.l1_loss(student_teacher_geom, teacher_geom)
        raise ValueError(f"Unsupported geometry loss type: {self.loss_type}")


class TeacherLogitConsistencyLoss(nn.Module):
    """Distill raw-reference identity logits into the adapted branch."""

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        if student_logits.numel() == 0 or teacher_logits.numel() == 0:
            return student_logits.new_tensor(0.0)

        temperature = max(float(self.temperature), 1e-6)
        student_log_prob = F.log_softmax(student_logits.float() / temperature, dim=1)
        teacher_prob = F.softmax(teacher_logits.float() / temperature, dim=1)
        return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature ** 2)


class CrossLightPrototypeLoss(nn.Module):
    """Align same-identity samples toward photometrically distant positive prototypes."""

    def __init__(
        self,
        similarity: str = "cosine",
        photometric_scale: float = 8.0,
        photometric_offset: float = 0.1,
        min_gap_weight: float = 0.1,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.similarity = str(similarity).lower()
        self.photometric_scale = float(photometric_scale)
        self.photometric_offset = float(photometric_offset)
        self.min_gap_weight = float(min_gap_weight)
        self.eps = float(eps)

    def _pair_loss(
        self,
        features: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        if self.similarity == "cosine":
            features = F.normalize(features.float(), p=2, dim=1, eps=self.eps)
            prototypes = F.normalize(prototypes.float(), p=2, dim=1, eps=self.eps)
            return 1.0 - (features * prototypes).sum(dim=1)
        if self.similarity == "mse":
            return torch.mean((features.float() - prototypes.float()) ** 2, dim=1)
        raise ValueError(f"Unsupported prototype similarity: {self.similarity}")

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        photometric_stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if features.size(0) <= 1:
            return features.new_tensor(0.0)

        features_f = features.float()
        labels = labels.view(-1)
        if photometric_stats is None:
            photometric_stats = features_f.new_zeros(features_f.size(0), 1)
        else:
            photometric_stats = photometric_stats.float().view(features_f.size(0), -1)

        sample_losses = []
        for sample_idx in range(features_f.size(0)):
            pos_mask = labels == labels[sample_idx]
            pos_mask[sample_idx] = False
            if not torch.any(pos_mask):
                continue

            positive_features = features_f[pos_mask].detach()
            positive_stats = photometric_stats[pos_mask]
            sample_stats = photometric_stats[sample_idx : sample_idx + 1]

            photometric_gap = torch.norm(positive_stats - sample_stats, dim=1)
            gap_weight = torch.sigmoid(
                self.photometric_scale * (photometric_gap - self.photometric_offset)
            )
            gap_weight = self.min_gap_weight + (1.0 - self.min_gap_weight) * gap_weight
            gap_weight = gap_weight / gap_weight.sum().clamp_min(self.eps)

            prototype = torch.sum(gap_weight.unsqueeze(1) * positive_features, dim=0, keepdim=True)
            sample_feature = features_f[sample_idx : sample_idx + 1]
            sample_losses.append(self._pair_loss(sample_feature, prototype))

        if not sample_losses:
            return features.new_tensor(0.0)
        return torch.cat(sample_losses, dim=0).mean()


class CrossLightMarginPreservingLoss(nn.Module):
    """Preserve or improve teacher ranking margins for cross-light positives."""

    def __init__(
        self,
        similarity: str = "cosine",
        photometric_scale: float = 8.0,
        photometric_offset: float = 0.1,
        topk_positive: int = 2,
        topk_negative: int = 4,
        margin_delta: float = 0.02,
        beta: float = 12.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.similarity = str(similarity).lower()
        self.photometric_scale = float(photometric_scale)
        self.photometric_offset = float(photometric_offset)
        self.topk_positive = max(int(topk_positive), 1)
        self.topk_negative = max(int(topk_negative), 1)
        self.margin_delta = float(margin_delta)
        self.beta = max(float(beta), 1e-6)
        self.eps = float(eps)

    def _pairwise_similarity(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        if self.similarity == "cosine":
            lhs = F.normalize(lhs.float(), p=2, dim=1, eps=self.eps)
            rhs = F.normalize(rhs.float(), p=2, dim=1, eps=self.eps)
            return lhs @ rhs.t()
        if self.similarity in {"l2", "negative_l2"}:
            return -torch.cdist(lhs.float(), rhs.float(), p=2)
        raise ValueError(f"Unsupported cross-light margin similarity: {self.similarity}")

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        labels: torch.Tensor,
        photometric_stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if student_features.size(0) <= 1 or teacher_features.size(0) <= 1:
            return student_features.new_tensor(0.0)

        student = student_features.float()
        teacher = teacher_features.float()
        labels = labels.view(-1)
        if photometric_stats is None:
            photometric_stats = teacher.new_zeros(teacher.size(0), 1)
        else:
            photometric_stats = photometric_stats.float().view(teacher.size(0), -1)

        teacher_sim = self._pairwise_similarity(teacher, teacher)
        student_sim = self._pairwise_similarity(student, student)
        sample_losses = []

        for anchor_idx in range(student.size(0)):
            pos_mask = labels == labels[anchor_idx]
            pos_mask[anchor_idx] = False
            neg_mask = labels != labels[anchor_idx]
            if not torch.any(pos_mask) or not torch.any(neg_mask):
                continue

            teacher_pos = teacher_sim[anchor_idx][pos_mask]
            student_pos = student_sim[anchor_idx][pos_mask]
            teacher_neg = teacher_sim[anchor_idx][neg_mask]
            student_neg = student_sim[anchor_idx][neg_mask]

            anchor_stats = photometric_stats[anchor_idx : anchor_idx + 1]
            positive_stats = photometric_stats[pos_mask]
            photometric_gap = torch.norm(positive_stats - anchor_stats, dim=1)
            positive_scores = torch.sigmoid(
                self.photometric_scale * (photometric_gap - self.photometric_offset)
            )
            positive_scores = positive_scores + self.eps

            k_pos = min(self.topk_positive, teacher_pos.numel())
            k_neg = min(self.topk_negative, teacher_neg.numel())
            _, pos_idx = torch.topk(positive_scores, k=k_pos, largest=True)
            teacher_pos = teacher_pos[pos_idx]
            student_pos = student_pos[pos_idx]
            positive_weights = positive_scores[pos_idx]
            positive_weights = positive_weights / positive_weights.sum().clamp_min(self.eps)

            teacher_neg, neg_idx = torch.topk(teacher_neg, k=k_neg, largest=True)
            student_neg = student_neg[neg_idx]
            negative_weights = F.softmax(teacher_neg.detach(), dim=0)

            teacher_margin = teacher_pos.unsqueeze(1) - teacher_neg.unsqueeze(0)
            student_margin = student_pos.unsqueeze(1) - student_neg.unsqueeze(0)
            target_margin = self.margin_delta + torch.relu(teacher_margin.detach())
            margin_gap = target_margin - student_margin
            loss_matrix = F.softplus(self.beta * margin_gap) / self.beta
            pair_weights = positive_weights.unsqueeze(1) * negative_weights.unsqueeze(0)
            sample_losses.append(
                (pair_weights * loss_matrix).sum() / pair_weights.sum().clamp_min(self.eps)
            )

        if not sample_losses:
            return student_features.new_tensor(0.0)
        return torch.stack(sample_losses).mean()


class SoftAPCrossLightLoss(nn.Module):
    """Optimize cross-light retrieval with a differentiable listwise AP surrogate."""

    def __init__(
        self,
        similarity: str = "cosine",
        photometric_scale: float = 8.0,
        photometric_offset: float = 0.1,
        min_positive_weight: float = 0.05,
        rank_temperature: float = 0.07,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.similarity = str(similarity).lower()
        self.photometric_scale = float(photometric_scale)
        self.photometric_offset = float(photometric_offset)
        self.min_positive_weight = float(min_positive_weight)
        self.rank_temperature = max(float(rank_temperature), 1e-6)
        self.eps = float(eps)

    def _pairwise_similarity(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        if self.similarity == "cosine":
            lhs = F.normalize(lhs.float(), p=2, dim=1, eps=self.eps)
            rhs = F.normalize(rhs.float(), p=2, dim=1, eps=self.eps)
            return lhs @ rhs.t()
        if self.similarity in {"l2", "negative_l2"}:
            return -torch.cdist(lhs.float(), rhs.float(), p=2)
        raise ValueError(f"Unsupported SoftAP similarity: {self.similarity}")

    def _positive_weights(
        self,
        anchor_stats: torch.Tensor,
        gallery_stats: torch.Tensor,
    ) -> torch.Tensor:
        photometric_gap = torch.norm(gallery_stats.float() - anchor_stats.float(), dim=1)
        gap_weight = torch.sigmoid(
            self.photometric_scale * (photometric_gap - self.photometric_offset)
        )
        return self.min_positive_weight + (1.0 - self.min_positive_weight) * gap_weight

    def forward(
        self,
        anchor_features: torch.Tensor,
        anchor_labels: torch.Tensor,
        anchor_photometric_stats: torch.Tensor,
        gallery_features: torch.Tensor,
        gallery_labels: torch.Tensor,
        gallery_photometric_stats: torch.Tensor,
        same_source_size: int = 0,
    ) -> torch.Tensor:
        if anchor_features.size(0) == 0 or gallery_features.size(0) == 0:
            return anchor_features.new_tensor(0.0)

        anchor_labels = anchor_labels.view(-1)
        gallery_labels = gallery_labels.view(-1)
        if anchor_labels.numel() != anchor_features.size(0):
            raise ValueError("Anchor labels must match anchor feature batch size")
        if gallery_labels.numel() != gallery_features.size(0):
            raise ValueError("Gallery labels must match gallery feature batch size")

        similarities = self._pairwise_similarity(anchor_features, gallery_features)
        sample_losses = []
        max_same_source = min(int(same_source_size), gallery_features.size(0))

        for anchor_idx in range(anchor_features.size(0)):
            pos_mask = gallery_labels == anchor_labels[anchor_idx]
            if anchor_idx < max_same_source:
                pos_mask[anchor_idx] = False
            neg_mask = gallery_labels != anchor_labels[anchor_idx]

            if not torch.any(pos_mask) or not torch.any(neg_mask):
                continue

            pos_scores = similarities[anchor_idx][pos_mask]
            neg_scores = similarities[anchor_idx][neg_mask]
            pos_weights = self._positive_weights(
                anchor_photometric_stats[anchor_idx : anchor_idx + 1],
                gallery_photometric_stats[pos_mask],
            ).clamp_min(self.eps)

            precision_terms = []
            for pos_idx in range(pos_scores.numel()):
                pos_score = pos_scores[pos_idx : pos_idx + 1]
                other_pos = torch.cat([pos_scores[:pos_idx], pos_scores[pos_idx + 1 :]], dim=0)
                pos_rank = 1.0
                if other_pos.numel() > 0:
                    pos_rank = pos_rank + torch.sigmoid(
                        (other_pos - pos_score) / self.rank_temperature
                    ).sum()
                neg_rank = torch.sigmoid(
                    (neg_scores - pos_score) / self.rank_temperature
                ).sum()
                precision = pos_rank / (pos_rank + neg_rank + self.eps)
                precision_terms.append(precision)

            precision_terms = torch.stack(precision_terms)
            ap = (pos_weights * precision_terms).sum() / pos_weights.sum().clamp_min(self.eps)
            sample_losses.append(1.0 - ap)

        if not sample_losses:
            return anchor_features.new_tensor(0.0)
        return torch.stack(sample_losses).mean()


class _TeacherManifoldBase(nn.Module):
    """Shared utilities for teacher-manifold geometry losses."""

    def __init__(
        self,
        similarity: str = "cosine",
        photometric_scale: float = 8.0,
        photometric_offset: float = 0.1,
        min_positive_weight: float = 0.05,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.similarity = str(similarity).lower()
        self.photometric_scale = float(photometric_scale)
        self.photometric_offset = float(photometric_offset)
        self.min_positive_weight = float(min_positive_weight)
        self.eps = float(eps)

    def _prepare_features(self, features: torch.Tensor) -> torch.Tensor:
        features = features.float()
        if self.similarity == "cosine":
            return F.normalize(features, p=2, dim=1, eps=self.eps)
        if self.similarity in {"l2", "negative_l2", "mse"}:
            return features
        raise ValueError(f"Unsupported teacher-manifold similarity: {self.similarity}")

    def _pairwise_similarity(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        lhs = self._prepare_features(lhs)
        rhs = self._prepare_features(rhs)
        if self.similarity == "cosine":
            return lhs @ rhs.t()
        return -torch.cdist(lhs, rhs, p=2)

    def _positive_weights(
        self,
        anchor_stats: torch.Tensor,
        teacher_stats: torch.Tensor,
    ) -> torch.Tensor:
        photometric_gap = torch.norm(teacher_stats.float() - anchor_stats.float(), dim=1)
        gap_weight = torch.sigmoid(
            self.photometric_scale * (photometric_gap - self.photometric_offset)
        )
        return self.min_positive_weight + (1.0 - self.min_positive_weight) * gap_weight

    def _select_teacher_positive_set(
        self,
        anchor_index: int,
        anchor_label: torch.Tensor,
        anchor_stats: torch.Tensor,
        teacher_features: torch.Tensor,
        teacher_labels: torch.Tensor,
        teacher_stats: torch.Tensor,
        same_source_size: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        pos_mask = teacher_labels == anchor_label
        if anchor_index < same_source_size and anchor_index < pos_mask.numel():
            pos_mask[anchor_index] = False
        if not torch.any(pos_mask):
            return None, None, None

        positive_features = self._prepare_features(teacher_features[pos_mask]).detach()
        positive_stats = teacher_stats[pos_mask].float()
        positive_weights = self._positive_weights(anchor_stats, positive_stats).clamp_min(self.eps)
        positive_weights = positive_weights / positive_weights.sum().clamp_min(self.eps)
        return positive_features, positive_stats, positive_weights

    def _weighted_centroid(
        self,
        positive_features: torch.Tensor,
        positive_weights: torch.Tensor,
    ) -> torch.Tensor:
        centroid = torch.sum(positive_weights.unsqueeze(1) * positive_features, dim=0, keepdim=True)
        if self.similarity == "cosine":
            centroid = F.normalize(centroid, p=2, dim=1, eps=self.eps)
        return centroid


class TeacherManifoldTubeLoss(_TeacherManifoldBase):
    """Keep corrected features inside a same-ID teacher manifold tube."""

    def __init__(
        self,
        similarity: str = "cosine",
        photometric_scale: float = 8.0,
        photometric_offset: float = 0.1,
        min_positive_weight: float = 0.05,
        shrinkage: float = 0.8,
        orthogonal_weight: float = 1.0,
        subspace_rank: int = 1,
        min_radius: float = 0.02,
        eps: float = 1e-12,
    ):
        super().__init__(
            similarity=similarity,
            photometric_scale=photometric_scale,
            photometric_offset=photometric_offset,
            min_positive_weight=min_positive_weight,
            eps=eps,
        )
        self.shrinkage = float(shrinkage)
        self.orthogonal_weight = float(orthogonal_weight)
        self.subspace_rank = max(int(subspace_rank), 0)
        self.min_radius = float(min_radius)

    def _teacher_subspace(
        self,
        positive_features: torch.Tensor,
        positive_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        centroid = self._weighted_centroid(positive_features, positive_weights)
        centered = positive_features - centroid
        if centered.size(0) == 0:
            return centroid, None, centroid.new_tensor(self.min_radius)

        weighted_norm_sq = (centered.pow(2).sum(dim=1) * positive_weights).sum()
        radius = torch.sqrt(weighted_norm_sq.clamp_min(self.eps))
        radius = torch.clamp(self.shrinkage * radius, min=self.min_radius)

        max_rank = min(self.subspace_rank, centered.size(0), centered.size(1))
        if max_rank <= 0 or centered.size(0) <= 1:
            return centroid, None, radius

        weighted_centered = centered * positive_weights.sqrt().unsqueeze(1)
        try:
            _, _, vh = torch.linalg.svd(weighted_centered, full_matrices=False)
        except RuntimeError:
            return centroid, None, radius
        basis = vh[:max_rank].t().contiguous()
        return centroid, basis, radius

    def forward(
        self,
        anchor_features: torch.Tensor,
        anchor_labels: torch.Tensor,
        anchor_photometric_stats: torch.Tensor,
        teacher_features: torch.Tensor,
        teacher_labels: torch.Tensor,
        teacher_photometric_stats: torch.Tensor,
        same_source_size: int = 0,
    ) -> torch.Tensor:
        if anchor_features.size(0) == 0 or teacher_features.size(0) == 0:
            return anchor_features.new_tensor(0.0)

        anchor = self._prepare_features(anchor_features)
        teacher_features = teacher_features.float()
        teacher_labels = teacher_labels.view(-1)
        anchor_labels = anchor_labels.view(-1)
        teacher_stats = teacher_photometric_stats.float().view(teacher_features.size(0), -1)
        anchor_stats = anchor_photometric_stats.float().view(anchor_features.size(0), -1)
        max_same_source = min(int(same_source_size), teacher_features.size(0))

        sample_losses = []
        for anchor_idx in range(anchor.size(0)):
            positive_features, _, positive_weights = self._select_teacher_positive_set(
                anchor_index=anchor_idx,
                anchor_label=anchor_labels[anchor_idx],
                anchor_stats=anchor_stats[anchor_idx : anchor_idx + 1],
                teacher_features=teacher_features,
                teacher_labels=teacher_labels,
                teacher_stats=teacher_stats,
                same_source_size=max_same_source,
            )
            if positive_features is None or positive_weights is None:
                continue

            centroid, basis, radius = self._teacher_subspace(positive_features, positive_weights)
            anchor_delta = anchor[anchor_idx : anchor_idx + 1] - centroid
            if basis is None:
                projected = torch.zeros_like(anchor_delta)
            else:
                projected = (anchor_delta @ basis) @ basis.t()
            orthogonal = anchor_delta - projected

            in_subspace_norm = torch.norm(projected, dim=1, p=2)
            orthogonal_norm = torch.norm(orthogonal, dim=1, p=2)
            radial_overflow = F.relu(in_subspace_norm - radius)
            sample_losses.append(
                radial_overflow.pow(2) + self.orthogonal_weight * orthogonal_norm.pow(2)
            )

        if not sample_losses:
            return anchor_features.new_tensor(0.0)
        return torch.cat(sample_losses, dim=0).mean()


class TeacherManifoldSeparationLoss(_TeacherManifoldBase):
    """Keep corrected features closer to same-ID teacher manifolds than to nearest negatives."""

    def __init__(
        self,
        similarity: str = "cosine",
        photometric_scale: float = 8.0,
        photometric_offset: float = 0.1,
        min_positive_weight: float = 0.05,
        margin: float = 0.08,
        eps: float = 1e-12,
    ):
        super().__init__(
            similarity=similarity,
            photometric_scale=photometric_scale,
            photometric_offset=photometric_offset,
            min_positive_weight=min_positive_weight,
            eps=eps,
        )
        self.margin = float(margin)

    def _score(
        self,
        anchor: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        if prototypes.size(0) == 0:
            return anchor.new_empty(0)
        return self._pairwise_similarity(anchor, prototypes).squeeze(0)

    def _negative_prototypes(
        self,
        teacher_features: torch.Tensor,
        teacher_labels: torch.Tensor,
        anchor_label: torch.Tensor,
    ) -> torch.Tensor:
        negative_mask = teacher_labels != anchor_label
        if not torch.any(negative_mask):
            return teacher_features.new_empty(0, teacher_features.size(1))

        negative_features = self._prepare_features(teacher_features[negative_mask]).detach()
        negative_labels = teacher_labels[negative_mask]
        prototypes = []
        for label in negative_labels.unique():
            class_mask = negative_labels == label
            class_prototype = negative_features[class_mask].mean(dim=0, keepdim=True)
            if self.similarity == "cosine":
                class_prototype = F.normalize(class_prototype, p=2, dim=1, eps=self.eps)
            prototypes.append(class_prototype)
        if not prototypes:
            return teacher_features.new_empty(0, teacher_features.size(1))
        return torch.cat(prototypes, dim=0)

    def forward(
        self,
        anchor_features: torch.Tensor,
        anchor_labels: torch.Tensor,
        anchor_photometric_stats: torch.Tensor,
        teacher_features: torch.Tensor,
        teacher_labels: torch.Tensor,
        teacher_photometric_stats: torch.Tensor,
        same_source_size: int = 0,
    ) -> torch.Tensor:
        if anchor_features.size(0) == 0 or teacher_features.size(0) == 0:
            return anchor_features.new_tensor(0.0)

        anchor = self._prepare_features(anchor_features)
        teacher_features = teacher_features.float()
        teacher_labels = teacher_labels.view(-1)
        anchor_labels = anchor_labels.view(-1)
        teacher_stats = teacher_photometric_stats.float().view(teacher_features.size(0), -1)
        anchor_stats = anchor_photometric_stats.float().view(anchor_features.size(0), -1)
        max_same_source = min(int(same_source_size), teacher_features.size(0))

        sample_losses = []
        for anchor_idx in range(anchor.size(0)):
            positive_features, _, positive_weights = self._select_teacher_positive_set(
                anchor_index=anchor_idx,
                anchor_label=anchor_labels[anchor_idx],
                anchor_stats=anchor_stats[anchor_idx : anchor_idx + 1],
                teacher_features=teacher_features,
                teacher_labels=teacher_labels,
                teacher_stats=teacher_stats,
                same_source_size=max_same_source,
            )
            if positive_features is None or positive_weights is None:
                continue

            positive_centroid = self._weighted_centroid(positive_features, positive_weights)
            negative_prototypes = self._negative_prototypes(
                teacher_features=teacher_features,
                teacher_labels=teacher_labels,
                anchor_label=anchor_labels[anchor_idx],
            )
            if negative_prototypes.size(0) == 0:
                continue

            anchor_feature = anchor[anchor_idx : anchor_idx + 1]
            positive_score = self._score(anchor_feature, positive_centroid).max()
            hardest_negative = self._score(anchor_feature, negative_prototypes).max()
            sample_losses.append(
                F.relu(self.margin + hardest_negative - positive_score).unsqueeze(0)
            )

        if not sample_losses:
            return anchor_features.new_tensor(0.0)
        return torch.cat(sample_losses, dim=0).mean()


class RankingTopologyPreservingLoss(_TeacherManifoldBase):
    """Preserve teacher local ranking topology after illumination adaptation."""

    def __init__(
        self,
        similarity: str = "cosine",
        photometric_scale: float = 8.0,
        photometric_offset: float = 0.1,
        min_positive_weight: float = 0.05,
        topk_positive: int = 2,
        topk_negative: int = 4,
        margin_slack: float = 0.01,
        beta: float = 12.0,
        eps: float = 1e-12,
    ):
        super().__init__(
            similarity=similarity,
            photometric_scale=photometric_scale,
            photometric_offset=photometric_offset,
            min_positive_weight=min_positive_weight,
            eps=eps,
        )
        self.topk_positive = max(int(topk_positive), 1)
        self.topk_negative = max(int(topk_negative), 1)
        self.margin_slack = float(margin_slack)
        self.beta = float(beta)

    def forward(
        self,
        anchor_features: torch.Tensor,
        anchor_labels: torch.Tensor,
        anchor_photometric_stats: torch.Tensor,
        teacher_features: torch.Tensor,
        teacher_labels: torch.Tensor,
        teacher_photometric_stats: torch.Tensor,
        same_source_size: int = 0,
    ) -> torch.Tensor:
        if anchor_features.size(0) == 0 or teacher_features.size(0) == 0:
            return anchor_features.new_tensor(0.0)

        anchor = self._prepare_features(anchor_features)
        teacher = self._prepare_features(teacher_features)
        teacher_labels = teacher_labels.view(-1)
        anchor_labels = anchor_labels.view(-1)
        teacher_stats = teacher_photometric_stats.float().view(teacher_features.size(0), -1)
        anchor_stats = anchor_photometric_stats.float().view(anchor_features.size(0), -1)
        max_same_source = min(int(same_source_size), teacher_features.size(0))
        sample_losses = []

        for anchor_idx in range(anchor.size(0)):
            positive_features, positive_stats, positive_weights = self._select_teacher_positive_set(
                anchor_index=anchor_idx,
                anchor_label=anchor_labels[anchor_idx],
                anchor_stats=anchor_stats[anchor_idx : anchor_idx + 1],
                teacher_features=teacher,
                teacher_labels=teacher_labels,
                teacher_stats=teacher_stats,
                same_source_size=max_same_source,
            )
            if (
                positive_features is None
                or positive_stats is None
                or positive_weights is None
                or positive_features.size(0) == 0
            ):
                continue

            negative_mask = teacher_labels != anchor_labels[anchor_idx]
            if not torch.any(negative_mask):
                continue

            negative_features = teacher[negative_mask].detach()
            negative_stats = teacher_stats[negative_mask]
            negative_weights = self._positive_weights(
                anchor_stats[anchor_idx : anchor_idx + 1],
                negative_stats,
            ).clamp_min(self.eps)

            if anchor_idx < max_same_source:
                teacher_anchor = teacher[anchor_idx : anchor_idx + 1].detach()
            else:
                teacher_anchor = self._weighted_centroid(positive_features, positive_weights)

            teacher_pos_scores = self._pairwise_similarity(teacher_anchor, positive_features).squeeze(0)
            teacher_neg_scores = self._pairwise_similarity(teacher_anchor, negative_features).squeeze(0)

            k_pos = min(self.topk_positive, teacher_pos_scores.numel())
            k_neg = min(self.topk_negative, teacher_neg_scores.numel())
            teacher_pos_scores, pos_idx = torch.topk(teacher_pos_scores, k=k_pos, largest=True)
            teacher_neg_scores, neg_idx = torch.topk(teacher_neg_scores, k=k_neg, largest=True)

            positive_features = positive_features[pos_idx]
            positive_weights = positive_weights[pos_idx]
            negative_features = negative_features[neg_idx]
            negative_weights = negative_weights[neg_idx]

            student_anchor = anchor[anchor_idx : anchor_idx + 1]
            student_pos_scores = self._pairwise_similarity(student_anchor, positive_features).squeeze(0)
            student_neg_scores = self._pairwise_similarity(student_anchor, negative_features).squeeze(0)

            teacher_margin = teacher_pos_scores.unsqueeze(1) - teacher_neg_scores.unsqueeze(0)
            valid_mask = teacher_margin > 0
            if not torch.any(valid_mask):
                continue

            student_margin = student_pos_scores.unsqueeze(1) - student_neg_scores.unsqueeze(0)
            pair_weights = positive_weights.unsqueeze(1) * negative_weights.unsqueeze(0)
            pair_weights = pair_weights / pair_weights.sum().clamp_min(self.eps)
            sample_loss = F.softplus(
                self.beta
                * (teacher_margin.detach() - self.margin_slack - student_margin)
            ) / self.beta
            sample_losses.append((pair_weights[valid_mask] * sample_loss[valid_mask]).sum())

        if not sample_losses:
            return anchor_features.new_tensor(0.0)
        return torch.stack(sample_losses).mean()


class AnisotropicIdentityProtectionLoss(_TeacherManifoldBase):
    """Penalize identity-sensitive feature motion more than nuisance motion."""

    def __init__(
        self,
        similarity: str = "cosine",
        photometric_scale: float = 8.0,
        photometric_offset: float = 0.1,
        min_positive_weight: float = 0.05,
        topk_positive: int = 2,
        topk_negative: int = 4,
        subspace_rank: int = 1,
        identity_weight: float = 1.0,
        nuisance_weight: float = 0.5,
        nuisance_radius: float = 0.12,
        eps: float = 1e-12,
    ):
        super().__init__(
            similarity=similarity,
            photometric_scale=photometric_scale,
            photometric_offset=photometric_offset,
            min_positive_weight=min_positive_weight,
            eps=eps,
        )
        self.topk_positive = max(int(topk_positive), 1)
        self.topk_negative = max(int(topk_negative), 1)
        self.subspace_rank = max(int(subspace_rank), 1)
        self.identity_weight = float(identity_weight)
        self.nuisance_weight = float(nuisance_weight)
        self.nuisance_radius = float(nuisance_radius)

    def _identity_basis(
        self,
        teacher_anchor: torch.Tensor,
        anchor_index: int,
        anchor_label: torch.Tensor,
        anchor_stats: torch.Tensor,
        teacher_features: torch.Tensor,
        teacher_labels: torch.Tensor,
        teacher_stats: torch.Tensor,
        same_source_size: int,
    ) -> Optional[torch.Tensor]:
        pos_mask = teacher_labels == anchor_label
        if anchor_index < same_source_size and anchor_index < pos_mask.numel():
            pos_mask[anchor_index] = False
        if not torch.any(pos_mask):
            return None
        positive_features = teacher_features[pos_mask].float().detach()
        positive_stats = teacher_stats[pos_mask].float()
        positive_weights = self._positive_weights(anchor_stats, positive_stats).clamp_min(self.eps)
        positive_weights = positive_weights / positive_weights.sum().clamp_min(self.eps)

        negative_mask = teacher_labels != anchor_label
        if not torch.any(negative_mask):
            return None

        negative_features = teacher_features[negative_mask].detach()
        negative_stats = teacher_stats[negative_mask]
        negative_weights = self._positive_weights(anchor_stats, negative_stats).clamp_min(self.eps)

        teacher_pos_scores = self._pairwise_similarity(
            self._prepare_features(teacher_anchor),
            self._prepare_features(positive_features),
        ).squeeze(0)
        teacher_neg_scores = self._pairwise_similarity(
            self._prepare_features(teacher_anchor),
            self._prepare_features(negative_features),
        ).squeeze(0)

        k_pos = min(self.topk_positive, teacher_pos_scores.numel())
        k_neg = min(self.topk_negative, teacher_neg_scores.numel())
        _, pos_idx = torch.topk(teacher_pos_scores, k=k_pos, largest=True)
        _, neg_idx = torch.topk(teacher_neg_scores, k=k_neg, largest=True)

        positive_features = positive_features[pos_idx]
        positive_weights = positive_weights[pos_idx]
        negative_features = negative_features[neg_idx]
        negative_weights = negative_weights[neg_idx]

        pair_diffs = []
        pair_weights = []
        for pos_idx in range(positive_features.size(0)):
            for neg_idx in range(negative_features.size(0)):
                pair_diffs.append(positive_features[pos_idx] - negative_features[neg_idx])
                pair_weights.append(positive_weights[pos_idx] * negative_weights[neg_idx])
        if not pair_diffs:
            return None

        diff_matrix = torch.stack(pair_diffs, dim=0)
        pair_weights = torch.stack(pair_weights).clamp_min(self.eps)
        weighted_diffs = diff_matrix * pair_weights.sqrt().unsqueeze(1)
        max_rank = min(self.subspace_rank, weighted_diffs.size(0), weighted_diffs.size(1))
        if max_rank <= 0:
            return None

        try:
            _, _, vh = torch.linalg.svd(weighted_diffs, full_matrices=False)
        except RuntimeError:
            return None
        return vh[:max_rank].t().contiguous()

    def forward(
        self,
        anchor_features: torch.Tensor,
        teacher_anchor_features: torch.Tensor,
        anchor_labels: torch.Tensor,
        anchor_photometric_stats: torch.Tensor,
        teacher_features: torch.Tensor,
        teacher_labels: torch.Tensor,
        teacher_photometric_stats: torch.Tensor,
        same_source_size: int = 0,
    ) -> torch.Tensor:
        if anchor_features.size(0) == 0 or teacher_features.size(0) == 0:
            return anchor_features.new_tensor(0.0)

        anchor = anchor_features.float()
        teacher_anchor = teacher_anchor_features.float()
        teacher_gallery = teacher_features.float()
        teacher_labels = teacher_labels.view(-1)
        anchor_labels = anchor_labels.view(-1)
        teacher_stats = teacher_photometric_stats.float().view(teacher_features.size(0), -1)
        anchor_stats = anchor_photometric_stats.float().view(anchor_features.size(0), -1)
        max_same_source = min(int(same_source_size), teacher_gallery.size(0))
        sample_losses = []

        for anchor_idx in range(anchor.size(0)):
            basis = self._identity_basis(
                teacher_anchor=teacher_anchor[anchor_idx : anchor_idx + 1],
                anchor_index=anchor_idx,
                anchor_label=anchor_labels[anchor_idx],
                anchor_stats=anchor_stats[anchor_idx : anchor_idx + 1],
                teacher_features=teacher_gallery,
                teacher_labels=teacher_labels,
                teacher_stats=teacher_stats,
                same_source_size=max_same_source,
            )

            delta = anchor[anchor_idx : anchor_idx + 1] - teacher_anchor[anchor_idx : anchor_idx + 1]
            if basis is None:
                identity_component = delta
                nuisance_component = torch.zeros_like(delta)
            else:
                identity_component = (delta @ basis) @ basis.t()
                nuisance_component = delta - identity_component

            identity_norm = torch.norm(identity_component, dim=1, p=2)
            nuisance_norm = torch.norm(nuisance_component, dim=1, p=2)
            sample_losses.append(
                self.identity_weight * identity_norm.pow(2)
                + self.nuisance_weight * F.relu(nuisance_norm - self.nuisance_radius).pow(2)
            )

        if not sample_losses:
            return anchor_features.new_tensor(0.0)
        return torch.cat(sample_losses, dim=0).mean()


class SemanticNonConfusionLoss(nn.Module):
    """Keep corrected logits from collapsing the teacher true-class margin."""

    def __init__(
        self,
        margin_delta: float = 0.02,
        squared: bool = True,
    ):
        super().__init__()
        self.margin_delta = float(margin_delta)
        self.squared = bool(squared)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if student_logits.numel() == 0 or teacher_logits.numel() == 0:
            return student_logits.new_tensor(0.0)
        if student_logits.size(1) <= 1 or teacher_logits.size(1) <= 1:
            return student_logits.new_tensor(0.0)

        labels = labels.view(-1, 1)
        mask = F.one_hot(labels.squeeze(1), num_classes=student_logits.size(1)).bool()
        fill_value = torch.finfo(student_logits.dtype).min

        student_true = student_logits.gather(1, labels).squeeze(1).float()
        teacher_true = teacher_logits.gather(1, labels).squeeze(1).float()
        student_neg = student_logits.float().masked_fill(mask, fill_value).max(dim=1).values
        teacher_neg = teacher_logits.float().masked_fill(mask, fill_value).max(dim=1).values

        target_margin = (teacher_true - teacher_neg) + self.margin_delta
        student_margin = student_true - student_neg
        loss = F.relu(target_margin.detach() - student_margin)
        if self.squared:
            loss = loss.pow(2)
        return loss.mean()


class CrossCovarianceDecorrelationLoss(nn.Module):
    """Suppress nuisance leakage by minimizing cross-covariance between embeddings."""

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        identity_features: torch.Tensor,
        nuisance_features: torch.Tensor,
    ) -> torch.Tensor:
        if identity_features.size(0) <= 1 or nuisance_features.size(0) <= 1:
            return identity_features.new_tensor(0.0)

        identity = identity_features.float() - identity_features.float().mean(dim=0, keepdim=True)
        nuisance = nuisance_features.float() - nuisance_features.float().mean(dim=0, keepdim=True)
        identity = identity / identity.std(dim=0, unbiased=False, keepdim=True).clamp_min(self.eps)
        nuisance = nuisance / nuisance.std(dim=0, unbiased=False, keepdim=True).clamp_min(self.eps)
        cross_cov = identity.t() @ nuisance / max(identity.size(0) - 1, 1)
        return cross_cov.pow(2).mean()


class FeatureTrustRegionLoss(nn.Module):
    """Keep adapted descriptors inside a teacher-defined local trust region."""

    def __init__(
        self,
        base_radius: float = 0.12,
        adaptive_scale: float = 0.0,
        class_spread_scale: float = 0.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.base_radius = float(base_radius)
        self.adaptive_scale = float(adaptive_scale)
        self.class_spread_scale = float(class_spread_scale)
        self.eps = float(eps)

    def _teacher_class_spread(
        self,
        teacher_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        spread = teacher_features.new_zeros(teacher_features.size(0))
        labels = labels.view(-1)
        for label in labels.unique():
            class_mask = labels == label
            if int(class_mask.sum().item()) <= 1:
                continue
            teacher_cls = teacher_features[class_mask]
            teacher_center = teacher_cls.mean(dim=0, keepdim=True)
            spread[class_mask] = torch.norm(teacher_cls - teacher_center, dim=1, p=2)
        return spread

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        severity: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if student_features.numel() == 0 or teacher_features.numel() == 0:
            return student_features.new_tensor(0.0)

        student = F.normalize(student_features.float(), p=2, dim=1, eps=self.eps)
        teacher = F.normalize(teacher_features.float(), p=2, dim=1, eps=self.eps)
        distance = torch.norm(student - teacher, dim=1, p=2)

        radius = distance.new_full(distance.shape, self.base_radius)
        if severity is not None and self.adaptive_scale != 0.0:
            severity = severity.float().view(distance.size(0), -1).mean(dim=1).clamp_min(0.0)
            radius = radius * (1.0 + self.adaptive_scale * severity)
        if labels is not None and self.class_spread_scale != 0.0:
            teacher_spread = self._teacher_class_spread(teacher, labels).clamp_min(0.0)
            radius = radius * (1.0 + self.class_spread_scale * teacher_spread)

        return F.relu(distance - radius).pow(2).mean()


class LocalRankPreservingLoss(nn.Module):
    """Preserve teacher local-neighborhood ordering after illumination adaptation."""

    def __init__(
        self,
        alpha: float = 0.9,
        k_positive: int = 1,
        k_negative: int = 1,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.k_positive = max(int(k_positive), 1)
        self.k_negative = max(int(k_negative), 1)
        self.eps = float(eps)

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if student_features.size(0) <= 1 or teacher_features.size(0) <= 1:
            return student_features.new_tensor(0.0)

        student = F.normalize(student_features.float(), p=2, dim=1, eps=self.eps)
        teacher = F.normalize(teacher_features.float(), p=2, dim=1, eps=self.eps)
        labels = labels.view(-1)

        teacher_sim = teacher @ teacher.t()
        student_sim = student @ student.t()
        sample_losses = []

        for anchor_idx in range(student.size(0)):
            pos_mask = labels == labels[anchor_idx]
            pos_mask[anchor_idx] = False
            neg_mask = labels != labels[anchor_idx]
            if not torch.any(pos_mask) or not torch.any(neg_mask):
                continue

            teacher_pos = teacher_sim[anchor_idx][pos_mask]
            teacher_neg = teacher_sim[anchor_idx][neg_mask]
            student_pos = student_sim[anchor_idx][pos_mask]
            student_neg = student_sim[anchor_idx][neg_mask]

            k_pos = min(self.k_positive, teacher_pos.numel())
            k_neg = min(self.k_negative, teacher_neg.numel())
            teacher_pos, pos_idx = torch.topk(teacher_pos, k=k_pos, largest=True)
            teacher_neg, neg_idx = torch.topk(teacher_neg, k=k_neg, largest=True)
            student_pos = student_pos[pos_idx]
            student_neg = student_neg[neg_idx]

            teacher_margin = torch.clamp(teacher_pos.unsqueeze(1) - teacher_neg.unsqueeze(0), min=0.0)
            if not torch.any(teacher_margin > 0):
                continue

            student_margin = student_pos.unsqueeze(1) - student_neg.unsqueeze(0)
            margin_loss = F.relu(self.alpha * teacher_margin.detach() - student_margin)
            sample_losses.append(margin_loss[teacher_margin > 0].mean())

        if not sample_losses:
            return student_features.new_tensor(0.0)
        return torch.stack(sample_losses).mean()


class NeighborhoodConsistencyLoss(nn.Module):
    """Distill retrieval-neighborhood structure into train-time descriptors.

    The default ``soft``/``uniform`` modes keep the earlier supervised listwise
    objective. ``teacher_target="reciprocal"`` switches to GL-NCD v2: the teacher
    builds a k-reciprocal neighborhood graph, converts reciprocal-overlap into a
    Jaccard-style target distribution, and trains the student descriptor to
    match that retrieval graph while suppressing selected hard negatives.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        topk: int = 6,
        positive_weight: float = 1.0,
        negative_weight: float = 0.25,
        local_weight: float = 0.35,
        use_global: bool = True,
        use_local: bool = True,
        use_hard_negatives: bool = True,
        teacher_target: str = "soft",
        eps: float = 1e-12,
    ):
        super().__init__()
        self.temperature = max(float(temperature), float(eps))
        self.topk = max(int(topk), 1)
        self.positive_weight = float(positive_weight)
        self.negative_weight = float(negative_weight)
        self.local_weight = float(local_weight)
        self.use_global = bool(use_global)
        self.use_local = bool(use_local)
        self.use_hard_negatives = bool(use_hard_negatives)
        self.teacher_target = str(teacher_target).lower()
        self.eps = float(eps)

    @property
    def _uses_reciprocal_graph(self) -> bool:
        return self.teacher_target in {
            "reciprocal",
            "jaccard",
            "graph",
            "reciprocal_graph",
            "glncd_v2",
        }

    def _normalize_global(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(features.float(), p=2, dim=1, eps=self.eps)

    def _normalize_local(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(features.float(), p=2, dim=2, eps=self.eps)

    def _score_local(
        self,
        anchor_features: torch.Tensor,
        gallery_features: torch.Tensor,
    ) -> torch.Tensor:
        if anchor_features.dim() != 3 or gallery_features.dim() != 3:
            return anchor_features.new_empty(0)
        if anchor_features.size(1) != gallery_features.size(1):
            shared_parts = min(anchor_features.size(1), gallery_features.size(1))
            anchor_features = anchor_features[:, :shared_parts, :]
            gallery_features = gallery_features[:, :shared_parts, :]
        return (anchor_features.unsqueeze(1) * gallery_features.unsqueeze(0)).sum(dim=3).mean(dim=2)

    def _teacher_graph_targets(
        self,
        teacher_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        exclude_self: bool = True,
    ) -> torch.Tensor:
        """Build square k-reciprocal/Jaccard teacher targets.

        Returned rows are probability distributions over graph-supported
        neighbors. Non-neighbor entries remain exactly zero, which lets the
        training loss add hard negatives explicitly without polluting the
        teacher target.
        """
        if teacher_features.dim() != 2 or teacher_features.size(0) <= 1:
            n = int(teacher_features.size(0)) if teacher_features.dim() >= 1 else 0
            return teacher_features.new_zeros((n, n), dtype=torch.float32)

        teacher = self._normalize_global(teacher_features)
        similarity = teacher @ teacher.t()
        distance = (1.0 - similarity).clamp_min(0.0)
        return self._graph_targets_from_distance(distance, labels=labels, exclude_self=exclude_self)

    def _graph_targets_from_distance(
        self,
        distance: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        exclude_self: bool = True,
    ) -> torch.Tensor:
        if distance.dim() != 2 or distance.size(0) != distance.size(1) or distance.size(0) <= 1:
            n = int(distance.size(0)) if distance.dim() >= 1 else 0
            return distance.new_zeros((n, n), dtype=torch.float32)

        n = distance.size(0)
        device = distance.device
        eye = torch.eye(n, dtype=torch.bool, device=device)
        rank_distance = distance.float().clone()
        if exclude_self:
            rank_distance = rank_distance.masked_fill(eye, float("inf"))

        k = min(self.topk, max(n - 1, 1))
        neighbor_indices = torch.topk(rank_distance, k=k, largest=False, dim=1).indices
        neighbor_mask = torch.zeros((n, n), dtype=torch.bool, device=device)
        neighbor_mask.scatter_(1, neighbor_indices, True)
        if exclude_self:
            neighbor_mask = neighbor_mask.masked_fill(eye, False)

        reciprocal_mask = neighbor_mask & neighbor_mask.t()
        support_mask = reciprocal_mask.clone()
        empty_rows = ~support_mask.any(dim=1)
        if torch.any(empty_rows):
            support_mask[empty_rows] = neighbor_mask[empty_rows]

        if isinstance(labels, torch.Tensor):
            flat_labels = labels.view(-1).to(device)
            if flat_labels.numel() == n:
                same_label_mask = flat_labels.unsqueeze(0) == flat_labels.unsqueeze(1)
                if exclude_self:
                    same_label_mask = same_label_mask.masked_fill(eye, False)
                support_mask = support_mask | same_label_mask

        if exclude_self:
            support_mask = support_mask.masked_fill(eye, False)

        weights = torch.exp(-distance.float() / self.temperature) * support_mask.float()
        # Jaccard over reciprocal-neighbor indicator weights, equivalent to the
        # overlap term used by k-reciprocal re-ranking but kept differentiability
        # out of the teacher graph.
        min_overlap = torch.minimum(weights.unsqueeze(1), weights.unsqueeze(0)).sum(dim=2)
        max_union = torch.maximum(weights.unsqueeze(1), weights.unsqueeze(0)).sum(dim=2)
        jaccard_similarity = min_overlap / max_union.clamp_min(self.eps)
        graph_distance = (1.0 - jaccard_similarity).clamp(0.0, 1.0)

        logits = -graph_distance / self.temperature
        if isinstance(labels, torch.Tensor):
            flat_labels = labels.view(-1).to(device)
            if flat_labels.numel() == n and self.positive_weight > 0:
                same_label_mask = flat_labels.unsqueeze(0) == flat_labels.unsqueeze(1)
                if exclude_self:
                    same_label_mask = same_label_mask.masked_fill(eye, False)
                logits = logits + same_label_mask.float() * math.log(max(self.positive_weight, self.eps))

        logits = logits.masked_fill(~support_mask, -1.0e9)
        targets = F.softmax(logits, dim=1)
        targets = targets * support_mask.float()
        row_sum = targets.sum(dim=1, keepdim=True)
        fallback = support_mask.float() / support_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        targets = torch.where(row_sum > 0, targets / row_sum.clamp_min(self.eps), fallback)
        if exclude_self:
            targets = targets.masked_fill(eye, 0.0)
        return targets.detach()

    def _listwise_anchor_loss(
        self,
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
        gallery_labels: torch.Tensor,
        anchor_label: torch.Tensor,
        exclude_index: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if student_scores.numel() == 0 or teacher_scores.numel() == 0:
            return None

        pos_mask = gallery_labels == anchor_label
        neg_mask = gallery_labels != anchor_label
        if exclude_index is not None and 0 <= exclude_index < pos_mask.numel():
            pos_mask = pos_mask.clone()
            pos_mask[exclude_index] = False
        if not torch.any(pos_mask) or not torch.any(neg_mask):
            return None

        pos_scores = teacher_scores[pos_mask]
        neg_scores = teacher_scores[neg_mask]
        k_pos = min(self.topk, pos_scores.numel())
        k_neg = min(self.topk, neg_scores.numel()) if self.use_hard_negatives else 0
        _, pos_rank = torch.topk(pos_scores, k=k_pos, largest=True)
        if k_neg > 0:
            _, neg_rank = torch.topk(neg_scores, k=k_neg, largest=True)

        pos_indices = torch.nonzero(pos_mask, as_tuple=False).view(-1)[pos_rank]
        if k_neg > 0:
            neg_indices = torch.nonzero(neg_mask, as_tuple=False).view(-1)[neg_rank]
            candidate_indices = torch.cat([pos_indices, neg_indices], dim=0)
        else:
            candidate_indices = pos_indices

        student_logits = student_scores[candidate_indices] / self.temperature
        if self.teacher_target in {"uniform", "flat", "hard"}:
            target_pos = torch.full(
                (k_pos,),
                1.0 / float(k_pos),
                device=student_logits.device,
                dtype=student_logits.dtype,
            )
        else:
            teacher_pos_logits = teacher_scores[pos_indices].detach() / self.temperature
            target_pos = F.softmax(teacher_pos_logits, dim=0)
        log_probs = F.log_softmax(student_logits, dim=0)

        positive_loss = -(target_pos * log_probs[:k_pos]).sum()
        if k_neg > 0 and self.negative_weight > 0:
            neg_logits = student_logits[k_pos:]
            pos_logits = student_logits[:k_pos]
            negative_loss = F.softplus(
                torch.logsumexp(neg_logits, dim=0) - torch.logsumexp(pos_logits, dim=0)
            )
        else:
            negative_loss = student_logits.new_tensor(0.0)
        return self.positive_weight * positive_loss + self.negative_weight * negative_loss

    def _graph_anchor_loss(
        self,
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
        target_probs: torch.Tensor,
        gallery_labels: Optional[torch.Tensor] = None,
        anchor_label: Optional[torch.Tensor] = None,
        exclude_index: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        support_mask = target_probs > 0
        if exclude_index is not None and 0 <= exclude_index < support_mask.numel():
            support_mask = support_mask.clone()
            support_mask[exclude_index] = False
            target_probs = target_probs.clone()
            target_probs[exclude_index] = 0.0
        if not torch.any(support_mask):
            return None

        hard_negative_mask = torch.zeros_like(support_mask)
        if (
            self.use_hard_negatives
            and self.negative_weight > 0
            and isinstance(gallery_labels, torch.Tensor)
            and isinstance(anchor_label, torch.Tensor)
        ):
            neg_mask = (gallery_labels.view(-1) != anchor_label) & ~support_mask
            if exclude_index is not None and 0 <= exclude_index < neg_mask.numel():
                neg_mask = neg_mask.clone()
                neg_mask[exclude_index] = False
            if torch.any(neg_mask):
                k_neg = min(self.topk, int(neg_mask.sum().item()))
                neg_scores = teacher_scores[neg_mask]
                _, neg_rank = torch.topk(neg_scores, k=k_neg, largest=True)
                neg_indices = torch.nonzero(neg_mask, as_tuple=False).view(-1)[neg_rank]
                hard_negative_mask[neg_indices] = True

        candidate_mask = support_mask | hard_negative_mask
        if not torch.any(candidate_mask):
            return None

        candidate_targets = target_probs[candidate_mask]
        candidate_targets = candidate_targets / candidate_targets.sum().clamp_min(self.eps)
        student_logits = student_scores[candidate_mask] / self.temperature
        log_probs = F.log_softmax(student_logits, dim=0)
        positive_loss = -(candidate_targets * log_probs).sum()

        if torch.any(hard_negative_mask):
            support_logits = student_scores[support_mask] / self.temperature
            negative_logits = student_scores[hard_negative_mask] / self.temperature
            negative_loss = F.softplus(
                torch.logsumexp(negative_logits, dim=0) - torch.logsumexp(support_logits, dim=0)
            )
        else:
            negative_loss = student_scores.new_tensor(0.0)
        return self.positive_weight * positive_loss + self.negative_weight * negative_loss

    def _global_loss(
        self,
        student_features: torch.Tensor,
        teacher_anchor_features: torch.Tensor,
        labels: torch.Tensor,
        teacher_gallery_features: torch.Tensor,
        teacher_gallery_labels: torch.Tensor,
        same_source_size: int,
    ) -> torch.Tensor:
        student = self._normalize_global(student_features)
        teacher_anchor = self._normalize_global(teacher_anchor_features)
        teacher_gallery = self._normalize_global(teacher_gallery_features).detach()
        labels = labels.view(-1)
        teacher_gallery_labels = teacher_gallery_labels.view(-1)

        student_scores = student @ teacher_gallery.t()
        teacher_scores = teacher_anchor.detach() @ teacher_gallery.t()
        max_same_source = min(int(same_source_size), teacher_gallery.size(0))
        losses = []
        for anchor_idx in range(student.size(0)):
            exclude_index = anchor_idx if anchor_idx < max_same_source else None
            anchor_loss = self._listwise_anchor_loss(
                student_scores[anchor_idx],
                teacher_scores[anchor_idx],
                teacher_gallery_labels,
                labels[anchor_idx],
                exclude_index=exclude_index,
            )
            if anchor_loss is not None:
                losses.append(anchor_loss)
        if not losses:
            return student_features.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _global_graph_loss(
        self,
        student_features: torch.Tensor,
        teacher_anchor_features: torch.Tensor,
        labels: torch.Tensor,
        teacher_gallery_features: torch.Tensor,
        teacher_gallery_labels: torch.Tensor,
        same_source_size: int,
    ) -> torch.Tensor:
        student = self._normalize_global(student_features)
        teacher_anchor = self._normalize_global(teacher_anchor_features)
        teacher_gallery = self._normalize_global(teacher_gallery_features).detach()
        labels = labels.view(-1)
        teacher_gallery_labels = teacher_gallery_labels.view(-1)

        student_scores = student @ teacher_gallery.t()
        teacher_scores = teacher_anchor.detach() @ teacher_gallery.t()
        graph_targets = self._teacher_graph_targets(
            teacher_gallery,
            labels=teacher_gallery_labels,
            exclude_self=True,
        )
        max_same_source = min(int(same_source_size), teacher_gallery.size(0))

        losses = []
        for anchor_idx in range(student.size(0)):
            if anchor_idx >= graph_targets.size(0):
                continue
            exclude_index = anchor_idx if anchor_idx < max_same_source else None
            anchor_loss = self._graph_anchor_loss(
                student_scores[anchor_idx],
                teacher_scores[anchor_idx],
                graph_targets[anchor_idx],
                gallery_labels=teacher_gallery_labels,
                anchor_label=labels[anchor_idx],
                exclude_index=exclude_index,
            )
            if anchor_loss is not None:
                losses.append(anchor_loss)
        if not losses:
            return student_features.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _local_loss(
        self,
        student_local_features: Optional[torch.Tensor],
        teacher_local_features: Optional[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(student_local_features, torch.Tensor) or not isinstance(teacher_local_features, torch.Tensor):
            return labels.new_tensor(0.0, dtype=torch.float32)
        if student_local_features.size(0) <= 1 or teacher_local_features.size(0) <= 1:
            return student_local_features.new_tensor(0.0)

        student_local = self._normalize_local(student_local_features)
        teacher_local = self._normalize_local(teacher_local_features).detach()
        labels = labels.view(-1)
        student_scores = self._score_local(student_local, teacher_local)
        teacher_scores = self._score_local(teacher_local, teacher_local)
        losses = []
        if self._uses_reciprocal_graph:
            graph_targets = self._graph_targets_from_distance(
                (1.0 - teacher_scores).clamp_min(0.0),
                labels=labels,
                exclude_self=True,
            )
            for anchor_idx in range(student_local.size(0)):
                anchor_loss = self._graph_anchor_loss(
                    student_scores[anchor_idx],
                    teacher_scores[anchor_idx],
                    graph_targets[anchor_idx],
                    gallery_labels=labels,
                    anchor_label=labels[anchor_idx],
                    exclude_index=anchor_idx,
                )
                if anchor_loss is not None:
                    losses.append(anchor_loss)
            if not losses:
                return student_local_features.new_tensor(0.0)
            return torch.stack(losses).mean()

        for anchor_idx in range(student_local.size(0)):
            anchor_loss = self._listwise_anchor_loss(
                student_scores[anchor_idx],
                teacher_scores[anchor_idx],
                labels,
                labels[anchor_idx],
                exclude_index=anchor_idx,
            )
            if anchor_loss is not None:
                losses.append(anchor_loss)
        if not losses:
            return student_local_features.new_tensor(0.0)
        return torch.stack(losses).mean()

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_anchor_features: torch.Tensor,
        labels: torch.Tensor,
        teacher_gallery_features: Optional[torch.Tensor] = None,
        teacher_gallery_labels: Optional[torch.Tensor] = None,
        student_local_features: Optional[torch.Tensor] = None,
        teacher_local_features: Optional[torch.Tensor] = None,
        same_source_size: int = 0,
    ) -> torch.Tensor:
        if student_features.size(0) <= 1 or teacher_anchor_features.size(0) <= 1:
            return student_features.new_tensor(0.0)

        if teacher_gallery_features is None:
            teacher_gallery_features = teacher_anchor_features
        if teacher_gallery_labels is None:
            teacher_gallery_labels = labels
        if same_source_size <= 0:
            same_source_size = teacher_anchor_features.size(0)

        if self.use_global:
            if self._uses_reciprocal_graph:
                global_loss = self._global_graph_loss(
                    student_features=student_features,
                    teacher_anchor_features=teacher_anchor_features,
                    labels=labels,
                    teacher_gallery_features=teacher_gallery_features,
                    teacher_gallery_labels=teacher_gallery_labels,
                    same_source_size=same_source_size,
                )
            else:
                global_loss = self._global_loss(
                    student_features=student_features,
                    teacher_anchor_features=teacher_anchor_features,
                    labels=labels,
                    teacher_gallery_features=teacher_gallery_features,
                    teacher_gallery_labels=teacher_gallery_labels,
                    same_source_size=same_source_size,
                )
        else:
            global_loss = student_features.new_tensor(0.0)
        if not self.use_local or self.local_weight <= 0:
            return global_loss

        local_loss = self._local_loss(
            student_local_features=student_local_features,
            teacher_local_features=teacher_local_features,
            labels=labels,
        )
        return global_loss + self.local_weight * local_loss


# ============================================================================
# ============================================================================

# ============================================================================
#                    IICL: illumination-variant feature consistency
# ============================================================================

class DataDrivenVariantGenerator(nn.Module):
    """
    Generate illumination variants from the current IPAID decomposition.

    Variants are built by mixing illumination fields across the batch and by
    applying gamma perturbation to the current illumination estimate. The goal
    is to regularize feature stability under illumination change, not to create
    generic augmentation noise.
    """

    def __init__(
        self,
        num_variants: int = 2,
        mix_ratio: float = 0.7,
        gamma_range: Tuple[float, float] = (0.6, 1.4),
    ):
        super().__init__()
        self.num_variants = num_variants
        self.mix_ratio = mix_ratio
        self.gamma_range = gamma_range

    def forward(
        self,
        reflectance: torch.Tensor,
        illumination: torch.Tensor,
        num_variants: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            reflectance: R [B, 3, H, W]
            illumination: L [B, 1|3, H, W]
        Returns:
            variants: list of illumination-shifted RGB images
        """
        n_var = num_variants or self.num_variants
        B = reflectance.shape[0]
        device = reflectance.device
        variants = []

        for _ in range(max(n_var - 1, 1)):
            perm = torch.randperm(B, device=device)
            L_shuffled = illumination[perm]
            L_variant = self.mix_ratio * illumination + (1 - self.mix_ratio) * L_shuffled
            variant = reflectance * L_variant
            variants.append(torch.clamp(variant, 0.01, 0.99))

        if n_var >= 2:
            gamma = torch.empty(B, 1, 1, 1, device=device).uniform_(
                self.gamma_range[0], self.gamma_range[1]
            )
            L_gamma = torch.pow(illumination + 1e-6, gamma)
            L_gamma = torch.clamp(L_gamma, 0.2, 3.0)
            variant = reflectance * L_gamma
            variants.append(torch.clamp(variant, 0.01, 0.99))

        return variants

class IlluminationFeatureConsistencyLoss(nn.Module):
    """
    Feature consistency loss for IICL-style illumination variants.

    Default training uses feature matching (`mse` or `cosine`) rather than a
    full InfoNCE objective. `infonce` remains available as an explicit option,
    but it is not the default behavior.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        loss_type: str = "cosine",  # "mse" or "cosine" or "infonce"
    ):
        super().__init__()
        self.temperature = temperature
        self.loss_type = loss_type
    
    def forward(
        self,
        features_orig: torch.Tensor,
        features_variants: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute feature consistency across illumination variants."""
        if len(features_variants) == 0:
            return torch.tensor(0.0, device=features_orig.device)
        
        if self.loss_type == "mse":
            loss = 0.0
            for feat_var in features_variants:
                loss = loss + F.mse_loss(features_orig, feat_var)
            loss = loss / len(features_variants)
            
        elif self.loss_type == "cosine":
            loss = 0.0
            for feat_var in features_variants:
                cos_sim = F.cosine_similarity(features_orig, feat_var, dim=1)
                loss = loss + (1 - cos_sim).mean()
            loss = loss / len(features_variants)
            
        elif self.loss_type == "infonce":
            loss = self._infonce_loss(features_orig, features_variants)
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def _infonce_loss(
        self,
        features_orig: torch.Tensor,
        features_variants: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute InfoNCE only when explicitly requested."""
        B, D = features_orig.shape
        device = features_orig.device
        
        all_variants = torch.stack(features_variants, dim=1)  # [B, num_var, D]
        
        features_orig_norm = F.normalize(features_orig, dim=1)  # [B, D]
        all_variants_norm = F.normalize(all_variants, dim=2)    # [B, num_var, D]
        
        loss = 0.0
        
        for i in range(B):
            anchor = features_orig_norm[i]  # [D]
            
            positives = all_variants_norm[i]  # [num_var, D]
            pos_sim = torch.sum(anchor.unsqueeze(0) * positives, dim=1) / self.temperature  # [num_var]
            
            neg_mask = torch.ones(B, dtype=torch.bool, device=device)
            neg_mask[i] = False
            negatives = features_orig_norm[neg_mask]  # [B-1, D]
            neg_sim = torch.sum(anchor.unsqueeze(0) * negatives, dim=1) / self.temperature  # [B-1]
            
            # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            for pos_s in pos_sim:
                logits = torch.cat([pos_s.unsqueeze(0), neg_sim])  # [1 + B-1]
                labels = torch.zeros(1, dtype=torch.long, device=device)
                loss = loss + F.cross_entropy(logits.unsqueeze(0), labels)
        
        loss = loss / (B * len(features_variants))
        return loss


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_ipaid_module():
    """Run a lightweight IPAID smoke test."""
    print("=" * 60)
    print("Testing the IPAID module")
    print("=" * 60)
    
    ipaid = IPAIDModule(
        base_channels=32,
        num_scales=3,
        refine_iterations=1,
        use_sensitivity=True,
        use_refinement=True,
    )
    
    params = count_parameters(ipaid)
    print(f"Module parameters: {params / 1e6:.2f}M")
    
    x = torch.rand(4, 3, 256, 256)
    
    with torch.no_grad():
        reflectance = ipaid(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {reflectance.shape}")
        print(f"Output range: [{reflectance.min():.3f}, {reflectance.max():.3f}]")
        
        details = ipaid.forward_with_details(x)
        print(f"Illumination shape: {details['illumination'].shape}")
        print(f"Illumination range: [{details['illumination'].min():.3f}, {details['illumination'].max():.3f}]")
        
        if details['sensitivity'] is not None:
            print(f"Sensitivity shape: {details['sensitivity'].shape}")
            print(f"Sensitivity range: [{details['sensitivity'].min():.3f}, {details['sensitivity'].max():.3f}]")
    
    loss_fn = IPAIDLoss()
    loss, loss_dict = loss_fn(details)
    print("\nLoss values:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")
    
    print("\nIPAID module smoke test passed.")
    return ipaid


if __name__ == "__main__":
    test_ipaid_module()


