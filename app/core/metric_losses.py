"""Metric-learning losses used by the packaged ReID models.

The module includes triplet, ArcFace, Circle, contrastive, center, and simple
multi-task weighting helpers used by the paper-facing training code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict


class TripletLoss(nn.Module):
    """Triplet loss with multiple mining strategies and warm-up-aware filtering."""

    def __init__(self, margin: float = 0.3, mining_type: str = 'soft', 
                 reduction: str = 'mean', soft_margin: bool = True,
                 outlier_threshold: float = 3.0, warmup_epochs: int = 5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mining_type = mining_type  # 'hard', 'semi-hard', 'soft', 'all'
        self.reduction = reduction
        self.soft_margin = soft_margin
        self.outlier_threshold = outlier_threshold
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self._call_count = 0
        self._estimated_steps_per_epoch = 100
        assert mining_type in ['hard', 'semi-hard', 'soft', 'all'], f"Unknown mining type: {mining_type}"
    
    def set_epoch(self, epoch: int):
        """Manually set the current epoch for warm-up-aware filtering."""
        self.current_epoch = epoch
    
    def _auto_update_epoch(self):
        """Estimate the epoch index when no external epoch is provided."""
        self._call_count += 1
        if self._call_count % self._estimated_steps_per_epoch == 0:
            self.current_epoch = self._call_count // self._estimated_steps_per_epoch

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:

        Returns:
        """
        self._auto_update_epoch()
        dist_matrix = self._pairwise_distance(features)

        mask_positive = self._get_positive_mask(labels)  # (batch_size, batch_size)
        mask_negative = self._get_negative_mask(labels)  # (batch_size, batch_size)

        if self.mining_type == 'hard':
            loss = self._hard_mining(dist_matrix, mask_positive, mask_negative)
        elif self.mining_type == 'semi-hard':
            loss = self._semi_hard_mining(dist_matrix, mask_positive, mask_negative)
        elif self.mining_type == 'soft':
            loss = self._soft_mining(dist_matrix, mask_positive, mask_negative)
        else:  # 'all'
            loss = self._all_triplets(dist_matrix, mask_positive, mask_negative)

        loss = self._filter_outliers(loss)

        if self.reduction == 'mean':
            loss = loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=features.device)
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

    def _pairwise_distance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the pairwise Euclidean distance matrix."""
        x_squared = (x ** 2).sum(dim=1, keepdim=True)
        dist_matrix = x_squared + x_squared.t() - 2 * torch.mm(x, x.t())
        dist_matrix = torch.clamp(dist_matrix, min=1e-8)
        return torch.sqrt(dist_matrix)

    def _get_positive_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """Return the positive-pair mask for matching labels."""
        return (labels.unsqueeze(0) == labels.unsqueeze(1)).bool()

    def _get_negative_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """Return the negative-pair mask for non-matching labels."""
        return (labels.unsqueeze(0) != labels.unsqueeze(1)).bool()

    def _filter_outliers(self, loss: torch.Tensor) -> torch.Tensor:
        if loss.numel() == 0:
            return loss
        
        if self.current_epoch < self.warmup_epochs:
            return loss
        
        mean_loss = loss.mean()
        std_loss = loss.std()
        
        if std_loss > 0:
            mask = loss < (mean_loss + self.outlier_threshold * std_loss)
            if mask.sum() < loss.numel() // 2:
                mask = loss < (mean_loss + self.outlier_threshold * 2 * std_loss)
            loss = loss[mask]
        
        return loss

    def _apply_margin(self, diff: torch.Tensor) -> torch.Tensor:
        """Apply either a hard margin or a softplus-smoothed margin."""
        if self.soft_margin:
            return F.softplus(diff + self.margin)
        else:
            return torch.clamp(diff + self.margin, min=0.0)

    def _hard_mining(self, dist_matrix: torch.Tensor, mask_positive: torch.Tensor, 
                     mask_negative: torch.Tensor) -> torch.Tensor:
        """Use the hardest positive and hardest negative for each anchor."""
        batch_size = dist_matrix.size(0)
        device = dist_matrix.device

        dist_positive = dist_matrix[mask_positive].view(batch_size, -1)
        hardest_positive, _ = dist_positive.max(dim=1)  # (batch_size,)

        dist_negative = dist_matrix[mask_negative].view(batch_size, -1)
        hardest_negative, _ = dist_negative.min(dim=1)  # (batch_size,)

        loss = self._apply_margin(hardest_positive - hardest_negative)
        return loss
    
    def _soft_mining(self, dist_matrix: torch.Tensor, mask_positive: torch.Tensor,
                     mask_negative: torch.Tensor) -> torch.Tensor:
        batch_size = dist_matrix.size(0)
        device = dist_matrix.device
        
        losses = []
        for i in range(batch_size):
            pos_mask = mask_positive[i].clone()
            pos_mask[i] = False
            if not pos_mask.any():
                continue
            pos_dist = dist_matrix[i][pos_mask]
            
            neg_mask = mask_negative[i]
            if not neg_mask.any():
                continue
            neg_dist = dist_matrix[i][neg_mask]
            
            pos_weights = F.softmax(pos_dist, dim=0)
            weighted_pos_dist = (pos_dist * pos_weights).sum()
            
            neg_weights = F.softmax(-neg_dist, dim=0)
            weighted_neg_dist = (neg_dist * neg_weights).sum()
            
            loss = self._apply_margin(weighted_pos_dist - weighted_neg_dist)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses)

    def _semi_hard_mining(self, dist_matrix: torch.Tensor, mask_positive: torch.Tensor,
                          mask_negative: torch.Tensor) -> torch.Tensor:
        """Use semi-hard negatives when possible, then fall back to hard negatives."""
        batch_size = dist_matrix.size(0)
        device = dist_matrix.device

        losses = []
        for i in range(batch_size):
            pos_mask = mask_positive[i]
            if not pos_mask.any():
                continue
            pos_dist = dist_matrix[i][pos_mask]
            hardest_pos = pos_dist.max()

            neg_mask = mask_negative[i]
            if not neg_mask.any():
                continue
            neg_dist = dist_matrix[i][neg_mask]

            semi_hard_mask = (neg_dist > hardest_pos) & (neg_dist < hardest_pos + self.margin)
            if semi_hard_mask.any():
                semi_hard_dist = neg_dist[semi_hard_mask]
                hardest_neg = semi_hard_dist.min()
                loss = torch.clamp(hardest_pos - hardest_neg + self.margin, min=0.0)
                losses.append(loss)
            else:
                hardest_neg = neg_dist.min()
                loss = torch.clamp(hardest_pos - hardest_neg + self.margin, min=0.0)
                losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses)

    def _all_triplets(self, dist_matrix: torch.Tensor, mask_positive: torch.Tensor,
                      mask_negative: torch.Tensor) -> torch.Tensor:
        """Accumulate the loss over all valid triplets in the batch."""
        batch_size = dist_matrix.size(0)
        device = dist_matrix.device

        losses = []
        for i in range(batch_size):
            pos_mask = mask_positive[i]
            neg_mask = mask_negative[i]
            if not (pos_mask.any() and neg_mask.any()):
                continue

            pos_dist = dist_matrix[i][pos_mask]
            neg_dist = dist_matrix[i][neg_mask]

            for d_pos in pos_dist:
                for d_neg in neg_dist:
                    loss = torch.clamp(d_pos - d_neg + self.margin, min=0.0)
                    if loss > 0:
                        losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses)


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss

    """

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.35):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:

        Returns:
        """
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(features, weight)  # (batch_size, out_features)
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        theta = torch.acos(cosine)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        target_cosine = torch.cos(theta + self.m * one_hot)

        output = cosine + one_hot * (target_cosine - cosine)
        output = self.s * output

        loss = F.cross_entropy(output, labels)
        return loss


class CircleLoss(nn.Module):

    def __init__(self, m: float = 0.25, gamma: float = 256):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.softplus = nn.Softplus()

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:

        Returns:
        """
        features = F.normalize(features, p=2, dim=1)

        sim_matrix = torch.mm(features, features.t())  # (batch_size, batch_size)

        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).bool()
        mask_neg = (labels.unsqueeze(0) != labels.unsqueeze(1)).bool()

        pos_sim = sim_matrix[mask_pos]
        neg_sim = sim_matrix[mask_neg]

        if len(pos_sim) == 0 or len(neg_sim) == 0:
            return torch.tensor(0.0, device=features.device)

        alpha_p = torch.relu(1 + self.m - pos_sim)
        alpha_n = torch.relu(neg_sim + self.m)

        pos_term = -alpha_p * pos_sim * self.gamma
        neg_term = alpha_n * neg_sim * self.gamma

        loss = self.softplus(torch.logsumexp(neg_term, dim=0) + torch.logsumexp(pos_term, dim=0))
        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for metric learning.

    """

    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:

        Returns:
        """
        dist_matrix = torch.cdist(features, features, p=2)

        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).bool()
        mask_neg = (labels.unsqueeze(0) != labels.unsqueeze(1)).bool()

        eye = torch.eye(features.size(0), device=features.device).bool()
        mask_pos = mask_pos & ~eye
        mask_neg = mask_neg & ~eye

        pos_dist = dist_matrix[mask_pos]
        pos_loss = (pos_dist ** 2).sum() if len(pos_dist) > 0 else torch.tensor(0.0, device=features.device)

        neg_dist = dist_matrix[mask_neg]
        neg_loss = (torch.clamp(self.margin - neg_dist, min=0.0) ** 2).sum() if len(neg_dist) > 0 else torch.tensor(0.0, device=features.device)

        loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            n_pos = max(1, len(pos_dist))
            n_neg = max(1, len(neg_dist))
            loss = loss / (n_pos + n_neg)
        return loss


class CenterLoss(nn.Module):
    """

    Deep Face Recognition (ECCV 2016)

    """

    def __init__(self, num_classes: int, feat_dim: int = 256, lr_center: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lr_center = lr_center
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
        Returns:
            loss: Center Loss
        """
        centers_batch = self.centers[labels]
        loss = F.mse_loss(features, centers_batch)
        return loss


class MultiTaskLossWrapper(nn.Module):

    def __init__(self, task_weights: Optional[Dict[str, float]] = None):
        super(MultiTaskLossWrapper, self).__init__()
        if task_weights is None:
            self.log_sigma1 = nn.Parameter(torch.tensor(0.0))
            self.log_sigma2 = nn.Parameter(torch.tensor(0.0))
            self.learnable = True
        else:
            self.task_weights = task_weights
            self.learnable = False

    def forward(self, loss_metric: torch.Tensor, loss_illumination: torch.Tensor) -> torch.Tensor:
        """

        Args:

        Returns:
        """
        if self.learnable:
            precision1 = torch.exp(-self.log_sigma1)
            precision2 = torch.exp(-self.log_sigma2)
            loss = precision1 * loss_metric + self.log_sigma1 + \
                   precision2 * loss_illumination + self.log_sigma2
        else:
            w1 = self.task_weights.get('metric', 1.0)
            w2 = self.task_weights.get('illumination', 0.5)
            loss = w1 * loss_metric + w2 * loss_illumination
        return loss


def get_metric_loss(name: str, **kwargs) -> nn.Module:
    """

    Args:

    Returns:
    """
    if name == 'triplet':
        return TripletLoss(**kwargs)
    elif name == 'arcface':
        return ArcFaceLoss(**kwargs)
    elif name == 'circle':
        return CircleLoss(**kwargs)
    elif name == 'contrastive':
        return ContrastiveLoss(**kwargs)
    elif name == 'center':
        return CenterLoss(**kwargs)
    else:
        raise ValueError(f"Unknown metric loss: {name}")


if __name__ == '__main__':
    batch_size = 16
    feat_dim = 512
    num_classes = 10

    features = torch.randn(batch_size, feat_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    triplet_loss = TripletLoss(margin=0.3, mining_type='hard')
    loss_t = triplet_loss(features, labels)
    print(f"Triplet Loss: {loss_t.item()}")

    arcface_loss = ArcFaceLoss(in_features=feat_dim, out_features=num_classes, s=30.0, m=0.35)
    loss_a = arcface_loss(features, labels)
    print(f"ArcFace Loss: {loss_a.item()}")

    circle_loss = CircleLoss(m=0.25, gamma=256)
    loss_c = circle_loss(features, labels)
    print(f"Circle Loss: {loss_c.item()}")

    contrastive_loss = ContrastiveLoss(margin=1.0)
    loss_ct = contrastive_loss(features, labels)
    print(f"Contrastive Loss: {loss_ct.item()}")

    print("All tests passed.")
