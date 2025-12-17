"""
检索式ReID度量学习损失函数集合

实现Triplet Loss、ArcFace Loss、Circle Loss、Contrastive Loss等，
支持难例挖掘和批次采样策略。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict


class TripletLoss(nn.Module):
    """
    Triplet Loss with hard negative mining (改进版).

    支持多种难例挖掘策略：
    - 'hard': 最难的负样本
    - 'semi-hard': 半难负样本（负样本距离大于正样本距离但小于margin）
    - 'soft': 软加权挖掘（新增，更稳定）
    - 'all': 所有负样本
    
    改进点：
    1. 添加soft mining策略，避免训练初期的不稳定
    2. 添加异常值过滤，跳过可能的噪声样本（带warm-up机制）
    3. 支持soft-margin（平滑边界）
    """

    def __init__(self, margin: float = 0.3, mining_type: str = 'soft', 
                 reduction: str = 'mean', soft_margin: bool = True,
                 outlier_threshold: float = 3.0, warmup_epochs: int = 5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mining_type = mining_type  # 'hard', 'semi-hard', 'soft', 'all'
        self.reduction = reduction
        self.soft_margin = soft_margin  # 使用softplus代替hard margin
        self.outlier_threshold = outlier_threshold  # 异常值阈值（标准差倍数）
        # 【新增】warm-up机制：训练前期禁用异常值过滤
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self._call_count = 0  # 用于自动估计epoch
        self._estimated_steps_per_epoch = 100  # 默认估计值
        assert mining_type in ['hard', 'semi-hard', 'soft', 'all'], f"Unknown mining type: {mining_type}"
    
    def set_epoch(self, epoch: int):
        """手动设置当前epoch，用于warm-up控制"""
        self.current_epoch = epoch
    
    def _auto_update_epoch(self):
        """自动估计当前epoch（如果未手动设置）"""
        self._call_count += 1
        # 每N步自动增加epoch计数
        if self._call_count % self._estimated_steps_per_epoch == 0:
            self.current_epoch = self._call_count // self._estimated_steps_per_epoch

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 特征张量，形状 (batch_size, feature_dim)
            labels: 标签张量，形状 (batch_size,)

        Returns:
            loss: 计算得到的Triplet Loss
        """
        # 自动更新epoch（如果未手动设置）
        self._auto_update_epoch()
        # 计算距离矩阵（欧氏距离）
        dist_matrix = self._pairwise_distance(features)

        # 获取正样本对和负样本对的掩码
        mask_positive = self._get_positive_mask(labels)  # (batch_size, batch_size)
        mask_negative = self._get_negative_mask(labels)  # (batch_size, batch_size)

        # 根据挖掘策略计算损失
        if self.mining_type == 'hard':
            loss = self._hard_mining(dist_matrix, mask_positive, mask_negative)
        elif self.mining_type == 'semi-hard':
            loss = self._semi_hard_mining(dist_matrix, mask_positive, mask_negative)
        elif self.mining_type == 'soft':
            loss = self._soft_mining(dist_matrix, mask_positive, mask_negative)
        else:  # 'all'
            loss = self._all_triplets(dist_matrix, mask_positive, mask_negative)

        # 异常值过滤
        loss = self._filter_outliers(loss)

        # 损失归约
        if self.reduction == 'mean':
            loss = loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=features.device)
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

    def _pairwise_distance(self, x: torch.Tensor) -> torch.Tensor:
        """计算欧氏距离矩阵"""
        x_squared = (x ** 2).sum(dim=1, keepdim=True)
        dist_matrix = x_squared + x_squared.t() - 2 * torch.mm(x, x.t())
        # 防止数值误差
        dist_matrix = torch.clamp(dist_matrix, min=1e-8)
        return torch.sqrt(dist_matrix)

    def _get_positive_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """生成正样本对掩码（相同标签）"""
        return (labels.unsqueeze(0) == labels.unsqueeze(1)).bool()

    def _get_negative_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """生成负样本对掩码（不同标签）"""
        return (labels.unsqueeze(0) != labels.unsqueeze(1)).bool()

    def _filter_outliers(self, loss: torch.Tensor) -> torch.Tensor:
        """
        过滤异常值损失，防止噪声样本影响训练
        
        【改进】添加warm-up机制：训练前期不进行异常值过滤，
        因为早期所有损失值都可能较高，过滤会移除有效样本
        """
        if loss.numel() == 0:
            return loss
        
        # 【新增】warm-up期间跳过异常值过滤
        if self.current_epoch < self.warmup_epochs:
            return loss
        
        # 计算均值和标准差
        mean_loss = loss.mean()
        std_loss = loss.std()
        
        # 过滤超过阈值的异常值
        if std_loss > 0:
            mask = loss < (mean_loss + self.outlier_threshold * std_loss)
            # 【改进】确保至少保留一半的样本，避免过度过滤
            if mask.sum() < loss.numel() // 2:
                # 如果过滤掉太多，使用更宽松的阈值
                mask = loss < (mean_loss + self.outlier_threshold * 2 * std_loss)
            loss = loss[mask]
        
        return loss

    def _apply_margin(self, diff: torch.Tensor) -> torch.Tensor:
        """应用margin（支持soft margin）"""
        if self.soft_margin:
            # 使用softplus进行平滑，避免梯度消失
            return F.softplus(diff + self.margin)
        else:
            return torch.clamp(diff + self.margin, min=0.0)

    def _hard_mining(self, dist_matrix: torch.Tensor, mask_positive: torch.Tensor, 
                     mask_negative: torch.Tensor) -> torch.Tensor:
        """最难负样本挖掘"""
        # 对每个anchor，选择最难的正样本（距离最大的）和最难的负样本（距离最小的）
        batch_size = dist_matrix.size(0)
        device = dist_matrix.device

        # 最难正样本距离
        dist_positive = dist_matrix[mask_positive].view(batch_size, -1)
        hardest_positive, _ = dist_positive.max(dim=1)  # (batch_size,)

        # 最难负样本距离
        dist_negative = dist_matrix[mask_negative].view(batch_size, -1)
        hardest_negative, _ = dist_negative.min(dim=1)  # (batch_size,)

        # 计算Triplet Loss
        loss = self._apply_margin(hardest_positive - hardest_negative)
        return loss
    
    def _soft_mining(self, dist_matrix: torch.Tensor, mask_positive: torch.Tensor,
                     mask_negative: torch.Tensor) -> torch.Tensor:
        """
        软加权挖掘（新增）：使用距离加权而非最难样本
        
        相比hard mining更稳定，不容易被噪声样本影响
        """
        batch_size = dist_matrix.size(0)
        device = dist_matrix.device
        
        losses = []
        for i in range(batch_size):
            # 正样本距离
            pos_mask = mask_positive[i].clone()
            pos_mask[i] = False  # 排除自身
            if not pos_mask.any():
                continue
            pos_dist = dist_matrix[i][pos_mask]
            
            # 负样本距离
            neg_mask = mask_negative[i]
            if not neg_mask.any():
                continue
            neg_dist = dist_matrix[i][neg_mask]
            
            # 软加权：正样本按距离加权（距离越大权重越高）
            pos_weights = F.softmax(pos_dist, dim=0)
            weighted_pos_dist = (pos_dist * pos_weights).sum()
            
            # 软加权：负样本按距离加权（距离越小权重越高）
            neg_weights = F.softmax(-neg_dist, dim=0)  # 负号使距离小的权重大
            weighted_neg_dist = (neg_dist * neg_weights).sum()
            
            # 计算损失
            loss = self._apply_margin(weighted_pos_dist - weighted_neg_dist)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses)

    def _semi_hard_mining(self, dist_matrix: torch.Tensor, mask_positive: torch.Tensor,
                          mask_negative: torch.Tensor) -> torch.Tensor:
        """半难负样本挖掘"""
        batch_size = dist_matrix.size(0)
        device = dist_matrix.device

        losses = []
        for i in range(batch_size):
            # 当前anchor的正样本距离
            pos_mask = mask_positive[i]
            if not pos_mask.any():
                continue
            pos_dist = dist_matrix[i][pos_mask]
            hardest_pos = pos_dist.max()  # 最难正样本

            # 负样本距离需要满足：dist_neg > dist_pos 且 dist_neg < dist_pos + margin
            neg_mask = mask_negative[i]
            if not neg_mask.any():
                continue
            neg_dist = dist_matrix[i][neg_mask]

            # 找到满足条件的负样本
            semi_hard_mask = (neg_dist > hardest_pos) & (neg_dist < hardest_pos + self.margin)
            if semi_hard_mask.any():
                semi_hard_dist = neg_dist[semi_hard_mask]
                # 选择最小的满足条件的负样本
                hardest_neg = semi_hard_dist.min()
                loss = torch.clamp(hardest_pos - hardest_neg + self.margin, min=0.0)
                losses.append(loss)
            else:
                # 如果没有满足条件的负样本，使用最难负样本
                hardest_neg = neg_dist.min()
                loss = torch.clamp(hardest_pos - hardest_neg + self.margin, min=0.0)
                losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses)

    def _all_triplets(self, dist_matrix: torch.Tensor, mask_positive: torch.Tensor,
                      mask_negative: torch.Tensor) -> torch.Tensor:
        """所有有效三元组的损失"""
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

            # 计算所有正负对组合
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

    参考文献: Jiankang Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.35):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # 缩放因子
        self.m = m  # 角度边际

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 特征张量，形状 (batch_size, in_features)，已归一化
            labels: 标签张量，形状 (batch_size,)

        Returns:
            loss: ArcFace损失
        """
        # 特征和权重归一化
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # 计算cosine相似度
        cosine = F.linear(features, weight)  # (batch_size, out_features)
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        # 计算theta
        theta = torch.acos(cosine)

        # 对目标类别添加角度边际
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        target_cosine = torch.cos(theta + self.m * one_hot)

        # 缩放
        output = cosine + one_hot * (target_cosine - cosine)
        output = self.s * output

        # 计算交叉熵损失
        loss = F.cross_entropy(output, labels)
        return loss


class CircleLoss(nn.Module):
    """
    Circle Loss: 更灵活的正负样本权重分配

    参考文献: Yifan Sun et al. "Circle Loss: A Unified Perspective of Pair Similarity Optimization"
    """

    def __init__(self, m: float = 0.25, gamma: float = 256):
        super(CircleLoss, self).__init__()
        self.m = m  # 边际
        self.gamma = gamma  # 缩放因子
        self.softplus = nn.Softplus()

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 特征张量，形状 (batch_size, feature_dim)，已归一化
            labels: 标签张量，形状 (batch_size,)

        Returns:
            loss: Circle损失
        """
        # 归一化特征
        features = F.normalize(features, p=2, dim=1)

        # 计算相似度矩阵（内积）
        sim_matrix = torch.mm(features, features.t())  # (batch_size, batch_size)

        # 构建正负样本掩码
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).bool()
        mask_neg = (labels.unsqueeze(0) != labels.unsqueeze(1)).bool()

        # 正负样本相似度
        pos_sim = sim_matrix[mask_pos]
        neg_sim = sim_matrix[mask_neg]

        if len(pos_sim) == 0 or len(neg_sim) == 0:
            return torch.tensor(0.0, device=features.device)

        # 自适应权重
        alpha_p = torch.relu(1 + self.m - pos_sim)
        alpha_n = torch.relu(neg_sim + self.m)

        # Circle损失公式
        pos_term = -alpha_p * pos_sim * self.gamma
        neg_term = alpha_n * neg_sim * self.gamma

        loss = self.softplus(torch.logsumexp(neg_term, dim=0) + torch.logsumexp(pos_term, dim=0))
        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for metric learning.

    最大化正样本对相似度，最小化负样本对相似度（如果距离小于边际）。
    """

    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 特征张量，形状 (batch_size, feature_dim)
            labels: 标签张量，形状 (batch_size,)

        Returns:
            loss: 对比损失
        """
        # 计算欧氏距离矩阵
        dist_matrix = torch.cdist(features, features, p=2)

        # 构建正负样本掩码
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).bool()
        mask_neg = (labels.unsqueeze(0) != labels.unsqueeze(1)).bool()

        # 对角线设为False（自身不算）
        eye = torch.eye(features.size(0), device=features.device).bool()
        mask_pos = mask_pos & ~eye
        mask_neg = mask_neg & ~eye

        # 正样本损失（距离应小）
        pos_dist = dist_matrix[mask_pos]
        pos_loss = (pos_dist ** 2).sum() if len(pos_dist) > 0 else torch.tensor(0.0, device=features.device)

        # 负样本损失（距离应大于边际）
        neg_dist = dist_matrix[mask_neg]
        neg_loss = (torch.clamp(self.margin - neg_dist, min=0.0) ** 2).sum() if len(neg_dist) > 0 else torch.tensor(0.0, device=features.device)

        # 总损失
        loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            n_pos = max(1, len(pos_dist))
            n_neg = max(1, len(neg_dist))
            loss = loss / (n_pos + n_neg)
        return loss


class MultiTaskLossWrapper(nn.Module):
    """
    多任务损失包装器，支持不确定性加权 (Uncertainty Weighting) 和固定权重。
    """

    def __init__(self, task_weights: Optional[Dict[str, float]] = None):
        super(MultiTaskLossWrapper, self).__init__()
        if task_weights is None:
            # 可学习的不确定性参数
            self.log_sigma1 = nn.Parameter(torch.tensor(0.0))  # 度量学习任务
            self.log_sigma2 = nn.Parameter(torch.tensor(0.0))  # 光照任务
            self.learnable = True
        else:
            self.task_weights = task_weights
            self.learnable = False

    def forward(self, loss_metric: torch.Tensor, loss_illumination: torch.Tensor) -> torch.Tensor:
        """
        组合两个任务的损失。

        Args:
            loss_metric: 度量学习损失
            loss_illumination: 光照不变性损失

        Returns:
            total_loss: 加权总损失
        """
        if self.learnable:
            # 不确定性加权
            precision1 = torch.exp(-self.log_sigma1)
            precision2 = torch.exp(-self.log_sigma2)
            loss = precision1 * loss_metric + self.log_sigma1 + \
                   precision2 * loss_illumination + self.log_sigma2
        else:
            # 固定权重
            w1 = self.task_weights.get('metric', 1.0)
            w2 = self.task_weights.get('illumination', 0.5)
            loss = w1 * loss_metric + w2 * loss_illumination
        return loss


# 快捷函数
def get_metric_loss(name: str, **kwargs) -> nn.Module:
    """
    根据名称获取度量学习损失函数。

    Args:
        name: 损失函数名称 ('triplet', 'arcface', 'circle', 'contrastive')
        **kwargs: 对应损失函数的参数

    Returns:
        损失函数实例
    """
    if name == 'triplet':
        return TripletLoss(**kwargs)
    elif name == 'arcface':
        return ArcFaceLoss(**kwargs)
    elif name == 'circle':
        return CircleLoss(**kwargs)
    elif name == 'contrastive':
        return ContrastiveLoss(**kwargs)
    else:
        raise ValueError(f"Unknown metric loss: {name}")


if __name__ == '__main__':
    # 简单测试
    batch_size = 16
    feat_dim = 512
    num_classes = 10

    features = torch.randn(batch_size, feat_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    # 测试Triplet Loss
    triplet_loss = TripletLoss(margin=0.3, mining_type='hard')
    loss_t = triplet_loss(features, labels)
    print(f"Triplet Loss: {loss_t.item()}")

    # 测试ArcFace Loss
    arcface_loss = ArcFaceLoss(in_features=feat_dim, out_features=num_classes, s=30.0, m=0.35)
    loss_a = arcface_loss(features, labels)
    print(f"ArcFace Loss: {loss_a.item()}")

    # 测试Circle Loss
    circle_loss = CircleLoss(m=0.25, gamma=256)
    loss_c = circle_loss(features, labels)
    print(f"Circle Loss: {loss_c.item()}")

    # 测试Contrastive Loss
    contrastive_loss = ContrastiveLoss(margin=1.0)
    loss_ct = contrastive_loss(features, labels)
    print(f"Contrastive Loss: {loss_ct.item()}")

    print("All tests passed.")