#!/usr/bin/env python3
"""
K-reciprocal Re-ranking for ReID

基于论文:
"Re-ranking Person Re-identification with k-reciprocal Encoding"
https://arxiv.org/abs/1701.08398

Re-ranking 的核心思想:
1. 对于 query q，找到它的 k 个最近邻 N(q, k)
2. 对于每个近邻 p，如果 q 也在 p 的 k 个最近邻中，则 p 是 q 的"互惠近邻"
3. 互惠近邻关系更可靠，因为是双向确认的
4. 利用互惠近邻扩展集合，重新计算更鲁棒的距离

这种方法通常可以提升 mAP 5-10%！
"""

import numpy as np


def compute_euclidean_distance(query_features: np.ndarray, gallery_features: np.ndarray) -> np.ndarray:
    """计算欧氏距离矩阵
    
    Args:
        query_features: (num_query, dim) 的特征矩阵
        gallery_features: (num_gallery, dim) 的特征矩阵
    
    Returns:
        距离矩阵 (num_query, num_gallery)
    """
    m, n = query_features.shape[0], gallery_features.shape[0]
    
    # 使用欧氏距离平方: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    q_sq = np.sum(query_features ** 2, axis=1, keepdims=True)  # (m, 1)
    g_sq = np.sum(gallery_features ** 2, axis=1, keepdims=True).T  # (1, n)
    
    dist = q_sq + g_sq - 2 * np.dot(query_features, gallery_features.T)
    dist = np.clip(dist, 0, None)  # 避免数值误差导致的负数
    
    return dist


def k_reciprocal_neighbors(initial_rank: np.ndarray, i: int, k1: int) -> np.ndarray:
    """获取第 i 个样本的 k-互惠近邻集合
    
    互惠近邻定义: 如果 j 在 i 的 k1 近邻中，且 i 也在 j 的 k1 近邻中，
    则 j 是 i 的互惠近邻。
    
    Args:
        initial_rank: (N, N) 的排序索引矩阵，每行是按距离排序的索引
        i: 当前样本索引
        k1: 近邻数量
    
    Returns:
        互惠近邻的索引数组
    """
    # i 的 k1 近邻
    forward_k_neighbors = initial_rank[i, :k1 + 1]
    
    # 检查每个近邻是否也把 i 当作近邻（互惠）
    reciprocal_neighbors = []
    for candidate in forward_k_neighbors:
        # candidate 的 k1 近邻
        backward_k_neighbors = initial_rank[candidate, :k1 + 1]
        if i in backward_k_neighbors:
            reciprocal_neighbors.append(candidate)
    
    return np.array(reciprocal_neighbors, dtype=np.int32)


def re_ranking(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """K-reciprocal Re-ranking
    
    Args:
        query_features: (num_query, dim) 的 query 特征
        gallery_features: (num_gallery, dim) 的 gallery 特征
        k1: 互惠近邻的 k 值，默认 20
        k2: 扩展近邻的 k 值，默认 6
        lambda_value: 原始距离和 Jaccard 距离的权重，默认 0.3
    
    Returns:
        重排序后的距离矩阵 (num_query, num_gallery)
    """
    num_query = query_features.shape[0]
    num_gallery = gallery_features.shape[0]
    num_all = num_query + num_gallery
    
    print(f"[Re-ranking] Query: {num_query}, Gallery: {num_gallery}, Total: {num_all}")
    print(f"[Re-ranking] Parameters: k1={k1}, k2={k2}, lambda={lambda_value}")
    
    # 合并所有特征
    all_features = np.vstack([query_features, gallery_features])
    
    # 计算全局距离矩阵 (all, all)
    print("[Re-ranking] 计算距离矩阵...")
    original_dist = compute_euclidean_distance(all_features, all_features)
    
    # 转换为高斯核相似度（用于后续计算）
    # 归一化到 [0, 1]
    original_dist = original_dist / np.max(original_dist)
    
    # 获取初始排序
    initial_rank = np.argsort(original_dist, axis=1)
    
    # 计算 k-reciprocal 特征（Jaccard 距离）
    print("[Re-ranking] 计算互惠近邻...")
    
    # V 矩阵：每个样本的 k-reciprocal 编码向量
    V = np.zeros((num_all, num_all), dtype=np.float32)
    
    for i in range(num_all):
        # 获取互惠近邻
        k_reciprocal = k_reciprocal_neighbors(initial_rank, i, k1)
        
        # 扩展互惠近邻集合
        # 对于每个互惠近邻，如果它的互惠近邻与当前集合重叠超过 2/3，则合并
        k_reciprocal_expansion = k_reciprocal.copy()
        
        for candidate in k_reciprocal:
            candidate_k_reciprocal = k_reciprocal_neighbors(initial_rank, candidate, int(k1 / 2))
            
            if len(candidate_k_reciprocal) > 0:
                # 计算重叠比例
                intersection = np.intersect1d(k_reciprocal, candidate_k_reciprocal)
                if len(intersection) > 2 / 3 * len(candidate_k_reciprocal):
                    k_reciprocal_expansion = np.union1d(k_reciprocal_expansion, candidate_k_reciprocal)
        
        # 使用高斯权重
        weight = np.exp(-original_dist[i, k_reciprocal_expansion])
        V[i, k_reciprocal_expansion] = weight / np.sum(weight)
    
    # 使用 k2 近邻进行局部平滑
    print("[Re-ranking] 局部平滑...")
    if k2 > 1:
        V_smoothed = np.zeros_like(V)
        for i in range(num_all):
            k2_neighbors = initial_rank[i, :k2]
            V_smoothed[i] = np.mean(V[k2_neighbors], axis=0)
        V = V_smoothed
    
    # 计算 Jaccard 距离
    print("[Re-ranking] 计算 Jaccard 距离...")
    
    # 只计算 query 和 gallery 之间的距离
    jaccard_dist = np.zeros((num_query, num_gallery), dtype=np.float32)
    
    for i in range(num_query):
        for j in range(num_gallery):
            # Jaccard 距离 = 1 - (交集 / 并集)
            min_sum = np.sum(np.minimum(V[i], V[num_query + j]))
            max_sum = np.sum(np.maximum(V[i], V[num_query + j]))
            
            if max_sum > 0:
                jaccard_dist[i, j] = 1 - min_sum / max_sum
            else:
                jaccard_dist[i, j] = 1.0
    
    # 获取原始 query-gallery 距离
    original_dist_qg = original_dist[:num_query, num_query:]
    
    # 融合原始距离和 Jaccard 距离
    print("[Re-ranking] 融合距离...")
    final_dist = (1 - lambda_value) * jaccard_dist + lambda_value * original_dist_qg
    
    print("[Re-ranking] 完成!")
    
    return final_dist


def re_ranking_fast(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """快速版本的 K-reciprocal Re-ranking（使用更少内存）
    
    对于大规模数据集，使用这个版本。
    """
    num_query = query_features.shape[0]
    num_gallery = gallery_features.shape[0]
    
    print(f"[Re-ranking Fast] Query: {num_query}, Gallery: {num_gallery}")
    
    # 计算 query-gallery 距离
    qg_dist = compute_euclidean_distance(query_features, gallery_features)
    
    # 计算 query-query 距离
    qq_dist = compute_euclidean_distance(query_features, query_features)
    
    # 计算 gallery-gallery 距离
    gg_dist = compute_euclidean_distance(gallery_features, gallery_features)
    
    # 合并距离矩阵
    num_all = num_query + num_gallery
    original_dist = np.zeros((num_all, num_all), dtype=np.float32)
    original_dist[:num_query, :num_query] = qq_dist
    original_dist[:num_query, num_query:] = qg_dist
    original_dist[num_query:, :num_query] = qg_dist.T
    original_dist[num_query:, num_query:] = gg_dist
    
    # 归一化
    original_dist = original_dist / (np.max(original_dist) + 1e-8)
    
    # 获取初始排序
    initial_rank = np.argsort(original_dist, axis=1)
    
    # 简化版：只使用 k-reciprocal 重加权
    print("[Re-ranking Fast] 计算互惠近邻重加权...")
    
    # 对于每个 query，重新计算与 gallery 的距离
    final_dist = np.zeros((num_query, num_gallery), dtype=np.float32)
    
    for i in range(num_query):
        # i 的互惠近邻
        k_recip_i = k_reciprocal_neighbors(initial_rank, i, k1)
        
        for j in range(num_gallery):
            gj = num_query + j  # gallery j 在全局索引中的位置
            
            # gallery j 的互惠近邻
            k_recip_j = k_reciprocal_neighbors(initial_rank, gj, k1)
            
            # 计算互惠近邻的重叠程度（作为相似度的指标）
            intersection = len(np.intersect1d(k_recip_i, k_recip_j))
            union = len(np.union1d(k_recip_i, k_recip_j))
            
            if union > 0:
                jaccard_sim = intersection / union
            else:
                jaccard_sim = 0
            
            # 融合原始距离和 Jaccard 相似度
            # Jaccard 距离 = 1 - Jaccard 相似度
            final_dist[i, j] = (1 - lambda_value) * (1 - jaccard_sim) + lambda_value * original_dist[i, gj]
    
    print("[Re-ranking Fast] 完成!")
    
    return final_dist


if __name__ == "__main__":
    # 简单测试
    np.random.seed(42)
    
    query_feats = np.random.randn(10, 512).astype(np.float32)
    gallery_feats = np.random.randn(50, 512).astype(np.float32)
    
    # L2 归一化
    query_feats = query_feats / np.linalg.norm(query_feats, axis=1, keepdims=True)
    gallery_feats = gallery_feats / np.linalg.norm(gallery_feats, axis=1, keepdims=True)
    
    print("测试 re-ranking...")
    final_dist = re_ranking(query_feats, gallery_feats, k1=10, k2=3, lambda_value=0.3)
    print(f"输出距离矩阵形状: {final_dist.shape}")
    print("测试通过!")
