#!/usr/bin/env python3
"""K-reciprocal re-ranking utilities for ReID retrieval.

Reference:
    "Re-ranking Person Re-identification with k-reciprocal Encoding"
    https://arxiv.org/abs/1701.08398

The high-level idea is:
    1. Find the k nearest neighbors of each sample.
    2. Keep neighbors that are mutual under the same k-neighbor rule.
    3. Use those reciprocal neighborhoods to build a more robust similarity
       estimate.
    4. Blend the reciprocal-neighborhood distance with the original distance.
"""

from __future__ import annotations

import numpy as np


def compute_euclidean_distance(query_features: np.ndarray, gallery_features: np.ndarray) -> np.ndarray:
    """Compute the pairwise squared Euclidean distance matrix."""
    q_sq = np.sum(query_features**2, axis=1, keepdims=True)
    g_sq = np.sum(gallery_features**2, axis=1, keepdims=True).T
    dist = q_sq + g_sq - 2 * np.dot(query_features, gallery_features.T)
    dist = np.clip(dist, 0, None)
    return dist


def k_reciprocal_neighbors(initial_rank: np.ndarray, i: int, k1: int) -> np.ndarray:
    """Return the mutual k-nearest-neighbor set for sample `i`."""
    forward_k_neighbors = initial_rank[i, : k1 + 1]
    reciprocal_neighbors = []
    for candidate in forward_k_neighbors:
        backward_k_neighbors = initial_rank[candidate, : k1 + 1]
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
    """Run full k-reciprocal re-ranking."""
    num_query = query_features.shape[0]
    num_gallery = gallery_features.shape[0]
    num_all = num_query + num_gallery

    print(f"[Re-ranking] Query: {num_query}, Gallery: {num_gallery}, Total: {num_all}")
    print(f"[Re-ranking] Parameters: k1={k1}, k2={k2}, lambda={lambda_value}")

    all_features = np.vstack([query_features, gallery_features])

    print("[Re-ranking] Computing the global distance matrix...")
    original_dist = compute_euclidean_distance(all_features, all_features)
    original_dist = original_dist / np.max(original_dist)
    initial_rank = np.argsort(original_dist, axis=1)

    print("[Re-ranking] Building reciprocal-neighborhood encodings...")
    V = np.zeros((num_all, num_all), dtype=np.float32)

    for i in range(num_all):
        k_reciprocal = k_reciprocal_neighbors(initial_rank, i, k1)
        k_reciprocal_expansion = k_reciprocal.copy()

        for candidate in k_reciprocal:
            candidate_k_reciprocal = k_reciprocal_neighbors(initial_rank, candidate, int(k1 / 2))
            if len(candidate_k_reciprocal) > 0:
                intersection = np.intersect1d(k_reciprocal, candidate_k_reciprocal)
                if len(intersection) > 2 / 3 * len(candidate_k_reciprocal):
                    k_reciprocal_expansion = np.union1d(k_reciprocal_expansion, candidate_k_reciprocal)

        weight = np.exp(-original_dist[i, k_reciprocal_expansion])
        V[i, k_reciprocal_expansion] = weight / np.sum(weight)

    print("[Re-ranking] Applying local smoothing...")
    if k2 > 1:
        V_smoothed = np.zeros_like(V)
        for i in range(num_all):
            k2_neighbors = initial_rank[i, :k2]
            V_smoothed[i] = np.mean(V[k2_neighbors], axis=0)
        V = V_smoothed

    print("[Re-ranking] Computing Jaccard distance...")
    jaccard_dist = np.zeros((num_query, num_gallery), dtype=np.float32)
    for i in range(num_query):
        for j in range(num_gallery):
            min_sum = np.sum(np.minimum(V[i], V[num_query + j]))
            max_sum = np.sum(np.maximum(V[i], V[num_query + j]))
            if max_sum > 0:
                jaccard_dist[i, j] = 1 - min_sum / max_sum
            else:
                jaccard_dist[i, j] = 1.0

    original_dist_qg = original_dist[:num_query, num_query:]

    print("[Re-ranking] Blending original and reciprocal distances...")
    final_dist = (1 - lambda_value) * jaccard_dist + lambda_value * original_dist_qg
    print("[Re-ranking] Done.")
    return final_dist


def re_ranking_fast(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """Run a lower-memory approximation of k-reciprocal re-ranking."""
    num_query = query_features.shape[0]
    num_gallery = gallery_features.shape[0]

    print(f"[Re-ranking Fast] Query: {num_query}, Gallery: {num_gallery}")

    qg_dist = compute_euclidean_distance(query_features, gallery_features)
    qq_dist = compute_euclidean_distance(query_features, query_features)
    gg_dist = compute_euclidean_distance(gallery_features, gallery_features)

    num_all = num_query + num_gallery
    original_dist = np.zeros((num_all, num_all), dtype=np.float32)
    original_dist[:num_query, :num_query] = qq_dist
    original_dist[:num_query, num_query:] = qg_dist
    original_dist[num_query:, :num_query] = qg_dist.T
    original_dist[num_query:, num_query:] = gg_dist
    original_dist = original_dist / (np.max(original_dist) + 1e-8)
    initial_rank = np.argsort(original_dist, axis=1)

    print("[Re-ranking Fast] Computing reciprocal overlap scores...")
    final_dist = np.zeros((num_query, num_gallery), dtype=np.float32)
    for i in range(num_query):
        k_recip_i = k_reciprocal_neighbors(initial_rank, i, k1)
        for j in range(num_gallery):
            gj = num_query + j
            k_recip_j = k_reciprocal_neighbors(initial_rank, gj, k1)
            intersection = len(np.intersect1d(k_recip_i, k_recip_j))
            union = len(np.union1d(k_recip_i, k_recip_j))
            jaccard_sim = intersection / union if union > 0 else 0.0
            final_dist[i, j] = (1 - lambda_value) * (1 - jaccard_sim) + lambda_value * original_dist[i, gj]

    print("[Re-ranking Fast] Done.")
    return final_dist


if __name__ == "__main__":
    np.random.seed(42)
    query_feats = np.random.randn(10, 512).astype(np.float32)
    gallery_feats = np.random.randn(50, 512).astype(np.float32)

    query_feats = query_feats / np.linalg.norm(query_feats, axis=1, keepdims=True)
    gallery_feats = gallery_feats / np.linalg.norm(gallery_feats, axis=1, keepdims=True)

    print("Testing re-ranking...")
    final_dist = re_ranking(query_feats, gallery_feats, k1=10, k2=3, lambda_value=0.3)
    print(f"Output distance-matrix shape: {final_dist.shape}")
    print("Re-ranking smoke test passed.")
