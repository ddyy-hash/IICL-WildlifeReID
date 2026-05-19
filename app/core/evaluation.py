#!/usr/bin/env python3
"""Unified ReID evaluation utilities."""

from __future__ import annotations

import gc
import json
import logging
import os
import platform
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

try:
    from tools.reranking import re_ranking

    RERANKING_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    RERANKING_AVAILABLE = False


class ReIDDataset(Dataset):
    """Generic folder-based ReID dataset.

    Supports:
    1) root/<identity>/<image>
    2) explicit samples: [(img_path, identity), ...]

    Returns (image_tensor, identity, camera_id, img_path).
    """

    def __init__(
        self,
        root: Optional[str] = None,
        samples: Optional[Sequence[Tuple[str, Any]]] = None,
        transform: Optional[Any] = None,
        extensions: Sequence[str] = (".jpg", ".jpeg", ".png"),
    ) -> None:
        self.transform = transform
        self.samples: List[Tuple[str, Any, int]] = []

        if samples is not None:
            for img_path, identity in samples:
                cam_id = self.extract_camera_id(os.path.basename(img_path))
                self.samples.append((str(img_path), identity, cam_id))
        elif root is not None:
            if not os.path.isdir(root):
                raise FileNotFoundError(f"dataset directory not found: {root}")
            for identity_name in sorted(os.listdir(root)):
                identity_dir = os.path.join(root, identity_name)
                if not os.path.isdir(identity_dir):
                    continue
                for fname in sorted(os.listdir(identity_dir)):
                    if any(fname.lower().endswith(ext) for ext in extensions):
                        img_path = os.path.join(identity_dir, fname)
                        cam_id = self.extract_camera_id(fname)
                        self.samples.append((img_path, identity_name, cam_id))
        else:
            raise ValueError("Either root or samples must be provided")

        ids = sorted(set(item[1] for item in self.samples))
        logger.info("Loaded evaluation dataset: %d images, %d identities", len(self.samples), len(ids))

    @staticmethod
    def extract_camera_id(img_name: str) -> int:
        """Extract camera id from common ReID filename patterns, else -1."""
        match = re.search(r"_c(\d+)[_s]", img_name)
        if match:
            return int(match.group(1))
        return -1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any, int, str]:
        img_path, identity, cam_id = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read evaluation image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        return img, identity, cam_id, img_path


def _release_cuda_memory() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:  # pragma: no cover - best effort only
        pass


def _forward_feature_batch(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    flip_test: bool,
) -> torch.Tensor:
    output = model(imgs, boxes_list=None, return_illuminated=False)
    feat = _get_feature_from_output(output)

    if flip_test:
        imgs_flip = torch.flip(imgs, dims=[3])
        output_flip = model(imgs_flip, boxes_list=None, return_illuminated=False)
        feat_flip = _get_feature_from_output(output_flip)
        feat = (feat + feat_flip) / 2.0

    return feat


def _get_local_feature_from_output(output: Any) -> Optional[torch.Tensor]:
    """Return local part descriptors from a model output when available."""
    if not isinstance(output, dict):
        return None
    local_feat = output.get("local_features")
    if local_feat is None:
        local_feat = output.get("part_features")
    if not isinstance(local_feat, torch.Tensor):
        return None
    if local_feat.ndim != 3:
        return None
    return local_feat


def _forward_global_local_batch(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    flip_test: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    output = model(imgs, boxes_list=None, return_illuminated=False, return_local_features=True)
    feat = _get_feature_from_output(output)
    local_feat = _get_local_feature_from_output(output)

    if flip_test:
        imgs_flip = torch.flip(imgs, dims=[3])
        output_flip = model(
            imgs_flip,
            boxes_list=None,
            return_illuminated=False,
            return_local_features=True,
        )
        feat_flip = _get_feature_from_output(output_flip)
        local_flip = _get_local_feature_from_output(output_flip)
        feat = (feat + feat_flip) / 2.0
        if local_feat is not None and local_flip is not None and local_feat.shape == local_flip.shape:
            local_feat = (local_feat + local_flip) / 2.0

    return feat, local_feat


def _forward_feature_batch_with_adaptive_split(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    flip_test: bool,
    min_chunk_size: int = 1,
) -> torch.Tensor:
    try:
        return _forward_feature_batch(model, imgs, flip_test=flip_test)
    except torch.OutOfMemoryError:
        batch_size = int(imgs.shape[0])
        if batch_size <= min_chunk_size:
            raise

        next_chunk_size = max(min_chunk_size, batch_size // 2)
        if next_chunk_size >= batch_size:
            next_chunk_size = batch_size - 1

        logger.warning(
            "CUDA OOM during feature extraction for batch_size=%d; retrying with chunk_size=%d",
            batch_size,
            next_chunk_size,
        )
        _release_cuda_memory()

        feat_chunks: List[torch.Tensor] = []
        for start in range(0, batch_size, next_chunk_size):
            stop = min(start + next_chunk_size, batch_size)
            feat_chunks.append(
                _forward_feature_batch_with_adaptive_split(
                    model,
                    imgs[start:stop],
                    flip_test=flip_test,
                    min_chunk_size=min_chunk_size,
                )
            )
            _release_cuda_memory()

        return torch.cat(feat_chunks, dim=0)


def _forward_global_local_batch_with_adaptive_split(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    flip_test: bool,
    min_chunk_size: int = 1,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    try:
        return _forward_global_local_batch(model, imgs, flip_test=flip_test)
    except torch.OutOfMemoryError:
        batch_size = int(imgs.shape[0])
        if batch_size <= min_chunk_size:
            raise

        next_chunk_size = max(min_chunk_size, batch_size // 2)
        if next_chunk_size >= batch_size:
            next_chunk_size = batch_size - 1

        logger.warning(
            "CUDA OOM during global-local feature extraction for batch_size=%d; retrying with chunk_size=%d",
            batch_size,
            next_chunk_size,
        )
        _release_cuda_memory()

        feat_chunks: List[torch.Tensor] = []
        local_chunks: List[torch.Tensor] = []
        saw_missing_local = False
        for start in range(0, batch_size, next_chunk_size):
            stop = min(start + next_chunk_size, batch_size)
            feat, local_feat = _forward_global_local_batch_with_adaptive_split(
                model,
                imgs[start:stop],
                flip_test=flip_test,
                min_chunk_size=min_chunk_size,
            )
            feat_chunks.append(feat)
            if local_feat is None:
                saw_missing_local = True
            else:
                local_chunks.append(local_feat)
            _release_cuda_memory()

        local_out = None if saw_missing_local or not local_chunks else torch.cat(local_chunks, dim=0)
        return torch.cat(feat_chunks, dim=0), local_out


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    flip_test: bool = False,
) -> Tuple[np.ndarray, List[Any], List[int], List[str]]:
    """Extract L2-normalized features and metadata (including image paths)."""
    was_training = model.training
    model.eval()
    feats_list: List[torch.Tensor] = []
    ids_list: List[Any] = []
    cams_list: List[int] = []
    paths_list: List[str] = []

    try:
        _release_cuda_memory()
        for batch in dataloader:
            imgs = batch[0].to(device)
            identities = batch[1]

            # Handle different dataset formats:
            # - ReIDDataset: (img, identity, cam_id, img_path) - 4 elements
            # - FullImageDataset: (img, label, img_path) - 3 elements
            if len(batch) == 3:
                # FullImageDataset format: batch[2] is img_path
                cam_ids: Any = [-1] * len(identities)
                img_paths: Any = batch[2]
            elif len(batch) >= 4:
                # ReIDDataset format: batch[2] is cam_id, batch[3] is img_path
                cam_ids = batch[2]
                img_paths = batch[3]
            else:
                # Fallback for unexpected formats
                cam_ids = [-1] * len(identities)
                img_paths = [""] * len(identities)

            feat = _forward_feature_batch_with_adaptive_split(
                model,
                imgs,
                flip_test=flip_test,
            )
            feat = F.normalize(feat, p=2, dim=1)
            feats_list.append(feat.cpu())

            if isinstance(identities, torch.Tensor):
                ids_list.extend(identities.tolist())
            else:
                ids_list.extend(list(identities))

            if isinstance(cam_ids, torch.Tensor):
                cams_list.extend([int(x) for x in cam_ids.tolist()])
            elif isinstance(cam_ids, (list, tuple)):
                cams_list.extend([int(x) for x in cam_ids])
            else:
                cams_list.extend([-1] * len(identities))

            if isinstance(img_paths, (list, tuple)):
                paths_list.extend([str(p) for p in img_paths])
            else:
                paths_list.extend([""] * len(identities))

            del imgs, feat
            _release_cuda_memory()
    finally:
        _release_cuda_memory()
        model.train(was_training)

    if not feats_list:
        return np.empty((0, 0), dtype=np.float32), ids_list, cams_list, paths_list

    features = torch.cat(feats_list, dim=0).numpy()
    return features, ids_list, cams_list, paths_list


@torch.no_grad()
def extract_global_local_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    flip_test: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[Any], List[int], List[str]]:
    """Extract normalized global features plus optional normalized local descriptors."""
    was_training = model.training
    model.eval()
    feats_list: List[torch.Tensor] = []
    local_list: List[torch.Tensor] = []
    ids_list: List[Any] = []
    cams_list: List[int] = []
    paths_list: List[str] = []
    saw_missing_local = False

    try:
        _release_cuda_memory()
        for batch in dataloader:
            imgs = batch[0].to(device)
            identities = batch[1]

            if len(batch) == 3:
                cam_ids: Any = [-1] * len(identities)
                img_paths: Any = batch[2]
            elif len(batch) >= 4:
                cam_ids = batch[2]
                img_paths = batch[3]
            else:
                cam_ids = [-1] * len(identities)
                img_paths = [""] * len(identities)

            feat, local_feat = _forward_global_local_batch_with_adaptive_split(
                model,
                imgs,
                flip_test=flip_test,
            )
            feat = F.normalize(feat, p=2, dim=1)
            feats_list.append(feat.cpu())

            if local_feat is None:
                saw_missing_local = True
            else:
                local_feat = F.normalize(local_feat, p=2, dim=2)
                local_list.append(local_feat.cpu())

            if isinstance(identities, torch.Tensor):
                ids_list.extend(identities.tolist())
            else:
                ids_list.extend(list(identities))

            if isinstance(cam_ids, torch.Tensor):
                cams_list.extend([int(x) for x in cam_ids.tolist()])
            elif isinstance(cam_ids, (list, tuple)):
                cams_list.extend([int(x) for x in cam_ids])
            else:
                cams_list.extend([-1] * len(identities))

            if isinstance(img_paths, (list, tuple)):
                paths_list.extend([str(p) for p in img_paths])
            else:
                paths_list.extend([""] * len(identities))

            del imgs, feat, local_feat
            _release_cuda_memory()
    finally:
        _release_cuda_memory()
        model.train(was_training)

    if not feats_list:
        return np.empty((0, 0), dtype=np.float32), None, ids_list, cams_list, paths_list

    global_features = torch.cat(feats_list, dim=0).numpy()
    local_features = None if saw_missing_local or not local_list else torch.cat(local_list, dim=0).numpy()
    return global_features, local_features, ids_list, cams_list, paths_list


def _get_feature_from_output(output: Any) -> torch.Tensor:
    """Handle multiple model output styles."""
    if isinstance(output, dict):
        if "features" in output:
            return output["features"]
        if "global_feat" in output:
            return output["global_feat"]
        raise KeyError("Model output dict does not contain 'features' or 'global_feat'")
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def compute_distance_matrix(
    query_feats: np.ndarray,
    gallery_feats: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """Compute query-gallery distance matrix."""
    if metric not in {"euclidean", "cosine"}:
        raise ValueError(f"Unsupported metric: {metric}")

    if metric == "cosine":
        similarity = np.dot(query_feats, gallery_feats.T)
        return 1.0 - similarity

    q_norm = np.sum(query_feats**2, axis=1, keepdims=True)
    g_norm = np.sum(gallery_feats**2, axis=1, keepdims=True).T
    return q_norm + g_norm - 2 * np.dot(query_feats, gallery_feats.T)


def compute_local_distance_matrix(
    query_local: np.ndarray,
    gallery_local: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """Compute a local part-aware distance matrix.

    Input shape is [N, P, D]. Matching is part-aligned by default because the
    model's local descriptors are stripe/part ordered. This is a deterministic
    retrieval-distance change, not a dataset-specific post-processing step.
    """
    if query_local.ndim != 3 or gallery_local.ndim != 3:
        raise ValueError("Local features must have shape [N, num_parts, dim]")
    if query_local.shape[1] != gallery_local.shape[1]:
        raise ValueError(
            f"Query/gallery local part count mismatch: {query_local.shape[1]} vs {gallery_local.shape[1]}"
        )
    if metric not in {"cosine", "euclidean"}:
        raise ValueError(f"Unsupported local metric: {metric}")

    if metric == "cosine":
        sim = np.einsum("qpd,gpd->qgp", query_local, gallery_local)
        return 1.0 - sim.mean(axis=2)

    diff = query_local[:, None, :, :] - gallery_local[None, :, :, :]
    return np.mean(np.sum(diff * diff, axis=3), axis=2)


def fuse_global_local_distance(
    global_dist: np.ndarray,
    local_dist: np.ndarray,
    local_weight: float = 0.35,
) -> np.ndarray:
    local_weight = float(np.clip(local_weight, 0.0, 1.0))
    if global_dist.shape != local_dist.shape:
        raise ValueError(f"Distance shape mismatch: {global_dist.shape} vs {local_dist.shape}")
    return (1.0 - local_weight) * global_dist + local_weight * local_dist


def compute_cmc_map(
    distmat: np.ndarray,
    query_ids: Sequence[Any],
    gallery_ids: Sequence[Any],
    query_cams: Optional[Sequence[int]] = None,
    gallery_cams: Optional[Sequence[int]] = None,
    query_paths: Optional[Sequence[str]] = None,
    gallery_paths: Optional[Sequence[str]] = None,
    max_rank: int = 10,
    exclude_same_camera: bool = True,
) -> Tuple[np.ndarray, float]:
    """Compute standard CMC + mAP.

    When query and gallery contain the same images (e.g., ATRW test set),
    this function automatically excludes each query image from its own gallery matches.
    """
    query_ids_arr = np.asarray(query_ids)
    gallery_ids_arr = np.asarray(gallery_ids)

    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)

    has_cam = (
        exclude_same_camera
        and query_cams is not None
        and gallery_cams is not None
    )
    if has_cam:
        query_cams_arr = np.asarray(query_cams)
        gallery_cams_arr = np.asarray(gallery_cams)
        has_cam = np.any(query_cams_arr != -1) and np.any(gallery_cams_arr != -1)
    else:
        query_cams_arr = None
        gallery_cams_arr = None

    # Check if we have image paths to exclude same images
    has_paths = query_paths is not None and gallery_paths is not None
    if has_paths:
        query_paths_arr = np.asarray([os.path.basename(str(p)) for p in query_paths])
        gallery_paths_arr = np.asarray([os.path.basename(str(p)) for p in gallery_paths])
    else:
        query_paths_arr = None
        gallery_paths_arr = None

    cmc = np.zeros(max_rank, dtype=float)
    aps: List[float] = []

    for i in range(num_q):
        qid = query_ids_arr[i]
        order = np.argsort(distmat[i])

        same_id = gallery_ids_arr[order] == qid

        if has_cam:
            qcam = query_cams_arr[i]
            same_cam = gallery_cams_arr[order] == qcam
            good_mask = same_id & ~same_cam
            junk_mask = same_id & same_cam
        else:
            good_mask = same_id
            junk_mask = np.zeros_like(same_id, dtype=bool)

            # CRITICAL: When no camera info, exclude same image by path
            if has_paths:
                qpath = query_paths_arr[i]
                same_image = gallery_paths_arr[order] == qpath
                junk_mask = same_image  # Mark same image as junk

        if good_mask.sum() == 0:
            continue

        if junk_mask.any():
            keep = ~junk_mask
            good_mask = good_mask[keep]

        if good_mask.sum() == 0:
            continue

        first_match_positions = np.where(good_mask)[0]
        if len(first_match_positions) > 0:
            first_match = int(first_match_positions[0])
            if first_match < max_rank:
                cmc[first_match:] += 1

        num_rel = int(good_mask.sum())
        tmp_cmc = good_mask.astype(np.float32).cumsum()
        precision = tmp_cmc / (np.arange(len(good_mask)) + 1)
        ap = float((precision * good_mask).sum() / num_rel)
        aps.append(ap)

    if not aps:
        logger.warning("No valid query has positive matches in gallery")
        return np.zeros(max_rank, dtype=float), 0.0

    cmc /= len(aps)
    m_ap = float(np.mean(aps))
    return cmc, m_ap


def evaluate_openset(
    distmat: np.ndarray,
    query_ids: Sequence[Any],
    gallery_ids: Sequence[Any],
    query_cams: Optional[Sequence[int]] = None,
    gallery_cams: Optional[Sequence[int]] = None,
    query_paths: Optional[Sequence[str]] = None,
    gallery_paths: Optional[Sequence[str]] = None,
    seen_ids: Optional[Iterable[Any]] = None,
    unseen_ids: Optional[Iterable[Any]] = None,
    max_rank: int = 10,
) -> Dict[str, float]:
    """Open-set evaluation with seen/unseen and single/cross camera groups."""
    seen_ids_set = set(seen_ids or [])
    unseen_ids_set = set(unseen_ids or [])

    q_ids = np.asarray(query_ids)
    g_ids = np.asarray(gallery_ids)

    has_cam = query_cams is not None and gallery_cams is not None
    if has_cam:
        q_cams = np.asarray(query_cams)
        g_cams = np.asarray(gallery_cams)
        has_cam = np.any(q_cams != -1) and np.any(g_cams != -1)
    else:
        q_cams = None
        g_cams = None

    has_paths = query_paths is not None and gallery_paths is not None
    if has_paths:
        q_paths = np.asarray([os.path.basename(str(p)) for p in query_paths])
        g_paths = np.asarray([os.path.basename(str(p)) for p in gallery_paths])
    else:
        q_paths = None
        g_paths = None

    cmc, m_ap = compute_cmc_map(
        distmat,
        q_ids,
        g_ids,
        q_cams,
        g_cams,
        query_paths,
        gallery_paths,
        max_rank=max_rank,
        exclude_same_camera=True,
    )

    results: Dict[str, float] = {
        "rank1": cmc[0] * 100,
        "rank5": cmc[min(4, len(cmc) - 1)] * 100,
        "rank10": cmc[min(9, len(cmc) - 1)] * 100,
        "mAP": m_ap * 100,
    }

    groups: Dict[str, List[Tuple[int, float]]] = {
        "seen": [],
        "unseen": [],
        "single": [],
        "cross": [],
    }

    for i in range(distmat.shape[0]):
        qid = q_ids[i]
        is_seen = qid in seen_ids_set
        is_unseen = qid in unseen_ids_set

        if seen_ids_set or unseen_ids_set:
            if not is_seen and not is_unseen:
                continue

        order = np.argsort(distmat[i])
        same_id = g_ids[order] == qid

        if has_cam:
            qcam = q_cams[i]
            same_cam = g_cams[order] == qcam
            good_mask = same_id & ~same_cam
            junk_mask = same_id & same_cam

            cams_for_id = set(g_cams[g_ids == qid])
            is_cross_cam = any(cam != qcam for cam in cams_for_id)
        else:
            good_mask = same_id
            junk_mask = np.zeros_like(same_id, dtype=bool)
            is_cross_cam = False

            if has_paths:
                qpath = q_paths[i]
                same_image = g_paths[order] == qpath
                junk_mask = same_image

        if good_mask.sum() == 0:
            continue

        if junk_mask.any():
            keep = ~junk_mask
            good_mask = good_mask[keep]

        if good_mask.sum() == 0:
            continue

        first_match_positions = np.where(good_mask)[0]
        first_match = int(first_match_positions[0]) if len(first_match_positions) > 0 else -1

        num_rel = int(good_mask.sum())
        tmp_cmc = good_mask.astype(np.float32).cumsum()
        precision = tmp_cmc / (np.arange(len(good_mask)) + 1)
        ap = float((precision * good_mask).sum() / num_rel)

        if is_seen:
            groups["seen"].append((first_match, ap))
        if is_unseen:
            groups["unseen"].append((first_match, ap))

        if has_cam:
            if is_cross_cam:
                groups["cross"].append((first_match, ap))
            else:
                groups["single"].append((first_match, ap))

    for group_name, entries in groups.items():
        if not entries:
            results[f"rank1_{group_name}"] = 0.0
            results[f"rank5_{group_name}"] = 0.0
            results[f"mAP_{group_name}"] = 0.0
            continue

        g_cmc = np.zeros(max_rank, dtype=float)
        g_ap: List[float] = []
        for first_match, ap in entries:
            if 0 <= first_match < max_rank:
                g_cmc[first_match:] += 1
            g_ap.append(ap)

        g_cmc /= len(g_ap)
        results[f"rank1_{group_name}"] = g_cmc[0] * 100
        results[f"rank5_{group_name}"] = g_cmc[min(4, len(g_cmc) - 1)] * 100
        results[f"mAP_{group_name}"] = float(np.mean(g_ap)) * 100

    return results


def _compute_binary_ap(matches: np.ndarray, scores: np.ndarray) -> float:
    """Binary AP without sklearn dependency."""
    if matches.sum() == 0:
        return 0.0
    order = np.argsort(-scores)
    sorted_matches = matches[order].astype(np.float32)
    cumsum = np.cumsum(sorted_matches)
    precision = cumsum / (np.arange(len(sorted_matches)) + 1)
    return float((precision * sorted_matches).sum() / sorted_matches.sum())


def load_atrw_gt(gt_file: str) -> Dict[str, Any]:
    """Load ATRW official ground-truth annotations."""
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"ATRW ground-truth file not found: {gt_file}")

    with open(gt_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    if not isinstance(annotations, list):
        raise ValueError("ATRW ground-truth must be a list of annotations")

    imgid_to_entity: Dict[int, int] = {}
    imgid_to_frame: Dict[int, Tuple[int, int, int]] = {}
    query_single: Set[int] = set()
    query_cross: Set[int] = set()

    for obj in annotations:
        imgid = int(obj["imgid"])
        entityid = int(obj["entityid"])
        frame_obj = obj.get("frame", [0, 0, 0])
        frame = (int(frame_obj[0]), int(frame_obj[1]), int(frame_obj[2]))
        query_type = str(obj.get("query", "")).strip().lower()

        imgid_to_entity[imgid] = entityid
        imgid_to_frame[imgid] = frame
        if query_type == "sing":
            query_single.add(imgid)
        elif query_type == "multi":
            query_cross.add(imgid)

    return {
        "annotations": annotations,
        "imgids": set(imgid_to_entity.keys()),
        "imgid_to_entity": imgid_to_entity,
        "imgid_to_frame": imgid_to_frame,
        "query_single": query_single,
        "query_cross": query_cross,
    }


def _normalize_atrw_gt(gt: Union[str, Dict[str, Any], Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    """Normalize ATRW GT input to standard internal structure."""
    if isinstance(gt, str):
        return load_atrw_gt(gt)

    if isinstance(gt, dict):
        required = {"imgid_to_entity", "imgid_to_frame", "query_single", "query_cross"}
        if required.issubset(set(gt.keys())):
            return gt
        if "annotations" in gt and isinstance(gt["annotations"], list):
            annotations = gt["annotations"]
        else:
            raise ValueError("Invalid ATRW GT dict format")
    elif isinstance(gt, (list, tuple)):
        annotations = list(gt)
    else:
        raise ValueError(f"Unsupported ATRW GT input type: {type(gt)}")

    imgid_to_entity: Dict[int, int] = {}
    imgid_to_frame: Dict[int, Tuple[int, int, int]] = {}
    query_single: Set[int] = set()
    query_cross: Set[int] = set()
    for obj in annotations:
        imgid = int(obj["imgid"])
        imgid_to_entity[imgid] = int(obj["entityid"])
        frame_obj = obj.get("frame", [0, 0, 0])
        imgid_to_frame[imgid] = (int(frame_obj[0]), int(frame_obj[1]), int(frame_obj[2]))
        query_type = str(obj.get("query", "")).strip().lower()
        if query_type == "sing":
            query_single.add(imgid)
        elif query_type == "multi":
            query_cross.add(imgid)

    return {
        "annotations": list(annotations),
        "imgids": set(imgid_to_entity.keys()),
        "imgid_to_entity": imgid_to_entity,
        "imgid_to_frame": imgid_to_frame,
        "query_single": query_single,
        "query_cross": query_cross,
    }


def build_submission_from_distance(
    imgids: Sequence[int],
    distmat: np.ndarray,
) -> List[Dict[str, Any]]:
    """Build ATRW-style retrieval submission from a square image-image distance matrix."""
    if len(imgids) != len(distmat):
        raise ValueError(f"imgids/distmat length mismatch: {len(imgids)} vs {len(distmat)}")
    if distmat.ndim != 2 or distmat.shape[0] != distmat.shape[1]:
        raise ValueError(f"distmat must be square, got shape={distmat.shape}")

    submission: List[Dict[str, Any]] = []
    for i, query_id in enumerate(imgids):
        dist = distmat[i].copy()
        dist[i] = np.inf
        order = np.argsort(dist)
        ans_ids = [int(imgids[j]) for j in order]
        submission.append({"query_id": int(query_id), "ans_ids": ans_ids})
    return submission


def build_submission_from_features(imgids: Sequence[int], features: np.ndarray) -> List[Dict[str, Any]]:
    """Build ATRW-style retrieval submission from features."""
    if len(imgids) != len(features):
        raise ValueError(f"imgids/features length mismatch: {len(imgids)} vs {len(features)}")

    sim = np.dot(features, features.T)
    distmat = 1.0 - sim
    return build_submission_from_distance(imgids, distmat)


def evaluate_atrw_official(
    gt: Union[str, Dict[str, Any], Sequence[Dict[str, Any]]],
    submission: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    """Evaluate ATRW official protocol with single/cross camera metrics."""
    gt_data = _normalize_atrw_gt(gt)
    imgid_to_entity: Dict[int, int] = gt_data["imgid_to_entity"]
    imgid_to_frame: Dict[int, Tuple[int, int, int]] = gt_data["imgid_to_frame"]
    query_single: Set[int] = gt_data["query_single"]
    query_cross: Set[int] = gt_data["query_cross"]

    gallery_size = max(len(imgid_to_entity), 1)
    cmc_single = np.zeros(gallery_size, dtype=np.int32)
    cmc_cross = np.zeros(gallery_size, dtype=np.int32)
    aps_single: List[float] = []
    aps_cross: List[float] = []

    for ans in submission:
        query_id = int(ans["query_id"])
        if query_id not in imgid_to_entity:
            continue

        if query_id in query_single:
            cmc_bucket = cmc_single
            ap_bucket = aps_single
        elif query_id in query_cross:
            cmc_bucket = cmc_cross
            ap_bucket = aps_cross
        else:
            continue

        ans_ids = [int(x) for x in ans.get("ans_ids", []) if int(x) in imgid_to_entity]
        if not ans_ids:
            continue

        q_entity = imgid_to_entity[query_id]
        q_frame = imgid_to_frame[query_id]

        ans_entities = np.asarray([imgid_to_entity[x] for x in ans_ids], dtype=np.int64)
        ans_frames = np.asarray([imgid_to_frame[x] for x in ans_ids], dtype=np.int64)

        pid_matches = ans_entities == q_entity
        same_track = (
            (ans_entities == q_entity)
            & (ans_frames[:, 0] == q_frame[0])
            & (ans_frames[:, 1] == q_frame[1])
            & (np.abs(ans_frames[:, 2] - q_frame[2]) <= 3)
        )
        junk = ans_entities == -1
        mask = same_track | junk

        ranks = np.arange(len(ans_ids), dtype=np.float32) + 1.0
        ranks[mask] = np.inf
        pid_matches[mask] = False

        scores = 1.0 / ranks
        ap = _compute_binary_ap(pid_matches.astype(np.int32), scores)
        ap_bucket.append(ap)

        sorted_matches = pid_matches[np.argsort(ranks)]
        if sorted_matches.sum() > 0:
            first = int(np.where(sorted_matches)[0][0])
            cmc_bucket[first:] += 1

    def _safe_rank(cmc: np.ndarray, rank_index: int, denom: int) -> float:
        if len(cmc) == 0 or denom <= 0:
            return 0.0
        idx = min(rank_index, len(cmc) - 1)
        return float(cmc[idx] / denom * 100.0)

    rank1_single = _safe_rank(cmc_single, 0, max(len(query_single), 1))
    rank5_single = _safe_rank(cmc_single, 4, max(len(query_single), 1))
    map_single = float(np.mean(aps_single) * 100.0) if aps_single else 0.0

    rank1_cross = _safe_rank(cmc_cross, 0, max(len(query_cross), 1))
    rank5_cross = _safe_rank(cmc_cross, 4, max(len(query_cross), 1))
    map_cross = float(np.mean(aps_cross) * 100.0) if aps_cross else 0.0

    mm_ap = (map_single + map_cross) / 2.0

    return {
        "rank1_single": rank1_single,
        "rank5_single": rank5_single,
        "mAP_single": map_single,
        "rank1_cross": rank1_cross,
        "rank5_cross": rank5_cross,
        "mAP_cross": map_cross,
        "mmAP": mm_ap,
    }


@dataclass
class ReIDEvaluator:
    """High-level evaluator used by both training and standalone scripts."""

    model: torch.nn.Module
    device: torch.device
    img_height: int = 256
    img_width: int = 256
    batch_size: int = 32
    flip_test: bool = False
    metric: str = "euclidean"
    rerank: bool = False
    rerank_params: Optional[Dict[str, Any]] = None
    use_local_distance: bool = False
    local_weight: float = 0.35
    local_metric: str = "cosine"
    num_workers: Optional[int] = None
    exclude_same_camera: bool = True

    def __post_init__(self) -> None:
        if self.rerank_params is None:
            self.rerank_params = {"k1": 20, "k2": 6, "lambda_value": 0.3}
        if self.num_workers is None:
            self.num_workers = 0 if platform.system() == "Windows" else 4

    def _build_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.img_height, self.img_width)),
                transforms.ToTensor(),
            ]
        )

    def _build_loader(self, root: str) -> DataLoader:
        ds = ReIDDataset(root=root, transform=self._build_transform())
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _maybe_rerank(self, q_feats: np.ndarray, g_feats: np.ndarray) -> Optional[np.ndarray]:
        if not self.rerank:
            return None
        if not RERANKING_AVAILABLE:
            logger.warning("reranking requested but tools.reranking is unavailable")
            return None
        logger.info("Applying k-reciprocal reranking")
        return re_ranking(q_feats, g_feats, **self.rerank_params)

    def _auto_detect_openset_info(self, query_dir: str) -> Optional[Dict[str, Any]]:
        if "_openset" not in query_dir.lower():
            return None
        path = os.path.join(os.path.dirname(query_dir), "openset_info.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(
                "Loaded openset metadata: seen=%d unseen=%d",
                len(data.get("seen_ids", [])),
                len(data.get("unseen_ids", [])),
            )
            return data
        except Exception as exc:  # pragma: no cover - invalid user file
            logger.warning("Failed to load openset metadata %s: %s", path, exc)
            return None

    def evaluate(
        self,
        query_dir: str,
        gallery_dir: str,
        openset_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, float]]:
        """Run query-vs-gallery evaluation."""
        if not os.path.exists(query_dir) or not os.path.exists(gallery_dir):
            logger.warning("evaluation directories missing: query=%s, gallery=%s", query_dir, gallery_dir)
            return None

        was_training = self.model.training
        self.model.eval()
        try:
            q_loader = self._build_loader(query_dir)
            g_loader = self._build_loader(gallery_dir)

            if self.use_local_distance:
                q_feats, q_local, q_ids, q_cams, q_paths = extract_global_local_features(
                    self.model,
                    q_loader,
                    self.device,
                    self.flip_test,
                )
                g_feats, g_local, g_ids, g_cams, g_paths = extract_global_local_features(
                    self.model,
                    g_loader,
                    self.device,
                    self.flip_test,
                )
            else:
                q_feats, q_ids, q_cams, q_paths = extract_features(self.model, q_loader, self.device, self.flip_test)
                g_feats, g_ids, g_cams, g_paths = extract_features(self.model, g_loader, self.device, self.flip_test)
                q_local = None
                g_local = None

            distmat = self._maybe_rerank(q_feats, g_feats)
            if distmat is None:
                distmat = compute_distance_matrix(q_feats, g_feats, metric=self.metric)
                if self.use_local_distance and q_local is not None and g_local is not None:
                    local_dist = compute_local_distance_matrix(q_local, g_local, metric=self.local_metric)
                    distmat = fuse_global_local_distance(
                        distmat,
                        local_dist,
                        local_weight=self.local_weight,
                    )
                elif self.use_local_distance:
                    logger.warning("Local distance requested but local descriptors were unavailable")

            if openset_info is None:
                openset_info = self._auto_detect_openset_info(query_dir)

            has_openset_partition = bool(openset_info) and (
                bool(openset_info.get("seen_ids"))
                or bool(openset_info.get("unseen_ids"))
            )

            if has_openset_partition:
                results = evaluate_openset(
                    distmat,
                    q_ids,
                    g_ids,
                    q_cams,
                    g_cams,
                    q_paths,
                    g_paths,
                    seen_ids=openset_info.get("seen_ids", []),
                    unseen_ids=openset_info.get("unseen_ids", []),
                )
                self._log_openset_results(results, len(q_ids), len(g_ids))
                return results

            cmc, m_ap = compute_cmc_map(
                distmat,
                q_ids,
                g_ids,
                q_cams,
                g_cams,
                q_paths,
                g_paths,
                exclude_same_camera=self.exclude_same_camera,
            )
            results = {
                "rank1": cmc[0] * 100,
                "rank5": cmc[min(4, len(cmc) - 1)] * 100,
                "rank10": cmc[min(9, len(cmc) - 1)] * 100,
                "mAP": m_ap * 100,
            }
            self._log_standard_results(results)
            return results
        finally:
            self.model.train(was_training)

    def evaluate_ipanda50(self, test_dir: str) -> Optional[Dict[str, float]]:
        """Run iPanda50 all-vs-all evaluation."""
        if not os.path.exists(test_dir):
            logger.warning("iPanda50 test dir missing: %s", test_dir)
            return None

        was_training = self.model.training
        self.model.eval()
        try:
            loader = self._build_loader(test_dir)
            feats, ids, _, _ = extract_features(self.model, loader, self.device, self.flip_test)

            num_images = len(ids)
            if num_images < 2:
                logger.warning("iPanda50 evaluation needs at least 2 images")
                return None

            id_arr = np.asarray(ids)
            distmat = compute_distance_matrix(feats, feats, metric=self.metric)

            cmc = np.zeros(10, dtype=float)
            aps: List[float] = []

            for i in range(num_images):
                dist = distmat[i].copy()
                order = np.argsort(dist)

                self_pos = np.where(order == i)[0]
                if len(self_pos) > 0:
                    order = np.delete(order, self_pos[0])

                matches = (id_arr[order] == id_arr[i]).astype(np.int32)
                if matches.sum() == 0:
                    continue

                first_match_positions = np.where(matches == 1)[0]
                if len(first_match_positions) > 0:
                    first_match = int(first_match_positions[0])
                    if first_match < len(cmc):
                        cmc[first_match:] += 1

                num_rel = int(matches.sum())
                tmp_cmc = matches.cumsum()
                precision = tmp_cmc * matches / (np.arange(len(matches)) + 1)
                aps.append(float(precision.sum() / num_rel))

            if not aps:
                logger.warning("iPanda50 evaluation failed: no valid query")
                return None

            cmc /= len(aps)
            m_ap = float(np.mean(aps))

            results = {
                "rank1": cmc[0] * 100,
                "rank5": cmc[min(4, len(cmc) - 1)] * 100,
                "rank10": cmc[min(9, len(cmc) - 1)] * 100,
                "mAP": m_ap * 100,
            }
            logger.info("===== iPanda50 Evaluation =====")
            logger.info("  Rank-1 : %.2f%%", results["rank1"])
            logger.info("  Rank-5 : %.2f%%", results["rank5"])
            logger.info("  mAP    : %.2f%%", results["mAP"])
            logger.info("================================")
            return results
        finally:
            self.model.train(was_training)

    def evaluate_auto(
        self,
        query_dir: str,
        gallery_dir: str,
        openset_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, float]]:
        """Auto protocol dispatch: iPanda50 or standard/open-set."""
        is_ipanda50 = "ipanda50" in query_dir.lower() or "ipanda50" in gallery_dir.lower()
        if is_ipanda50:
            test_dir = os.path.join(os.path.dirname(query_dir), "test")
            if os.path.exists(test_dir):
                return self.evaluate_ipanda50(test_dir)
        return self.evaluate(query_dir, gallery_dir, openset_info=openset_info)

    @staticmethod
    def _log_standard_results(results: Dict[str, float]) -> None:
        logger.info("===== ReID Evaluation Results =====")
        logger.info("  Rank-1  : %.2f%%", results["rank1"])
        logger.info("  Rank-5  : %.2f%%", results["rank5"])
        logger.info("  Rank-10 : %.2f%%", results["rank10"])
        logger.info("  mAP     : %.2f%%", results["mAP"])
        logger.info("===================================")

    @staticmethod
    def _log_openset_results(results: Dict[str, float], num_q: int, num_g: int) -> None:
        logger.info("=" * 70)
        logger.info("           Open-Set ReID Evaluation Results")
        logger.info("=" * 70)
        logger.info("Query: %d, Gallery: %d", num_q, num_g)
        logger.info("-" * 70)
        logger.info("%-12s %10s %10s %10s", "Metric", "Overall", "Seen", "Unseen")
        logger.info("-" * 70)
        logger.info(
            "%-12s %9.2f%% %9.2f%% %9.2f%%",
            "Rank-1",
            results.get("rank1", 0.0),
            results.get("rank1_seen", 0.0),
            results.get("rank1_unseen", 0.0),
        )
        logger.info(
            "%-12s %9.2f%% %9.2f%% %9.2f%%",
            "mAP",
            results.get("mAP", 0.0),
            results.get("mAP_seen", 0.0),
            results.get("mAP_unseen", 0.0),
        )
        if "rank1_single" in results or "rank1_cross" in results:
            logger.info("-" * 70)
            logger.info("%-12s %12s %12s", "Metric", "Single-cam", "Cross-cam")
            logger.info(
                "%-12s %11.2f%% %11.2f%%",
                "Rank-1",
                results.get("rank1_single", 0.0),
                results.get("rank1_cross", 0.0),
            )
            logger.info(
                "%-12s %11.2f%% %11.2f%%",
                "mAP",
                results.get("mAP_single", 0.0),
                results.get("mAP_cross", 0.0),
            )
        logger.info("=" * 70)
