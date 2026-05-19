#!/usr/bin/env python3
"""Shared data utilities for joint training."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from ultralytics import YOLO


class YOLODetectorWrapper:
    """YOLO wrapper that returns detection boxes per image."""

    def __init__(self, model_path: str = "./fea_data/yolov8m-seg.pt", conf: float = 0.5):
        self.model = YOLO(model_path)
        self.conf = conf

    @torch.no_grad()
    def detect_batch(self, images: torch.Tensor) -> List[Optional[torch.Tensor]]:
        """Detect boxes for a tensor batch and return [N,4] boxes per image."""
        batch_size = images.shape[0]
        device = images.device

        images_np = images.cpu().numpy()
        images_np = (images_np * 255).astype(np.uint8)
        images_np = images_np.transpose(0, 2, 3, 1)

        boxes_list: List[torch.Tensor] = []
        for i in range(batch_size):
            img = images_np[i]
            if img.shape[-1] == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img

            results = self.model.predict(img_bgr, conf=self.conf, verbose=False)

            det_boxes = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                det_boxes.append([x1, y1, x2, y2])

            if det_boxes:
                boxes = torch.tensor(det_boxes, dtype=torch.float32, device=device)
            else:
                h, w = img_bgr.shape[:2]
                boxes = torch.tensor([[0.0, 0.0, float(w - 1), float(h - 1)]], dtype=torch.float32, device=device)
            boxes_list.append(boxes)

        return boxes_list


class FullImageDataset(Dataset):
    """Folder-structured training dataset: data_dir/identity/image."""

    def __init__(
        self,
        data_dir: str,
        transform=None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ):
        self.data_dir = data_dir
        self.transform = transform

        self.samples: List[Tuple[str, str]] = []
        self.identity_to_idx: Dict[str, int] = {}
        self.idx_to_samples: Dict[int, List[int]] = {}

        self._load_samples(data_dir, extensions)
        print(f"[INFO] Dataset loaded: {len(self.samples)} images, {len(self.identity_to_idx)} identities")

    def _load_samples(self, data_dir: str, extensions: Tuple[str, ...]) -> None:
        idx = 0
        for identity_dir in sorted(os.listdir(data_dir)):
            identity_path = os.path.join(data_dir, identity_dir)
            if not os.path.isdir(identity_path):
                continue

            identity_id = identity_dir
            if identity_id not in self.identity_to_idx:
                self.identity_to_idx[identity_id] = idx
                self.idx_to_samples[idx] = []
                idx += 1

            label_idx = self.identity_to_idx[identity_id]
            for img_file in os.listdir(identity_path):
                if img_file.lower().endswith(extensions):
                    img_path = os.path.join(identity_path, img_file)
                    sample_idx = len(self.samples)
                    self.samples.append((img_path, identity_id))
                    self.idx_to_samples[label_idx].append(sample_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, identity_id = self.samples[idx]
        label = self.identity_to_idx[identity_id]

        image = cv2.imread(img_path)
        if image is None:
            image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label, img_path

    @property
    def num_classes(self) -> int:
        return len(self.identity_to_idx)


class PKSampler(Sampler[int]):
    """Sample P identities with K samples each in one mini-batch."""

    def __init__(self, dataset: FullImageDataset, p: int = 8, k: int = 4, seed: int = 42):
        self.dataset = dataset
        self.p = p
        self.k = k
        self.seed = int(seed)
        self.epoch = 0
        self.batch_size = p * k

        self.valid_ids = [
            idx for idx, samples in dataset.idx_to_samples.items() if len(samples) >= k
        ]
        if len(self.valid_ids) < p:
            print(f"[WARNING] Valid identities {len(self.valid_ids)} < P={p}; using all identities.")
            self.valid_ids = list(dataset.idx_to_samples.keys())

        print(f"[INFO] PK sampler: P={p}, K={k}, valid_ids={len(self.valid_ids)}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        batch_indices: List[int] = []
        num_batches = len(self.dataset) // self.batch_size

        for _ in range(num_batches):
            selected_ids = rng.choice(
                self.valid_ids,
                size=min(self.p, len(self.valid_ids)),
                replace=False,
            )

            batch: List[int] = []
            for pid in selected_ids:
                samples = self.dataset.idx_to_samples[pid]
                if len(samples) >= self.k:
                    selected = rng.choice(samples, size=self.k, replace=False)
                else:
                    selected = rng.choice(samples, size=self.k, replace=True)
                batch.extend(selected.tolist())

            batch_indices.extend(batch)

        return iter(batch_indices)

    def __len__(self) -> int:
        return (len(self.dataset) // self.batch_size) * self.batch_size


class DistributedPKSampler(Sampler[int]):
    """Distributed PK sampler with a global P x K layout sharded across ranks."""

    def __init__(
        self,
        dataset: FullImageDataset,
        p: int = 8,
        k: int = 4,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
    ):
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        if p % num_replicas != 0:
            raise ValueError(
                f"Global P={p} must be divisible by num_replicas={num_replicas} for distributed PK sampling"
            )

        self.dataset = dataset
        self.p = p
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        self.local_p = p // num_replicas
        self.global_batch_size = p * k
        self.local_batch_size = self.local_p * k

        self.valid_ids = [
            idx for idx, samples in dataset.idx_to_samples.items() if len(samples) >= k
        ]
        if len(self.valid_ids) < p:
            print(
                f"[WARNING] Valid identities {len(self.valid_ids)} < global P={p}; "
                "distributed PK sampler will sample identities with replacement."
            )
            self.valid_ids = list(dataset.idx_to_samples.keys())

        self.num_global_batches = len(self.dataset) // self.global_batch_size
        print(
            f"[INFO] Distributed PK sampler: global_P={p}, local_P={self.local_p}, "
            f"K={k}, num_replicas={num_replicas}, valid_ids={len(self.valid_ids)}"
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        batch_indices: List[int] = []

        for _ in range(self.num_global_batches):
            replace_ids = len(self.valid_ids) < self.p
            selected_ids = rng.choice(
                self.valid_ids,
                size=self.p,
                replace=replace_ids,
            )

            local_start = self.rank * self.local_p
            local_end = local_start + self.local_p
            local_ids = selected_ids[local_start:local_end]

            batch: List[int] = []
            for pid in local_ids:
                samples = self.dataset.idx_to_samples[int(pid)]
                if len(samples) >= self.k:
                    selected = rng.choice(samples, size=self.k, replace=False)
                else:
                    selected = rng.choice(samples, size=self.k, replace=True)
                batch.extend(selected.tolist())

            batch_indices.extend(batch)

        return iter(batch_indices)

    def __len__(self) -> int:
        return self.num_global_batches * self.local_batch_size
