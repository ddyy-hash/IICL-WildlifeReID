#!/usr/bin/env python3
"""Frozen-feature external baselines for animal ReID datasets.

The baselines are training-free references: extract L2-normalized descriptors
from a fixed model, compute cosine distance, and report standard CMC/mAP.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.evaluation import ReIDDataset, compute_cmc_map, compute_distance_matrix


RUN_ID = "tmm_missing_evidence_20260517"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
METRIC_FIELDS = ["rank1", "rank5", "rank10", "mAP"]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    query: str
    gallery: str


DATASETS: Dict[str, DatasetSpec] = {
    "atrw": DatasetSpec("atrw", "ATRW", "data/processed/atrw/query", "data/processed/atrw/gallery"),
    "gzgc_zebra": DatasetSpec(
        "gzgc_zebra",
        "GZGC Zebra stress",
        "data/processed/gzgc_zebra/query",
        "data/processed/gzgc_zebra/gallery",
    ),
    "leopard": DatasetSpec(
        "leopard",
        "LeopardID2022",
        "data/processed/leopard/query",
        "data/processed/leopard/gallery",
    ),
    "whaleshark": DatasetSpec(
        "whaleshark",
        "WhaleSharkID",
        "data/processed/whaleshark/query",
        "data/processed/whaleshark/gallery",
    ),
    "stripespotter": DatasetSpec(
        "stripespotter",
        "StripeSpotter",
        "data/processed/stripespotter/query",
        "data/processed/stripespotter/gallery",
    ),
}


class PILReIDDataset(Dataset):
    """Folder-based ReID dataset returning PIL images for model-specific processors."""

    def __init__(self, root: str, limit: Optional[int] = None) -> None:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"dataset directory not found: {root}")
        self.samples: List[Tuple[str, str, int]] = []
        for identity_name in sorted(os.listdir(root)):
            identity_dir = os.path.join(root, identity_name)
            if not os.path.isdir(identity_dir):
                continue
            for fname in sorted(os.listdir(identity_dir)):
                if fname.lower().endswith(IMAGE_EXTENSIONS):
                    img_path = os.path.join(identity_dir, fname)
                    cam_id = ReIDDataset.extract_camera_id(fname)
                    self.samples.append((img_path, identity_name, cam_id))
        if limit is not None and limit > 0:
            self.samples = self.samples[: int(limit)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str, int, str]:
        img_path, identity, cam_id = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return image, identity, cam_id, img_path


def collate_pil(batch: Sequence[Tuple[Image.Image, str, int, str]]) -> Tuple[List[Image.Image], List[str], List[int], List[str]]:
    images, identities, cams, paths = zip(*batch)
    return list(images), list(identities), [int(cam) for cam in cams], list(paths)


class FrozenFeatureModel:
    key: str
    display_name: str

    def __call__(self, images: Sequence[Image.Image]) -> torch.Tensor:
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError


class TimmFeatureModel(FrozenFeatureModel):
    def __init__(
        self,
        key: str,
        display_name: str,
        model_id: str,
        image_size: int,
        device: torch.device,
    ) -> None:
        import timm

        self.key = key
        self.display_name = display_name
        self.model_id = model_id
        self.image_size = int(image_size)
        self.device = device
        self.model = timm.create_model(model_id, pretrained=True, num_classes=0).to(device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @torch.no_grad()
    def __call__(self, images: Sequence[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.transform(image) for image in images], dim=0).to(self.device)
        output = self.model(batch)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim == 4:
            output = output.mean(dim=(2, 3))
        output = F.normalize(output.float(), p=2, dim=1)
        return output.cpu()

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_key": self.key,
            "display_name": self.display_name,
            "backend": "timm",
            "model_id": self.model_id,
            "image_size": self.image_size,
            "preprocessing": "resize, ImageNet mean/std, L2 normalize",
            "weights_source": self.model_id,
        }


class DINOv2FeatureModel(FrozenFeatureModel):
    def __init__(self, device: torch.device, model_id: str = "facebook/dinov2-base") -> None:
        self.key = "dinov2_base"
        self.display_name = "DINOv2-base"
        self.model_id = model_id
        self.device = device
        self.backend = "transformers"
        self.processor = None
        self.transform = None
        try:
            from transformers import AutoImageProcessor, AutoModel

            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id).to(device)
        except Exception as exc:
            import timm

            self.backend = "timm"
            self.model_id = "vit_base_patch14_dinov2.lvd142m"
            self.transformers_error = repr(exc)
            self.model = timm.create_model(self.model_id, pretrained=True, num_classes=0).to(device)
            self.transform = transforms.Compose(
                [
                    transforms.Resize((518, 518)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        self.model.eval()

    @torch.no_grad()
    def __call__(self, images: Sequence[Image.Image]) -> torch.Tensor:
        if self.backend == "transformers":
            assert self.processor is not None
            inputs = self.processor(images=list(images), return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            output = self.model(**inputs)
            feat = output.last_hidden_state[:, 0, :]
        else:
            assert self.transform is not None
            batch = torch.stack([self.transform(image) for image in images], dim=0).to(self.device)
            feat = self.model(batch)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            if feat.ndim == 4:
                feat = feat.mean(dim=(2, 3))
        feat = F.normalize(feat.float(), p=2, dim=1)
        return feat.cpu()

    def metadata(self) -> Dict[str, Any]:
        metadata = {
            "model_key": self.key,
            "display_name": self.display_name,
            "backend": self.backend,
            "model_id": self.model_id,
            "image_size": "processor-default" if self.backend == "transformers" else 518,
            "preprocessing": (
                "AutoImageProcessor, CLS embedding, L2 normalize"
                if self.backend == "transformers"
                else "resize 518, ImageNet mean/std, pooled descriptor, L2 normalize"
            ),
            "weights_source": self.model_id,
        }
        if hasattr(self, "transformers_error"):
            metadata["transformers_fallback_error"] = self.transformers_error
        return metadata


def build_model(model_key: str, device: torch.device) -> FrozenFeatureModel:
    if model_key == "mega_l_384":
        return TimmFeatureModel(
            key="mega_l_384",
            display_name="MegaDescriptor-L-384",
            model_id="hf-hub:BVRA/MegaDescriptor-L-384",
            image_size=384,
            device=device,
        )
    if model_key == "mega_l_224":
        return TimmFeatureModel(
            key="mega_l_224",
            display_name="MegaDescriptor-L-224",
            model_id="hf-hub:BVRA/MegaDescriptor-L-224",
            image_size=224,
            device=device,
        )
    if model_key == "dinov2_base":
        return DINOv2FeatureModel(device=device)
    raise ValueError(f"Unknown external baseline model: {model_key}")


def resolve_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(PROJECT_ROOT / candidate)


def build_loader(root: str, batch_size: int, num_workers: int, limit: Optional[int]) -> DataLoader:
    dataset = PILReIDDataset(root, limit=limit)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_pil,
    )


@torch.no_grad()
def extract_dataset_features(
    model: FrozenFeatureModel,
    root: str,
    batch_size: int,
    num_workers: int,
    limit: Optional[int],
) -> Tuple[np.ndarray, List[str], List[int], List[str]]:
    loader = build_loader(root, batch_size=batch_size, num_workers=num_workers, limit=limit)
    feats: List[torch.Tensor] = []
    identities: List[str] = []
    cams: List[int] = []
    paths: List[str] = []
    for images, batch_ids, batch_cams, batch_paths in loader:
        feat = model(images)
        feats.append(feat)
        identities.extend(batch_ids)
        cams.extend(batch_cams)
        paths.extend(batch_paths)
    if not feats:
        return np.empty((0, 0), dtype=np.float32), identities, cams, paths
    return torch.cat(feats, dim=0).numpy().astype(np.float32), identities, cams, paths


def metrics_from_features(
    query_feats: np.ndarray,
    gallery_feats: np.ndarray,
    query_ids: Sequence[str],
    gallery_ids: Sequence[str],
    query_cams: Sequence[int],
    gallery_cams: Sequence[int],
    query_paths: Sequence[str],
    gallery_paths: Sequence[str],
) -> Dict[str, float]:
    distmat = compute_distance_matrix(query_feats, gallery_feats, metric="cosine")
    cmc, m_ap = compute_cmc_map(
        distmat,
        query_ids,
        gallery_ids,
        query_cams,
        gallery_cams,
        query_paths,
        gallery_paths,
        max_rank=10,
        exclude_same_camera=False,
    )
    if len(cmc) == 0:
        return {"rank1": 0.0, "rank5": 0.0, "rank10": 0.0, "mAP": 0.0}
    return {
        "rank1": float(cmc[0] * 100.0),
        "rank5": float(cmc[min(4, len(cmc) - 1)] * 100.0),
        "rank10": float(cmc[min(9, len(cmc) - 1)] * 100.0),
        "mAP": float(m_ap * 100.0),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_dry_schema(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "dataset": "dummy",
            "display_name": "Dummy",
            "model": "dinov2_base",
            "model_display": "DINOv2-base",
            "status": "ok",
            "rank1": 100.0,
            "rank5": 100.0,
            "rank10": 100.0,
            "mAP": 100.0,
            "error": "",
        }
    ]
    write_csv(
        output_dir / "external_baselines.csv",
        rows,
        ["dataset", "display_name", "model", "model_display", "status", *METRIC_FIELDS, "error"],
    )
    (output_dir / "model_metadata.json").write_text("[]\n", encoding="utf-8")
    (output_dir / "external_baselines.md").write_text("# External Baselines\n\nDry run schema.\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=str, default=f"outputs/{RUN_ID}/external_baselines")
    parser.add_argument(
        "--datasets",
        type=str,
        default="atrw,gzgc_zebra,leopard,whaleshark,stripespotter",
        help=f"Comma-separated dataset keys. Available: {','.join(DATASETS)}",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="mega_l_384,dinov2_base",
        help="Comma-separated model keys: mega_l_384, mega_l_224, dinov2_base",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--smoke_limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    if args.dry_run:
        run_dry_schema(output_dir)
        print(f"[dry-run] wrote schema outputs to {output_dir}")
        return

    selected_datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown_datasets = [key for key in selected_datasets if key not in DATASETS]
    if unknown_datasets:
        raise ValueError(f"Unknown dataset keys: {unknown_datasets}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_rows: List[Dict[str, Any]] = []
    metadata_rows: List[Dict[str, Any]] = []
    for model_key in selected_models:
        try:
            model = build_model(model_key, device)
            metadata_rows.append(model.metadata())
        except Exception as exc:
            error = repr(exc)
            metadata_rows.append({"model_key": model_key, "status": "load_failed", "error": error})
            for dataset_key in selected_datasets:
                spec = DATASETS[dataset_key]
                result_rows.append(
                    {
                        "dataset": dataset_key,
                        "display_name": spec.display_name,
                        "model": model_key,
                        "model_display": model_key,
                        "status": "model_load_failed",
                        "rank1": 0.0,
                        "rank5": 0.0,
                        "rank10": 0.0,
                        "mAP": 0.0,
                        "error": error,
                    }
                )
            if not args.continue_on_error:
                raise
            continue

        for dataset_key in selected_datasets:
            spec = DATASETS[dataset_key]
            try:
                query_feats, query_ids, query_cams, query_paths = extract_dataset_features(
                    model,
                    resolve_path(spec.query),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    limit=args.smoke_limit,
                )
                gallery_feats, gallery_ids, gallery_cams, gallery_paths = extract_dataset_features(
                    model,
                    resolve_path(spec.gallery),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    limit=args.smoke_limit,
                )
                metrics = metrics_from_features(
                    query_feats,
                    gallery_feats,
                    query_ids,
                    gallery_ids,
                    query_cams,
                    gallery_cams,
                    query_paths,
                    gallery_paths,
                )
                result_rows.append(
                    {
                        "dataset": dataset_key,
                        "display_name": spec.display_name,
                        "model": model.key,
                        "model_display": model.display_name,
                        "status": "ok",
                        **metrics,
                        "error": "",
                    }
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                result_rows.append(
                    {
                        "dataset": dataset_key,
                        "display_name": spec.display_name,
                        "model": model.key,
                        "model_display": model.display_name,
                        "status": "eval_failed",
                        "rank1": 0.0,
                        "rank5": 0.0,
                        "rank10": 0.0,
                        "mAP": 0.0,
                        "error": repr(exc),
                    }
                )

    write_csv(
        output_dir / "external_baselines.csv",
        result_rows,
        ["dataset", "display_name", "model", "model_display", "status", *METRIC_FIELDS, "error"],
    )
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = ["# External Frozen-Feature Baselines", ""]
    for row in result_rows:
        if row["status"] == "ok":
            lines.append(
                f"- {row['dataset']} / {row['model_display']}: "
                f"Rank-1 {float(row['rank1']):.2f}, Rank-5 {float(row['rank5']):.2f}, "
                f"Rank-10 {float(row['rank10']):.2f}, mAP {float(row['mAP']):.2f}"
            )
        else:
            lines.append(f"- {row['dataset']} / {row['model']}: {row['status']} ({row['error']})")
    (output_dir / "external_baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[external-baselines] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
