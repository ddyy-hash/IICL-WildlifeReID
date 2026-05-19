#!/usr/bin/env python3
"""Analyze ReID performance by query-image light bins."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal local envs
    cv2 = None

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def compute_light_stats(image_path: str) -> Dict[str, float]:
    if cv2 is not None:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "q10": float(np.quantile(gray, 0.10)),
        "q90": float(np.quantile(gray, 0.90)),
        "dark_pixel_ratio": float((gray < 0.15).mean()),
        "bright_pixel_ratio": float((gray > 0.85).mean()),
    }


def collect_folder_light_stats(root: str, split: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset split not found: {root}")

    for identity_dir in sorted(item for item in root_path.iterdir() if item.is_dir()):
        for image_path in sorted(identity_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            stats = compute_light_stats(str(image_path))
            rows.append(
                {
                    "split": split,
                    "identity": identity_dir.name,
                    "path": str(image_path),
                    **stats,
                }
            )
    return rows


def add_light_bins(
    rows: List[Dict[str, Any]],
    low_quantile: float = 0.33,
    high_quantile: float = 0.67,
    contrast_quantile: float = 0.75,
) -> Dict[str, float]:
    if not rows:
        return {"brightness_low": 0.0, "brightness_high": 0.0, "contrast_high": 0.0}

    brightness = np.asarray([float(row["brightness"]) for row in rows], dtype=np.float64)
    contrast = np.asarray([float(row["contrast"]) for row in rows], dtype=np.float64)
    brightness_low = float(np.quantile(brightness, low_quantile))
    brightness_high = float(np.quantile(brightness, high_quantile))
    contrast_high = float(np.quantile(contrast, contrast_quantile))

    for row in rows:
        b = float(row["brightness"])
        c = float(row["contrast"])
        if b <= brightness_low:
            light_bin = "dark"
        elif b >= brightness_high:
            light_bin = "bright"
        else:
            light_bin = "mid"
        row["light_bin"] = light_bin
        row["hard_light"] = bool(light_bin in {"dark", "bright"} or c >= contrast_high)

    return {
        "brightness_low": brightness_low,
        "brightness_high": brightness_high,
        "contrast_high": contrast_high,
    }


def summarize_light_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {
        "all": list(rows),
        "dark": [row for row in rows if row.get("light_bin") == "dark"],
        "mid": [row for row in rows if row.get("light_bin") == "mid"],
        "bright": [row for row in rows if row.get("light_bin") == "bright"],
        "hard_light": [row for row in rows if row.get("hard_light")],
    }
    summary: Dict[str, Any] = {}
    for name, group_rows in groups.items():
        entry: Dict[str, Any] = {"count": len(group_rows)}
        for key in ("brightness", "contrast", "dark_pixel_ratio", "bright_pixel_ratio"):
            values = np.asarray([float(row[key]) for row in group_rows], dtype=np.float64)
            if values.size == 0:
                entry[key] = {"mean": None, "median": None}
            else:
                entry[key] = {"mean": float(values.mean()), "median": float(np.median(values))}
        summary[name] = entry
    return summary


def _build_eval_transform(img_height: int, img_width: int) -> Any:
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _metrics_from_subset(
    distmat: np.ndarray,
    query_indices: Sequence[int],
    query_ids: Sequence[Any],
    gallery_ids: Sequence[Any],
    query_cams: Sequence[int],
    gallery_cams: Sequence[int],
    query_paths: Sequence[str],
    gallery_paths: Sequence[str],
    exclude_same_camera: bool,
) -> Dict[str, float]:
    from app.core.evaluation import compute_cmc_map

    if not query_indices:
        return {"count": 0, "rank1": 0.0, "rank5": 0.0, "mAP": 0.0}

    idx = np.asarray(query_indices, dtype=np.int64)
    cmc, m_ap = compute_cmc_map(
        distmat[idx],
        [query_ids[i] for i in idx],
        gallery_ids,
        [query_cams[i] for i in idx],
        gallery_cams,
        [query_paths[i] for i in idx],
        gallery_paths,
        max_rank=10,
        exclude_same_camera=exclude_same_camera,
    )
    return {
        "count": int(len(idx)),
        "rank1": float(cmc[0] * 100.0),
        "rank5": float(cmc[min(4, len(cmc) - 1)] * 100.0),
        "mAP": float(m_ap * 100.0),
    }


def evaluate_checkpoint_by_light_bin(
    checkpoint_path: str,
    query_dir: str,
    gallery_dir: str,
    query_rows: Sequence[Dict[str, Any]],
    device_arg: str,
    batch_size: int,
    num_workers: Optional[int],
    img_height: Optional[int],
    img_width: Optional[int],
    exclude_same_camera: bool,
) -> Dict[str, Any]:
    import torch

    from app.core.evaluation import ReIDDataset, compute_cmc_map, compute_distance_matrix, extract_features
    from app.core.model_factory import extract_config_from_checkpoint, resolve_eval_input_size
    from tools.evaluate_reid import _build_model, _resolve_device

    device = _resolve_device(device_arg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        checkpoint = {"state_dict": checkpoint}
    model = _build_model(checkpoint, baseline=False, device=device)

    cfg = extract_config_from_checkpoint(checkpoint)
    cfg_h, cfg_w = resolve_eval_input_size(cfg)
    eval_h = int(img_height) if img_height is not None else cfg_h
    eval_w = int(img_width) if img_width is not None else cfg_w
    cfg_workers = int(_safe_get(cfg, ("hardware", "num_workers"), 0))
    workers = int(num_workers) if num_workers is not None else cfg_workers

    transform = _build_eval_transform(eval_h, eval_w)
    query_dataset = ReIDDataset(root=query_dir, transform=transform)
    gallery_dataset = ReIDDataset(root=gallery_dir, transform=transform)
    query_loader = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    gallery_loader = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    query_feats, query_ids, query_cams, query_paths = extract_features(model, query_loader, device)
    gallery_feats, gallery_ids, gallery_cams, gallery_paths = extract_features(model, gallery_loader, device)
    distmat = compute_distance_matrix(query_feats, gallery_feats, metric="cosine")

    path_to_row = {str(Path(row["path"]).resolve()): row for row in query_rows}
    bins_by_index: Dict[str, List[int]] = {"all": [], "dark": [], "mid": [], "bright": [], "hard_light": []}
    for idx, path in enumerate(query_paths):
        row = path_to_row.get(str(Path(path).resolve()))
        if row is None:
            continue
        bins_by_index["all"].append(idx)
        bins_by_index[str(row["light_bin"])].append(idx)
        if row.get("hard_light"):
            bins_by_index["hard_light"].append(idx)

    metrics = {
        name: _metrics_from_subset(
            distmat,
            indices,
            query_ids,
            gallery_ids,
            query_cams,
            gallery_cams,
            query_paths,
            gallery_paths,
            exclude_same_camera=exclude_same_camera,
        )
        for name, indices in bins_by_index.items()
    }
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "input_size": {"height": eval_h, "width": eval_w},
        "metrics": metrics,
    }


def _safe_get(mapping: Any, path: Sequence[str], default: Any) -> Any:
    value = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Light-Bin Analysis",
        "",
        "## Query Light Distribution",
        "",
        "| Bin | Count | Brightness Mean | Contrast Mean | Dark Pixel Ratio | Bright Pixel Ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("all", "dark", "mid", "bright", "hard_light"):
        row = summary["query_summary"][name]
        lines.append(
            "| {name} | {count} | {brightness} | {contrast} | {dark_ratio} | {bright_ratio} |".format(
                name=name,
                count=row["count"],
                brightness=_fmt(row["brightness"]["mean"]),
                contrast=_fmt(row["contrast"]["mean"]),
                dark_ratio=_fmt(row["dark_pixel_ratio"]["mean"]),
                bright_ratio=_fmt(row["bright_pixel_ratio"]["mean"]),
            )
        )

    if "checkpoint_eval" in summary:
        lines.extend(
            [
                "",
                "## Retrieval By Query Light Bin",
                "",
                "| Bin | Queries | Rank-1 | Rank-5 | mAP |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        metrics = summary["checkpoint_eval"]["metrics"]
        for name in ("all", "dark", "mid", "bright", "hard_light"):
            row = metrics[name]
            lines.append(
                f"| {name} | {row['count']} | {_fmt(row['rank1'])} | {_fmt(row['rank5'])} | {_fmt(row['mAP'])} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze query/gallery lighting and optional checkpoint metrics by bin.")
    parser.add_argument("--query_dir", type=str, required=True)
    parser.add_argument("--gallery_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional ReID checkpoint for retrieval-by-bin metrics")
    parser.add_argument("--output_dir", type=str, default="outputs/light_bin_analysis")
    parser.add_argument("--low_quantile", type=float, default=0.33)
    parser.add_argument("--high_quantile", type=float, default=0.67)
    parser.add_argument("--contrast_quantile", type=float, default=0.75)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--img_height", type=int, default=None)
    parser.add_argument("--img_width", type=int, default=None)
    parser.add_argument("--include_same_camera", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_rows = collect_folder_light_stats(args.query_dir, split="query")
    gallery_rows = collect_folder_light_stats(args.gallery_dir, split="gallery")
    query_thresholds = add_light_bins(
        query_rows,
        low_quantile=args.low_quantile,
        high_quantile=args.high_quantile,
        contrast_quantile=args.contrast_quantile,
    )
    gallery_thresholds = add_light_bins(
        gallery_rows,
        low_quantile=args.low_quantile,
        high_quantile=args.high_quantile,
        contrast_quantile=args.contrast_quantile,
    )

    summary: Dict[str, Any] = {
        "query_dir": str(Path(args.query_dir).resolve()),
        "gallery_dir": str(Path(args.gallery_dir).resolve()),
        "thresholds": {
            "query": query_thresholds,
            "gallery": gallery_thresholds,
        },
        "query_summary": summarize_light_rows(query_rows),
        "gallery_summary": summarize_light_rows(gallery_rows),
    }

    if args.checkpoint:
        summary["checkpoint_eval"] = evaluate_checkpoint_by_light_bin(
            checkpoint_path=args.checkpoint,
            query_dir=args.query_dir,
            gallery_dir=args.gallery_dir,
            query_rows=query_rows,
            device_arg=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_height=args.img_height,
            img_width=args.img_width,
            exclude_same_camera=not args.include_same_camera,
        )

    write_rows_csv(output_dir / "query_light_stats.csv", query_rows)
    write_rows_csv(output_dir / "gallery_light_stats.csv", gallery_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_md(output_dir / "summary.md", summary)

    print(json.dumps({"thresholds": summary["thresholds"], "query_summary": summary["query_summary"]}, ensure_ascii=False, indent=2))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
