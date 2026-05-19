#!/usr/bin/env python3
"""Prepare match-aware identity-disjoint ReID splits for TMM experiments.

The script rebuilds processed ReID directories from the original COCO-style
datasets, instead of reusing earlier mixed or closed-set splits.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageStat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    annotation_path: Path
    image_dir: Path
    output_dir: Path
    category_id: Optional[int]
    stress_metric: str
    min_train_images: int = 2
    min_eval_images: int = 3
    margin: float = 0.10


@dataclass
class Sample:
    identity: str
    image_id: int
    annotation_id: int
    file_name: str
    src_path: Path
    bbox: Tuple[float, float, float, float]
    viewpoint: str
    metric_value: Optional[float] = None


def _project_path(path: str) -> Path:
    return (PROJECT_ROOT / path).resolve()


DATASETS: Dict[str, DatasetSpec] = {
    "gzgc_zebra": DatasetSpec(
        key="gzgc_zebra",
        display_name="GZGC Zebra",
        annotation_path=_project_path("orignal_data/gzgc.coco/gzgc.coco/annotations/instances_train2020.json"),
        image_dir=_project_path("orignal_data/gzgc.coco/gzgc.coco/images/train2020"),
        output_dir=_project_path("data/processed/gzgc_zebra"),
        category_id=1,
        stress_metric="brightness",
    ),
    "leopard": DatasetSpec(
        key="leopard",
        display_name="LeopardID2022",
        annotation_path=_project_path("orignal_data/leopard.coco/annotations/instances_train2022.json"),
        image_dir=_project_path("orignal_data/leopard.coco/images/train2022"),
        output_dir=_project_path("data/processed/leopard"),
        category_id=None,
        stress_metric="brightness",
    ),
    "whaleshark": DatasetSpec(
        key="whaleshark",
        display_name="WhaleSharkID",
        annotation_path=_project_path(
            "orignal_data/whaleshark.coco/whaleshark.coco/annotations/instances_train2020.json"
        ),
        image_dir=_project_path("orignal_data/whaleshark.coco/whaleshark.coco/images/train2020"),
        output_dir=_project_path("data/processed/whaleshark"),
        category_id=None,
        stress_metric="color_cast",
    ),
}


def _safe_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _sanitize_identity(identity: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in identity.strip())
    return cleaned or "unknown"


def _expanded_bbox(
    bbox: Sequence[float],
    image_size: Tuple[int, int],
    margin: float,
) -> Tuple[int, int, int, int]:
    x, y, w, h = [float(v) for v in bbox]
    img_w, img_h = image_size
    margin_w = w * margin
    margin_h = h * margin
    x1 = max(0, int(x - margin_w))
    y1 = max(0, int(y - margin_h))
    x2 = min(img_w, int(x + w + margin_w))
    y2 = min(img_h, int(y + h + margin_h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid expanded bbox: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def _crop_image(src_path: Path, bbox: Sequence[float], margin: float) -> Image.Image:
    with Image.open(src_path) as image:
        image = image.convert("RGB")
        crop_box = _expanded_bbox(bbox, image.size, margin)
        return image.crop(crop_box)


def _compute_metric(sample: Sample, metric: str, margin: float) -> float:
    crop = _crop_image(sample.src_path, sample.bbox, margin)
    crop.thumbnail((160, 160))
    if metric == "brightness":
        gray = crop.convert("L")
        return float(ImageStat.Stat(gray).mean[0])
    if metric == "color_cast":
        stat = ImageStat.Stat(crop)
        channels = stat.mean[:3]
        avg = sum(channels) / 3.0
        if avg <= 1e-6:
            return 0.0
        channel_mean = avg
        variance = sum((channel - channel_mean) ** 2 for channel in channels) / 3.0
        return float((variance ** 0.5) / channel_mean)
    raise ValueError(f"Unsupported stress metric: {metric}")


def _save_crop(task: Tuple[Sample, Path, float]) -> Tuple[str, bool, str]:
    sample, dst_path, margin = task
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        crop = _crop_image(sample.src_path, sample.bbox, margin)
        crop.save(dst_path, "JPEG", quality=95)
        return dst_path.as_posix(), True, ""
    except Exception as exc:  # pragma: no cover - recorded in split_info
        return dst_path.as_posix(), False, str(exc)


def load_identity_samples(spec: DatasetSpec) -> Tuple[Dict[str, List[Sample]], Dict[str, Any]]:
    if not spec.annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {spec.annotation_path}")
    if not spec.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {spec.image_dir}")

    with open(spec.annotation_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    images = {int(item["id"]): item for item in data.get("images", [])}
    identities: Dict[str, List[Sample]] = defaultdict(list)
    skipped = Counter()

    for ann in data.get("annotations", []):
        if spec.category_id is not None and ann.get("category_id") != spec.category_id:
            skipped["category_filter"] += 1
            continue
        identity = str(ann.get("name") or "").strip()
        if not identity:
            skipped["missing_identity"] += 1
            continue
        image_id = int(ann.get("image_id"))
        image_info = images.get(image_id)
        if not image_info:
            skipped["missing_image_record"] += 1
            continue
        file_name = str(image_info.get("file_name") or "")
        if not file_name:
            skipped["missing_file_name"] += 1
            continue
        src_path = spec.image_dir / file_name
        if not src_path.exists() or src_path.suffix.lower() not in VALID_EXTENSIONS:
            skipped["missing_source_file"] += 1
            continue
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            skipped["invalid_bbox"] += 1
            continue

        identities[identity].append(
            Sample(
                identity=identity,
                image_id=image_id,
                annotation_id=int(ann.get("id")),
                file_name=file_name,
                src_path=src_path,
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                viewpoint=str(ann.get("viewpoint") or "unknown"),
            )
        )

    metadata = {
        "source_annotation_path": _safe_relative(spec.annotation_path),
        "source_image_dir": _safe_relative(spec.image_dir),
        "raw_images": len(data.get("images", [])),
        "raw_annotations": len(data.get("annotations", [])),
        "skipped": dict(skipped),
    }
    return dict(identities), metadata


def _stratified_identity_split(
    identities: Dict[str, List[Sample]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    min_train_images: int,
    min_eval_images: int,
) -> Dict[str, List[str]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    rng = random.Random(seed)
    valid = {name: samples for name, samples in identities.items() if len(samples) >= min_train_images}
    eval_eligible = [name for name, samples in valid.items() if len(samples) >= min_eval_images]
    train_only = [name for name, samples in valid.items() if len(samples) < min_eval_images]

    # Balance long-tail identities across splits by assigning count-sorted groups.
    eval_eligible.sort(key=lambda name: (-len(valid[name]), name))
    bins: List[List[str]] = [[] for _ in range(10)]
    for idx, name in enumerate(eval_eligible):
        bins[idx % len(bins)].append(name)

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []
    for bucket in bins:
        rng.shuffle(bucket)
        n = len(bucket)
        n_val = round(n * val_ratio)
        n_test = round(n * test_ratio)
        # Keep non-empty val/test when the whole dataset supports it.
        val_ids.extend(bucket[:n_val])
        test_ids.extend(bucket[n_val : n_val + n_test])
        train_ids.extend(bucket[n_val + n_test :])

    train_ids.extend(train_only)
    for collection in (train_ids, val_ids, test_ids):
        collection.sort()

    return {"train": train_ids, "val": val_ids, "test": test_ids, "excluded": sorted(set(identities) - set(valid))}


def _metric_samples(samples: List[Sample], spec: DatasetSpec) -> List[Sample]:
    enriched: List[Sample] = []
    for sample in samples:
        sample.metric_value = _compute_metric(sample, spec.stress_metric, spec.margin)
        enriched.append(sample)
    return enriched


def _choose_query_and_gallery(samples: List[Sample], spec: DatasetSpec) -> Tuple[Sample, List[Sample], Dict[str, Any]]:
    enriched = _metric_samples(samples, spec)
    sorted_samples = sorted(enriched, key=lambda item: (float(item.metric_value or 0.0), item.image_id, item.annotation_id))
    low = sorted_samples[0]
    high = sorted_samples[-1]

    def avg_distance(candidate: Sample) -> float:
        value = float(candidate.metric_value or 0.0)
        others = [float(item.metric_value or 0.0) for item in sorted_samples if item is not candidate]
        return sum(abs(value - other) for other in others) / max(1, len(others))

    query = low if avg_distance(low) >= avg_distance(high) else high
    gallery = [item for item in sorted_samples if item is not query]
    if not gallery:
        raise ValueError("Evaluation identity must have at least one gallery image")

    metric_values = [float(item.metric_value or 0.0) for item in sorted_samples]
    return query, gallery, {
        "query_metric": float(query.metric_value or 0.0),
        "gallery_metric_min": min(float(item.metric_value or 0.0) for item in gallery),
        "gallery_metric_max": max(float(item.metric_value or 0.0) for item in gallery),
        "metric_min": min(metric_values),
        "metric_max": max(metric_values),
        "metric_range": max(metric_values) - min(metric_values),
    }


def _pid_mapping(split_ids: Dict[str, List[str]]) -> Dict[str, str]:
    all_ids = sorted(set(split_ids["train"]) | set(split_ids["val"]) | set(split_ids["test"]))
    return {identity: f"{idx:04d}" for idx, identity in enumerate(all_ids)}


def _sample_filename(pid: str, split: str, index: int, sample: Sample) -> str:
    split_code = {
        "train": "tr",
        "selection_query": "vq",
        "selection_gallery": "vg",
        "query": "tq",
        "gallery": "tg",
    }[split]
    return f"{pid}_{split_code}_{index:04d}_img{sample.image_id}_ann{sample.annotation_id}.jpg"


def build_crop_tasks(
    identities: Dict[str, List[Sample]],
    split_ids: Dict[str, List[str]],
    pid_map: Dict[str, str],
    spec: DatasetSpec,
) -> Tuple[List[Tuple[Sample, Path, float]], Dict[str, Any]]:
    tasks: List[Tuple[Sample, Path, float]] = []
    eval_metric_records: Dict[str, Dict[str, Any]] = {"val": {}, "test": {}}

    for identity in split_ids["train"]:
        pid = pid_map[identity]
        samples = sorted(identities[identity], key=lambda item: (item.image_id, item.annotation_id))
        for index, sample in enumerate(samples):
            dst = spec.output_dir / "train" / pid / _sample_filename(pid, "train", index, sample)
            tasks.append((sample, dst, spec.margin))

    for logical_split, query_dir, gallery_dir in [
        ("val", "selection_query", "selection_gallery"),
        ("test", "query", "gallery"),
    ]:
        for identity in split_ids[logical_split]:
            pid = pid_map[identity]
            query, gallery, metric_record = _choose_query_and_gallery(identities[identity], spec)
            eval_metric_records[logical_split][identity] = metric_record
            query_dst = spec.output_dir / query_dir / pid / _sample_filename(pid, query_dir, 0, query)
            tasks.append((query, query_dst, spec.margin))
            for index, sample in enumerate(gallery):
                gallery_dst = spec.output_dir / gallery_dir / pid / _sample_filename(pid, gallery_dir, index, sample)
                tasks.append((sample, gallery_dst, spec.margin))

    return tasks, eval_metric_records


def _count_split(root: Path) -> Dict[str, int]:
    if not root.exists():
        return {"ids": 0, "images": 0}
    ids = 0
    images = 0
    for identity_dir in root.iterdir():
        if not identity_dir.is_dir():
            continue
        count = sum(1 for file in identity_dir.iterdir() if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS)
        if count:
            ids += 1
            images += count
    return {"ids": ids, "images": images}


def _distribution(values: Iterable[int]) -> Dict[str, Any]:
    counts = list(values)
    if not counts:
        return {"ids": 0}
    return {
        "ids": len(counts),
        "images": sum(counts),
        "min": min(counts),
        "median": median(counts),
        "mean": round(mean(counts), 3),
        "max": max(counts),
        "ge2": sum(value >= 2 for value in counts),
        "ge3": sum(value >= 3 for value in counts),
        "ge5": sum(value >= 5 for value in counts),
        "ge8": sum(value >= 8 for value in counts),
    }


def _backup_existing(output_dirs: Sequence[Path], backup_root: Path) -> None:
    backup_root.mkdir(parents=True, exist_ok=True)
    processed_root = (PROJECT_ROOT / "data" / "processed").resolve()
    for output_dir in output_dirs:
        resolved = output_dir.resolve()
        if not resolved.exists():
            continue
        try:
            resolved.relative_to(processed_root)
        except ValueError as exc:
            raise RuntimeError(f"Refusing to move directory outside data/processed: {resolved}") from exc
        target = backup_root / output_dir.name
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(resolved), str(target))


def prepare_dataset(spec: DatasetSpec, args: argparse.Namespace) -> Dict[str, Any]:
    print(f"\n=== Preparing {spec.display_name} ({spec.key}) ===")
    identities, source_metadata = load_identity_samples(spec)
    raw_distribution = _distribution(len(samples) for samples in identities.values())
    split_ids = _stratified_identity_split(
        identities,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_train_images=spec.min_train_images,
        min_eval_images=spec.min_eval_images,
    )
    pid_map = _pid_mapping(split_ids)

    for subdir in ["train", "selection_query", "selection_gallery", "query", "gallery"]:
        (spec.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    tasks, eval_metric_records = build_crop_tasks(identities, split_ids, pid_map, spec)
    print(f"Crop tasks: {len(tasks)}")

    failures: List[Dict[str, str]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_save_crop, task) for task in tasks]
        for future in as_completed(futures):
            path, ok, error = future.result()
            completed += 1
            if not ok:
                failures.append({"path": path, "error": error})
            if completed % args.progress_interval == 0 or completed == len(tasks):
                print(f"  {spec.key}: {completed}/{len(tasks)} crops complete")

    split_counts = {
        "train": _count_split(spec.output_dir / "train"),
        "selection_query": _count_split(spec.output_dir / "selection_query"),
        "selection_gallery": _count_split(spec.output_dir / "selection_gallery"),
        "query": _count_split(spec.output_dir / "query"),
        "gallery": _count_split(spec.output_dir / "gallery"),
    }

    identity_counts = {identity: len(samples) for identity, samples in identities.items()}
    split_info: Dict[str, Any] = {
        "dataset": spec.key,
        "display_name": spec.display_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "match_aware_identity_disjoint_query_gallery",
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "validation_selection": args.val_ratio,
            "test": args.test_ratio,
        },
        "min_train_images_per_identity": spec.min_train_images,
        "min_eval_images_per_identity": spec.min_eval_images,
        "stress_metric": spec.stress_metric,
        "query_selection": (
            "for each validation/test identity, choose the image at the metric extreme with the largest "
            "average metric distance to the remaining same-identity images; remaining images form gallery"
        ),
        "source": source_metadata,
        "raw_identity_distribution": raw_distribution,
        "split_identity_counts": {key: len(value) for key, value in split_ids.items()},
        "split_image_counts_before_crop": {
            key: sum(identity_counts[identity] for identity in value)
            for key, value in split_ids.items()
            if key in {"train", "val", "test"}
        },
        "output_counts": split_counts,
        "identity_sets": {
            "train": split_ids["train"],
            "validation_selection": split_ids["val"],
            "test": split_ids["test"],
            "excluded": split_ids["excluded"],
        },
        "pid_mapping": pid_map,
        "eval_metric_summary": {
            split: {
                "identities": len(records),
                "mean_metric_range": round(mean(item["metric_range"] for item in records.values()), 6)
                if records
                else 0.0,
                "max_metric_range": round(max((item["metric_range"] for item in records.values()), default=0.0), 6),
            }
            for split, records in eval_metric_records.items()
        },
        "crop_failures": failures[:50],
        "crop_failure_count": len(failures),
    }

    with open(spec.output_dir / "split_info.json", "w", encoding="utf-8") as handle:
        json.dump(split_info, handle, indent=2, ensure_ascii=False)
    with open(spec.output_dir / "id_mapping.json", "w", encoding="utf-8") as handle:
        json.dump({"identity_to_pid": pid_map, "pid_to_identity": {v: k for k, v in pid_map.items()}}, handle, indent=2)
    with open(spec.output_dir / "dataset_stats.json", "w", encoding="utf-8") as handle:
        json.dump({"output_counts": split_counts, "raw_identity_distribution": raw_distribution}, handle, indent=2)

    print(f"Saved split metadata: {_safe_relative(spec.output_dir / 'split_info.json')}")
    return split_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="gzgc_zebra,leopard,whaleshark",
        help="Comma-separated dataset keys, or 'all'.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-interval", type=int, default=500)
    parser.add_argument("--no-backup", action="store_true", help="Do not move existing processed dirs to backup.")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned source/output paths.")
    return parser.parse_args()


def selected_specs(dataset_arg: str) -> List[DatasetSpec]:
    keys = list(DATASETS) if dataset_arg.strip().lower() == "all" else [item.strip() for item in dataset_arg.split(",")]
    unknown = [key for key in keys if key not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Available: {list(DATASETS)}")
    return [DATASETS[key] for key in keys]


def main() -> None:
    args = parse_args()
    specs = selected_specs(args.datasets)
    print("Selected datasets:", ", ".join(spec.key for spec in specs))
    for spec in specs:
        print(f"  {spec.key}: {_safe_relative(spec.annotation_path)} -> {_safe_relative(spec.output_dir)}")
    if args.dry_run:
        return

    if not args.no_backup:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_root = (PROJECT_ROOT / "data" / "processed" / f"_backup_match_aware_{timestamp}").resolve()
        print(f"Backing up existing processed directories to {_safe_relative(backup_root)}")
        _backup_existing([spec.output_dir for spec in specs], backup_root)
    else:
        for spec in specs:
            _reset_dir(spec.output_dir)

    summaries = [prepare_dataset(spec, args) for spec in specs]
    summary_path = PROJECT_ROOT / "data" / "processed" / "match_aware_split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "protocol": "match_aware_identity_disjoint_query_gallery",
                "datasets": summaries,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved global summary: {_safe_relative(summary_path)}")


if __name__ == "__main__":
    main()

