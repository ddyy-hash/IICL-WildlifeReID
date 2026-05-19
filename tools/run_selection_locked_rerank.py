#!/usr/bin/env python3
"""Selection-locked k-reciprocal graph calibration for TMM evidence.

The script tunes reranking hyperparameters only on each dataset's selection
query/gallery split, then applies the selected tuple exactly once on the final
test query/gallery split.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.evaluation import ReIDDataset, compute_cmc_map, compute_distance_matrix, extract_features
from app.core.model_factory import extract_config_from_checkpoint, resolve_eval_input_size
from tools.evaluate_reid import _build_model, _resolve_device
from tools.reranking import re_ranking


RUN_ID = "tmm_missing_evidence_20260517"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    selection_query: Optional[str]
    selection_gallery: Optional[str]
    query: str
    gallery: str
    checkpoint: str
    has_selection: bool = True


DEFAULT_DATASETS: Dict[str, DatasetSpec] = {
    "gzgc_zebra": DatasetSpec(
        key="gzgc_zebra",
        display_name="GZGC Zebra stress",
        selection_query="data/processed/gzgc_zebra/selection_query",
        selection_gallery="data/processed/gzgc_zebra/selection_gallery",
        query="data/processed/gzgc_zebra/query",
        gallery="data/processed/gzgc_zebra/gallery",
        checkpoint="checkpoints/tmm_formal48g/gzgc_zebra_primary/joint_phase3_epoch60.pth",
    ),
    "leopard": DatasetSpec(
        key="leopard",
        display_name="LeopardID2022",
        selection_query="data/processed/leopard/selection_query",
        selection_gallery="data/processed/leopard/selection_gallery",
        query="data/processed/leopard/query",
        gallery="data/processed/leopard/gallery",
        checkpoint="checkpoints/tmm_finalbestopt_20260512/leopard/low_wd_smooth/joint_best.pth",
    ),
    "whaleshark": DatasetSpec(
        key="whaleshark",
        display_name="WhaleSharkID",
        selection_query="data/processed/whaleshark/selection_query",
        selection_gallery="data/processed/whaleshark/selection_gallery",
        query="data/processed/whaleshark/query",
        gallery="data/processed/whaleshark/gallery",
        checkpoint="checkpoints/tmm_formal48g/whaleshark_primary/joint_phase3_epoch30.pth",
    ),
    "atrw": DatasetSpec(
        key="atrw",
        display_name="ATRW auxiliary",
        selection_query=None,
        selection_gallery=None,
        query="data/processed/atrw/query",
        gallery="data/processed/atrw/gallery",
        checkpoint=(
            "checkpoints/ablation/20260509_critique_original_data_20260509_021547/"
            "atrw_switch_level_modules/full_model/joint_best.pth"
        ),
        has_selection=False,
    ),
}


METRIC_FIELDS = ["rank1", "rank5", "rank10", "mAP"]


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def relative_or_absolute(path: str, root: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(root / candidate)


def metrics_from_distance(
    distmat: np.ndarray,
    query_ids: Sequence[Any],
    gallery_ids: Sequence[Any],
    query_cams: Sequence[int],
    gallery_cams: Sequence[int],
    query_paths: Sequence[str],
    gallery_paths: Sequence[str],
    exclude_same_camera: bool = False,
) -> Dict[str, float]:
    cmc, m_ap = compute_cmc_map(
        distmat,
        query_ids,
        gallery_ids,
        query_cams,
        gallery_cams,
        query_paths,
        gallery_paths,
        max_rank=10,
        exclude_same_camera=exclude_same_camera,
    )
    if len(cmc) == 0:
        return {"rank1": 0.0, "rank5": 0.0, "rank10": 0.0, "mAP": 0.0}
    return {
        "rank1": float(cmc[0] * 100.0),
        "rank5": float(cmc[min(4, len(cmc) - 1)] * 100.0),
        "rank10": float(cmc[min(9, len(cmc) - 1)] * 100.0),
        "mAP": float(m_ap * 100.0),
    }


def select_best_grid_row(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot select rerank parameters from an empty grid.")
    return sorted(
        rows,
        key=lambda row: (
            -float(row["mAP"]),
            -float(row["rank1"]),
            int(row["k1"]),
            int(row["k2"]),
            float(row["lambda"]),
        ),
    )[0]


def load_checkpoint_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any], int, int, bool]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        checkpoint = {"state_dict": checkpoint}
    model = _build_model(checkpoint, baseline=False, device=device)
    cfg = extract_config_from_checkpoint(checkpoint)
    img_h, img_w = resolve_eval_input_size(cfg)
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg, dict) else {}
    feature_cfg = eval_cfg.get("feature_extraction", {}) if isinstance(eval_cfg, dict) else {}
    flip_test = bool(feature_cfg.get("flip_test", eval_cfg.get("flip_test", True)))
    return model, cfg, int(img_h), int(img_w), flip_test


def build_loader(
    root: str,
    img_h: int,
    img_w: int,
    batch_size: int,
    num_workers: int,
    limit: Optional[int] = None,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
        ]
    )
    dataset = ReIDDataset(root=root, transform=transform)
    if limit is not None and limit > 0:
        dataset.samples = dataset.samples[: int(limit)]
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def extract_split_features(
    model: torch.nn.Module,
    query_dir: str,
    gallery_dir: str,
    device: torch.device,
    img_h: int,
    img_w: int,
    batch_size: int,
    num_workers: int,
    flip_test: bool,
    subset_query: Optional[int] = None,
    subset_gallery: Optional[int] = None,
) -> Dict[str, Any]:
    q_loader = build_loader(query_dir, img_h, img_w, batch_size, num_workers, subset_query)
    g_loader = build_loader(gallery_dir, img_h, img_w, batch_size, num_workers, subset_gallery)
    q_feats, q_ids, q_cams, q_paths = extract_features(model, q_loader, device, flip_test)
    g_feats, g_ids, g_cams, g_paths = extract_features(model, g_loader, device, flip_test)
    return {
        "query_features": q_feats,
        "gallery_features": g_feats,
        "query_ids": q_ids,
        "gallery_ids": g_ids,
        "query_cams": q_cams,
        "gallery_cams": g_cams,
        "query_paths": q_paths,
        "gallery_paths": g_paths,
    }


def evaluate_plain(features: Dict[str, Any]) -> Dict[str, float]:
    distmat = compute_distance_matrix(
        features["query_features"],
        features["gallery_features"],
        metric="euclidean",
    )
    return metrics_from_distance(
        distmat,
        features["query_ids"],
        features["gallery_ids"],
        features["query_cams"],
        features["gallery_cams"],
        features["query_paths"],
        features["gallery_paths"],
        exclude_same_camera=False,
    )


def evaluate_rerank(
    features: Dict[str, Any],
    k1: int,
    k2: int,
    lambda_value: float,
) -> Dict[str, float]:
    distmat = re_ranking(
        features["query_features"],
        features["gallery_features"],
        k1=int(k1),
        k2=int(k2),
        lambda_value=float(lambda_value),
    )
    return metrics_from_distance(
        distmat,
        features["query_ids"],
        features["gallery_ids"],
        features["query_cams"],
        features["gallery_cams"],
        features["query_paths"],
        features["gallery_paths"],
        exclude_same_camera=False,
    )


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_dry_schema(output_dir: Path) -> None:
    ensure_dir(output_dir)
    grid_rows = [
        {
            "dataset": "dummy",
            "display_name": "Dummy",
            "split": "selection",
            "k1": 6,
            "k2": 1,
            "lambda": 0.3,
            "rank1": 100.0,
            "rank5": 100.0,
            "rank10": 100.0,
            "mAP": 100.0,
        }
    ]
    selected = {
        "dummy": {
            "dataset": "dummy",
            "selected_on": "selection",
            "k1": 6,
            "k2": 1,
            "lambda": 0.3,
            "selection_metrics": {field: 100.0 for field in METRIC_FIELDS},
        }
    }
    test_rows = [
        {
            "dataset": "dummy",
            "display_name": "Dummy",
            "split": "test",
            "method": "plain",
            "k1": "",
            "k2": "",
            "lambda": "",
            "rank1": 100.0,
            "rank5": 100.0,
            "rank10": 100.0,
            "mAP": 100.0,
        }
    ]
    write_csv(
        output_dir / "selection_grid.csv",
        grid_rows,
        ["dataset", "display_name", "split", "k1", "k2", "lambda", *METRIC_FIELDS],
    )
    write_csv(
        output_dir / "test_plain_vs_graph.csv",
        test_rows,
        ["dataset", "display_name", "split", "method", "k1", "k2", "lambda", *METRIC_FIELDS],
    )
    (output_dir / "selected_params.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_dataset(
    spec: DatasetSpec,
    project_root: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    k1_values: Sequence[int],
    k2_values: Sequence[int],
    lambda_values: Sequence[float],
    subset_query: Optional[int],
    subset_gallery: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    checkpoint_path = relative_or_absolute(spec.checkpoint, project_root)
    model, _, img_h, img_w, flip_test = load_checkpoint_model(checkpoint_path, device)

    grid_rows: List[Dict[str, Any]] = []
    selected_record: Dict[str, Any] = {}
    test_rows: List[Dict[str, Any]] = []

    if spec.has_selection:
        selection_features = extract_split_features(
            model,
            relative_or_absolute(spec.selection_query or "", project_root),
            relative_or_absolute(spec.selection_gallery or "", project_root),
            device,
            img_h,
            img_w,
            batch_size,
            num_workers,
            flip_test,
            subset_query=subset_query,
            subset_gallery=subset_gallery,
        )
        for k1 in k1_values:
            for k2 in k2_values:
                for lambda_value in lambda_values:
                    metrics = evaluate_rerank(selection_features, k1, k2, lambda_value)
                    grid_rows.append(
                        {
                            "dataset": spec.key,
                            "display_name": spec.display_name,
                            "split": "selection",
                            "k1": int(k1),
                            "k2": int(k2),
                            "lambda": float(lambda_value),
                            **metrics,
                        }
                    )
        best = select_best_grid_row(grid_rows)
        selected_record = {
            "dataset": spec.key,
            "display_name": spec.display_name,
            "selected_on": "selection",
            "checkpoint": spec.checkpoint,
            "k1": int(best["k1"]),
            "k2": int(best["k2"]),
            "lambda": float(best["lambda"]),
            "selection_metrics": {field: float(best[field]) for field in METRIC_FIELDS},
            "tie_break": "max selection mAP, then rank1, then smaller k1/k2/lambda",
        }

    test_features = extract_split_features(
        model,
        relative_or_absolute(spec.query, project_root),
        relative_or_absolute(spec.gallery, project_root),
        device,
        img_h,
        img_w,
        batch_size,
        num_workers,
        flip_test,
        subset_query=subset_query,
        subset_gallery=subset_gallery,
    )
    plain_metrics = evaluate_plain(test_features)
    test_rows.append(
        {
            "dataset": spec.key,
            "display_name": spec.display_name,
            "split": "test",
            "method": "plain",
            "checkpoint": spec.checkpoint,
            "k1": "",
            "k2": "",
            "lambda": "",
            **plain_metrics,
        }
    )
    if selected_record:
        graph_metrics = evaluate_rerank(
            test_features,
            int(selected_record["k1"]),
            int(selected_record["k2"]),
            float(selected_record["lambda"]),
        )
        test_rows.append(
            {
                "dataset": spec.key,
                "display_name": spec.display_name,
                "split": "test",
                "method": "selection_locked_graph",
                "checkpoint": spec.checkpoint,
                "k1": int(selected_record["k1"]),
                "k2": int(selected_record["k2"]),
                "lambda": float(selected_record["lambda"]),
                **graph_metrics,
            }
        )
    return grid_rows, selected_record, test_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=str, default=f"outputs/{RUN_ID}/selection_locked_graph")
    parser.add_argument(
        "--datasets",
        type=str,
        default="gzgc_zebra,leopard,whaleshark",
        help=f"Comma-separated dataset keys. Available: {','.join(DEFAULT_DATASETS)}",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--k1_values", type=str, default="6,10,14,20")
    parser.add_argument("--k2_values", type=str, default="1,3,6")
    parser.add_argument("--lambda_values", type=str, default="0.3,0.5,0.7")
    parser.add_argument("--subset_query", type=int, default=None, help="Optional smoke-test query limit")
    parser.add_argument("--subset_gallery", type=int, default=None, help="Optional smoke-test gallery limit")
    parser.add_argument("--dry_run", action="store_true", help="Write schema-only dummy outputs")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    if args.dry_run:
        run_dry_schema(output_dir)
        print(f"[dry-run] wrote schema outputs to {output_dir}")
        return

    selected_keys = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = [key for key in selected_keys if key not in DEFAULT_DATASETS]
    if unknown:
        raise ValueError(f"Unknown dataset keys: {unknown}")

    ensure_dir(output_dir)
    device = _resolve_device(args.device)
    k1_values = parse_int_list(args.k1_values)
    k2_values = parse_int_list(args.k2_values)
    lambda_values = parse_float_list(args.lambda_values)

    all_grid_rows: List[Dict[str, Any]] = []
    all_test_rows: List[Dict[str, Any]] = []
    selected_records: Dict[str, Any] = {}

    for key in selected_keys:
        spec = DEFAULT_DATASETS[key]
        print(f"[selection-locked] dataset={spec.key} checkpoint={spec.checkpoint}")
        grid_rows, selected_record, test_rows = run_dataset(
            spec=spec,
            project_root=PROJECT_ROOT,
            output_dir=output_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            k1_values=k1_values,
            k2_values=k2_values,
            lambda_values=lambda_values,
            subset_query=args.subset_query,
            subset_gallery=args.subset_gallery,
        )
        all_grid_rows.extend(grid_rows)
        all_test_rows.extend(test_rows)
        if selected_record:
            selected_records[key] = selected_record

    write_csv(
        output_dir / "selection_grid.csv",
        all_grid_rows,
        ["dataset", "display_name", "split", "k1", "k2", "lambda", *METRIC_FIELDS],
    )
    write_csv(
        output_dir / "test_plain_vs_graph.csv",
        all_test_rows,
        ["dataset", "display_name", "split", "method", "checkpoint", "k1", "k2", "lambda", *METRIC_FIELDS],
    )
    (output_dir / "selected_params.json").write_text(
        json.dumps(selected_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_lines = ["# Selection-Locked Graph Calibration", ""]
    for row in all_test_rows:
        summary_lines.append(
            f"- {row['dataset']} {row['method']}: "
            f"Rank-1 {float(row['rank1']):.2f}, Rank-5 {float(row['rank5']):.2f}, "
            f"Rank-10 {float(row['rank10']):.2f}, mAP {float(row['mAP']):.2f}"
        )
    (output_dir / "selection_locked_graph.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    print(f"[selection-locked] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
