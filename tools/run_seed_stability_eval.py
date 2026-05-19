#!/usr/bin/env python3
"""Evaluate seed-stability checkpoints and summarize mean/std metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_selection_locked_rerank import (
    DEFAULT_DATASETS,
    DatasetSpec,
    METRIC_FIELDS,
    run_dataset,
    write_csv,
)
from tools.evaluate_reid import _resolve_device


RUN_ID = "tmm_missing_evidence_20260517"


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def find_checkpoint(output_dir: str) -> str:
    root = PROJECT_ROOT / output_dir
    candidates = [
        root / "joint_best.pth",
        root / "joint_best_reid_best.pth",
        root / "baseline_best.pth",
        root / "baseline_best_reid_best.pth",
        root / "osnet_ain_x1_0" / "baseline_best.pth",
        root / "osnet_ain_x1_0" / "baseline_best_reid_best.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.relative_to(PROJECT_ROOT)).replace("\\", "/")
    raise FileNotFoundError(f"No reportable checkpoint found under {root}")


def build_spec(row: Dict[str, str], checkpoint: str) -> DatasetSpec:
    dataset = row["dataset"]
    base = DEFAULT_DATASETS.get(dataset)
    if base is None:
        raise ValueError(f"Unsupported dataset for seed stability: {dataset}")
    return DatasetSpec(
        key=dataset,
        display_name=base.display_name,
        selection_query=base.selection_query,
        selection_gallery=base.selection_gallery,
        query=row["query_dir"],
        gallery=row["gallery_dir"],
        checkpoint=checkpoint,
        has_selection=base.has_selection,
    )


def add_seed_fields(rows: Sequence[Dict[str, Any]], manifest_row: Dict[str, str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["variant"] = manifest_row["variant"]
        enriched["seed"] = int(manifest_row["seed"])
        out.append(enriched)
    return out


def summarize(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("split") != "test":
            continue
        key = (str(row["dataset"]), str(row.get("variant", "")), str(row.get("method", "")))
        groups.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (dataset, variant, method), group_rows in sorted(groups.items()):
        summary: Dict[str, Any] = {
            "dataset": dataset,
            "variant": variant,
            "method": method,
            "n": len(group_rows),
            "seeds": ",".join(str(int(row["seed"])) for row in sorted(group_rows, key=lambda r: int(r["seed"]))),
        }
        for metric in METRIC_FIELDS:
            values = [float(row[metric]) for row in group_rows]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    return summary_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=str,
        default="config/tmm_seed_stability_20260517/seed_stability_manifest.tsv",
    )
    parser.add_argument("--output_dir", type=str, default=f"outputs/{RUN_ID}/seed_stability")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--k1_values", type=str, default="6,10,14,20")
    parser.add_argument("--k2_values", type=str, default="1,3,6")
    parser.add_argument("--lambda_values", type=str, default="0.3,0.5,0.7")
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        write_csv(
            output_dir / "seed_stability_full.csv",
            [
                {
                    "dataset": "dummy",
                    "display_name": "Dummy",
                    "variant": "full",
                    "seed": 42,
                    "split": "test",
                    "method": "plain",
                    "checkpoint": "dummy.pth",
                    "k1": "",
                    "k2": "",
                    "lambda": "",
                    "rank1": 100.0,
                    "rank5": 100.0,
                    "rank10": 100.0,
                    "mAP": 100.0,
                }
            ],
            [
                "dataset",
                "display_name",
                "variant",
                "seed",
                "split",
                "method",
                "checkpoint",
                "k1",
                "k2",
                "lambda",
                *METRIC_FIELDS,
            ],
        )
        (output_dir / "seed_stability_summary.md").write_text("# Seed Stability\n\nDry run schema.\n", encoding="utf-8")
        return

    manifest_rows = read_manifest(PROJECT_ROOT / args.manifest)
    device = _resolve_device(args.device)
    k1_values = parse_int_list(args.k1_values)
    k2_values = parse_int_list(args.k2_values)
    lambda_values = parse_float_list(args.lambda_values)

    all_grid_rows: List[Dict[str, Any]] = []
    all_test_rows: List[Dict[str, Any]] = []
    selected_params: Dict[str, Any] = {}
    missing_rows: List[Dict[str, Any]] = []

    for manifest_row in manifest_rows:
        try:
            checkpoint = find_checkpoint(manifest_row["output_dir"])
        except FileNotFoundError as exc:
            if not args.allow_missing:
                raise
            missing_rows.append({**manifest_row, "error": repr(exc)})
            continue

        spec = build_spec(manifest_row, checkpoint)
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
            subset_query=None,
            subset_gallery=None,
        )
        all_grid_rows.extend(add_seed_fields(grid_rows, manifest_row))
        all_test_rows.extend(add_seed_fields(test_rows, manifest_row))
        if selected_record:
            selected_params[
                f"{manifest_row['dataset']}_{manifest_row['variant']}_seed{manifest_row['seed']}"
            ] = selected_record

    fields = [
        "dataset",
        "display_name",
        "variant",
        "seed",
        "split",
        "method",
        "checkpoint",
        "k1",
        "k2",
        "lambda",
        *METRIC_FIELDS,
    ]
    write_csv(output_dir / "seed_stability_full.csv", all_test_rows, fields)
    write_csv(
        output_dir / "seed_stability_selection_grid.csv",
        all_grid_rows,
        ["dataset", "display_name", "variant", "seed", "split", "k1", "k2", "lambda", *METRIC_FIELDS],
    )
    summary_rows = summarize(all_test_rows)
    write_csv(
        output_dir / "seed_stability_summary.csv",
        summary_rows,
        [
            "dataset",
            "variant",
            "method",
            "n",
            "seeds",
            "rank1_mean",
            "rank1_std",
            "rank5_mean",
            "rank5_std",
            "rank10_mean",
            "rank10_std",
            "mAP_mean",
            "mAP_std",
        ],
    )
    (output_dir / "selected_seed_graph_params.json").write_text(
        json.dumps(selected_params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "missing_checkpoints.json").write_text(
        json.dumps(missing_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = ["# Seed Stability", ""]
    for row in summary_rows:
        lines.append(
            f"- {row['dataset']} / {row['variant']} / {row['method']}: "
            f"Rank-1 {float(row['rank1_mean']):.2f}+/-{float(row['rank1_std']):.2f}, "
            f"mAP {float(row['mAP_mean']):.2f}+/-{float(row['mAP_std']):.2f} "
            f"(n={row['n']}, seeds={row['seeds']})"
        )
    if missing_rows:
        lines.append("")
        lines.append(f"Missing checkpoints: {len(missing_rows)}")
    (output_dir / "seed_stability_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[seed-stability] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
