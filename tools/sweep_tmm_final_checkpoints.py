#!/usr/bin/env python3
"""Sweep saved TMM checkpoints on fixed final query/gallery splits."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "gzgc_zebra": {
        "checkpoint_dir": "checkpoints/tmm_formal48g/gzgc_zebra_primary",
        "query_dir": "data/processed/gzgc_zebra/query",
        "gallery_dir": "data/processed/gzgc_zebra/gallery",
    },
    "leopard": {
        "checkpoint_dir": "checkpoints/tmm_formal48g/leopard_primary",
        "query_dir": "data/processed/leopard/query",
        "gallery_dir": "data/processed/leopard/gallery",
    },
    "whaleshark": {
        "checkpoint_dir": "checkpoints/tmm_formal48g/whaleshark_primary",
        "query_dir": "data/processed/whaleshark/query",
        "gallery_dir": "data/processed/whaleshark/gallery",
    },
}

METRIC_PATTERNS = {
    "rank1": re.compile(r"Rank-1\s*:\s*([0-9.]+)%"),
    "rank5": re.compile(r"Rank-5\s*:\s*([0-9.]+)%"),
    "rank10": re.compile(r"Rank-10\s*:\s*([0-9.]+)%"),
    "mAP": re.compile(r"mAP\s*:\s*([0-9.]+)%"),
}

PREFERRED_ORDER = [
    "joint_best.pth",
    "joint_best_reid_best.pth",
    "joint_phase1_epoch10.pth",
    "joint_phase3_epoch10.pth",
    "joint_phase3_epoch20.pth",
    "joint_phase3_epoch30.pth",
    "joint_phase3_epoch40.pth",
    "joint_phase3_epoch50.pth",
    "joint_phase3_epoch60.pth",
]


def _project_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _iter_checkpoints(checkpoint_dir: Path) -> List[Path]:
    found = {path.name: path for path in checkpoint_dir.glob("*.pth")}
    ordered = [found[name] for name in PREFERRED_ORDER if name in found]
    remaining = sorted(path for name, path in found.items() if name not in set(PREFERRED_ORDER))
    return ordered + remaining


def _parse_metrics(stdout: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = pattern.search(stdout)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def _run_eval(
    checkpoint: Path,
    query_dir: Path,
    gallery_dir: Path,
    batch_size: int,
    device: str,
    num_workers: int,
    rerank: bool,
) -> Dict[str, object]:
    cmd = [
        sys.executable,
        "tools/evaluate_reid.py",
        "--checkpoint",
        str(checkpoint),
        "--query_dir",
        str(query_dir),
        "--gallery_dir",
        str(gallery_dir),
        "--batch_size",
        str(batch_size),
        "--device",
        device,
        "--num_workers",
        str(num_workers),
    ]
    if rerank:
        cmd.append("--rerank")
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    metrics = _parse_metrics(proc.stdout)
    return {
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_name": checkpoint.name,
        "exit_code": proc.returncode,
        "rank1": metrics.get("rank1"),
        "rank5": metrics.get("rank5"),
        "rank10": metrics.get("rank10"),
        "mAP": metrics.get("mAP"),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-30:]),
    }


def _write_outputs(rows: List[Dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tmm_final_checkpoint_sweep_results.json"
    csv_path = out_dir / "tmm_final_checkpoint_sweep_results.csv"
    md_path = out_dir / "tmm_final_checkpoint_sweep_results.md"

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "dataset",
        "checkpoint_name",
        "checkpoint",
        "exit_code",
        "rank1",
        "rank5",
        "rank10",
        "mAP",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    lines = ["# TMM Final Query/Gallery Checkpoint Sweep", ""]
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        ds_rows = [row for row in rows if row["dataset"] == dataset and row.get("exit_code") == 0]
        ds_rows.sort(key=lambda row: (float(row.get("mAP") or -1), float(row.get("rank1") or -1)), reverse=True)
        lines.extend([f"## {dataset}", "", "| checkpoint | Rank-1 | Rank-5 | Rank-10 | mAP |", "|---|---:|---:|---:|---:|"])
        for row in ds_rows:
            lines.append(
                "| {checkpoint_name} | {rank1:.2f} | {rank5:.2f} | {rank10:.2f} | {mAP:.2f} |".format(
                    checkpoint_name=row["checkpoint_name"],
                    rank1=float(row.get("rank1") or 0.0),
                    rank5=float(row.get("rank5") or 0.0),
                    rank10=float(row.get("rank10") or 0.0),
                    mAP=float(row.get("mAP") or 0.0),
                )
            )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=sorted(DATASETS))
    parser.add_argument("--output_dir", default="outputs/tmm_final_sweep_20260512")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        checkpoint_dir = _project_path(spec["checkpoint_dir"])
        query_dir = _project_path(spec["query_dir"])
        gallery_dir = _project_path(spec["gallery_dir"])
        checkpoints = _iter_checkpoints(checkpoint_dir)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found under {checkpoint_dir}")
        for checkpoint in checkpoints:
            print(f"[SWEEP] {dataset} {checkpoint.name}", flush=True)
            row = _run_eval(
                checkpoint=checkpoint,
                query_dir=query_dir,
                gallery_dir=gallery_dir,
                batch_size=args.batch_size,
                device=args.device,
                num_workers=args.num_workers,
                rerank=bool(args.rerank),
            )
            row["dataset"] = dataset
            rows.append(row)
            if row["exit_code"] == 0:
                print(
                    "[RESULT] {dataset} {checkpoint} Rank-1={rank1} Rank-5={rank5} mAP={mAP}".format(
                        dataset=dataset,
                        checkpoint=checkpoint.name,
                        rank1=row.get("rank1"),
                        rank5=row.get("rank5"),
                        mAP=row.get("mAP"),
                    ),
                    flush=True,
                )
            else:
                print(f"[ERROR] {dataset} {checkpoint.name} exit={row['exit_code']}", flush=True)

    out_dir = _project_path(args.output_dir)
    _write_outputs(rows, out_dir)
    print(f"[DONE] wrote {out_dir}")


if __name__ == "__main__":
    main()
