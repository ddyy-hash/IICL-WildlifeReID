#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent


@dataclass
class DatasetConfig:
    name: str
    train: str
    query: str
    gallery: str
    config: str


DATASETS: Dict[str, DatasetConfig] = {
    "atrw": DatasetConfig(
        name="ATRW (Closed-Set)",
        train="data/processed/atrw/train",
        query="data/processed/atrw/query",
        gallery="data/processed/atrw/gallery",
        config="config/illumination_config.yaml",
    ),
    "stripespotter": DatasetConfig(
        name="StripeSpotter (Closed-Set)",
        train="data/processed/stripespotter/train",
        query="data/processed/stripespotter/query",
        gallery="data/processed/stripespotter/gallery",
        config="config/illumination_config_stripespotter.yaml",
    ),
    "gzgc_zebra": DatasetConfig(
        name="GZGC Zebra (Closed-Set)",
        train="data/processed/gzgc_zebra/train",
        query="data/processed/gzgc_zebra/query",
        gallery="data/processed/gzgc_zebra/gallery",
        config="config/illumination_config_gzgc.yaml",
    ),
    "gzgc_giraffe": DatasetConfig(
        name="GZGC Giraffe (Closed-Set)",
        train="data/processed/gzgc_giraffe/train",
        query="data/processed/gzgc_giraffe/query",
        gallery="data/processed/gzgc_giraffe/gallery",
        config="config/illumination_config_gzgc_giraffe.yaml",
    ),
    "czechlynx": DatasetConfig(
        name="CzechLynx (Official Time-Closed)",
        train="data/processed/czechlynx/train",
        query="data/processed/czechlynx/query",
        gallery="data/processed/czechlynx/gallery",
        config="config/illumination_config_czechlynx_actual.yaml",
    ),
    "atrw_openset": DatasetConfig(
        name="ATRW (Open-Set)",
        train="data/processed/atrw_openset/train",
        query="data/processed/atrw_openset/query",
        gallery="data/processed/atrw_openset/gallery",
        config="config/illumination_config.yaml",
    ),
    "stripespotter_openset": DatasetConfig(
        name="StripeSpotter (Open-Set)",
        train="data/processed/stripespotter_openset/train",
        query="data/processed/stripespotter_openset/query",
        gallery="data/processed/stripespotter_openset/gallery",
        config="config/illumination_config_stripespotter.yaml",
    ),
    "gzgc_zebra_openset": DatasetConfig(
        name="GZGC Zebra (Open-Set)",
        train="data/processed/gzgc_zebra_openset/train",
        query="data/processed/gzgc_zebra_openset/query",
        gallery="data/processed/gzgc_zebra_openset/gallery",
        config="config/illumination_config_gzgc.yaml",
    ),
    "gzgc_giraffe_openset": DatasetConfig(
        name="GZGC Giraffe (Open-Set)",
        train="data/processed/gzgc_giraffe_openset/train",
        query="data/processed/gzgc_giraffe_openset/query",
        gallery="data/processed/gzgc_giraffe_openset/gallery",
        config="config/illumination_config_gzgc_giraffe.yaml",
    ),
}

VARIANTS = ("baseline", "ipaid_only", "full")


def _parse_datasets(dataset_arg: str) -> List[str]:
    if dataset_arg.lower() == "all":
        return list(DATASETS.keys())
    selected = [x.strip() for x in dataset_arg.split(",") if x.strip()]
    invalid = [x for x in selected if x not in DATASETS]
    if invalid:
        raise ValueError(f"Unknown datasets: {invalid}. Available options: {list(DATASETS.keys())}")
    return selected


def _parse_variants(variants_arg: str) -> List[str]:
    selected = [x.strip() for x in variants_arg.split(",") if x.strip()]
    invalid = [x for x in selected if x not in VARIANTS]
    if invalid:
        raise ValueError(f"Unknown variants: {invalid}. Available options: {list(VARIANTS)}")
    return selected


def _load_phase_epochs(config_path: str) -> tuple[int, int]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    phases = (cfg.get("training", {}) or {}).get("phases", {}) or {}
    p1 = int((phases.get("phase1", {}) or {}).get("epochs", 10))
    p2 = int((phases.get("phase2", {}) or {}).get("epochs", 100))
    return p1, p2


def _run_command(cmd: List[str], capture: bool = False) -> tuple[int, str]:
    print("[CMD]", " ".join(cmd))
    if capture:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, output

    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    return proc.returncode, ""


def _create_ipaid_only_config(base_config: str, out_path: Path) -> None:
    with open(base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    training = cfg.setdefault("training", {})

    iicl = training.setdefault("iicl", {})
    iicl["enabled"] = False

    losses = training.get("losses", {})
    if isinstance(losses, dict):
        iicl_loss = losses.get("iicl")
        if isinstance(iicl_loss, dict):
            iicl_loss["enabled"] = False

    metric_learning = training.get("metric_learning", {})
    if isinstance(metric_learning, dict):
        iicl_metric = metric_learning.get("iicl")
        if isinstance(iicl_metric, dict):
            iicl_metric["enabled"] = False

    training["output_dir"] = str(out_path.parent).replace("\\", "/")

    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def _find_checkpoint(variant: str, out_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    if variant == "baseline":
        candidates = [
            out_dir / "baseline_best.pth",
        ]
    else:
        candidates = [
            out_dir / "joint_best_reid_best.pth",
            out_dir / "joint_best.pth",
        ]

    for path in candidates:
        if path.exists():
            return path

    pattern = "baseline_epoch*.pth" if variant == "baseline" else "joint_phase*_epoch*.pth"
    files = sorted(out_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _parse_eval_metrics(output: str) -> Dict[str, float]:
    rank1 = re.search(r"Rank-1\s*:\s*([0-9.]+)%", output)
    rank5 = re.search(r"Rank-5\s*:\s*([0-9.]+)%", output)
    rank10 = re.search(r"Rank-10\s*:\s*([0-9.]+)%", output)
    m_ap = re.search(r"mAP\s*:\s*([0-9.]+)%", output)
    if not rank1 or not m_ap:
        return {}
    return {
        "rank1": float(rank1.group(1)),
        "rank5": float(rank5.group(1)) if rank5 else 0.0,
        "rank10": float(rank10.group(1)) if rank10 else 0.0,
        "mAP": float(m_ap.group(1)),
    }


def _evaluate_variant(variant: str, checkpoint_path: Path, ds: DatasetConfig, device: str) -> Dict[str, float]:
    cmd = [
        sys.executable,
        "tools/evaluate_reid.py",
        "--checkpoint",
        str(checkpoint_path),
        "--query_dir",
        ds.query,
        "--gallery_dir",
        ds.gallery,
        "--device",
        device,
    ]
    if variant == "baseline":
        cmd.append("--baseline")

    code, output = _run_command(cmd, capture=True)
    print(output)
    if code != 0:
        return {}
    return _parse_eval_metrics(output)


def _run_variant(
    variant: str,
    ds_key: str,
    ds: DatasetConfig,
    output_root: Path,
    device: str,
    num_workers: int,
    quick: bool,
) -> Dict[str, object]:
    out_dir = output_root / ds_key / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"[{ds_key}] Running variant: {variant}")
    print("=" * 70)

    p1, p2 = _load_phase_epochs(str(PROJECT_ROOT / ds.config))
    baseline_epochs = p1 + p2

    train_cmd: List[str]
    if variant == "baseline":
        epochs = 15 if quick else baseline_epochs
        train_cmd = [
            sys.executable,
            "tools/train_baselines.py",
            "--data_dir",
            ds.train,
            "--config",
            ds.config,
            "--query_dir",
            ds.query,
            "--gallery_dir",
            ds.gallery,
            "--output_dir",
            str(out_dir),
            "--backbone",
            "osnet_ain_x1_0",
            "--epochs",
            str(epochs),
            "--device",
            device,
        ]
    else:
        config_path = Path(PROJECT_ROOT / ds.config)
        if variant == "ipaid_only":
            config_path = out_dir / "config_ipaid_only.yaml"
            _create_ipaid_only_config(str(PROJECT_ROOT / ds.config), config_path)

        train_cmd = [
            sys.executable,
            "tools/train_joint.py",
            "--config",
            str(config_path),
            "--data_dir",
            ds.train,
            "--query_dir",
            ds.query,
            "--gallery_dir",
            ds.gallery,
            "--output_dir",
            str(out_dir),
            "--device",
            device,
            "--num_workers",
            str(num_workers),
        ]

        if variant == "ipaid_only":
            train_cmd.append("--no_iicl")
        if variant == "full":
            train_cmd.append("--use_iicl")

        if quick:
            train_cmd.extend(["--phase1_epochs", "3", "--phase2_epochs", "10"])

    start = time.time()
    code, _ = _run_command(train_cmd, capture=False)
    duration = time.time() - start

    result: Dict[str, object] = {
        "status": "ok" if code == 0 else "failed",
        "train_return_code": code,
        "duration_sec": round(duration, 2),
        "output_dir": str(out_dir),
        "train_command": " ".join(train_cmd),
    }

    if code != 0:
        return result

    checkpoint_path = _find_checkpoint(variant, out_dir)
    if checkpoint_path is None:
        result["status"] = "failed"
        result["error"] = "checkpoint_not_found"
        return result

    result["checkpoint"] = str(checkpoint_path)
    metrics = _evaluate_variant(variant, checkpoint_path, ds, device)
    if not metrics:
        result["status"] = "failed"
        result["error"] = "evaluation_failed"
        return result

    result["metrics"] = metrics
    return result


def _safe_metric(dataset_results: Dict[str, object], variant: str, metric: str) -> Optional[float]:
    record = dataset_results.get(variant, {})
    if not isinstance(record, dict):
        return None
    metrics = record.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _write_report(results: Dict[str, Dict[str, object]], report_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Ablation Results Summary")
    lines.append("")

    for dataset_key, dataset_results in results.items():
        lines.append(f"## {dataset_key}")
        lines.append("")
        lines.append("| Variant | Rank-1 | mAP | Status |")
        lines.append("|---|---:|---:|---|")

        for variant in VARIANTS:
            record = dataset_results.get(variant, {}) if isinstance(dataset_results, dict) else {}
            if not isinstance(record, dict):
                record = {}
            metrics = record.get("metrics", {}) if isinstance(record.get("metrics"), dict) else {}
            rank1 = metrics.get("rank1")
            m_ap = metrics.get("mAP")
            status = record.get("status", "missing")
            rank1_text = f"{rank1:.2f}" if isinstance(rank1, (int, float)) else "N/A"
            map_text = f"{m_ap:.2f}" if isinstance(m_ap, (int, float)) else "N/A"
            lines.append(f"| {variant} | {rank1_text} | {map_text} | {status} |")

        baseline_rank1 = _safe_metric(dataset_results, "baseline", "rank1")
        baseline_map = _safe_metric(dataset_results, "baseline", "mAP")
        ipaid_rank1 = _safe_metric(dataset_results, "ipaid_only", "rank1")
        ipaid_map = _safe_metric(dataset_results, "ipaid_only", "mAP")
        full_rank1 = _safe_metric(dataset_results, "full", "rank1")
        full_map = _safe_metric(dataset_results, "full", "mAP")

        lines.append("")
        lines.append("Contribution Breakdown:")

        if baseline_rank1 is not None and ipaid_rank1 is not None:
            lines.append(
                f"- IPAID gain: Rank-1 {ipaid_rank1 - baseline_rank1:+.2f}, mAP {(ipaid_map or 0) - (baseline_map or 0):+.2f}"
            )
        else:
            lines.append("- IPAID gain: N/A")

        if ipaid_rank1 is not None and full_rank1 is not None:
            lines.append(
                f"- IICL gain: Rank-1 {full_rank1 - ipaid_rank1:+.2f}, mAP {(full_map or 0) - (ipaid_map or 0):+.2f}"
            )
        else:
            lines.append("- IICL gain: N/A")

        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified ablation runner")
    parser.add_argument(
        "--dataset",
        type=str,
        default="atrw",
        help=(
            "Dataset names: atrw/stripespotter/gzgc_zebra/gzgc_giraffe "
            "plus open-set variants; use commas or 'all'"
        ),
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="baseline,ipaid_only,full",
        help="Comma-separated ablation variants: baseline,ipaid_only,full",
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode: baseline=15 epochs, ipaid_only/full=phase1=3 phase2=10")
    parser.add_argument("--output_dir", type=str, default="checkpoints/ablation", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data-loading workers")
    args = parser.parse_args()

    datasets = _parse_datasets(args.dataset)
    variants = _parse_variants(args.variants)

    output_root = (PROJECT_ROOT / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict[str, object]] = {}

    for ds_key in datasets:
        ds = DATASETS[ds_key]
        ds_results: Dict[str, object] = {}
        for variant in variants:
            ds_results[variant] = _run_variant(
                variant=variant,
                ds_key=ds_key,
                ds=ds,
                output_root=output_root,
                device=args.device,
                num_workers=args.num_workers,
                quick=args.quick,
            )
        all_results[ds_key] = ds_results

    results_path = output_root / "ablation_results.json"
    report_path = output_root / "ablation_report.md"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    _write_report(all_results, report_path)

    print("\n" + "=" * 70)
    print("Ablation run complete.")
    print(f"Results file: {results_path}")
    print(f"Report file: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
