#!/usr/bin/env python3
"""Materialize TMM algorithm-improvement configs with train-time neighborhood consistency."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "config" / "tmm_neighborhood_20260513"
RUN_ROOT = "checkpoints/tmm_neighborhood_20260513"


SOURCES: Dict[str, Dict[str, Any]] = {
    "leopard": {
        "base": "config/tmm_finalbestopt_20260512/illumination_config_leopard_low_wd_smooth.yaml",
        "variant": "low_wd_smooth_neighborhood",
        "weight": 0.075,
        "topk": 6,
        "local_weight": 0.45,
        "negative_weight": 0.25,
    },
    "whaleshark": {
        "base": "config/tmm_finalbestopt_20260512/illumination_config_whaleshark_shark_k5_metric.yaml",
        "variant": "shark_k5_neighborhood",
        "weight": 0.09,
        "topk": 5,
        "local_weight": 0.55,
        "negative_weight": 0.20,
    },
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _set(root: Dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    cursor = root
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def materialize() -> List[Dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []
    for dataset, spec in SOURCES.items():
        cfg = deepcopy(_load_yaml(PROJECT_ROOT / spec["base"]))
        variant = str(spec["variant"])
        output_dir = f"{RUN_ROOT}/{dataset}/{variant}"

        _set(cfg, ("output_dir",), output_dir)
        _set(cfg, ("training", "output_dir"), output_dir)
        _set(cfg, ("output", "checkpoint_dir"), output_dir)
        _set(cfg, ("output", "log_dir"), output_dir)
        _set(cfg, ("training", "neighborhood_consistency", "enabled"), True)
        _set(cfg, ("training", "neighborhood_consistency", "weight"), float(spec["weight"]))
        _set(cfg, ("training", "neighborhood_consistency", "temperature"), 0.07)
        _set(cfg, ("training", "neighborhood_consistency", "topk"), int(spec["topk"]))
        _set(cfg, ("training", "neighborhood_consistency", "positive_weight"), 1.0)
        _set(cfg, ("training", "neighborhood_consistency", "negative_weight"), float(spec["negative_weight"]))
        _set(cfg, ("training", "neighborhood_consistency", "local_weight"), float(spec["local_weight"]))
        _set(cfg, ("training", "ranking_topology", "enabled"), False)
        _set(cfg, ("training", "local_rank_preserving", "enabled"), False)
        _set(cfg, ("evaluation", "best_metric"), "mAP")
        _set(cfg, ("evaluation", "protocol"), "query_gallery")
        cfg["tmm_neighborhood_20260513"] = {
            "dataset": dataset,
            "variant": variant,
            "algorithm_change": "train-time global/local teacher neighborhood consistency",
            "motivation": "local reranking improved mAP, so the model is trained to internalize neighborhood calibration",
        }

        path = OUT_DIR / f"illumination_config_{dataset}_{variant}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
        summary.append(
            {
                "dataset": dataset,
                "variant": variant,
                "path": path.as_posix(),
                "base": spec["base"],
                "output_dir": output_dir,
                "neighborhood_consistency": cfg["training"]["neighborhood_consistency"],
            }
        )

    summary_path = OUT_DIR / "neighborhood_config_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    print(json.dumps(materialize(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
