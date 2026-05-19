#!/usr/bin/env python3
"""Materialize TMM GL-NCD v2 full and ablation configs for cloud training."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "config" / "tmm_glncd_v2_20260513"
RUN_ROOT = "checkpoints/tmm_glncd_v2_20260513"


DATASETS: Dict[str, Dict[str, Any]] = {
    "atrw": {
        "base": "config/illumination_config_atrw.yaml",
        "data_dir": "data/processed/atrw/train",
        "protocol": "atrw_openset",
        "best_metric": "mmAP",
        "phase3_epochs": 66,
        "weight": 0.07,
        "topk": 6,
        "local_weight": 0.40,
        "negative_weight": 0.22,
        "eval_interval": 2,
        "priority": "core",
    },
    "leopard": {
        "base": "config/tmm_finalbestopt_20260512/illumination_config_leopard_low_wd_smooth.yaml",
        "data_dir": "data/processed/leopard/train",
        "query": "data/processed/leopard/query",
        "gallery": "data/processed/leopard/gallery",
        "protocol": "query_gallery",
        "best_metric": "mAP",
        "phase3_epochs": 60,
        "weight": 0.075,
        "topk": 6,
        "local_weight": 0.45,
        "negative_weight": 0.25,
        "eval_interval": 5,
        "priority": "main_ablation",
    },
    "whaleshark": {
        "base": "config/tmm_finalbestopt_20260512/illumination_config_whaleshark_shark_k5_metric.yaml",
        "data_dir": "data/processed/whaleshark/train",
        "query": "data/processed/whaleshark/query",
        "gallery": "data/processed/whaleshark/gallery",
        "protocol": "query_gallery",
        "best_metric": "mAP",
        "phase3_epochs": 60,
        "weight": 0.09,
        "topk": 5,
        "local_weight": 0.55,
        "negative_weight": 0.20,
        "eval_interval": 5,
        "priority": "main_ablation",
    },
    "gzgc_zebra": {
        "base": "config/tmm_finalbestopt_20260512/illumination_config_gzgc_zebra_final_primary.yaml",
        "data_dir": "data/processed/gzgc_zebra/train",
        "query": "data/processed/gzgc_zebra/query",
        "gallery": "data/processed/gzgc_zebra/gallery",
        "protocol": "query_gallery",
        "best_metric": "mAP",
        "phase3_epochs": 60,
        "weight": 0.065,
        "topk": 6,
        "local_weight": 0.40,
        "negative_weight": 0.22,
        "eval_interval": 5,
        "priority": "appendix",
    },
}


Mutator = Callable[[Dict[str, Any], Dict[str, Any]], None]


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _set(root: Dict[str, Any], keys: Tuple[str, ...], value: Any) -> None:
    cursor = root
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _get(root: Dict[str, Any], keys: Tuple[str, ...], default: Any = None) -> Any:
    cursor: Any = root
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def _set_common(cfg: Dict[str, Any], dataset: str, variant: str, spec: Dict[str, Any]) -> None:
    output_dir = f"{RUN_ROOT}/{dataset}/{variant}"
    _set(cfg, ("output_dir",), output_dir)
    _set(cfg, ("training", "output_dir"), output_dir)
    _set(cfg, ("output", "checkpoint_dir"), output_dir)
    _set(cfg, ("output", "log_dir"), output_dir)
    _set(cfg, ("training", "data_dir"), spec["data_dir"])
    _set(cfg, ("training", "phases", "phase3", "epochs"), int(spec["phase3_epochs"]))
    _set(cfg, ("training", "eval_interval"), int(spec["eval_interval"]))
    _set(cfg, ("evaluation", "eval_interval"), int(spec["eval_interval"]))
    _set(cfg, ("evaluation", "protocol"), spec["protocol"])
    _set(cfg, ("evaluation", "best_metric"), spec["best_metric"])
    _set(cfg, ("evaluation", "strict_protocol_check"), True)
    if spec["protocol"] == "query_gallery":
        _set(cfg, ("training", "query_dir"), spec["query"])
        _set(cfg, ("training", "gallery_dir"), spec["gallery"])
        _set(cfg, ("evaluation", "query_dir"), spec["query"])
        _set(cfg, ("evaluation", "gallery_dir"), spec["gallery"])
        _set(cfg, ("evaluation", "additional_protocols"), [])
    else:
        _set(cfg, ("evaluation", "additional_protocols"), ["atrw_closedset_train70_val30"])
    _set(cfg, ("checkpointing", "save_interval"), 5)
    _set(cfg, ("checkpointing", "max_keep"), 6)
    _set(cfg, ("checkpointing", "save_best_only"), False)
    _set(cfg, ("hardware", "use_ddp"), False)
    _set(cfg, ("hardware", "num_workers"), 4)
    _set(cfg, ("hardware", "use_amp"), True)
    _set(cfg, ("hardware", "amp_dtype"), "bfloat16")


def _set_glncd(cfg: Dict[str, Any], spec: Dict[str, Any]) -> None:
    _set(cfg, ("training", "neighborhood_consistency", "enabled"), True)
    _set(cfg, ("training", "neighborhood_consistency", "weight"), float(spec["weight"]))
    _set(cfg, ("training", "neighborhood_consistency", "temperature"), 0.07)
    _set(cfg, ("training", "neighborhood_consistency", "topk"), int(spec["topk"]))
    _set(cfg, ("training", "neighborhood_consistency", "positive_weight"), 1.0)
    _set(cfg, ("training", "neighborhood_consistency", "negative_weight"), float(spec["negative_weight"]))
    _set(cfg, ("training", "neighborhood_consistency", "local_weight"), float(spec["local_weight"]))
    _set(cfg, ("training", "neighborhood_consistency", "use_global"), True)
    _set(cfg, ("training", "neighborhood_consistency", "use_local"), True)
    _set(cfg, ("training", "neighborhood_consistency", "use_hard_negatives"), True)
    _set(cfg, ("training", "neighborhood_consistency", "teacher_target"), "reciprocal")
    _set(cfg, ("training", "ranking_topology", "enabled"), False)
    _set(cfg, ("training", "local_rank_preserving", "enabled"), False)


def v_full(cfg: Dict[str, Any], spec: Dict[str, Any]) -> None:
    _set_glncd(cfg, spec)


def v_global_only(cfg: Dict[str, Any], spec: Dict[str, Any]) -> None:
    _set_glncd(cfg, spec)
    _set(cfg, ("training", "neighborhood_consistency", "use_local"), False)
    _set(cfg, ("training", "neighborhood_consistency", "local_weight"), 0.0)


def v_local_only(cfg: Dict[str, Any], spec: Dict[str, Any]) -> None:
    _set_glncd(cfg, spec)
    _set(cfg, ("training", "neighborhood_consistency", "use_global"), False)
    _set(cfg, ("training", "neighborhood_consistency", "local_weight"), 1.0)


def v_no_hard_negative(cfg: Dict[str, Any], spec: Dict[str, Any]) -> None:
    _set_glncd(cfg, spec)
    _set(cfg, ("training", "neighborhood_consistency", "use_hard_negatives"), False)
    _set(cfg, ("training", "neighborhood_consistency", "negative_weight"), 0.0)


def v_soft_target(cfg: Dict[str, Any], spec: Dict[str, Any]) -> None:
    _set_glncd(cfg, spec)
    _set(cfg, ("training", "neighborhood_consistency", "teacher_target"), "soft")


VARIANTS: Dict[str, List[Tuple[str, Mutator, str]]] = {
    "atrw": [
        ("glncd_v2_full", v_full, "core ATRW validation of reciprocal-graph GL-NCD v2"),
        ("glncd_v2_global_only", v_global_only, "ATRW support ablation without stripe-local neighborhood"),
    ],
    "leopard": [
        ("glncd_v2_full", v_full, "full reciprocal global-local neighborhood consistency"),
        ("glncd_v2_global_only", v_global_only, "remove local stripe neighborhood term"),
        ("glncd_v2_local_only", v_local_only, "remove global embedding neighborhood term"),
        ("glncd_v2_no_hard_negative", v_no_hard_negative, "remove hard-negative suppression"),
        ("glncd_v2_soft_target", v_soft_target, "replace reciprocal graph target with the earlier supervised soft target"),
    ],
    "whaleshark": [
        ("glncd_v2_full", v_full, "full reciprocal global-local neighborhood consistency"),
        ("glncd_v2_global_only", v_global_only, "remove local stripe neighborhood term"),
        ("glncd_v2_local_only", v_local_only, "remove global embedding neighborhood term"),
        ("glncd_v2_no_hard_negative", v_no_hard_negative, "remove hard-negative suppression"),
        ("glncd_v2_soft_target", v_soft_target, "replace reciprocal graph target with the earlier supervised soft target"),
    ],
    "gzgc_zebra": [
        ("glncd_v2_full", v_full, "appendix full GL-NCD v2 on existing cross-species zebra evidence"),
        ("glncd_v2_global_only", v_global_only, "appendix support ablation without stripe-local neighborhood"),
    ],
}


def materialize() -> List[Dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []
    for dataset, spec in DATASETS.items():
        base_cfg = _load_yaml(PROJECT_ROOT / str(spec["base"]))
        for variant, mutator, rationale in VARIANTS[dataset]:
            cfg = deepcopy(base_cfg)
            _set_common(cfg, dataset, variant, spec)
            mutator(cfg, spec)
            cfg["tmm_glncd_v2_20260513"] = {
                "dataset": dataset,
                "variant": variant,
                "priority": spec["priority"],
                "rationale": rationale,
                "algorithm_change": "GL-NCD v2 reciprocal/Jaccard global-local neighborhood distillation",
                "baseline_config": spec["base"],
                "selection_protocol": spec["protocol"],
                "best_metric": spec["best_metric"],
            }
            path = OUT_DIR / f"illumination_config_{dataset}_{variant}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
            summary.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "path": path.as_posix(),
                    "base": spec["base"],
                    "data_dir": spec["data_dir"],
                    "query_dir": spec.get("query", ""),
                    "gallery_dir": spec.get("gallery", ""),
                    "protocol": spec["protocol"],
                    "best_metric": spec["best_metric"],
                    "output_dir": _get(cfg, ("output_dir",)),
                    "neighborhood_consistency": _get(cfg, ("training", "neighborhood_consistency")),
                    "rationale": rationale,
                }
            )
    summary_path = OUT_DIR / "glncd_config_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    print(json.dumps(materialize(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
