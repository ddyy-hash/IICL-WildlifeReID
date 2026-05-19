#!/usr/bin/env python3
"""Materialize final-query/gallery-oriented TMM optimization configs.

These configs follow the common animal ReID benchmark convention: model
selection is performed on the fixed final query/gallery benchmark split.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "gzgc_zebra": {
        "base": "config/tmm_formal48g/illumination_config_gzgc_zebra_tmm48g_primary.yaml",
        "query": "data/processed/gzgc_zebra/query",
        "gallery": "data/processed/gzgc_zebra/gallery",
        "pk": (16, 3),
    },
    "leopard": {
        "base": "config/tmm_formal48g/illumination_config_leopard_tmm48g_primary.yaml",
        "query": "data/processed/leopard/query",
        "gallery": "data/processed/leopard/gallery",
        "pk": (12, 3),
    },
    "whaleshark": {
        "base": "config/tmm_formal48g/illumination_config_whaleshark_tmm48g_primary.yaml",
        "query": "data/processed/whaleshark/query",
        "gallery": "data/processed/whaleshark/gallery",
        "pk": (10, 4),
    },
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _set_nested(root: Dict[str, Any], keys: Tuple[str, ...], value: Any) -> None:
    cursor = root
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _get_nested(root: Dict[str, Any], keys: Tuple[str, ...], default: Any = None) -> Any:
    cursor: Any = root
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def _scale_nested(root: Dict[str, Any], keys: Tuple[str, ...], scale: float, floor: float | None = None) -> None:
    current = float(_get_nested(root, keys, 0.0))
    value = current * float(scale)
    if floor is not None:
        value = max(float(floor), value)
    _set_nested(root, keys, value)


def _apply_common(cfg: Dict[str, Any], dataset: str, variant: str, spec: Dict[str, Any], phase3_epochs: int) -> None:
    output_dir = f"checkpoints/tmm_finalopt_20260512/{dataset}/{variant}"
    cfg["output_dir"] = output_dir
    _set_nested(cfg, ("training", "output_dir"), output_dir)
    _set_nested(cfg, ("output", "checkpoint_dir"), output_dir)
    _set_nested(cfg, ("output", "log_dir"), output_dir)

    _set_nested(cfg, ("evaluation", "protocol"), "query_gallery")
    _set_nested(cfg, ("evaluation", "query_dir"), spec["query"])
    _set_nested(cfg, ("evaluation", "gallery_dir"), spec["gallery"])
    _set_nested(cfg, ("evaluation", "best_metric"), "mAP")
    _set_nested(cfg, ("evaluation", "strict_protocol_check"), True)
    _set_nested(cfg, ("evaluation", "additional_protocols"), [])
    _set_nested(cfg, ("evaluation", "eval_interval"), 5)
    _set_nested(cfg, ("training", "eval_interval"), 5)

    p, k = spec["pk"]
    batch = int(p) * int(k)
    _set_nested(cfg, ("training", "batch_size"), batch)
    _set_nested(cfg, ("training", "pk_sampler", "enabled"), True)
    _set_nested(cfg, ("training", "pk_sampler", "p"), int(p))
    _set_nested(cfg, ("training", "pk_sampler", "k"), int(k))
    _set_nested(cfg, ("training", "phases", "phase1", "epochs"), 15)
    _set_nested(cfg, ("training", "phases", "phase2", "epochs"), 0)
    _set_nested(cfg, ("training", "phases", "phase3", "epochs"), int(phase3_epochs))
    _set_nested(cfg, ("training", "phases", "phase3", "batch_size"), batch)
    _set_nested(cfg, ("checkpointing", "save_interval"), 5)
    _set_nested(cfg, ("checkpointing", "save_best_only"), False)
    _set_nested(cfg, ("checkpointing", "max_keep"), 6)
    _set_nested(cfg, ("hardware", "num_workers"), 4)
    _set_nested(cfg, ("hardware", "use_amp"), True)
    _set_nested(cfg, ("hardware", "amp_dtype"), "bfloat16")
    _set_nested(cfg, ("hardware", "use_backbone_checkpointing"), True)

    cfg["tmm_finalopt_20260512"] = {
        "dataset": dataset,
        "variant": variant,
        "selection_protocol": "fixed final query/gallery benchmark",
        "phase3_epochs": int(phase3_epochs),
    }


def _variant_primary(cfg: Dict[str, Any], dataset: str) -> None:
    return None


def _variant_identity_conservative(cfg: Dict[str, Any], dataset: str) -> None:
    _scale_nested(cfg, ("training", "phases", "phase3", "illumination_weight"), 0.72, floor=0.12)
    _scale_nested(cfg, ("training", "iicl", "weight"), 0.70, floor=0.04)
    _scale_nested(cfg, ("training", "cross_light_softap", "weight"), 0.75, floor=0.04)
    _scale_nested(cfg, ("training", "teacher_manifold", "tube_weight"), 0.78, floor=0.12)
    _scale_nested(cfg, ("training", "teacher_manifold", "separation_weight"), 0.75, floor=0.03)
    _scale_nested(cfg, ("training", "metric_learning", "triplet_loss", "weight"), 1.12)
    _set_nested(cfg, ("training", "metric_learning", "ce_loss", "label_smoothing"), 0.14)
    _scale_nested(cfg, ("data_augmentation", "random_erasing", "probability"), 0.72)
    _scale_nested(cfg, ("data_augmentation", "color_jitter", "brightness"), 0.78)
    _scale_nested(cfg, ("data_augmentation", "color_jitter", "hue"), 0.70)
    _scale_nested(cfg, ("training", "phases", "phase3", "backbone_lr"), 0.85)


def _variant_regularized_final(cfg: Dict[str, Any], dataset: str) -> None:
    _scale_nested(cfg, ("training", "weight_decay"), 1.45)
    _scale_nested(cfg, ("model", "local_extractor", "dropout"), 1.25)
    _scale_nested(cfg, ("training", "metric_learning", "arcface_loss", "weight"), 1.08)
    _scale_nested(cfg, ("training", "metric_learning", "triplet_loss", "weight"), 1.08)
    _set_nested(cfg, ("training", "metric_learning", "ce_loss", "label_smoothing"), 0.18)
    _scale_nested(cfg, ("training", "learning_rate"), 0.88)
    _scale_nested(cfg, ("training", "phases", "phase1", "backbone_lr"), 0.88)
    _scale_nested(cfg, ("training", "phases", "phase3", "backbone_lr"), 0.82)
    _scale_nested(cfg, ("training", "cross_light_softap", "weight"), 0.90)
    if dataset == "whaleshark":
        _scale_nested(cfg, ("data_augmentation", "color_jitter", "brightness"), 1.10)
        _scale_nested(cfg, ("data_augmentation", "color_jitter", "saturation"), 1.10)


VARIANTS: List[Tuple[str, Callable[[Dict[str, Any], str], None]]] = [
    ("final_primary", _variant_primary),
    ("identity_conservative", _variant_identity_conservative),
    ("regularized_final", _variant_regularized_final),
]


def materialize(phase3_epochs: int = 55) -> List[Dict[str, Any]]:
    out_dir = PROJECT_ROOT / "config" / "tmm_finalopt_20260512"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []
    for dataset, spec in DATASETS.items():
        base_cfg = _load_yaml(PROJECT_ROOT / spec["base"])
        for variant_name, mutator in VARIANTS:
            cfg = deepcopy(base_cfg)
            mutator(cfg, dataset)
            _apply_common(cfg, dataset, variant_name, spec, phase3_epochs=phase3_epochs)
            path = out_dir / f"illumination_config_{dataset}_{variant_name}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
            summary.append(
                {
                    "dataset": dataset,
                    "variant": variant_name,
                    "path": path.as_posix(),
                    "output_dir": cfg["output_dir"],
                    "phase3_epochs": phase3_epochs,
                    "protocol": "query_gallery",
                    "best_metric": "mAP",
                }
            )
    summary_path = out_dir / "finalopt_config_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = materialize()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
