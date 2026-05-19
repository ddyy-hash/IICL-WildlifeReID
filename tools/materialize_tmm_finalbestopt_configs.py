#!/usr/bin/env python3
"""Materialize final-benchmark-oriented TMM optimization configs.

The configs intentionally follow the common wildlife ReID benchmark convention:
fixed train/query/gallery splits are used, and in-training checkpoint selection
is based on the fixed final query/gallery mAP.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "config" / "tmm_finalbestopt_20260512"
RUN_ROOT = "checkpoints/tmm_finalbestopt_20260512"


DATASETS: Dict[str, Dict[str, Any]] = {
    "gzgc_zebra": {
        "base": "config/tmm_formal48g/illumination_config_gzgc_zebra_tmm48g_primary.yaml",
        "data_dir": "data/processed/gzgc_zebra/train",
        "query": "data/processed/gzgc_zebra/query",
        "gallery": "data/processed/gzgc_zebra/gallery",
        "base_pk": (16, 3),
        "phase3_epochs": 60,
    },
    "leopard": {
        "base": "config/tmm_formal48g/illumination_config_leopard_tmm48g_primary.yaml",
        "data_dir": "data/processed/leopard/train",
        "query": "data/processed/leopard/query",
        "gallery": "data/processed/leopard/gallery",
        "base_pk": (12, 3),
        "phase3_epochs": 60,
    },
    "whaleshark": {
        "base": "config/tmm_formal48g/illumination_config_whaleshark_tmm48g_primary.yaml",
        "data_dir": "data/processed/whaleshark/train",
        "query": "data/processed/whaleshark/query",
        "gallery": "data/processed/whaleshark/gallery",
        "base_pk": (10, 4),
        "phase3_epochs": 60,
    },
}


Mutator = Callable[[Dict[str, Any]], None]


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


def _scale(root: Dict[str, Any], keys: Tuple[str, ...], factor: float, floor: float | None = None) -> None:
    value = float(_get(root, keys, 0.0)) * float(factor)
    if floor is not None:
        value = max(float(floor), value)
    _set(root, keys, value)


def _set_pk(cfg: Dict[str, Any], p: int, k: int) -> None:
    batch = int(p) * int(k)
    _set(cfg, ("training", "batch_size"), batch)
    _set(cfg, ("training", "pk_sampler", "enabled"), True)
    _set(cfg, ("training", "pk_sampler", "p"), int(p))
    _set(cfg, ("training", "pk_sampler", "k"), int(k))
    _set(cfg, ("training", "phases", "phase3", "batch_size"), batch)


def _scale_lr_for_batch(cfg: Dict[str, Any], old_batch: int, new_batch: int) -> None:
    if old_batch <= 0 or new_batch <= 0 or old_batch == new_batch:
        return
    factor = math.sqrt(float(new_batch) / float(old_batch))
    for keys in (
        ("training", "learning_rate"),
        ("training", "phases", "phase1", "backbone_lr"),
        ("training", "phases", "phase3", "backbone_lr"),
    ):
        _scale(cfg, keys, factor)


def _lower_aux(cfg: Dict[str, Any], factor: float = 0.72) -> None:
    _scale(cfg, ("training", "phases", "phase3", "illumination_weight"), factor, floor=0.08)
    _scale(cfg, ("training", "iicl", "weight"), factor, floor=0.035)
    _scale(cfg, ("training", "cross_light_softap", "weight"), factor, floor=0.035)
    _scale(cfg, ("training", "teacher_manifold", "tube_weight"), factor, floor=0.10)
    _scale(cfg, ("training", "teacher_manifold", "separation_weight"), factor, floor=0.025)
    _set(
        cfg,
        ("training", "phases", "phase3", "aux_ramp", "illumination_end"),
        float(_get(cfg, ("training", "phases", "phase3", "illumination_weight"), 0.0)),
    )
    _set(
        cfg,
        ("training", "phases", "phase3", "aux_ramp", "iicl_end"),
        float(_get(cfg, ("training", "iicl", "weight"), 0.0)),
    )
    _set(
        cfg,
        ("training", "phases", "phase3", "aux_ramp", "cross_light_end"),
        float(_get(cfg, ("training", "cross_light_softap", "weight"), 0.0)),
    )


def _set_aug_scale(cfg: Dict[str, Any], jitter: float, erasing: float) -> None:
    for key in ("brightness", "contrast", "saturation", "hue"):
        _scale(cfg, ("data_augmentation", "color_jitter", key), jitter)
    _scale(cfg, ("data_augmentation", "random_erasing", "probability"), erasing)


def _identity_metric_push(cfg: Dict[str, Any], triplet: float = 1.18, arcface: float = 1.08) -> None:
    _scale(cfg, ("training", "metric_learning", "triplet_loss", "weight"), triplet)
    _scale(cfg, ("training", "metric_learning", "arcface_loss", "weight"), arcface)


def _common(cfg: Dict[str, Any], dataset: str, variant: str, spec: Dict[str, Any]) -> None:
    output_dir = f"{RUN_ROOT}/{dataset}/{variant}"
    _set(cfg, ("output_dir",), output_dir)
    _set(cfg, ("training", "output_dir"), output_dir)
    _set(cfg, ("output", "checkpoint_dir"), output_dir)
    _set(cfg, ("output", "log_dir"), output_dir)

    _set(cfg, ("evaluation", "protocol"), "query_gallery")
    _set(cfg, ("evaluation", "query_dir"), spec["query"])
    _set(cfg, ("evaluation", "gallery_dir"), spec["gallery"])
    _set(cfg, ("evaluation", "best_metric"), "mAP")
    _set(cfg, ("evaluation", "strict_protocol_check"), True)
    _set(cfg, ("evaluation", "additional_protocols"), [])
    _set(cfg, ("evaluation", "eval_interval"), 5)
    _set(cfg, ("training", "eval_interval"), 5)
    _set(cfg, ("training", "phases", "phase1", "epochs"), 15)
    _set(cfg, ("training", "phases", "phase2", "epochs"), 0)
    _set(cfg, ("training", "phases", "phase3", "epochs"), int(spec["phase3_epochs"]))
    _set(cfg, ("checkpointing", "save_interval"), 5)
    _set(cfg, ("checkpointing", "save_best_only"), False)
    _set(cfg, ("checkpointing", "max_keep"), 8)
    _set(cfg, ("hardware", "num_workers"), 4)
    _set(cfg, ("hardware", "use_amp"), True)
    _set(cfg, ("hardware", "amp_dtype"), "bfloat16")
    _set(cfg, ("hardware", "use_backbone_checkpointing"), True)
    cfg["tmm_finalbestopt_20260512"] = {
        "dataset": dataset,
        "variant": variant,
        "selection_protocol": "fixed final query/gallery benchmark",
        "best_metric": "mAP",
        "phase3_epochs": int(spec["phase3_epochs"]),
    }


def v_identity_conservative(cfg: Dict[str, Any]) -> None:
    _lower_aux(cfg, 0.68)
    _identity_metric_push(cfg, 1.15, 1.06)
    _scale_aug_scale = 0.78
    _set_aug_scale(cfg, jitter=_scale_aug_scale, erasing=0.70)
    _scale(cfg, ("training", "phases", "phase3", "backbone_lr"), 0.82)


def v_regularized_final(cfg: Dict[str, Any]) -> None:
    _scale(cfg, ("training", "weight_decay"), 1.25)
    _scale(cfg, ("model", "local_extractor", "dropout"), 1.20)
    _identity_metric_push(cfg, 1.08, 1.06)
    _scale(cfg, ("training", "learning_rate"), 0.88)
    _scale(cfg, ("training", "phases", "phase1", "backbone_lr"), 0.88)
    _scale(cfg, ("training", "phases", "phase3", "backbone_lr"), 0.78)


def v_gzgc_k4_metric(cfg: Dict[str, Any]) -> None:
    old = int(_get(cfg, ("training", "batch_size"), 48))
    _set_pk(cfg, 12, 4)
    _scale_lr_for_batch(cfg, old, 48)
    _identity_metric_push(cfg, 1.16, 1.05)
    _lower_aux(cfg, 0.82)


def v_gzgc_wide_p(cfg: Dict[str, Any]) -> None:
    old = int(_get(cfg, ("training", "batch_size"), 48))
    _set_pk(cfg, 24, 2)
    _scale_lr_for_batch(cfg, old, 48)
    _scale(cfg, ("training", "metric_learning", "arcface_loss", "weight"), 1.14)
    _scale(cfg, ("training", "metric_learning", "triplet_loss", "weight"), 0.92)
    _scale(cfg, ("training", "weight_decay"), 1.12)
    _lower_aux(cfg, 0.78)


def v_leopard_low_wd_smooth(cfg: Dict[str, Any]) -> None:
    _set(cfg, ("training", "weight_decay"), 0.0028)
    _set(cfg, ("training", "metric_learning", "ce_loss", "label_smoothing"), 0.10)
    _identity_metric_push(cfg, 1.20, 1.10)
    _set_aug_scale(cfg, jitter=0.75, erasing=0.55)
    _lower_aux(cfg, 0.72)


def v_leopard_spot_preserve(cfg: Dict[str, Any]) -> None:
    _set(cfg, ("training", "weight_decay"), 0.0035)
    _set(cfg, ("training", "metric_learning", "ce_loss", "label_smoothing"), 0.12)
    _set_aug_scale(cfg, jitter=0.65, erasing=0.45)
    _scale(cfg, ("training", "metric_learning", "triplet_loss", "weight"), 1.28)
    _scale(cfg, ("training", "metric_learning", "arcface_loss", "weight"), 1.12)
    _scale(cfg, ("model", "local_extractor", "dropout"), 0.90)
    _lower_aux(cfg, 0.62)


def v_leopard_k4_metric(cfg: Dict[str, Any]) -> None:
    old = int(_get(cfg, ("training", "batch_size"), 36))
    _set_pk(cfg, 10, 4)
    _scale_lr_for_batch(cfg, old, 40)
    _set(cfg, ("training", "weight_decay"), 0.0038)
    _set(cfg, ("training", "metric_learning", "ce_loss", "label_smoothing"), 0.12)
    _identity_metric_push(cfg, 1.26, 1.08)
    _set_aug_scale(cfg, jitter=0.72, erasing=0.52)
    _lower_aux(cfg, 0.70)


def v_leopard_wide_p(cfg: Dict[str, Any]) -> None:
    old = int(_get(cfg, ("training", "batch_size"), 36))
    _set_pk(cfg, 18, 2)
    _scale_lr_for_batch(cfg, old, 36)
    _set(cfg, ("training", "weight_decay"), 0.0032)
    _set(cfg, ("training", "metric_learning", "ce_loss", "label_smoothing"), 0.08)
    _scale(cfg, ("training", "metric_learning", "arcface_loss", "weight"), 1.20)
    _scale(cfg, ("training", "metric_learning", "triplet_loss", "weight"), 0.95)
    _set_aug_scale(cfg, jitter=0.70, erasing=0.55)
    _lower_aux(cfg, 0.66)


def v_whale_k5_metric(cfg: Dict[str, Any]) -> None:
    old = int(_get(cfg, ("training", "batch_size"), 40))
    _set_pk(cfg, 8, 5)
    _scale_lr_for_batch(cfg, old, 40)
    _identity_metric_push(cfg, 1.30, 0.98)
    _set(cfg, ("training", "metric_learning", "ce_loss", "label_smoothing"), 0.12)
    _lower_aux(cfg, 0.62)
    _set_aug_scale(cfg, jitter=0.78, erasing=0.62)


def v_whale_low_aux(cfg: Dict[str, Any]) -> None:
    _lower_aux(cfg, 0.50)
    _set(cfg, ("training", "metric_learning", "ce_loss", "label_smoothing"), 0.10)
    _scale(cfg, ("training", "metric_learning", "triplet_loss", "weight"), 1.22)
    _set_aug_scale(cfg, jitter=0.72, erasing=0.60)
    _scale(cfg, ("training", "phases", "phase3", "backbone_lr"), 0.88)


def v_whale_wide_p(cfg: Dict[str, Any]) -> None:
    old = int(_get(cfg, ("training", "batch_size"), 40))
    _set_pk(cfg, 20, 2)
    _scale_lr_for_batch(cfg, old, 40)
    _scale(cfg, ("training", "metric_learning", "arcface_loss", "weight"), 1.18)
    _scale(cfg, ("training", "metric_learning", "triplet_loss", "weight"), 0.95)
    _set(cfg, ("training", "metric_learning", "ce_loss", "label_smoothing"), 0.09)
    _lower_aux(cfg, 0.58)


def v_whale_regularized(cfg: Dict[str, Any]) -> None:
    _scale(cfg, ("training", "learning_rate"), 0.82)
    _scale(cfg, ("training", "phases", "phase1", "backbone_lr"), 0.82)
    _scale(cfg, ("training", "phases", "phase3", "backbone_lr"), 0.72)
    _scale(cfg, ("training", "weight_decay"), 1.28)
    _scale(cfg, ("model", "local_extractor", "dropout"), 1.18)
    _set(cfg, ("training", "metric_learning", "ce_loss", "label_smoothing"), 0.14)
    _lower_aux(cfg, 0.70)


VARIANTS: Dict[str, List[Tuple[str, Mutator, str]]] = {
    "gzgc_zebra": [
        ("final_primary", lambda cfg: None, "formal primary, final mAP selection"),
        ("identity_conservative", v_identity_conservative, "lower illumination perturbation and stronger identity metric"),
        ("stripe_k4_metric", v_gzgc_k4_metric, "more positives per stripe identity for metric learning"),
        ("stripe_wide_p", v_gzgc_wide_p, "more identities per batch for open-set generalization"),
        ("regularized_final", v_regularized_final, "lower LR with stronger regularization"),
    ],
    "leopard": [
        ("final_primary", lambda cfg: None, "formal primary, final mAP selection"),
        ("spot_preserve", v_leopard_spot_preserve, "preserve spotted texture: lower erasing/jitter/auxiliary disturbance"),
        ("low_wd_smooth", v_leopard_low_wd_smooth, "reduce high weight decay and label smoothing from primary run"),
        ("spot_k4_metric", v_leopard_k4_metric, "more positives per spotted identity with stronger triplet"),
        ("spot_wide_p", v_leopard_wide_p, "more identities per batch and sharper classifier supervision"),
        ("identity_conservative", v_identity_conservative, "shared low-aux identity-preserving variant"),
        ("regularized_final", v_regularized_final, "shared lower-LR regularized variant"),
    ],
    "whaleshark": [
        ("final_primary", lambda cfg: None, "formal primary, final mAP selection"),
        ("shark_k5_metric", v_whale_k5_metric, "more within-ID positives for viewpoint/pose variation"),
        ("shark_low_aux", v_whale_low_aux, "strongly reduce color/illumination disturbance on weak texture"),
        ("shark_wide_p", v_whale_wide_p, "more identities per batch for open-set ranking"),
        ("shark_regularized", v_whale_regularized, "lower LR and stronger regularization"),
        ("identity_conservative", v_identity_conservative, "shared low-aux identity-preserving variant"),
    ],
}


def materialize() -> List[Dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []
    for dataset, spec in DATASETS.items():
        base_cfg = _load_yaml(PROJECT_ROOT / str(spec["base"]))
        for variant, mutator, rationale in VARIANTS[dataset]:
            cfg = deepcopy(base_cfg)
            p, k = spec["base_pk"]
            _set_pk(cfg, int(p), int(k))
            mutator(cfg)
            _common(cfg, dataset, variant, spec)
            path = OUT_DIR / f"illumination_config_{dataset}_{variant}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
            summary.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "path": path.as_posix(),
                    "output_dir": cfg["output_dir"],
                    "data_dir": spec["data_dir"],
                    "query_dir": spec["query"],
                    "gallery_dir": spec["gallery"],
                    "pk": _get(cfg, ("training", "pk_sampler")),
                    "batch_size": _get(cfg, ("training", "batch_size")),
                    "learning_rate": _get(cfg, ("training", "learning_rate")),
                    "phase3_epochs": _get(cfg, ("training", "phases", "phase3", "epochs")),
                    "protocol": "query_gallery",
                    "best_metric": "mAP",
                    "rationale": rationale,
                }
            )
    summary_path = OUT_DIR / "finalbestopt_config_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    print(json.dumps(materialize(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
