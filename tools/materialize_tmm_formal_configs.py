#!/usr/bin/env python3
"""Materialize 48GB formal configs from validated local TMM trials."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


SELECTED = {
    "gzgc_zebra": {
        "base": "config/illumination_config_gzgc_zebra_match12g.yaml",
        "trial": (
            "checkpoints/tuning/tmm_multifidelity_zebra_leopard_phase1_20260511_122703/"
            "gzgc_zebra/trial_005/trial_result.json"
        ),
        "formal_p": 16,
        "formal_k": 3,
        "lr_cap": 4.5e-4,
    },
    "leopard": {
        "base": "config/illumination_config_leopard_match12g.yaml",
        "trial": (
            "checkpoints/tuning/tmm_multifidelity_zebra_leopard_phase1_20260511_122703/"
            "leopard/trial_007/trial_result.json"
        ),
        "formal_p": 12,
        "formal_k": 3,
        "lr_cap": 4.2e-4,
    },
    "whaleshark": {
        "base": "config/illumination_config_whaleshark_match12g.yaml",
        "trial": (
            "checkpoints/tuning/tmm_whaleshark_fullsel_phase1_20260511_132656/"
            "whaleshark/trial_001/trial_result.json"
        ),
        "formal_p": 10,
        "formal_k": 4,
        "lr_cap": 3.2e-4,
    },
}


def _set_nested(root: Dict[str, Any], *keys: str, value: Any) -> None:
    cursor = root
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _load_trial(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("params"), dict):
        raise ValueError(f"Expected trial_result with params in {path}")
    return data


def materialize_one(dataset: str, spec: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    cfg = _load_yaml(PROJECT_ROOT / str(spec["base"]))
    trial = _load_trial(PROJECT_ROOT / str(spec["trial"]))
    params = trial["params"]

    local_batch = int(params["batch_size"])
    formal_p = int(spec["formal_p"])
    formal_k = int(spec["formal_k"])
    formal_batch = formal_p * formal_k
    lr_scale = math.sqrt(formal_batch / local_batch)
    formal_lr = min(float(params["learning_rate"]) * lr_scale, float(spec["lr_cap"]))
    phase3_ratio = float(params["phase3_backbone_lr"]) / float(params["learning_rate"])
    phase3_lr = formal_lr * phase3_ratio

    cfg = deepcopy(cfg)
    output_dir = f"checkpoints/tmm_formal48g/{dataset}_primary"

    _set_nested(cfg, "training", "output_dir", value=output_dir)
    _set_nested(cfg, "training", "batch_size", value=formal_batch)
    _set_nested(cfg, "training", "learning_rate", value=formal_lr)
    _set_nested(cfg, "training", "weight_decay", value=float(params["weight_decay"]))
    _set_nested(cfg, "training", "eval_interval", value=5)
    _set_nested(cfg, "training", "pk_sampler", "enabled", value=True)
    _set_nested(cfg, "training", "pk_sampler", "p", value=formal_p)
    _set_nested(cfg, "training", "pk_sampler", "k", value=formal_k)

    _set_nested(cfg, "training", "metric_learning", "ce_loss", "label_smoothing", value=float(params["label_smoothing"]))
    _set_nested(cfg, "training", "metric_learning", "arcface_loss", "weight", value=float(params["arcface_weight"]))
    _set_nested(cfg, "training", "metric_learning", "arcface_loss", "m", value=float(params["arcface_margin"]))
    _set_nested(cfg, "training", "metric_learning", "triplet_loss", "weight", value=float(params["triplet_weight"]))
    _set_nested(cfg, "training", "iicl", "weight", value=float(params["iicl_weight"]))
    _set_nested(cfg, "training", "iicl", "num_variants", value=2)
    _set_nested(cfg, "training", "iicl", "num_grad_variants", value=int(params["num_grad_variants"]))
    _set_nested(cfg, "training", "cross_light_softap", "weight", value=float(params["softap_weight"]))
    _set_nested(cfg, "training", "teacher_manifold", "tube_weight", value=float(params["teacher_tube_weight"]))
    _set_nested(
        cfg,
        "training",
        "teacher_manifold",
        "separation_weight",
        value=float(params["teacher_separation_weight"]),
    )

    _set_nested(cfg, "training", "phases", "phase1", "epochs", value=15)
    _set_nested(cfg, "training", "phases", "phase1", "backbone_lr", value=formal_lr)
    _set_nested(cfg, "training", "phases", "phase2", "epochs", value=0)
    _set_nested(cfg, "training", "phases", "phase3", "epochs", value=66)
    _set_nested(cfg, "training", "phases", "phase3", "batch_size", value=formal_batch)
    _set_nested(cfg, "training", "phases", "phase3", "backbone_lr", value=phase3_lr)
    _set_nested(cfg, "training", "phases", "phase3", "illumination_lr", value=float(params["illumination_lr"]))
    _set_nested(
        cfg,
        "training",
        "phases",
        "phase3",
        "illumination_weight",
        value=float(params["phase3_illumination_weight"]),
    )
    _set_nested(cfg, "training", "phases", "phase3", "aux_ramp", "enabled", value=True)
    _set_nested(cfg, "training", "phases", "phase3", "aux_ramp", "epochs", value=8)
    _set_nested(
        cfg,
        "training",
        "phases",
        "phase3",
        "aux_ramp",
        "illumination_end",
        value=float(params["phase3_illumination_weight"]),
    )
    _set_nested(
        cfg,
        "training",
        "phases",
        "phase3",
        "aux_ramp",
        "iicl_end",
        value=float(params["iicl_weight"]),
    )
    _set_nested(
        cfg,
        "training",
        "phases",
        "phase3",
        "aux_ramp",
        "cross_light_end",
        value=float(params["softap_weight"]),
    )

    _set_nested(cfg, "model", "local_extractor", "dropout", value=float(params["dropout"]))
    _set_nested(cfg, "illumination_module", "module_params", "base_channels", value=int(params["base_channels"]))
    _set_nested(cfg, "illumination_module", "module_params", "refine_iterations", value=int(params["refine_iterations"]))
    _set_nested(
        cfg,
        "illumination_module",
        "module_params",
        "num_grad_variants",
        value=int(params["num_grad_variants"]),
    )

    # Keep species-specific augmentation from the selected trial.
    trial_cfg = _load_yaml(PROJECT_ROOT / str(trial["config"]))
    for section in ("data_augmentation",):
        if section in trial_cfg:
            cfg[section] = deepcopy(trial_cfg[section])
    for section, key in (
        ("model", "feature_fusion"),
        ("model", "branch_attention_fusion"),
    ):
        if section in trial_cfg and key in trial_cfg[section]:
            cfg.setdefault(section, {})[key] = deepcopy(trial_cfg[section][key])

    _set_nested(cfg, "evaluation", "eval_interval", value=5)
    _set_nested(cfg, "evaluation", "protocol", value="self_defined_train_qg")
    _set_nested(cfg, "evaluation", "best_metric", value="mAP")
    _set_nested(cfg, "checkpointing", "max_keep", value=3)
    _set_nested(cfg, "hardware", "num_workers", value=4)
    _set_nested(cfg, "hardware", "use_amp", value=True)
    _set_nested(cfg, "hardware", "amp_dtype", value="bfloat16")
    _set_nested(cfg, "hardware", "use_backbone_checkpointing", value=True)
    cfg["output_dir"] = output_dir

    cfg.setdefault("tmm_formal48g_source", {})
    cfg["tmm_formal48g_source"] = {
        "selected_trial_result": str(spec["trial"]),
        "local_selection_metrics": trial.get("metrics", {}),
        "local_proxy_score": trial.get("score"),
        "local_batch": local_batch,
        "formal_batch": formal_batch,
        "lr_scaling": "sqrt(formal_batch/local_batch) with dataset cap",
        "lr_scale": lr_scale,
        "formal_learning_rate": formal_lr,
        "formal_phase3_backbone_lr": phase3_lr,
    }

    out_path = out_dir / f"illumination_config_{dataset}_tmm48g_primary.yaml"
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=False)
    return {
        "dataset": dataset,
        "path": out_path.as_posix(),
        "local_trial": str(spec["trial"]),
        "local_metrics": trial.get("metrics", {}),
        "formal_p": formal_p,
        "formal_k": formal_k,
        "formal_batch": formal_batch,
        "formal_learning_rate": formal_lr,
        "formal_phase3_backbone_lr": phase3_lr,
    }


def main() -> None:
    out_dir = PROJECT_ROOT / "config" / "tmm_formal48g"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = [materialize_one(dataset, spec, out_dir) for dataset, spec in SELECTED.items()]
    summary_path = out_dir / "formal48g_config_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
