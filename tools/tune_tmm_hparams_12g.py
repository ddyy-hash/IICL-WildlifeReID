#!/usr/bin/env python3
"""Local 12GB hyperparameter tuning for the TMM cross-species configs.

The script runs short proxy trials on stratified subsets. It is intended for
local screening of unstable or memory-heavy choices before full 48GB training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import optuna
import torch
import yaml


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import load_config


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    base_config: Path
    train_dir: Path
    selection_query_dir: Path
    selection_gallery_dir: Path
    selection_info: Path
    query_dir: Path
    gallery_dir: Path
    default_proxy_ids: int
    default_proxy_max_images: int
    pk_options: Tuple[Tuple[int, int], ...]
    lr_range: Tuple[float, float]
    dropout_range: Tuple[float, float]
    jitter_brightness_range: Tuple[float, float]
    jitter_hue_range: Tuple[float, float]
    erasing_prob_range: Tuple[float, float]
    iicl_weight_range: Tuple[float, float]
    illum_weight_range: Tuple[float, float]


DATASETS: Dict[str, DatasetSpec] = {
    "gzgc_zebra": DatasetSpec(
        key="gzgc_zebra",
        base_config=PROJECT_ROOT / "config" / "illumination_config_gzgc_zebra_match12g.yaml",
        train_dir=PROJECT_ROOT / "data" / "processed" / "gzgc_zebra" / "train",
        selection_query_dir=PROJECT_ROOT / "data" / "processed" / "gzgc_zebra" / "selection_query",
        selection_gallery_dir=PROJECT_ROOT / "data" / "processed" / "gzgc_zebra" / "selection_gallery",
        selection_info=PROJECT_ROOT / "data" / "processed" / "gzgc_zebra" / "split_info.json",
        query_dir=PROJECT_ROOT / "data" / "processed" / "gzgc_zebra" / "query",
        gallery_dir=PROJECT_ROOT / "data" / "processed" / "gzgc_zebra" / "gallery",
        default_proxy_ids=96,
        default_proxy_max_images=4,
        pk_options=((8, 2), (10, 2), (12, 2), (8, 3)),
        lr_range=(1.8e-4, 4.2e-4),
        dropout_range=(0.05, 0.22),
        jitter_brightness_range=(0.04, 0.14),
        jitter_hue_range=(0.0, 0.02),
        erasing_prob_range=(0.25, 0.45),
        iicl_weight_range=(0.05, 0.18),
        illum_weight_range=(0.22, 0.38),
    ),
    "leopard": DatasetSpec(
        key="leopard",
        base_config=PROJECT_ROOT / "config" / "illumination_config_leopard_match12g.yaml",
        train_dir=PROJECT_ROOT / "data" / "processed" / "leopard" / "train",
        selection_query_dir=PROJECT_ROOT / "data" / "processed" / "leopard" / "selection_query",
        selection_gallery_dir=PROJECT_ROOT / "data" / "processed" / "leopard" / "selection_gallery",
        selection_info=PROJECT_ROOT / "data" / "processed" / "leopard" / "split_info.json",
        query_dir=PROJECT_ROOT / "data" / "processed" / "leopard" / "query",
        gallery_dir=PROJECT_ROOT / "data" / "processed" / "leopard" / "gallery",
        default_proxy_ids=72,
        default_proxy_max_images=6,
        pk_options=((6, 3), (8, 3), (6, 4), (10, 2)),
        lr_range=(1.6e-4, 3.8e-4),
        dropout_range=(0.08, 0.32),
        jitter_brightness_range=(0.10, 0.28),
        jitter_hue_range=(0.01, 0.06),
        erasing_prob_range=(0.32, 0.58),
        iicl_weight_range=(0.06, 0.22),
        illum_weight_range=(0.24, 0.42),
    ),
    "whaleshark": DatasetSpec(
        key="whaleshark",
        base_config=PROJECT_ROOT / "config" / "illumination_config_whaleshark_match12g.yaml",
        train_dir=PROJECT_ROOT / "data" / "processed" / "whaleshark" / "train",
        selection_query_dir=PROJECT_ROOT / "data" / "processed" / "whaleshark" / "selection_query",
        selection_gallery_dir=PROJECT_ROOT / "data" / "processed" / "whaleshark" / "selection_gallery",
        selection_info=PROJECT_ROOT / "data" / "processed" / "whaleshark" / "split_info.json",
        query_dir=PROJECT_ROOT / "data" / "processed" / "whaleshark" / "query",
        gallery_dir=PROJECT_ROOT / "data" / "processed" / "whaleshark" / "gallery",
        default_proxy_ids=72,
        default_proxy_max_images=6,
        pk_options=((5, 4), (4, 5), (6, 4), (8, 3), (10, 2)),
        lr_range=(1.6e-4, 3.8e-4),
        dropout_range=(0.05, 0.24),
        jitter_brightness_range=(0.12, 0.30),
        jitter_hue_range=(0.02, 0.07),
        erasing_prob_range=(0.25, 0.50),
        iicl_weight_range=(0.06, 0.22),
        illum_weight_range=(0.26, 0.45),
    ),
}


def _as_project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _image_paths(identity_dir: Path) -> List[Path]:
    return sorted(p for p in identity_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _identity_image_counts(train_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for identity_dir in sorted(train_dir.iterdir()):
        if identity_dir.is_dir():
            counts[identity_dir.name] = len(_image_paths(identity_dir))
    return counts


def _dataset_stats(train_dir: Path) -> Dict[str, Any]:
    counts = list(_identity_image_counts(train_dir).values())
    counts.sort()
    if not counts:
        return {"ids": 0, "images": 0}

    def quantile(q: float) -> float:
        if len(counts) == 1:
            return float(counts[0])
        pos = q * (len(counts) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(counts[lo])
        return float(counts[lo] * (hi - pos) + counts[hi] * (pos - lo))

    return {
        "ids": len(counts),
        "images": int(sum(counts)),
        "min": int(counts[0]),
        "q25": quantile(0.25),
        "median": quantile(0.5),
        "q75": quantile(0.75),
        "mean": float(sum(counts) / len(counts)),
        "max": int(counts[-1]),
        "ge2": int(sum(v >= 2 for v in counts)),
        "ge3": int(sum(v >= 3 for v in counts)),
        "ge4": int(sum(v >= 4 for v in counts)),
        "ge5": int(sum(v >= 5 for v in counts)),
    }


def _safe_replace_dir(path: Path) -> None:
    resolved = path.resolve()
    allowed = (PROJECT_ROOT / "checkpoints" / "tuning").resolve()
    if not str(resolved).lower().startswith(str(allowed).lower()):
        raise RuntimeError(f"Refusing to remove path outside checkpoints/tuning: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _copy_or_link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_proxy_train_dir(
    spec: DatasetSpec,
    output_root: Path,
    proxy_ids: Optional[int],
    proxy_max_images: Optional[int],
    seed: int,
    refresh: bool = False,
) -> Path:
    n_ids = int(proxy_ids or spec.default_proxy_ids)
    max_images = int(proxy_max_images or spec.default_proxy_max_images)
    proxy_root = output_root / "proxy_data" / f"{spec.key}_ids{n_ids}_imgs{max_images}_seed{seed}"
    proxy_train = proxy_root / "train"
    manifest_path = proxy_root / "manifest.json"

    expected = {
        "dataset": spec.key,
        "source_train_dir": str(spec.train_dir.resolve()),
        "proxy_ids": n_ids,
        "proxy_max_images": max_images,
        "seed": seed,
    }
    if proxy_train.exists() and manifest_path.exists() and not refresh:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        if all(manifest.get(k) == v for k, v in expected.items()):
            return proxy_train

    _safe_replace_dir(proxy_root)
    proxy_train.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    count_map = _identity_image_counts(spec.train_dir)
    identities = [identity for identity, count in count_map.items() if count >= 2]
    identities.sort(key=lambda item: (count_map[item], item))

    if len(identities) > n_ids:
        # Stratify by count rank so the proxy keeps both tail and high-sample IDs.
        bins: List[List[str]] = [[], [], [], []]
        for idx, identity in enumerate(identities):
            bin_idx = min(3, int(idx * 4 / len(identities)))
            bins[bin_idx].append(identity)
        selected: List[str] = []
        base_take = n_ids // len(bins)
        remainder = n_ids % len(bins)
        for bin_idx, bucket in enumerate(bins):
            take = base_take + (1 if bin_idx < remainder else 0)
            rng.shuffle(bucket)
            selected.extend(bucket[:take])
        if len(selected) < n_ids:
            remaining = [identity for identity in identities if identity not in set(selected)]
            rng.shuffle(remaining)
            selected.extend(remaining[: n_ids - len(selected)])
    else:
        selected = identities
    selected = sorted(selected[:n_ids])

    copied_images = 0
    per_id_counts: Dict[str, int] = {}
    for identity in selected:
        src_dir = spec.train_dir / identity
        images = _image_paths(src_dir)
        rng.shuffle(images)
        chosen = sorted(images[: max(2, min(max_images, len(images)))])
        dst_dir = proxy_train / identity
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in chosen:
            dst = dst_dir / src.name
            _copy_or_link(src, dst)
            copied_images += 1
        per_id_counts[identity] = len(chosen)

    manifest = {
        **expected,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selected_ids": selected,
        "per_id_counts": per_id_counts,
        "images": copied_images,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return proxy_train


def build_proxy_selection_dirs(
    spec: DatasetSpec,
    output_root: Path,
    selection_ids: Optional[int],
    seed: int,
    refresh: bool = False,
) -> Tuple[Path, Path, Path]:
    """Build fixed query/gallery subsets for low-fidelity model selection."""
    if not selection_ids or int(selection_ids) <= 0:
        return spec.selection_query_dir, spec.selection_gallery_dir, spec.selection_info

    n_ids = int(selection_ids)
    proxy_root = output_root / "proxy_selection" / f"{spec.key}_sel{n_ids}_seed{seed}"
    proxy_query = proxy_root / "selection_query"
    proxy_gallery = proxy_root / "selection_gallery"
    manifest_path = proxy_root / "manifest.json"
    proxy_info = proxy_root / "selection_subset_info.json"

    expected = {
        "dataset": spec.key,
        "source_selection_query_dir": str(spec.selection_query_dir.resolve()),
        "source_selection_gallery_dir": str(spec.selection_gallery_dir.resolve()),
        "selection_ids": n_ids,
        "seed": seed,
    }
    if proxy_query.exists() and proxy_gallery.exists() and manifest_path.exists() and not refresh:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        if all(manifest.get(k) == v for k, v in expected.items()):
            return proxy_query, proxy_gallery, proxy_info

    _safe_replace_dir(proxy_root)
    proxy_query.mkdir(parents=True, exist_ok=True)
    proxy_gallery.mkdir(parents=True, exist_ok=True)

    query_ids = {p.name for p in spec.selection_query_dir.iterdir() if p.is_dir()}
    gallery_ids = {p.name for p in spec.selection_gallery_dir.iterdir() if p.is_dir()}
    identities = sorted(query_ids & gallery_ids)
    rng = random.Random(seed)
    rng.shuffle(identities)
    selected = sorted(identities[: min(n_ids, len(identities))])

    copied = {"selection_query": 0, "selection_gallery": 0}
    for identity in selected:
        for src_root, dst_root, key in (
            (spec.selection_query_dir, proxy_query, "selection_query"),
            (spec.selection_gallery_dir, proxy_gallery, "selection_gallery"),
        ):
            src_dir = src_root / identity
            dst_dir = dst_root / identity
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src in _image_paths(src_dir):
                _copy_or_link(src, dst_dir / src.name)
                copied[key] += 1

    manifest = {
        **expected,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selected_ids": selected,
        "copied_images": copied,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    proxy_info.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return proxy_query, proxy_gallery, proxy_info


def _set_output_dir(config: Dict[str, Any], output_dir: Path) -> None:
    rel = _as_project_path(output_dir)
    config["output_dir"] = rel
    config.setdefault("training", {})["output_dir"] = rel
    config.setdefault("output", {})["checkpoint_dir"] = rel
    config.setdefault("output", {})["log_dir"] = rel


def _set_train_and_eval_paths(
    config: Dict[str, Any],
    spec: DatasetSpec,
    train_dir: Path,
    selection_query_dir: Optional[Path] = None,
    selection_gallery_dir: Optional[Path] = None,
    selection_info: Optional[Path] = None,
) -> None:
    sel_query = selection_query_dir or spec.selection_query_dir
    sel_gallery = selection_gallery_dir or spec.selection_gallery_dir
    sel_info = selection_info or spec.selection_info

    train_cfg = config.setdefault("training", {})
    train_cfg["data_dir"] = _as_project_path(train_dir)
    train_cfg["query_dir"] = _as_project_path(spec.query_dir)
    train_cfg["gallery_dir"] = _as_project_path(spec.gallery_dir)

    eval_cfg = config.setdefault("evaluation", {})
    eval_cfg["protocol"] = "self_defined_train_qg"
    eval_cfg["selection_query_dir"] = _as_project_path(sel_query)
    eval_cfg["selection_gallery_dir"] = _as_project_path(sel_gallery)
    eval_cfg["selection_info"] = _as_project_path(sel_info)
    eval_cfg["query_dir"] = _as_project_path(spec.query_dir)
    eval_cfg["gallery_dir"] = _as_project_path(spec.gallery_dir)
    eval_cfg["best_metric"] = "mAP"
    eval_cfg["strict_protocol_check"] = True
    eval_cfg["additional_protocols"] = []


def _apply_common_trial_limits(
    config: Dict[str, Any],
    trial_output_dir: Path,
    proxy_train_dir: Path,
    spec: DatasetSpec,
    trial_phase1_epochs: int,
    trial_phase3_epochs: int,
    trial_eval_interval: int,
    num_workers: int,
    selection_query_dir: Optional[Path] = None,
    selection_gallery_dir: Optional[Path] = None,
    selection_info: Optional[Path] = None,
) -> None:
    _set_output_dir(config, trial_output_dir)
    _set_train_and_eval_paths(
        config,
        spec,
        proxy_train_dir,
        selection_query_dir=selection_query_dir,
        selection_gallery_dir=selection_gallery_dir,
        selection_info=selection_info,
    )
    config.setdefault("logging", {})["tensorboard"] = False
    config.setdefault("logging", {}).setdefault("wandb", {})["enabled"] = False
    config.setdefault("checkpointing", {})["max_keep"] = 1
    config.setdefault("hardware", {})["num_workers"] = int(num_workers)
    config.setdefault("hardware", {})["use_amp"] = True
    config.setdefault("hardware", {})["amp_dtype"] = "bfloat16"
    config.setdefault("hardware", {})["use_backbone_checkpointing"] = True
    config.setdefault("hardware", {})["use_ddp"] = False

    train_cfg = config.setdefault("training", {})
    train_cfg["eval_interval"] = int(trial_eval_interval)
    phases = train_cfg.setdefault("phases", {})
    phases.setdefault("phase1", {})["epochs"] = int(trial_phase1_epochs)
    phases.setdefault("phase2", {})["epochs"] = 0
    phases.setdefault("phase3", {})["epochs"] = int(trial_phase3_epochs)
    phases.setdefault("phase3", {})["use_backbone_checkpointing"] = True
    # Proxy trials are very short; use the sampled auxiliary weights immediately.
    # The full configs keep their gradual ramp for formal training stability.
    phases.setdefault("phase3", {}).setdefault("aux_ramp", {})["enabled"] = False
    eval_cfg = config.setdefault("evaluation", {})
    eval_cfg["eval_interval"] = int(trial_eval_interval)


def _parse_pk_override(raw: Optional[str], spec: DatasetSpec) -> Tuple[Tuple[int, int], ...]:
    if not raw:
        return spec.pk_options
    values: List[Tuple[int, int]] = []
    for part in raw.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if "x" not in token:
            raise ValueError(f"Invalid PK token '{token}', expected e.g. 8x3")
        p_raw, k_raw = token.split("x", 1)
        values.append((int(p_raw), int(k_raw)))
    return tuple(values) or spec.pk_options


def _bounded_choices(values: Iterable[int], max_value: Optional[int]) -> List[int]:
    choices = sorted(set(int(v) for v in values))
    if max_value is not None:
        choices = [v for v in choices if v <= int(max_value)]
    if not choices:
        raise ValueError("No choices left after applying max bound")
    return choices


def _apply_trial_params(
    config: Dict[str, Any],
    spec: DatasetSpec,
    trial: optuna.Trial,
    pk_options: Optional[Tuple[Tuple[int, int], ...]] = None,
    force_num_grad_variants: Optional[int] = None,
    max_base_channels: Optional[int] = None,
    max_refine_iterations: Optional[int] = None,
) -> Dict[str, Any]:
    resolved_pk_options = pk_options or spec.pk_options
    pk_label = trial.suggest_categorical("pk", [f"{p}x{k}" for p, k in resolved_pk_options])
    p, k = (int(part) for part in pk_label.split("x"))
    batch_size = p * k

    lr = trial.suggest_float("learning_rate", spec.lr_range[0], spec.lr_range[1], log=True)
    phase3_lr_ratio = trial.suggest_float("phase3_lr_ratio", 0.45, 0.65)
    illum_lr = trial.suggest_float("illumination_lr", 3.0e-5, 8.0e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 5.0e-4, 2.5e-2, log=True)
    dropout = trial.suggest_float("dropout", spec.dropout_range[0], spec.dropout_range[1])
    label_smoothing = trial.suggest_float("label_smoothing", 0.06, 0.22)
    arcface_weight = trial.suggest_float("arcface_weight", 0.30, 0.75)
    arcface_margin = trial.suggest_float("arcface_margin", 0.22, 0.34)
    triplet_weight = trial.suggest_float("triplet_weight", 0.55, 1.15)
    iicl_weight = trial.suggest_float("iicl_weight", spec.iicl_weight_range[0], spec.iicl_weight_range[1])
    illumination_weight = trial.suggest_float(
        "phase3_illumination_weight",
        spec.illum_weight_range[0],
        spec.illum_weight_range[1],
    )
    softap_weight = trial.suggest_float("softap_weight", 0.06, 0.16)
    tube_weight = trial.suggest_float("teacher_tube_weight", 0.20, 0.34)
    separation_weight = trial.suggest_float("teacher_separation_weight", 0.05, 0.12)
    base_channels = trial.suggest_categorical(
        "base_channels",
        _bounded_choices([24, 32], max_base_channels),
    )
    refine_iterations = trial.suggest_categorical(
        "refine_iterations",
        _bounded_choices([1, 2], max_refine_iterations),
    )
    if force_num_grad_variants is None:
        num_grad_variants = trial.suggest_categorical("num_grad_variants", [0, 1])
    else:
        num_grad_variants = int(force_num_grad_variants)
    init_corrected_bias = trial.suggest_float("init_corrected_bias", 1.05, 1.70)
    max_residual_scale = trial.suggest_float("max_residual_scale", 0.25, 0.45)
    brightness = trial.suggest_float(
        "brightness",
        spec.jitter_brightness_range[0],
        spec.jitter_brightness_range[1],
    )
    contrast = trial.suggest_float("contrast", max(0.04, brightness * 0.70), min(0.35, brightness * 1.20))
    saturation = trial.suggest_float("saturation", 0.04, min(0.25, brightness * 0.90 + 0.04))
    hue = trial.suggest_float("hue", spec.jitter_hue_range[0], spec.jitter_hue_range[1])
    erasing_prob = trial.suggest_float(
        "erasing_prob",
        spec.erasing_prob_range[0],
        spec.erasing_prob_range[1],
    )

    train_cfg = config.setdefault("training", {})
    train_cfg["batch_size"] = batch_size
    train_cfg["learning_rate"] = float(lr)
    train_cfg["weight_decay"] = float(weight_decay)
    train_cfg.setdefault("pk_sampler", {}).update({"enabled": True, "p": p, "k": k})

    phases = train_cfg.setdefault("phases", {})
    phases.setdefault("phase1", {})["backbone_lr"] = float(lr)
    phases.setdefault("phase3", {})["batch_size"] = batch_size
    phases.setdefault("phase3", {})["backbone_lr"] = float(lr * phase3_lr_ratio)
    phases.setdefault("phase3", {})["illumination_lr"] = float(illum_lr)
    phases.setdefault("phase3", {})["illumination_weight"] = float(illumination_weight)
    aux_ramp = phases.setdefault("phase3", {}).setdefault("aux_ramp", {})
    aux_ramp["illumination_end"] = float(illumination_weight)
    aux_ramp["iicl_end"] = float(iicl_weight)
    aux_ramp["cross_light_end"] = float(softap_weight)

    metric_cfg = train_cfg.setdefault("metric_learning", {})
    metric_cfg.setdefault("ce_loss", {})["label_smoothing"] = float(label_smoothing)
    metric_cfg.setdefault("arcface_loss", {})["weight"] = float(arcface_weight)
    metric_cfg.setdefault("arcface_loss", {})["m"] = float(arcface_margin)
    metric_cfg.setdefault("triplet_loss", {})["weight"] = float(triplet_weight)

    iicl_cfg = train_cfg.setdefault("iicl", {})
    iicl_cfg["weight"] = float(iicl_weight)
    iicl_cfg["num_variants"] = 2
    iicl_cfg["num_grad_variants"] = int(num_grad_variants)

    train_cfg.setdefault("cross_light_softap", {})["weight"] = float(softap_weight)
    train_cfg.setdefault("teacher_manifold", {})["tube_weight"] = float(tube_weight)
    train_cfg.setdefault("teacher_manifold", {})["separation_weight"] = float(separation_weight)

    module_params = config.setdefault("illumination_module", {}).setdefault("module_params", {})
    module_params["base_channels"] = int(base_channels)
    module_params["refine_iterations"] = int(refine_iterations)
    module_params["num_grad_variants"] = int(num_grad_variants)

    model_cfg = config.setdefault("model", {})
    model_cfg.setdefault("local_extractor", {})["dropout"] = float(dropout)
    model_cfg.setdefault("feature_fusion", {})["init_corrected_bias"] = float(init_corrected_bias)
    model_cfg.setdefault("feature_fusion", {})["max_residual_scale"] = float(max_residual_scale)

    aug_cfg = config.setdefault("data_augmentation", {})
    aug_cfg.setdefault("color_jitter", {})["brightness"] = float(brightness)
    aug_cfg.setdefault("color_jitter", {})["contrast"] = float(contrast)
    aug_cfg.setdefault("color_jitter", {})["saturation"] = float(saturation)
    aug_cfg.setdefault("color_jitter", {})["hue"] = float(hue)
    aug_cfg.setdefault("random_erasing", {})["probability"] = float(erasing_prob)

    return {
        "p": p,
        "k": k,
        "batch_size": batch_size,
        "learning_rate": lr,
        "phase3_backbone_lr": lr * phase3_lr_ratio,
        "illumination_lr": illum_lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "label_smoothing": label_smoothing,
        "arcface_weight": arcface_weight,
        "arcface_margin": arcface_margin,
        "triplet_weight": triplet_weight,
        "iicl_weight": iicl_weight,
        "phase3_illumination_weight": illumination_weight,
        "softap_weight": softap_weight,
        "teacher_tube_weight": tube_weight,
        "teacher_separation_weight": separation_weight,
        "base_channels": base_channels,
        "refine_iterations": refine_iterations,
        "num_grad_variants": num_grad_variants,
    }


def materialize_trial_config(
    spec: DatasetSpec,
    trial: optuna.Trial,
    trial_output_dir: Path,
    proxy_train_dir: Path,
    trial_phase1_epochs: int,
    trial_phase3_epochs: int,
    trial_eval_interval: int,
    num_workers: int,
    selection_query_dir: Optional[Path] = None,
    selection_gallery_dir: Optional[Path] = None,
    selection_info: Optional[Path] = None,
    pk_options: Optional[Tuple[Tuple[int, int], ...]] = None,
    force_num_grad_variants: Optional[int] = None,
    max_base_channels: Optional[int] = None,
    max_refine_iterations: Optional[int] = None,
) -> Tuple[Path, Dict[str, Any]]:
    config = load_config(str(spec.base_config))
    params = _apply_trial_params(
        config,
        spec,
        trial,
        pk_options=pk_options,
        force_num_grad_variants=force_num_grad_variants,
        max_base_channels=max_base_channels,
        max_refine_iterations=max_refine_iterations,
    )
    _apply_common_trial_limits(
        config,
        trial_output_dir=trial_output_dir,
        proxy_train_dir=proxy_train_dir,
        spec=spec,
        trial_phase1_epochs=trial_phase1_epochs,
        trial_phase3_epochs=trial_phase3_epochs,
        trial_eval_interval=trial_eval_interval,
        num_workers=num_workers,
        selection_query_dir=selection_query_dir,
        selection_gallery_dir=selection_gallery_dir,
        selection_info=selection_info,
    )
    config.setdefault("tuning", {})["trial_params"] = params
    trial_output_dir.mkdir(parents=True, exist_ok=True)
    trial_config_path = trial_output_dir / "trial_config.yaml"
    trial_config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return trial_config_path, params


def _parse_metrics_from_log(log_text: str) -> Dict[str, float]:
    matches = re.findall(
        r"Evaluation Results .*?Rank-1:\s*([0-9.]+)%,\s*Rank-5:\s*([0-9.]+)%,\s*mAP:\s*([0-9.]+)%",
        log_text,
    )
    if not matches:
        return {}
    rank1, rank5, m_ap = matches[-1]
    return {"rank1": float(rank1), "rank5": float(rank5), "mAP": float(m_ap)}


def _extract_trial_metrics(trial_output_dir: Path) -> Dict[str, float]:
    candidates = [
        trial_output_dir / "joint_best.pth",
        trial_output_dir / "joint_best_reid_best.pth",
    ]
    for checkpoint_path in candidates:
        if not checkpoint_path.exists():
            continue
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        metrics = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
        eval_metrics = metrics.get("eval", {}) if isinstance(metrics, dict) else {}
        if isinstance(eval_metrics, dict) and eval_metrics:
            return {k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))}

    log_path = trial_output_dir / "joint_training.log"
    if log_path.exists():
        return _parse_metrics_from_log(log_path.read_text(encoding="utf-8", errors="ignore"))
    return {}


def _run_trial_command(
    trial_config_path: Path,
    proxy_train_dir: Path,
    trial_output_dir: Path,
    timeout_minutes: float,
    device: str,
    num_workers: int,
) -> Tuple[int, str, float]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "train_joint.py"),
        "--data_dir",
        _as_project_path(proxy_train_dir),
        "--output_dir",
        _as_project_path(trial_output_dir),
        "--config",
        _as_project_path(trial_config_path),
        "--device",
        device,
        "--num_workers",
        str(num_workers),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(60, int(timeout_minutes * 60)),
        )
        output = completed.stdout
        return_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + f"\n[TIMEOUT] exceeded {timeout_minutes} minutes\n"
        return_code = 124
    elapsed = time.time() - started
    (trial_output_dir / "subprocess.log").write_text(output, encoding="utf-8", errors="ignore")
    return return_code, output, elapsed


def _score_metrics(metrics: Dict[str, float]) -> float:
    if not metrics:
        return 0.0
    # Primary target is retrieval mAP; small rank1 tie-breaker stabilizes early proxy runs.
    return float(metrics.get("mAP", 0.0)) + 0.02 * float(metrics.get("rank1", 0.0))


def tune_dataset(
    spec: DatasetSpec,
    output_root: Path,
    trials: int,
    timeout_minutes: float,
    proxy_ids: Optional[int],
    proxy_max_images: Optional[int],
    selection_ids: Optional[int],
    seed: int,
    trial_phase1_epochs: int,
    trial_phase3_epochs: int,
    trial_eval_interval: int,
    device: str,
    num_workers: int,
    refresh_proxy: bool,
    dry_run: bool,
    pk_override: Optional[str],
    force_num_grad_variants: Optional[int],
    max_base_channels: Optional[int],
    max_refine_iterations: Optional[int],
) -> List[Dict[str, Any]]:
    dataset_root = output_root / spec.key
    dataset_root.mkdir(parents=True, exist_ok=True)
    proxy_train_dir = build_proxy_train_dir(
        spec,
        output_root=output_root,
        proxy_ids=proxy_ids,
        proxy_max_images=proxy_max_images,
        seed=seed,
        refresh=refresh_proxy,
    )
    selection_query_dir, selection_gallery_dir, selection_info = build_proxy_selection_dirs(
        spec,
        output_root=output_root,
        selection_ids=selection_ids,
        seed=seed,
        refresh=refresh_proxy,
    )
    resolved_pk_options = _parse_pk_override(pk_override, spec)

    storage_path = dataset_root / "optuna_study.db"
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0),
        study_name=f"{spec.key}_match12g",
        storage=f"sqlite:///{storage_path.as_posix()}",
        load_if_exists=True,
    )

    records: List[Dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        trial_dir = dataset_root / f"trial_{trial.number:03d}"
        trial_config_path, params = materialize_trial_config(
            spec,
            trial,
            trial_output_dir=trial_dir,
            proxy_train_dir=proxy_train_dir,
            trial_phase1_epochs=trial_phase1_epochs,
            trial_phase3_epochs=trial_phase3_epochs,
            trial_eval_interval=trial_eval_interval,
            num_workers=num_workers,
            selection_query_dir=selection_query_dir,
            selection_gallery_dir=selection_gallery_dir,
            selection_info=selection_info,
            pk_options=resolved_pk_options,
            force_num_grad_variants=force_num_grad_variants,
            max_base_channels=max_base_channels,
            max_refine_iterations=max_refine_iterations,
        )
        if dry_run:
            record = {
                "dataset": spec.key,
                "trial": trial.number,
                "status": "dry_run",
                "config": str(trial_config_path),
                "params": params,
            }
            records.append(record)
            (trial_dir / "trial_result.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return 0.0

        return_code, output, elapsed = _run_trial_command(
            trial_config_path=trial_config_path,
            proxy_train_dir=proxy_train_dir,
            trial_output_dir=trial_dir,
            timeout_minutes=timeout_minutes,
            device=device,
            num_workers=num_workers,
        )
        metrics = _extract_trial_metrics(trial_dir)
        score = _score_metrics(metrics)
        failed = return_code != 0
        oom = "out of memory" in output.lower() or "cuda error" in output.lower()
        if failed:
            score = 0.0
        trial.set_user_attr("return_code", return_code)
        trial.set_user_attr("elapsed_sec", elapsed)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("params_resolved", params)
        trial.set_user_attr("oom_or_cuda_error", oom)
        record = {
            "dataset": spec.key,
            "trial": trial.number,
            "status": "ok" if return_code == 0 else ("oom_or_cuda_error" if oom else "failed"),
            "score": score,
            "metrics": metrics,
            "elapsed_sec": elapsed,
            "return_code": return_code,
            "config": str(trial_config_path),
            "params": params,
        }
        records.append(record)
        (trial_dir / "trial_result.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return score

    study.optimize(objective, n_trials=trials, gc_after_trial=True, show_progress_bar=False)
    summary_path = dataset_root / "trial_summary.json"
    all_records: List[Dict[str, Any]] = []
    for t in study.trials:
        all_records.append(
            {
                "dataset": spec.key,
                "trial": t.number,
                "state": str(t.state),
                "value": t.value,
                "params": t.params,
                "metrics": t.user_attrs.get("metrics"),
                "return_code": t.user_attrs.get("return_code"),
                "elapsed_sec": t.user_attrs.get("elapsed_sec"),
                "oom_or_cuda_error": t.user_attrs.get("oom_or_cuda_error"),
            }
        )
    summary_path.write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv_summary(dataset_root / "trial_summary.csv", all_records)

    if not dry_run:
        try:
            best_trial = study.best_trial
        except ValueError:
            best_trial = None
        if best_trial is not None and best_trial.value is not None and best_trial.value > 0:
            materialize_best_config(
                spec,
                best_trial,
                dataset_root / "best_config_from_optuna.yaml",
                pk_options=resolved_pk_options,
                force_num_grad_variants=force_num_grad_variants,
                max_base_channels=max_base_channels,
                max_refine_iterations=max_refine_iterations,
            )

    return records


def write_csv_summary(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    rows = list(records)
    if not rows:
        return
    fieldnames = [
        "dataset",
        "trial",
        "state",
        "value",
        "mAP",
        "rank1",
        "return_code",
        "elapsed_sec",
        "oom_or_cuda_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in rows:
            metrics = record.get("metrics") or {}
            writer.writerow(
                {
                    "dataset": record.get("dataset"),
                    "trial": record.get("trial"),
                    "state": record.get("state"),
                    "value": record.get("value"),
                    "mAP": metrics.get("mAP"),
                    "rank1": metrics.get("rank1"),
                    "return_code": record.get("return_code"),
                    "elapsed_sec": record.get("elapsed_sec"),
                    "oom_or_cuda_error": record.get("oom_or_cuda_error"),
                }
            )


def _fixed_trial_from_params(params: Dict[str, Any]) -> optuna.trial.FixedTrial:
    return optuna.trial.FixedTrial(params)


def materialize_best_config(
    spec: DatasetSpec,
    best_trial: optuna.trial.FrozenTrial,
    output_path: Path,
    pk_options: Optional[Tuple[Tuple[int, int], ...]] = None,
    force_num_grad_variants: Optional[int] = None,
    max_base_channels: Optional[int] = None,
    max_refine_iterations: Optional[int] = None,
) -> None:
    config = load_config(str(spec.base_config))
    fixed_trial = _fixed_trial_from_params(dict(best_trial.params))
    _apply_trial_params(
        config,
        spec,
        fixed_trial,
        pk_options=pk_options,
        force_num_grad_variants=force_num_grad_variants,
        max_base_channels=max_base_channels,
        max_refine_iterations=max_refine_iterations,
    )
    _set_train_and_eval_paths(config, spec, spec.train_dir)
    _set_output_dir(config, PROJECT_ROOT / "checkpoints" / f"tmm_{spec.key}_match12g_optuna_best")
    config.setdefault("tuning", {})["source_best_trial"] = {
        "number": best_trial.number,
        "value": best_trial.value,
        "params": best_trial.params,
        "metrics": best_trial.user_attrs.get("metrics"),
    }
    output_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    config_target = PROJECT_ROOT / "config" / f"illumination_config_{spec.key}_match12g_optuna_best.yaml"
    config_target.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def parse_datasets(raw: str) -> List[DatasetSpec]:
    if raw.strip().lower() == "all":
        return [DATASETS[key] for key in ("gzgc_zebra", "leopard", "whaleshark")]
    keys = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [key for key in keys if key not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}; valid keys: {sorted(DATASETS)}")
    return [DATASETS[key] for key in keys]


def write_math_report(specs: List[DatasetSpec], output_root: Path) -> None:
    lines = [
        "# 12GB TMM Hyperparameter Search Rationale",
        "",
        "Local GPU target: 12GB class card. The configs keep input at 256x256, AMP bf16, backbone checkpointing, and 0-1 gradient-carrying illumination variants.",
        "The 48GB ATRW Route-B reference used batch 32 at 256x384. Pixel scaling gives 32 * (256*256)/(256*384) ~= 21.3 samples for the same activation footprint before GPU-memory scaling; using a 12GB card with desktop overhead makes 16-20 samples the safe local range.",
        "",
    ]
    for spec in specs:
        stats = _dataset_stats(spec.train_dir)
        pk_text = ", ".join(f"{p}x{k}={p*k}" for p, k in spec.pk_options)
        lines.extend(
            [
                f"## {spec.key}",
                f"- Train IDs/images: {stats['ids']} / {stats['images']}",
                f"- Per-ID images: min {stats['min']}, median {stats['median']:.1f}, q75 {stats['q75']:.1f}, max {stats['max']}",
                f"- Positive-support counts: ge2={stats['ge2']}, ge3={stats['ge3']}, ge4={stats['ge4']}, ge5={stats['ge5']}",
                f"- PK search: {pk_text}",
                f"- LR search: [{spec.lr_range[0]:.2e}, {spec.lr_range[1]:.2e}], log scale",
                "",
            ]
        )
    (output_root / "math_rationale.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune TMM match-aware configs on a 12GB local GPU")
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset keys or 'all'")
    parser.add_argument("--trials", type=int, default=3, help="Optuna trials per dataset")
    parser.add_argument("--timeout_minutes", type=float, default=45.0, help="Timeout per trial")
    parser.add_argument("--proxy_ids", type=int, default=None, help="Override number of proxy train identities")
    parser.add_argument("--proxy_max_images", type=int, default=None, help="Override max images per proxy identity")
    parser.add_argument(
        "--selection_ids",
        type=int,
        default=None,
        help="Use a fixed subset of selection identities for low-fidelity evaluation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trial_phase1_epochs", type=int, default=1)
    parser.add_argument("--trial_phase3_epochs", type=int, default=1)
    parser.add_argument("--trial_eval_interval", type=int, default=1)
    parser.add_argument("--output_root", default="checkpoints/tuning/tmm_match12g")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--refresh_proxy", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Generate trial configs but do not train")
    parser.add_argument("--pk_override", type=str, default=None, help="Comma-separated PK choices, e.g. 8x3,10x2")
    parser.add_argument("--force_num_grad_variants", type=int, default=None)
    parser.add_argument("--max_base_channels", type=int, default=None)
    parser.add_argument("--max_refine_iterations", type=int, default=None)
    args = parser.parse_args()

    specs = parse_datasets(args.datasets)
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_math_report(specs, output_root)

    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] datasets={[spec.key for spec in specs]}")
    print(f"[INFO] torch={torch.__version__}, cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"[INFO] gpu={props.name}, total_gb={props.total_memory / 1024**3:.2f}")

    all_records: List[Dict[str, Any]] = []
    for spec in specs:
        print(f"[INFO] tuning dataset={spec.key}")
        all_records.extend(
            tune_dataset(
                spec=spec,
                output_root=output_root,
                trials=args.trials,
                timeout_minutes=args.timeout_minutes,
                proxy_ids=args.proxy_ids,
                proxy_max_images=args.proxy_max_images,
                selection_ids=args.selection_ids,
                seed=args.seed,
                trial_phase1_epochs=args.trial_phase1_epochs,
                trial_phase3_epochs=args.trial_phase3_epochs,
                trial_eval_interval=args.trial_eval_interval,
                device=args.device,
                num_workers=args.num_workers,
                refresh_proxy=args.refresh_proxy,
                dry_run=args.dry_run,
                pk_override=args.pk_override,
                force_num_grad_variants=args.force_num_grad_variants,
                max_base_channels=args.max_base_channels,
                max_refine_iterations=args.max_refine_iterations,
            )
        )

    (output_root / "last_run_records.json").write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
