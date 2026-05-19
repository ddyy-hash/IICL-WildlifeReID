#!/usr/bin/env python3
"""Configuration loader for joint training/evaluation."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "illumination_module": {
        "loss_params": {
            "lambda_recon": 1.0,
            "lambda_smooth": 0.15,
            "lambda_edge": 0.08,
            "lambda_structure": 0.08,
            "lambda_sensitivity": 0.02,
            "lambda_identity": 0.1,
            "lambda_lab_chroma": 0.1,
            "lambda_high_freq": 0.05,
            "lambda_log_chroma": 0.0,
            "chroma_mode": "dual",
        },
        "module_params": {
            "base_channels": 32,
            "num_scales": 3,
            "refine_iterations": 2,
            "use_sensitivity": True,
            "use_refinement": True,
            "color_illumination_mode": "local_rgb",
            "clamp_input_range": False,
            "wb_max_shift": 0.12,
            "enable_task_aware_rollback": True,
            "rollback_hidden_dim": 64,
            "rollback_min_alpha": 0.05,
            "rollback_max_alpha": 0.98,
            "rollback_granularity": "global",
            "rollback_num_stripes": 6,
            "use_model_aware_residual": False,
            "model_residual_hidden_dim": 64,
            "model_residual_scale": 0.15,
            "enable_coarse_task_grad": True,
            "coarse_guidance_mode": "safe",
            "num_grad_variants": 1,
        },
    },
    "model": {
        "backbone": "osnet_ain_x1_0",
        "local_extractor": {
            "num_parts": 6,
            "dropout": 0.0,
        },
        "feature_fusion": {
            "enabled": True,
            "hidden_dim": 128,
            "init_corrected_bias": 2.0,
            "include_illum_stats": True,
            "max_residual_scale": 0.5,
        },
        "branch_attention_fusion": {
            "enabled": False,
            "hidden_dim": 128,
            "num_branches": 3,
            "temperature": 1.0,
            "include_illum_stats": True,
        },
        "nuisance_head": {
            "enabled": False,
            "hidden_dim": 128,
            "nuisance_dim": 64,
            "photometric_dim": 4,
        },
        "illumination_module": {
            "enabled": True,
            "module_params": {},
        },
    },
    "training": {
        "output_dir": "./checkpoints/joint",
        "batch_size": 32,
        "learning_rate": 3.5e-4,
        "image_height": 256,
        "image_width": 256,
        "weight_decay": 5e-4,
        "gradient_clip": 1.0,
        "eval_interval": 5,
        "pk_sampler": {
            "enabled": False,
            "p": 8,
            "k": 2,
        },
        "center_loss": {
            "enabled": True,
            "weight": 0.0005,
            "feat_dim": 256,
            "lr_scale": 0.5,
        },
        "metric_learning": {
            "ce_loss": {"weight": 1.0, "label_smoothing": 0.1},
            "arcface_loss": {"weight": 1.0, "s": 30.0, "m": 0.35},
            "triplet_loss": {"weight": 0.0, "margin": 0.3, "mining_type": "soft"},
            "circle_loss": {"weight": 0.0, "margin": 0.25, "gamma": 256},
        },
        "iicl": {
            "enabled": True,
            "weight": 0.5,
            "num_variants": 2,
            "num_grad_variants": 1,
            "temperature": 0.1,
            "loss_type": "cosine",
        },
        "photo_prior": {
            "initial_weight": 1.0,
            "min_weight": 0.35,
            "decay_power": 1.0,
        },
        "cross_light_prototype": {
            "enabled": True,
            "weight": 0.12,
            "similarity": "cosine",
            "photometric_scale": 8.0,
            "photometric_offset": 0.1,
            "min_gap_weight": 0.1,
        },
        "cross_light_margin_preserving": {
            "enabled": False,
            "weight": 0.15,
            "similarity": "cosine",
            "photometric_scale": 8.0,
            "photometric_offset": 0.1,
            "topk_positive": 2,
            "topk_negative": 4,
            "margin_delta": 0.02,
            "beta": 12.0,
        },
        "cross_light_softap": {
            "enabled": False,
            "weight": 0.18,
            "similarity": "cosine",
            "photometric_scale": 8.0,
            "photometric_offset": 0.1,
            "min_positive_weight": 0.05,
            "rank_temperature": 0.07,
            "queue_size": 192,
        },
        "teacher_manifold": {
            "enabled": False,
            "tube_weight": 0.24,
            "separation_weight": 0.08,
            "similarity": "cosine",
            "photometric_scale": 8.0,
            "photometric_offset": 0.1,
            "min_positive_weight": 0.05,
            "shrinkage": 0.8,
            "orthogonal_weight": 1.0,
            "subspace_rank": 1,
            "min_radius": 0.02,
            "margin": 0.08,
            "queue_size": 192,
        },
        "ranking_topology": {
            "enabled": False,
            "weight": 0.18,
            "similarity": "cosine",
            "photometric_scale": 8.0,
            "photometric_offset": 0.1,
            "min_positive_weight": 0.05,
            "topk_positive": 2,
            "topk_negative": 4,
            "margin_slack": 0.01,
            "beta": 12.0,
            "queue_size": 192,
        },
        "anisotropic_identity_protection": {
            "enabled": False,
            "weight": 0.10,
            "similarity": "cosine",
            "photometric_scale": 8.0,
            "photometric_offset": 0.1,
            "min_positive_weight": 0.05,
            "topk_positive": 2,
            "topk_negative": 4,
            "subspace_rank": 1,
            "identity_weight": 1.0,
            "nuisance_weight": 0.5,
            "nuisance_radius": 0.12,
        },
        "semantic_non_confusion": {
            "enabled": True,
            "weight": 0.06,
            "margin_delta": 0.02,
            "squared": True,
        },
        "nuisance_decoupling": {
            "enabled": True,
            "weight": 0.04,
            "regression_weight": 1.0,
            "decorrelation_weight": 0.5,
        },
        "teacher_prototype_anchor": {
            "enabled": False,
            "weight": 0.0,
            "metric": "cosine",
        },
        "relative_class_structure": {
            "enabled": True,
            "weight": 0.08,
            "metric": "cosine",
            "radial_weight": 0.5,
        },
        "feature_trust_region": {
            "enabled": True,
            "weight": 0.05,
            "base_radius": 0.12,
            "adaptive_scale": 0.5,
            "class_spread_scale": 0.0,
        },
        "local_rank_preserving": {
            "enabled": True,
            "weight": 0.1,
            "alpha": 0.9,
            "k_positive": 2,
            "k_negative": 4,
        },
        "neighborhood_consistency": {
            "enabled": False,
            "weight": 0.08,
            "temperature": 0.07,
            "topk": 6,
            "positive_weight": 1.0,
            "negative_weight": 0.25,
            "local_weight": 0.35,
            "use_global": True,
            "use_local": True,
            "use_hard_negatives": True,
            "teacher_target": "soft",
        },
        "identity_image_preserving": {
            "enabled": True,
            "weight": 0.05,
        },
        "aux_gradient_gate": {
            "enabled": True,
            "eps": 1e-8,
        },
        "identity_preserving": {
            "mode": "geometry",
            "phase2_scale": 1.0,
            "phase3_scale": 0.35,
            "anchor_weight": 1.0,
            "geometry_weight": 0.5,
            "logit_weight": 0.15,
            "detach_reference": True,
            "teacher_temperature": 2.0,
            "similarity": "cosine",
            "geometry_loss": "mse",
        },
        "phases": {
            "phase1": {
                "epochs": 10,
                "illumination_weight": 1.0,
                "reid_weight": 0.3,
                "illumination_lr": 1e-4,
                "warmup_epochs": 5,
            },
            "phase2": {
                "epochs": 100,
                "illumination_weight": 1.0,
                "reid_weight": 0.0,
                "illumination_lr": 5e-5,
                "backbone_lr": 3.5e-4,
            },
            "phase3": {},
        },
        "early_stopping": {
            "enabled": True,
            "patience": 30,
            "monitor": "mAP",
            "min_delta": 0.001,
        },
    },
    "data_augmentation": {
        "image_size": 256,
        "random_horizontal_flip": 0.5,
        "color_jitter": {
            "brightness": 0.2,
            "contrast": 0.15,
            "saturation": 0.15,
            "hue": 0.03,
        },
        "random_crop": {
            "enabled": True,
            "scale": [0.85, 1.0],
        },
        "random_erasing": {
            "enabled": True,
            "probability": 0.5,
            "scale": [0.02, 0.25],
            "ratio": [0.3, 3.3],
        },
    },
    "yolo": {
        "model_path": "fea_data/yolov8m-seg.pt",
        "conf_threshold": 0.3,
    },
    "evaluation": {
        "eval_interval": 5,
        "flip_test": True,
        "rerank": False,
        "rerank_params": {
            "k1": 25,
            "k2": 6,
            "lambda_value": 0.2,
        },
        "protocol": "val_split_70_30",
        "additional_protocols": [],
        "best_metric": "rank1",
        "strict_protocol_check": True,
        "atrw": {
            "data_root": None,
            "test_dir": None,
            "eval_script_dir": "ATRWEvalScript-main",
            "batch_size": 64,
            "num_workers": 4,
            "train_ratio": 0.7,
            "seed": 42,
        },
    },
    "checkpointing": {
        "save_interval": 10,
        "max_keep": 5,
    },
    "hardware": {
        "num_workers": 4,
        "use_amp": True,
        "amp_dtype": "float16",
        "use_backbone_checkpointing": True,
        "use_ddp": True,
        "ddp_backend": "nccl",
        "ddp_find_unused_parameters": False,
        "ddp_timeout_minutes": 30,
    },
}


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries and return merged object."""
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def normalize_legacy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize legacy config schema into current structure."""
    training = config.setdefault("training", {})
    phases = training.setdefault("phases", {})
    iicl_cfg = training.setdefault("iicl", {})
    illum_cfg = config.setdefault("illumination_module", {})
    module_params = illum_cfg.setdefault("module_params", {})

    # Legacy phase epoch keys.
    if "phase1_epochs" in training:
        phase1 = phases.setdefault("phase1", {})
        phase1["epochs"] = training["phase1_epochs"]
    if "phase2_epochs" in training:
        phase2 = phases.setdefault("phase2", {})
        phase2["epochs"] = training["phase2_epochs"]

    # Legacy losses section -> metric/iicl/center sections.
    losses = training.get("losses", {})
    if isinstance(losses, dict):
        metric_learning = training.setdefault("metric_learning", {})
        loss_key_map = {
            "cross_entropy": "ce_loss",
            "ce_loss": "ce_loss",
            "arcface": "arcface_loss",
            "arcface_loss": "arcface_loss",
            "triplet": "triplet_loss",
            "triplet_loss": "triplet_loss",
            "circle": "circle_loss",
            "circle_loss": "circle_loss",
        }
        for old_key, new_key in loss_key_map.items():
            if old_key in losses:
                if isinstance(losses[old_key], dict):
                    metric_target = metric_learning.setdefault(new_key, {})
                    deep_merge(metric_target, losses[old_key])
                else:
                    metric_learning[new_key] = losses[old_key]

        if "iicl" in losses:
            iicl_target = training.setdefault("iicl", {})
            if isinstance(losses["iicl"], dict):
                deep_merge(iicl_target, losses["iicl"])
            else:
                training["iicl"] = losses["iicl"]

        if "center_loss" in losses:
            center_target = training.setdefault("center_loss", {})
            if isinstance(losses["center_loss"], dict):
                deep_merge(center_target, losses["center_loss"])
            else:
                training["center_loss"] = losses["center_loss"]
        elif "center" in losses:
            center_target = training.setdefault("center_loss", {})
            if isinstance(losses["center"], dict):
                deep_merge(center_target, losses["center"])
            else:
                training["center_loss"] = losses["center"]

        # Legacy key alias.
        triplet_cfg = metric_learning.get("triplet_loss", {})
        if isinstance(triplet_cfg, dict) and "mining" in triplet_cfg and "mining_type" not in triplet_cfg:
            triplet_cfg["mining_type"] = triplet_cfg["mining"]

    if isinstance(iicl_cfg, dict):
        if "variants" in iicl_cfg and "num_variants" not in iicl_cfg:
            iicl_cfg["num_variants"] = iicl_cfg["variants"]
        if "grad_variants" in iicl_cfg and "num_grad_variants" not in iicl_cfg:
            iicl_cfg["num_grad_variants"] = iicl_cfg["grad_variants"]

    if isinstance(module_params, dict):
        if "num_grad_variants" not in module_params and isinstance(iicl_cfg, dict):
            module_params["num_grad_variants"] = iicl_cfg.get("num_grad_variants", 1)

    # Legacy augmentation block under training.
    if isinstance(training.get("augmentation"), dict):
        config.setdefault("data_augmentation", {})
        legacy_aug = training["augmentation"]
        aug_target = config["data_augmentation"]

        # Compatibility mapping for old key names.
        if "random_flip" in legacy_aug and "random_horizontal_flip" not in legacy_aug:
            legacy_aug["random_horizontal_flip"] = 0.5 if legacy_aug["random_flip"] else 0.0
        if isinstance(legacy_aug.get("random_crop"), bool):
            legacy_aug["random_crop"] = {"enabled": legacy_aug["random_crop"]}
        if isinstance(legacy_aug.get("color_jitter"), dict):
            color_jitter = legacy_aug["color_jitter"]
            if color_jitter.get("enabled") is False:
                color_jitter.setdefault("brightness", 0.0)
                color_jitter.setdefault("contrast", 0.0)
                color_jitter.setdefault("saturation", 0.0)
                color_jitter.setdefault("hue", 0.0)

        deep_merge(aug_target, legacy_aug)

    # Legacy worker key under training.
    if "num_workers" in training:
        config.setdefault("hardware", {})
        config["hardware"]["num_workers"] = training["num_workers"]

    evaluation = config.setdefault("evaluation", {})
    additional_protocols = evaluation.get("additional_protocols")
    if isinstance(additional_protocols, str):
        evaluation["additional_protocols"] = [additional_protocols]
    elif additional_protocols is None:
        evaluation["additional_protocols"] = []

    feature_extraction = evaluation.get("feature_extraction")
    if isinstance(feature_extraction, dict) and "flip_test" in feature_extraction:
        evaluation["flip_test"] = bool(feature_extraction["flip_test"])

    # evaluation.eval_freq -> training.eval_interval
    if "eval_freq" in evaluation:
        training["eval_interval"] = evaluation["eval_freq"]
    elif "eval_interval" in evaluation:
        training["eval_interval"] = evaluation["eval_interval"]

    # Legacy reranking schema.
    reranking = evaluation.get("reranking")
    if isinstance(reranking, dict):
        evaluation["rerank"] = bool(reranking.get("enabled", False))
        rerank_params = evaluation.setdefault("rerank_params", {})
        for key in ("k1", "k2", "lambda_value"):
            if key in reranking:
                rerank_params[key] = reranking[key]

    return config


def _load_yaml_file(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def cli_args_to_config(args: Any) -> Dict[str, Any]:
    """Convert argparse Namespace into nested config overrides."""
    cfg: Dict[str, Any] = {}

    def set_if_not_none(path: str, value: Any) -> None:
        if value is None:
            return
        cur = cfg
        parts = path.split(".")
        for key in parts[:-1]:
            cur = cur.setdefault(key, {})
        cur[parts[-1]] = value

    set_if_not_none("output_dir", getattr(args, "output_dir", None))
    set_if_not_none("training.output_dir", getattr(args, "output_dir", None))
    set_if_not_none("model.backbone", getattr(args, "backbone", None))
    set_if_not_none("training.batch_size", getattr(args, "batch_size", None))
    set_if_not_none("training.phases.phase1.epochs", getattr(args, "phase1_epochs", None))
    set_if_not_none("training.phases.phase2.epochs", getattr(args, "phase2_epochs", None))
    set_if_not_none("training.optimizer.lr", getattr(args, "learning_rate", None))
    set_if_not_none("training.learning_rate", getattr(args, "learning_rate", None))
    set_if_not_none("model.local_extractor.num_parts", getattr(args, "num_stripes", None))
    set_if_not_none("training.eval_interval", getattr(args, "eval_interval", None))
    set_if_not_none("training.pk_sampler.p", getattr(args, "p_size", None))
    set_if_not_none("training.pk_sampler.k", getattr(args, "k_size", None))
    set_if_not_none("training.metric_learning.circle_loss.gamma", getattr(args, "circle_gamma", None))
    set_if_not_none("training.image_height", getattr(args, "img_height", None))
    set_if_not_none("training.image_width", getattr(args, "img_width", None))
    set_if_not_none("hardware.num_workers", getattr(args, "num_workers", None))
    set_if_not_none("evaluation.protocol", getattr(args, "eval_protocol", None))
    set_if_not_none("evaluation.best_metric", getattr(args, "best_metric", None))
    set_if_not_none("evaluation.strict_protocol_check", getattr(args, "strict_protocol_check", None))

    use_iicl = getattr(args, "use_iicl", None)
    if use_iicl is not None:
        set_if_not_none("training.iicl.enabled", bool(use_iicl))

    set_if_not_none("training.iicl.weight", getattr(args, "iicl_weight", None))
    set_if_not_none("training.iicl.num_variants", getattr(args, "iicl_variants", None))

    return cfg


def load_config(config_path: Optional[str], cli_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Three-layer merge: defaults <- YAML <- CLI overrides."""
    merged = copy.deepcopy(DEFAULT_CONFIG)
    deep_merge(merged, _load_yaml_file(config_path))
    normalize_legacy_config(merged)
    if cli_overrides:
        deep_merge(merged, cli_overrides)
    normalize_legacy_config(merged)
    return merged
