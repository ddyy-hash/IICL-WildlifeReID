#!/usr/bin/env python3
"""Run the paper-oriented ATRW main ablation with official open-set evaluation."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import load_config


DEFAULT_VARIANT_ORDER = (
    "plain_baseline",
    "naive_illumination",
    "illumination_only",
    "full_model",
)

DEFAULT_BASELINE_HEAD = "plain_global"
SUPPORTED_BASELINE_HEADS = (
    "plain_global",
    "local_stripe",
)

_BACKBONE_MID_DIM_FALLBACK = {
    "osnet_x1_0": 384,
    "osnet_x0_75": 288,
    "osnet_x0_5": 192,
    "osnet_x0_25": 96,
    "osnet_ain_x1_0": 384,
    "osnet_ain_x0_75": 288,
    "osnet_ain_x0_5": 192,
    "osnet_ain_x0_25": 96,
    "osnet_ibn_x1_0": 384,
    "resnet50": 512,
    "resnet50_fc512": 512,
    "resnet101": 512,
    "resnet152": 512,
    "resnet50_ibn_a": 512,
    "resnet50_ibn_b": 512,
    "resnet101_ibn_a": 512,
    "densenet121": 512,
    "mobilenetv2_x1_0": 32,
    "mobilenetv2_x1_4": 48,
    "shufflenet": 240,
    "squeezenet1_0": 256,
    "squeezenet1_1": 256,
}

ZERO_ILLUMINATION_LOSS_KEYS = (
    "lambda_recon",
    "lambda_smooth",
    "lambda_edge",
    "lambda_structure",
    "lambda_sensitivity",
    "lambda_identity",
    "lambda_lab_chroma",
    "lambda_high_freq",
    "lambda_log_chroma",
)

ROUTE_B_DISABLE_KEYS = (
    "cross_light_prototype",
    "cross_light_margin_preserving",
    "cross_light_softap",
    "teacher_manifold",
    "ranking_topology",
    "anisotropic_identity_protection",
    "semantic_non_confusion",
    "nuisance_decoupling",
    "teacher_prototype_anchor",
    "relative_class_structure",
    "feature_trust_region",
    "local_rank_preserving",
    "identity_image_preserving",
)


@dataclass
class AblationJob:
    variant_key: str
    display_name: str
    train_entrypoint: str
    eval_mode: str
    config: Dict[str, Any]
    output_dir: Path
    config_path: Path
    train_log_path: Path
    eval_log_path: Path
    result_path: Path
    train_command: List[str] = field(default_factory=list)
    checkpoint_candidates: List[str] = field(default_factory=list)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _phase_cfg(config: Dict[str, Any], phase_name: str) -> Dict[str, Any]:
    training_cfg = _as_dict(config.get("training"))
    phases_cfg = _as_dict(training_cfg.get("phases"))
    return _as_dict(phases_cfg.get(phase_name))


def _sum_phase_epochs(config: Dict[str, Any]) -> int:
    return sum(int(_phase_cfg(config, name).get("epochs", 0)) for name in ("phase1", "phase2", "phase3"))


def _set_output_dir(config: Dict[str, Any], output_dir: Path) -> None:
    output_text = output_dir.as_posix()
    config["output_dir"] = output_text
    config.setdefault("training", {})["output_dir"] = output_text


def _collapse_to_single_reid_stage(config: Dict[str, Any], total_epochs: Optional[int] = None) -> None:
    training_cfg = config.setdefault("training", {})
    hardware_cfg = config.setdefault("hardware", {})
    baseline_total_epochs = int(total_epochs) if total_epochs is not None else _sum_phase_epochs(config)
    learning_rate = float(_as_dict(training_cfg.get("optimizer")).get("lr", training_cfg.get("learning_rate", 3.5e-4)))
    source_phase1_cfg = _phase_cfg(config, "phase1")
    training_cfg["phases"] = {
        "phase1": {
            "name": "baseline_reid",
            "epochs": baseline_total_epochs,
            "freeze_backbone": False,
            "freeze_illumination": True,
            "illumination_weight": 0.0,
            "reid_weight": 1.0,
            "backbone_lr": float(source_phase1_cfg.get("backbone_lr", learning_rate)),
            "use_backbone_checkpointing": bool(
                source_phase1_cfg.get(
                    "use_backbone_checkpointing",
                    hardware_cfg.get("use_backbone_checkpointing", True),
                )
            ),
        },
        "phase2": {
            "name": "baseline_phase2_disabled",
            "epochs": 0,
            "freeze_backbone": True,
            "freeze_illumination": True,
            "illumination_weight": 0.0,
            "reid_weight": 0.0,
        },
        "phase3": {
            "name": "baseline_phase3_disabled",
            "epochs": 0,
            "freeze_backbone": False,
            "freeze_illumination": True,
            "illumination_weight": 0.0,
            "reid_weight": 1.0,
            "backbone_lr": float(source_phase1_cfg.get("backbone_lr", learning_rate)),
        },
    }


def _disable_illumination_branch(config: Dict[str, Any]) -> None:
    model_cfg = config.setdefault("model", {})
    illum_top_cfg = config.setdefault("illumination_module", {})
    model_illum_cfg = model_cfg.setdefault("illumination_module", {})
    illum_loss_cfg = illum_top_cfg.setdefault("loss_params", {})
    illum_module_params = illum_top_cfg.setdefault("module_params", {})
    illum_model_params = model_illum_cfg.setdefault("module_params", {})

    illum_top_cfg["module_type"] = "disabled"
    model_illum_cfg["enabled"] = False
    for loss_key in ZERO_ILLUMINATION_LOSS_KEYS:
        illum_loss_cfg[loss_key] = 0.0

    for params_cfg in (illum_module_params, illum_model_params):
        params_cfg["use_sensitivity"] = False
        params_cfg["use_refinement"] = False
        params_cfg["use_feature_guided"] = False
        params_cfg["use_color_illumination"] = False
        params_cfg["enable_task_aware_rollback"] = False
        params_cfg["enable_coarse_task_grad"] = False
        params_cfg["safe_color_enabled"] = False
        params_cfg["clamp_input_range"] = False
        params_cfg["num_grad_variants"] = 0
        params_cfg["refine_iterations"] = 0


def _disable_method_heads(config: Dict[str, Any]) -> None:
    model_cfg = config.setdefault("model", {})
    model_cfg.setdefault("feature_fusion", {})["enabled"] = False
    model_cfg.setdefault("branch_attention_fusion", {})["enabled"] = False
    model_cfg.setdefault("nuisance_head", {})["enabled"] = False


def _update_illumination_module_params(config: Dict[str, Any], **updates: Any) -> None:
    illum_top_cfg = config.setdefault("illumination_module", {})
    model_cfg = config.setdefault("model", {})
    model_illum_cfg = model_cfg.setdefault("illumination_module", {})

    for params_cfg in (
        illum_top_cfg.setdefault("module_params", {}),
        model_illum_cfg.setdefault("module_params", {}),
    ):
        params_cfg.update(updates)


def _disable_route_b_losses(config: Dict[str, Any]) -> None:
    training_cfg = config.setdefault("training", {})
    training_cfg.setdefault("iicl", {}).update({
        "enabled": False,
        "weight": 0.0,
        "num_variants": 0,
        "num_grad_variants": 0,
    })
    training_cfg.setdefault("aux_gradient_gate", {})["enabled"] = False

    identity_cfg = training_cfg.setdefault("identity_preserving", {})
    identity_cfg["phase2_scale"] = 0.0
    identity_cfg["phase3_scale"] = 0.0
    identity_cfg["anchor_weight"] = 0.0
    identity_cfg["geometry_weight"] = 0.0
    identity_cfg["logit_weight"] = 0.0

    for key in ROUTE_B_DISABLE_KEYS:
        section = training_cfg.setdefault(key, {})
        section["enabled"] = False
        if "weight" in section:
            section["weight"] = 0.0
        if key == "teacher_manifold":
            section["tube_weight"] = 0.0
            section["separation_weight"] = 0.0


def _disable_reid_recipe_boosters_for_plain_baseline(config: Dict[str, Any]) -> None:
    training_cfg = config.setdefault("training", {})
    metric_cfg = training_cfg.setdefault("metric_learning", {})

    training_cfg.setdefault("pk_sampler", {})["enabled"] = False

    center_cfg = training_cfg.setdefault("center_loss", {})
    center_cfg["enabled"] = False
    center_cfg["weight"] = 0.0

    ce_cfg = metric_cfg.setdefault("ce_loss", {})
    ce_cfg["weight"] = 1.0
    ce_cfg["label_smoothing"] = 0.0

    for loss_name in ("triplet_loss", "arcface_loss", "circle_loss"):
        loss_cfg = metric_cfg.setdefault(loss_name, {})
        loss_cfg["enabled"] = False
        loss_cfg["weight"] = 0.0


def _disable_overlap_augmentations_for_plain_baseline(config: Dict[str, Any]) -> None:
    aug_cfg = config.setdefault("data_augmentation", {})
    random_erasing_cfg = aug_cfg.setdefault("random_erasing", {})
    random_erasing_cfg["enabled"] = False

    color_jitter_cfg = aug_cfg.setdefault("color_jitter", {})
    color_jitter_cfg["enabled"] = False
    color_jitter_cfg["brightness"] = 0.0
    color_jitter_cfg["contrast"] = 0.0
    color_jitter_cfg["saturation"] = 0.0
    color_jitter_cfg["hue"] = 0.0


def _set_flip_test(config: Dict[str, Any], enabled: bool) -> None:
    eval_cfg = config.setdefault("evaluation", {})
    eval_cfg["flip_test"] = bool(enabled)
    eval_cfg.setdefault("feature_extraction", {})["flip_test"] = bool(enabled)


def _mark_baseline_metadata(config: Dict[str, Any], baseline_type: str, disabled_components: Iterable[str]) -> None:
    model_cfg = _as_dict(config.get("model"))
    eval_cfg = _as_dict(config.get("evaluation"))
    config["baseline"] = {
        "enabled": True,
        "type": baseline_type,
        "source_backbone": model_cfg.get("backbone"),
        "baseline_backbone": model_cfg.get("backbone"),
        "source_total_epochs": _sum_phase_epochs(config),
        "baseline_total_epochs": _sum_phase_epochs(config),
        "disabled_components": list(disabled_components),
        "evaluation_protocol": eval_cfg.get("protocol"),
        "best_metric": eval_cfg.get("best_metric"),
    }


def _resolve_baseline_head(baseline_head: Optional[str]) -> str:
    resolved = str(baseline_head or DEFAULT_BASELINE_HEAD).strip().lower()
    if resolved not in SUPPORTED_BASELINE_HEADS:
        raise ValueError(
            f"Unsupported baseline head: {resolved}. "
            f"Expected one of {SUPPORTED_BASELINE_HEADS}."
        )
    return resolved


def _resolve_backbone_mid_dim(backbone_name: str) -> int:
    try:
        from app.core.joint_model import get_backbone_mid_dim as runtime_get_backbone_mid_dim
    except ModuleNotFoundError:
        runtime_get_backbone_mid_dim = None

    if runtime_get_backbone_mid_dim is not None:
        return int(runtime_get_backbone_mid_dim(backbone_name))

    if backbone_name in _BACKBONE_MID_DIM_FALLBACK:
        return int(_BACKBONE_MID_DIM_FALLBACK[backbone_name])
    if "resnet" in backbone_name.lower():
        return 512
    return 384


def _apply_backbone_override(config: Dict[str, Any], backbone_override: Optional[str]) -> None:
    if not backbone_override:
        return

    model_cfg = config.setdefault("model", {})
    model_cfg["backbone"] = backbone_override

    backbone_mid_channels = _resolve_backbone_mid_dim(backbone_override)
    illum_top_cfg = config.setdefault("illumination_module", {})
    illum_model_cfg = model_cfg.setdefault("illumination_module", {})

    illum_top_cfg.setdefault("module_params", {})["backbone_mid_channels"] = backbone_mid_channels
    illum_model_cfg.setdefault("module_params", {})["backbone_mid_channels"] = backbone_mid_channels


def derive_plain_baseline_config(
    full_config: Dict[str, Any],
    backbone_override: Optional[str] = None,
    total_epochs: Optional[int] = None,
    baseline_head: Optional[str] = None,
) -> Dict[str, Any]:
    derived = copy.deepcopy(_as_dict(full_config))
    model_cfg = derived.setdefault("model", {})
    training_cfg = derived.setdefault("training", {})
    baseline_head = _resolve_baseline_head(baseline_head)

    _apply_backbone_override(derived, backbone_override)

    reid_head_cfg = model_cfg.setdefault("reid_head", {})
    local_extractor_cfg = model_cfg.setdefault("local_extractor", {})
    if baseline_head == "plain_global":
        reid_head_cfg["type"] = "plain_global"
        local_extractor_cfg["num_parts"] = 1
        local_extractor_cfg["dropout"] = 0.0
        baseline_type = "plain_global_reid"
    else:
        reid_head_cfg["type"] = "local_stripe"
        baseline_type = "stripe_head_ce_reid"

    _disable_illumination_branch(derived)
    _disable_method_heads(derived)
    _disable_route_b_losses(derived)
    _disable_reid_recipe_boosters_for_plain_baseline(derived)
    _disable_overlap_augmentations_for_plain_baseline(derived)
    _set_flip_test(derived, enabled=False)

    training_cfg.setdefault("photo_prior", {}).update({
        "initial_weight": 0.0,
        "min_weight": 0.0,
    })

    _collapse_to_single_reid_stage(derived, total_epochs=total_epochs)
    _mark_baseline_metadata(
        derived,
        baseline_type=baseline_type,
        disabled_components=[
            "illumination_module",
            "feature_fusion",
            "branch_attention_fusion",
            "nuisance_head",
            "pk_sampler",
            "center_loss",
            "arcface_loss",
            "triplet_loss",
            "flip_test",
            "random_erasing",
            "color_jitter",
            "iicl",
            "route_b_metric_geometry",
        ],
    )
    return derived


def derive_illumination_only_config(
    full_config: Dict[str, Any],
    backbone_override: Optional[str] = None,
) -> Dict[str, Any]:
    derived = copy.deepcopy(_as_dict(full_config))
    model_cfg = derived.setdefault("model", {})
    training_cfg = derived.setdefault("training", {})

    _apply_backbone_override(derived, backbone_override)

    model_cfg.setdefault("illumination_module", {})["enabled"] = True
    model_cfg.setdefault("feature_fusion", {})["enabled"] = True
    model_cfg.setdefault("branch_attention_fusion", {})["enabled"] = False
    model_cfg.setdefault("nuisance_head", {})["enabled"] = False

    _disable_route_b_losses(derived)

    training_cfg.setdefault("photo_prior", {}).update(
        _as_dict(_as_dict(full_config).get("training")).get("photo_prior", {})
    )
    derived.setdefault("baseline", {})
    derived["baseline"] = {
        "enabled": False,
        "type": "illumination_only",
        "source_backbone": model_cfg.get("backbone"),
        "baseline_backbone": model_cfg.get("backbone"),
        "source_total_epochs": _sum_phase_epochs(full_config),
        "baseline_total_epochs": _sum_phase_epochs(derived),
        "disabled_components": [
            "branch_attention_fusion",
            "nuisance_head",
            "iicl",
            "route_b_metric_geometry",
        ],
        "evaluation_protocol": _as_dict(derived.get("evaluation")).get("protocol"),
        "best_metric": _as_dict(derived.get("evaluation")).get("best_metric"),
    }
    return derived


def derive_naive_illumination_config(
    full_config: Dict[str, Any],
    backbone_override: Optional[str] = None,
) -> Dict[str, Any]:
    derived = copy.deepcopy(_as_dict(full_config))
    model_cfg = derived.setdefault("model", {})
    training_cfg = derived.setdefault("training", {})

    _apply_backbone_override(derived, backbone_override)

    model_cfg.setdefault("illumination_module", {})["enabled"] = True
    _disable_method_heads(derived)
    _disable_route_b_losses(derived)
    _update_illumination_module_params(
        derived,
        use_feature_guided=False,
        enable_task_aware_rollback=False,
        use_model_aware_residual=False,
        enable_coarse_task_grad=False,
    )

    training_cfg.setdefault("photo_prior", {}).update(
        _as_dict(_as_dict(full_config).get("training")).get("photo_prior", {})
    )
    derived["baseline"] = {
        "enabled": False,
        "type": "naive_illumination",
        "source_backbone": model_cfg.get("backbone"),
        "baseline_backbone": model_cfg.get("backbone"),
        "source_total_epochs": _sum_phase_epochs(full_config),
        "baseline_total_epochs": _sum_phase_epochs(derived),
        "disabled_components": [
            "feature_guided_refinement",
            "task_aware_rollback",
            "model_aware_residual",
            "coarse_task_grad",
            "feature_fusion",
            "branch_attention_fusion",
            "nuisance_head",
            "iicl",
            "route_b_metric_geometry",
        ],
        "evaluation_protocol": _as_dict(derived.get("evaluation")).get("protocol"),
        "best_metric": _as_dict(derived.get("evaluation")).get("best_metric"),
    }
    return derived


def _derive_full_model_config(full_config: Dict[str, Any], backbone_override: Optional[str] = None) -> Dict[str, Any]:
    derived = copy.deepcopy(_as_dict(full_config))
    _apply_backbone_override(derived, backbone_override)
    return derived


def _materialize_job(job: AblationJob) -> None:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    with open(job.config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(job.config, handle, sort_keys=False, allow_unicode=True)


def _build_baseline_command(job: AblationJob, data_dir: str, device: str) -> List[str]:
    backbone = str(_as_dict(job.config.get("model")).get("backbone", "osnet_ain_x1_0"))
    return [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "train_baselines.py"),
        "--config",
        str(job.config_path),
        "--data_dir",
        data_dir,
        "--output_dir",
        str(job.output_dir),
        "--backbone",
        backbone,
        "--device",
        device,
    ]


def _build_joint_command(job: AblationJob, data_dir: str, device: str, num_workers: int) -> List[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "train_joint.py"),
        "--config",
        str(job.config_path),
        "--data_dir",
        data_dir,
        "--output_dir",
        str(job.output_dir),
        "--device",
        device,
        "--num_workers",
        str(num_workers),
    ]


def build_atrw_main_ablation_jobs(
    config_path: str,
    data_dir: str,
    output_root: Path,
    device: str,
    backbone_override: Optional[str] = None,
    num_workers: int = 4,
    baseline_head: Optional[str] = None,
) -> List[AblationJob]:
    full_cfg = load_config(config_path)
    jobs: List[AblationJob] = []
    baseline_head = _resolve_baseline_head(baseline_head)
    baseline_display_name = "Plain Baseline" if baseline_head == "plain_global" else "Stripe-head Baseline"

    variant_defs = [
        (
            "plain_baseline",
            baseline_display_name,
            derive_plain_baseline_config(
                full_cfg,
                backbone_override=backbone_override,
                baseline_head=baseline_head,
            ),
            (PROJECT_ROOT / "tools" / "train_baselines.py").as_posix(),
            ["baseline_best.pth", "baseline_best_reid_best.pth"],
        ),
        (
            "naive_illumination",
            "Naive Illumination",
            derive_naive_illumination_config(full_cfg, backbone_override=backbone_override),
            (PROJECT_ROOT / "tools" / "train_joint.py").as_posix(),
            ["joint_best.pth", "joint_best_reid_best.pth"],
        ),
        (
            "illumination_only",
            "Illumination Only",
            derive_illumination_only_config(full_cfg, backbone_override=backbone_override),
            (PROJECT_ROOT / "tools" / "train_joint.py").as_posix(),
            ["joint_best.pth", "joint_best_reid_best.pth"],
        ),
        (
            "full_model",
            "Full Model",
            _derive_full_model_config(full_cfg, backbone_override=backbone_override),
            (PROJECT_ROOT / "tools" / "train_joint.py").as_posix(),
            ["joint_best.pth", "joint_best_reid_best.pth"],
        ),
    ]

    for variant_key, display_name, config, entrypoint, checkpoint_candidates in variant_defs:
        output_dir = output_root / variant_key
        _set_output_dir(config, output_dir)
        job = AblationJob(
            variant_key=variant_key,
            display_name=display_name,
            train_entrypoint=entrypoint,
            eval_mode="atrw_openset",
            config=config,
            output_dir=output_dir,
            config_path=output_dir / "derived_config.yaml",
            train_log_path=output_dir / "train.log",
            eval_log_path=output_dir / "eval.log",
            result_path=output_dir / "result.json",
            checkpoint_candidates=checkpoint_candidates,
        )
        if entrypoint.endswith("train_baselines.py"):
            job.train_command = _build_baseline_command(job, data_dir=data_dir, device=device)
        else:
            job.train_command = _build_joint_command(job, data_dir=data_dir, device=device, num_workers=num_workers)
        jobs.append(job)

    return jobs


def _run_logged_command(command: List[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(process.returncode)


def _find_checkpoint(job: AblationJob) -> Optional[Path]:
    for filename in job.checkpoint_candidates:
        direct_candidate = job.output_dir / filename
        if direct_candidate.exists():
            return direct_candidate
        recursive_candidates = sorted(
            job.output_dir.rglob(filename),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if recursive_candidates:
            return recursive_candidates[0]

    patterns = ("baseline_epoch*.pth", "joint_phase*_epoch*.pth", "joint_epoch*.pth")
    for pattern in patterns:
        matches = sorted(job.output_dir.rglob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    return None


def _parse_atrw_eval_log(eval_text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    single_match = re.search(
        r"Single-camera\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%",
        eval_text,
        flags=re.MULTILINE,
    )
    cross_match = re.search(
        r"Cross-camera\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%",
        eval_text,
        flags=re.MULTILINE,
    )
    mmap_match = re.search(r"mmAP\s+.*?([0-9.]+)%", eval_text, flags=re.MULTILINE)

    if single_match:
        metrics["rank1_single"] = float(single_match.group(1))
        metrics["rank5_single"] = float(single_match.group(2))
        metrics["mAP_single"] = float(single_match.group(3))
    if cross_match:
        metrics["rank1_cross"] = float(cross_match.group(1))
        metrics["rank5_cross"] = float(cross_match.group(2))
        metrics["mAP_cross"] = float(cross_match.group(3))
    if mmap_match:
        metrics["mmAP"] = float(mmap_match.group(1))
    return metrics


def _evaluate_job(
    job: AblationJob,
    checkpoint_path: Path,
    data_root: str,
    eval_script_dir: str,
) -> Dict[str, Any]:
    atrw_cfg = _as_dict(_as_dict(job.config.get("evaluation")).get("atrw"))
    model_cfg = _as_dict(job.config.get("model"))
    eval_command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "eval_atrw_openset.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--data_root",
        data_root,
        "--eval_script_dir",
        eval_script_dir,
        "--output",
        str(job.output_dir / "submission_openset.json"),
        "--backbone",
        str(model_cfg.get("backbone", "osnet_ain_x1_0")),
        "--batch_size",
        str(int(atrw_cfg.get("batch_size", 64))),
        "--num_workers",
        str(int(atrw_cfg.get("num_workers", 4))),
    ]
    return_code = _run_logged_command(eval_command, log_path=job.eval_log_path, cwd=PROJECT_ROOT)
    eval_text = job.eval_log_path.read_text(encoding="utf-8", errors="replace")
    metrics = _parse_atrw_eval_log(eval_text)
    return {
        "return_code": return_code,
        "command": " ".join(eval_command),
        "metrics": metrics,
    }


def _write_summary(output_root: Path, results: List[Dict[str, Any]]) -> None:
    summary_json = output_root / "atrw_main_ablation_results.json"
    summary_csv = output_root / "atrw_main_ablation_table.csv"
    summary_md = output_root / "atrw_main_ablation_table.md"

    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    csv_rows = []
    for result in results:
        metrics = result.get("metrics", {})
        csv_rows.append(
            {
                "variant": result.get("variant_key"),
                "display_name": result.get("display_name"),
                "rank1_single": metrics.get("rank1_single"),
                "mAP_single": metrics.get("mAP_single"),
                "rank1_cross": metrics.get("rank1_cross"),
                "mAP_cross": metrics.get("mAP_cross"),
                "mmAP": metrics.get("mmAP"),
                "status": result.get("status"),
            }
        )

    with open(summary_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "display_name",
                "rank1_single",
                "mAP_single",
                "rank1_cross",
                "mAP_cross",
                "mmAP",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    md_lines = [
        "# ATRW Main Ablation",
        "",
        "| Variant | Rank-1(single) | mAP(single) | Rank-1(cross) | mAP(cross) | mmAP | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in csv_rows:
        md_lines.append(
            "| {display_name} | {rank1_single} | {mAP_single} | {rank1_cross} | {mAP_cross} | {mmAP} | {status} |".format(
                display_name=row["display_name"],
                rank1_single=_fmt_metric(row["rank1_single"]),
                mAP_single=_fmt_metric(row["mAP_single"]),
                rank1_cross=_fmt_metric(row["rank1_cross"]),
                mAP_cross=_fmt_metric(row["mAP_cross"]),
                mmAP=_fmt_metric(row["mmAP"]),
                status=row["status"],
            )
        )
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")


def _fmt_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return "-"


def _select_variants(requested: str) -> List[str]:
    if requested.strip().lower() == "all":
        return list(DEFAULT_VARIANT_ORDER)
    variants = [item.strip() for item in requested.split(",") if item.strip()]
    unknown = [item for item in variants if item not in DEFAULT_VARIANT_ORDER]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Expected subset of {DEFAULT_VARIANT_ORDER}")
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ATRW paper-oriented main ablation")
    parser.add_argument("--config", type=str, default="config/illumination_config_atrw.yaml", help="Source ATRW config")
    parser.add_argument("--data_dir", type=str, required=True, help="ATRW train directory")
    parser.add_argument(
        "--data_root",
        type=str,
        default="orignal_data/Amur Tiger Re-identification",
        help="ATRW original data root for official open-set evaluation",
    )
    parser.add_argument("--eval_script_dir", type=str, default="ATRWEvalScript-main", help="ATRW official eval script dir")
    parser.add_argument("--output_dir", type=str, default="checkpoints/ablation/atrw_main", help="Ablation output root")
    parser.add_argument("--device", type=str, default="cuda", help="Training device")
    parser.add_argument("--num_workers", type=int, default=4, help="Training dataloader workers")
    parser.add_argument("--backbone", type=str, default=None, help="Optional backbone override")
    parser.add_argument(
        "--baseline_head",
        type=str,
        default=DEFAULT_BASELINE_HEAD,
        choices=list(SUPPORTED_BASELINE_HEADS),
        help="Baseline ReID head style: keep the default global GAP head or preserve the local stripe head.",
    )
    parser.add_argument("--variants", type=str, default="all", help="Comma-separated variant keys or 'all'")
    parser.add_argument("--dry_run", action="store_true", help="Only materialize configs and print commands")
    args = parser.parse_args()

    output_root = (PROJECT_ROOT / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = build_atrw_main_ablation_jobs(
        config_path=args.config,
        data_dir=args.data_dir,
        output_root=output_root,
        device=args.device,
        backbone_override=args.backbone,
        num_workers=args.num_workers,
        baseline_head=args.baseline_head,
    )
    selected = set(_select_variants(args.variants))
    jobs = [job for job in jobs if job.variant_key in selected]

    results: List[Dict[str, Any]] = []
    for job in jobs:
        _materialize_job(job)
        if args.dry_run:
            print(f"[Dry Run] {job.display_name}")
            print("  Train:", " ".join(job.train_command))
            continue

        train_code = _run_logged_command(job.train_command, log_path=job.train_log_path, cwd=PROJECT_ROOT)
        result_record: Dict[str, Any] = {
            "variant_key": job.variant_key,
            "display_name": job.display_name,
            "train_command": " ".join(job.train_command),
            "train_log": str(job.train_log_path),
            "status": "trained" if train_code == 0 else "train_failed",
        }

        if train_code != 0:
            result_record["train_return_code"] = train_code
            results.append(result_record)
            continue

        checkpoint_path = _find_checkpoint(job)
        if checkpoint_path is None:
            result_record["status"] = "checkpoint_missing"
            results.append(result_record)
            continue

        result_record["checkpoint"] = str(checkpoint_path)
        eval_result = _evaluate_job(
            job,
            checkpoint_path=checkpoint_path,
            data_root=args.data_root,
            eval_script_dir=args.eval_script_dir,
        )
        result_record["eval_command"] = eval_result["command"]
        result_record["eval_log"] = str(job.eval_log_path)
        result_record["metrics"] = eval_result["metrics"]
        result_record["eval_return_code"] = eval_result["return_code"]
        result_record["status"] = "done" if eval_result["return_code"] == 0 and eval_result["metrics"] else "eval_failed"

        with open(job.result_path, "w", encoding="utf-8") as handle:
            json.dump(result_record, handle, ensure_ascii=False, indent=2)
        results.append(result_record)

    if not args.dry_run:
        _write_summary(output_root, results)


if __name__ == "__main__":
    main()
