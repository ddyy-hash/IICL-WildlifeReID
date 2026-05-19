#!/usr/bin/env python3
"""
Rigorous white-box baseline runner derived from the full-model config.

This script keeps the baseline maximally fair to the main pipeline by:
- loading the same full-model YAML config
- preserving the same data augmentation / PK sampler / AMP / evaluation protocol
- instantiating the same JointReIDModel backbone + local extractor stack
- disabling only illumination-specific mechanisms

The result is a strict no-illumination baseline that stays on the same training
and evaluation path as ``tools/train_joint.py`` instead of maintaining a stale,
parallel training implementation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import cli_args_to_config, load_config
from tools.selection_protocols import CROSS_SPECIES_SELECTION_PROTOCOL, QUERY_GALLERY_PROTOCOL

_HEAVY_IMPORT_ERROR: Optional[ModuleNotFoundError] = None
try:
    import torch

    from tools.train_joint import (
        JointTrainer,
        cleanup_distributed,
        init_distributed_mode,
        setup_logging,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in minimal test environments
    _HEAVY_IMPORT_ERROR = exc
    torch = None  # type: ignore[assignment]
    JointTrainer = object  # type: ignore[misc,assignment]

    def _missing_dependency(*_args: Any, **_kwargs: Any) -> Any:
        raise ModuleNotFoundError(
            f"Baseline training dependencies missing: {_HEAVY_IMPORT_ERROR}. "
            "Dry-run/config-derivation utilities remain available."
        ) from _HEAVY_IMPORT_ERROR

    cleanup_distributed = _missing_dependency  # type: ignore[assignment]
    init_distributed_mode = _missing_dependency  # type: ignore[assignment]
    setup_logging = _missing_dependency  # type: ignore[assignment]


DEFAULT_BASELINE_BACKBONES = [
    "osnet_ain_x1_0",
    "osnet_x1_0",
    "resnet50",
]

_ZERO_ILLUMINATION_LOSS_KEYS = [
    "lambda_recon",
    "lambda_smooth",
    "lambda_edge",
    "lambda_structure",
    "lambda_sensitivity",
    "lambda_identity",
    "lambda_lab_chroma",
    "lambda_high_freq",
    "lambda_log_chroma",
]

_DISABLE_MODULE_BOOL_KEYS = [
    "use_sensitivity",
    "use_refinement",
    "use_feature_guided",
    "use_color_illumination",
    "enable_task_aware_rollback",
    "enable_coarse_task_grad",
    "safe_color_enabled",
    "clamp_input_range",
]

_ZERO_IDENTITY_KEYS = [
    "phase2_scale",
    "phase3_scale",
    "anchor_weight",
    "geometry_weight",
    "logit_weight",
]


def _require_training_dependencies() -> None:
    if _HEAVY_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            f"Baseline training dependencies missing: {_HEAVY_IMPORT_ERROR}"
        ) from _HEAVY_IMPORT_ERROR


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _phase_cfg(config: Dict[str, Any], phase_name: str) -> Dict[str, Any]:
    training_cfg = _as_dict(config.get("training"))
    phases_cfg = _as_dict(training_cfg.get("phases"))
    return _as_dict(phases_cfg.get(phase_name))


def _sum_phase_epochs(config: Dict[str, Any]) -> int:
    total = 0
    for phase_name in ("phase1", "phase2", "phase3"):
        total += int(_phase_cfg(config, phase_name).get("epochs", 0))
    return total


def resolve_requested_backbones(
    backbone: Optional[str],
    all_backbones: bool,
) -> List[str]:
    if all_backbones:
        return list(DEFAULT_BASELINE_BACKBONES)
    chosen = backbone or DEFAULT_BASELINE_BACKBONES[0]
    if chosen not in DEFAULT_BASELINE_BACKBONES:
        raise ValueError(
            f"Unsupported baseline backbone: {chosen}. "
            f"Expected one of {DEFAULT_BASELINE_BACKBONES}."
        )
    return [chosen]


def derive_baseline_config(
    full_config: Dict[str, Any],
    backbone_override: Optional[str] = None,
    total_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Derive a rigorous white-box baseline config from a full-model config.

    Fairness principle:
    - preserve backbone/local extractor training and evaluation settings
    - disable only illumination-specific modules and losses
    - collapse the three-stage schedule into a single ReID-only stage
    """
    source_cfg = _as_dict(full_config)
    derived = copy.deepcopy(source_cfg)

    model_cfg = derived.setdefault("model", {})
    training_cfg = derived.setdefault("training", {})
    hardware_cfg = derived.setdefault("hardware", {})
    illum_top_cfg = derived.setdefault("illumination_module", {})
    eval_cfg = derived.setdefault("evaluation", {})

    model_illum_cfg = model_cfg.setdefault("illumination_module", {})
    feature_fusion_cfg = model_cfg.setdefault("feature_fusion", {})
    illum_loss_cfg = illum_top_cfg.setdefault("loss_params", {})
    illum_module_params = illum_top_cfg.setdefault("module_params", {})
    illum_model_params = model_illum_cfg.setdefault("module_params", {})
    iicl_cfg = training_cfg.setdefault("iicl", {})
    photo_prior_cfg = training_cfg.setdefault("photo_prior", {})
    identity_cfg = training_cfg.setdefault("identity_preserving", {})

    if backbone_override:
        model_cfg["backbone"] = backbone_override

    illum_top_cfg["module_type"] = "disabled"
    model_illum_cfg["enabled"] = False
    feature_fusion_cfg["enabled"] = False

    for key in _ZERO_ILLUMINATION_LOSS_KEYS:
        illum_loss_cfg[key] = 0.0

    for params_cfg in (illum_module_params, illum_model_params):
        for key in _DISABLE_MODULE_BOOL_KEYS:
            params_cfg[key] = False
        params_cfg["num_grad_variants"] = 0
        params_cfg["refine_iterations"] = 0

    iicl_cfg["enabled"] = False
    iicl_cfg["weight"] = 0.0
    iicl_cfg["num_variants"] = 0
    iicl_cfg["num_grad_variants"] = 0

    photo_prior_cfg["initial_weight"] = 0.0
    photo_prior_cfg["min_weight"] = 0.0

    for key in _ZERO_IDENTITY_KEYS:
        identity_cfg[key] = 0.0

    baseline_total_epochs = int(total_epochs) if total_epochs is not None else _sum_phase_epochs(source_cfg)
    if baseline_total_epochs <= 0:
        raise ValueError("Baseline total epochs must be positive after derivation.")

    learning_rate = float(
        _as_dict(training_cfg.get("optimizer")).get(
            "lr",
            training_cfg.get("learning_rate", 3.5e-4),
        )
    )
    source_phase1_cfg = _phase_cfg(source_cfg, "phase1")
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

    derived["baseline"] = {
        "enabled": True,
        "type": "whitebox_no_illumination",
        "source_backbone": _as_dict(source_cfg.get("model")).get("backbone"),
        "baseline_backbone": model_cfg.get("backbone"),
        "source_total_epochs": _sum_phase_epochs(source_cfg),
        "baseline_total_epochs": baseline_total_epochs,
        "disabled_components": [
            "illumination_module",
            "feature_fusion",
            "iicl",
            "photo_prior",
            "identity_preserving_geometry",
        ],
        "evaluation_protocol": eval_cfg.get("protocol"),
        "best_metric": eval_cfg.get("best_metric"),
    }
    return derived


def summarize_baseline_derivation(
    full_config: Dict[str, Any],
    baseline_config: Dict[str, Any],
) -> Dict[str, Any]:
    full_model_cfg = _as_dict(full_config.get("model"))
    baseline_model_cfg = _as_dict(baseline_config.get("model"))
    baseline_training_cfg = _as_dict(baseline_config.get("training"))
    eval_cfg = _as_dict(baseline_config.get("evaluation"))
    hardware_cfg = _as_dict(baseline_config.get("hardware"))
    metric_cfg = _as_dict(baseline_training_cfg.get("metric_learning"))

    return {
        "baseline_mode": "whitebox_no_illumination",
        "source_backbone": full_model_cfg.get("backbone"),
        "baseline_backbone": baseline_model_cfg.get("backbone"),
        "source_total_epochs": _sum_phase_epochs(full_config),
        "baseline_total_epochs": _sum_phase_epochs(baseline_config),
        "source_phase_epochs": {
            phase: int(_phase_cfg(full_config, phase).get("epochs", 0))
            for phase in ("phase1", "phase2", "phase3")
        },
        "baseline_phase_epochs": {
            phase: int(_phase_cfg(baseline_config, phase).get("epochs", 0))
            for phase in ("phase1", "phase2", "phase3")
        },
        "batch_size": int(baseline_training_cfg.get("batch_size", 0)),
        "pk_sampler": copy.deepcopy(_as_dict(baseline_training_cfg.get("pk_sampler"))),
        "image_height": int(baseline_training_cfg.get("image_height", baseline_training_cfg.get("image_size", 0))),
        "image_width": int(baseline_training_cfg.get("image_width", baseline_training_cfg.get("image_size", 0))),
        "learning_rate": float(
            _as_dict(baseline_training_cfg.get("optimizer")).get(
                "lr",
                baseline_training_cfg.get("learning_rate", 0.0),
            )
        ),
        "amp": {
            "enabled": bool(hardware_cfg.get("use_amp", False)),
            "dtype": hardware_cfg.get("amp_dtype", "float16"),
        },
        "evaluation": {
            "protocol": eval_cfg.get("protocol"),
            "additional_protocols": list(eval_cfg.get("additional_protocols", []) or []),
            "best_metric": eval_cfg.get("best_metric"),
            "strict_protocol_check": bool(eval_cfg.get("strict_protocol_check", False)),
        },
        "retained_reid_losses": {
            "ce_weight": float(_as_dict(metric_cfg.get("ce_loss")).get("weight", 0.0)),
            "triplet_weight": float(_as_dict(metric_cfg.get("triplet_loss")).get("weight", 0.0)),
            "circle_weight": float(_as_dict(metric_cfg.get("circle_loss")).get("weight", 0.0)),
            "arcface_weight": float(_as_dict(metric_cfg.get("arcface_loss")).get("weight", 0.0)),
            "center_weight": float(_as_dict(baseline_training_cfg.get("center_loss")).get("weight", 0.0)),
        },
        "disabled_components": [
            "illumination_module",
            "feature_fusion",
            "iicl",
            "photo_prior",
            "identity_preserving_geometry",
        ],
    }


def format_baseline_summary_lines(summary: Dict[str, Any]) -> List[str]:
    evaluation = _as_dict(summary.get("evaluation"))
    amp_cfg = _as_dict(summary.get("amp"))
    pk_sampler = _as_dict(summary.get("pk_sampler"))
    source_phase_epochs = _as_dict(summary.get("source_phase_epochs"))
    baseline_phase_epochs = _as_dict(summary.get("baseline_phase_epochs"))

    def _format_phase_epochs(phase_epochs: Dict[str, Any]) -> str:
        if not phase_epochs:
            return "n/a"
        ordered = []
        for phase_name in ("phase1", "phase2", "phase3"):
            if phase_name in phase_epochs:
                ordered.append(f"{phase_name}={phase_epochs.get(phase_name)}")
        for phase_name, value in phase_epochs.items():
            if phase_name not in {"phase1", "phase2", "phase3"}:
                ordered.append(f"{phase_name}={value}")
        return ", ".join(ordered) if ordered else "n/a"

    return [
        f"  source_backbone: {summary.get('source_backbone')}",
        f"  baseline_backbone: {summary.get('baseline_backbone')}",
        f"  total_epochs: {summary.get('source_total_epochs')} -> {summary.get('baseline_total_epochs')}",
        f"  source_phase_epochs: {_format_phase_epochs(source_phase_epochs)}",
        f"  baseline_phase_epochs: {_format_phase_epochs(baseline_phase_epochs)}",
        f"  batch_size: {summary.get('batch_size')}",
        f"  p_size: {pk_sampler.get('p')}, k_size: {pk_sampler.get('k')}",
        f"  img_size: {summary.get('image_height')}x{summary.get('image_width')}",
        f"  learning_rate: {summary.get('learning_rate')}",
        f"  eval_protocol: {evaluation.get('protocol')}",
        f"  additional_eval_protocols: {evaluation.get('additional_protocols')}",
        f"  best_metric: {evaluation.get('best_metric')}",
        f"  amp: enabled={amp_cfg.get('enabled')}, dtype={amp_cfg.get('dtype')}",
        f"  disabled_components: {', '.join(summary.get('disabled_components', []))}",
    ]


def save_baseline_artifacts(
    output_dir: str,
    baseline_config: Dict[str, Any],
    summary: Dict[str, Any],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "derived_baseline_config.yaml")
    summary_path = os.path.join(output_dir, "baseline_summary.json")

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(baseline_config, f, sort_keys=False, allow_unicode=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _set_output_dir(config: Dict[str, Any], output_dir: str) -> None:
    config["output_dir"] = output_dir
    config.setdefault("training", {})["output_dir"] = output_dir


def _extract_runtime_params(config: Dict[str, Any]) -> Dict[str, Any]:
    training_cfg = _as_dict(config.get("training"))
    model_cfg = _as_dict(config.get("model"))
    phase1_cfg = _phase_cfg(config, "phase1")
    phase2_cfg = _phase_cfg(config, "phase2")
    phase3_cfg = _phase_cfg(config, "phase3")
    pk_cfg = _as_dict(training_cfg.get("pk_sampler"))
    metric_cfg = _as_dict(training_cfg.get("metric_learning"))
    circle_cfg = _as_dict(metric_cfg.get("circle_loss"))
    hardware_cfg = _as_dict(config.get("hardware"))

    return {
        "backbone": model_cfg.get("backbone", DEFAULT_BASELINE_BACKBONES[0]),
        "batch_size": int(training_cfg.get("batch_size", 32)),
        "phase1_epochs": int(phase1_cfg.get("epochs", 0)),
        "phase2_epochs": int(phase2_cfg.get("epochs", 0)),
        "phase3_epochs": int(phase3_cfg.get("epochs", 0)),
        "learning_rate": float(
            _as_dict(training_cfg.get("optimizer")).get("lr", training_cfg.get("learning_rate", 3.5e-4))
        ),
        "num_stripes": int(_as_dict(model_cfg.get("local_extractor")).get("num_parts", 6)),
        "eval_interval": int(training_cfg.get("eval_interval", 5)),
        "p_size": pk_cfg.get("p"),
        "k_size": pk_cfg.get("k", 4),
        "circle_gamma": int(circle_cfg.get("gamma", 256)),
        "img_height": int(training_cfg.get("image_height", training_cfg.get("image_size", 256))),
        "img_width": int(training_cfg.get("image_width", training_cfg.get("image_size", 256))),
        "num_workers": int(hardware_cfg.get("num_workers", 4)),
    }


class BaselineTrainer(JointTrainer):
    """Single-stage white-box baseline built on the current JointTrainer stack."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        checkpoint_cfg = _as_dict(self.config.get("checkpointing"))
        configured_interval = checkpoint_cfg.get("save_interval", 10)
        try:
            self.checkpoint_save_interval = max(0, int(configured_interval))
        except (TypeError, ValueError):
            self.checkpoint_save_interval = 10

    def _init_model(self) -> None:
        super()._init_model()
        if getattr(self, "use_ipaid", True):
            raise RuntimeError(
                "Baseline config derivation failed: illumination branch is still enabled."
            )
        if getattr(self.model, "feature_fusion", None) is not None:
            raise RuntimeError(
                "Baseline config derivation failed: feature fusion is still enabled."
            )
        self.logger.info(
            "White-box baseline initialized: illumination=%s, feature_fusion=%s",
            "disabled",
            "disabled",
        )

    def _save_emergency_checkpoint(self, reason: str) -> None:
        if not self.is_main_process:
            return
        if not hasattr(self, "optimizer") or not hasattr(self, "scheduler"):
            return

        path = os.path.join(
            self.output_dir,
            f"baseline_{reason}_epoch{max(self.current_epoch + 1, 0)}.pth",
        )
        checkpoint = {
            "epoch": self.current_epoch,
            "phase": self.current_phase,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "num_classes": self.num_classes,
            "config": self.config,
            "best_acc": self.best_acc,
            "best_rank1": self.best_rank1,
            "best_map": self.best_map,
            "best_metric_name": self.best_metric_name,
            "best_metric_value": self.best_metric_value,
            "reason": reason,
            "baseline_mode": "whitebox_no_illumination",
        }
        torch.save(checkpoint, path)
        self.logger.warning("Emergency checkpoint saved: %s", path)

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        phase: int,
        is_best: bool = False,
        suffix: str = "",
    ) -> None:
        if not self.is_main_process:
            return

        checkpoint = {
            "epoch": epoch,
            "phase": phase,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "num_classes": self.num_classes,
            "metrics": metrics,
            "config": self.config,
            "best_acc": self.best_acc,
            "best_rank1": self.best_rank1,
            "best_map": self.best_map,
            "best_metric_name": self.best_metric_name,
            "best_metric_value": self.best_metric_value,
            "eval_protocol": self.eval_protocol,
            "additional_eval_protocols": list(self.additional_eval_protocols),
            "baseline_mode": "whitebox_no_illumination",
        }

        if is_best:
            path = os.path.join(self.output_dir, f"baseline_best{suffix}.pth")
        else:
            path = os.path.join(self.output_dir, f"baseline_epoch{epoch + 1}.pth")

        torch.save(checkpoint, path)
        self.logger.info("Checkpoint saved: %s", path)

    def train(self) -> None:
        def maybe_run_eval(
            epoch: int,
            total_epochs: int,
            metrics: Dict[str, float],
        ) -> None:
            should_eval = ((epoch + 1) % self.eval_interval == 0) or ((epoch + 1) == total_epochs)
            if not should_eval:
                return

            eval_error: Optional[Exception] = None
            error_flag = torch.zeros(
                1,
                device=self.device if self.device.type == "cuda" else torch.device("cpu"),
                dtype=torch.int64,
            )
            self._barrier()
            try:
                eval_metrics = self.evaluate_for_model_selection()
            except Exception as exc:
                eval_error = exc
                error_flag.fill_(1)

            if self.is_distributed:
                torch.distributed.broadcast(error_flag, src=0)
            if int(error_flag.item()) > 0:
                if eval_error is not None and not self.is_distributed:
                    raise eval_error
                raise RuntimeError(f"Rank-0 evaluation failed at baseline epoch={epoch + 1}")

            if self.is_main_process and eval_metrics:
                self.logger.info(
                    "Evaluation Results (%s): %s",
                    self.eval_protocol,
                    self._format_eval_metrics(eval_metrics),
                )
                self._log_additional_eval_results()

                metric_value = self._extract_primary_metric(eval_metrics)
                current_rank1, current_map = self._extract_reid_selection_metrics(eval_metrics)
                is_new_best_metric = metric_value > self.best_metric_value
                rank_eps = 1e-12
                rank1_tied = abs(current_rank1 - self.best_rank1) <= rank_eps
                is_new_best_reid = current_rank1 > self.best_rank1 or (rank1_tied and current_map > self.best_map)

                if is_new_best_metric or is_new_best_reid:
                    checkpoint_metrics = dict(metrics)
                    checkpoint_metrics["eval"] = eval_metrics

                    if is_new_best_metric:
                        self.best_metric_value = metric_value
                        self.save_checkpoint(epoch, checkpoint_metrics, phase=1, is_best=True, suffix="")
                        self.logger.info(
                            "New best baseline_best by %s: %.2f",
                            self.best_metric_name,
                            self.best_metric_value,
                        )

                    if is_new_best_reid:
                        self.best_rank1 = current_rank1
                        self.best_map = current_map
                        self.save_checkpoint(epoch, checkpoint_metrics, phase=1, is_best=True, suffix="_reid_best")
                        self.logger.info(
                            "New best baseline_best_reid_best: Rank-1=%.2f%%, mAP=%.2f%%",
                            self.best_rank1,
                            self.best_map,
                        )
            self._barrier()

        self.logger.info("=" * 70)
        self.logger.info("Starting White-Box Baseline Training")
        self.logger.info(
            "Single stage: %d epochs (ReID-only, illumination/fusion consistency disabled in config)",
            self.phase1_epochs,
        )
        self.logger.info("=" * 70)

        best_acc = self.best_acc
        start_epoch = min(self.resume_epoch, self.phase1_epochs) if self.resume_phase else 0

        try:
            self._setup_optimizer_phase1()

            if self.resume_optimizer_state and self.resume_scheduler_state:
                self.optimizer.load_state_dict(self.resume_optimizer_state)
                self.scheduler.load_state_dict(self.resume_scheduler_state)

            for epoch in range(start_epoch, self.phase1_epochs):
                self.current_phase = 1
                self.current_epoch = epoch
                self._set_sampler_epoch(epoch)
                metrics = self.train_epoch(epoch, phase=1)
                self.scheduler.step()
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    "Baseline Epoch [%d/%d] | Loss: %.4f | Acc: %.2f%% | LR: %.6f",
                    epoch + 1,
                    self.phase1_epochs,
                    metrics["total_loss"],
                    metrics["accuracy"],
                    lr,
                )

                if metrics["accuracy"] > best_acc:
                    best_acc = metrics["accuracy"]
                    self.best_acc = best_acc

                try:
                    checkpoint_interval = max(0, int(getattr(self, "checkpoint_save_interval", 10)))
                except (TypeError, ValueError):
                    checkpoint_interval = 10
                if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
                    self.save_checkpoint(epoch, metrics, phase=1, is_best=False)

                maybe_run_eval(epoch, total_epochs=self.phase1_epochs, metrics=metrics)
        except KeyboardInterrupt:
            self._save_emergency_checkpoint("interrupt")
            raise
        except Exception:
            self._save_emergency_checkpoint("exception")
            raise
        finally:
            self.resume_optimizer_state = None
            self.resume_scheduler_state = None

        self.best_acc = best_acc
        self.logger.info("=" * 70)
        self.logger.info("Baseline Training Complete! Best train accuracy: %.2f%%", best_acc)
        if self.best_metric_value > 0:
            self.logger.info(
                "Best Metric: %s: %.2f | Best ReID Rank-1: %.2f%% | mAP: %.2f%%",
                self.best_metric_name,
                self.best_metric_value,
                self.best_rank1,
                self.best_map,
            )
        self.logger.info("=" * 70)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rigorous white-box baseline runner derived from a full-model config",
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Training data root directory")
    parser.add_argument("--config", type=str, required=True, help="Full-model YAML config path")
    parser.add_argument("--output_dir", type=str, default=None, help="Root output directory for baseline runs")
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=DEFAULT_BASELINE_BACKBONES,
        help="Single baseline backbone to run",
    )
    parser.add_argument(
        "--all-backbones",
        action="store_true",
        help="Run all standard baselines: osnet_ain_x1_0, osnet_x1_0, resnet50",
    )
    parser.add_argument("--total_epochs", type=int, default=None, help="Override total baseline epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size before derivation")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate before derivation")
    parser.add_argument("--num_stripes", type=int, default=None, help="Override number of local stripes")
    parser.add_argument("--eval_interval", type=int, default=None, help="Run evaluation every N epochs")
    parser.add_argument("--p_size", type=int, default=None, help="PK sampler P value")
    parser.add_argument("--k_size", type=int, default=None, help="PK sampler K value")
    parser.add_argument("--circle_gamma", type=int, default=None, help="Circle loss gamma")
    parser.add_argument("--img_height", type=int, default=None, help="Input image height")
    parser.add_argument("--img_width", type=int, default=None, help="Input image width")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cpu/cuda")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for torchrun compatibility")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument(
        "--eval_protocol",
        type=str,
        choices=[
            "val_split_70_30",
            QUERY_GALLERY_PROTOCOL,
            CROSS_SPECIES_SELECTION_PROTOCOL,
            "atrw_openset",
            "atrw_closedset_train70_val30",
            "atrw_closedset_animals_701",
        ],
        default=None,
        help="Override in-training evaluation protocol before derivation",
    )
    parser.add_argument("--best_metric", type=str, default=None, help="Override primary model-selection metric")
    parser.add_argument(
        "--strict_protocol_check",
        dest="strict_protocol_check",
        action="store_true",
        help="Enable strict protocol checks",
    )
    parser.add_argument(
        "--no_strict_protocol_check",
        dest="strict_protocol_check",
        action="store_false",
        help="Disable strict protocol checks",
    )
    parser.set_defaults(strict_protocol_check=None)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only derive and save baseline config/summary without training",
    )
    parser.add_argument(
        "--strict_plain_reid",
        action="store_true",
        help="Use the standard strict plain-global ReID baseline derivation.",
    )
    return parser


def _resolve_output_root(config_path: str, output_dir: Optional[str]) -> str:
    if output_dir:
        return output_dir
    config_stem = Path(config_path).stem
    return os.path.join(".", "checkpoints", "baselines", config_stem)


def _print_run_header(
    config_path: str,
    backbone_runs: Sequence[str],
    summary: Dict[str, Any],
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Baseline source config: {config_path}")
    print(f"Backbones: {list(backbone_runs)}")
    print(f"{'=' * 60}")
    for line in format_baseline_summary_lines(summary):
        print(line)
    print(f"{'=' * 60}\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        parser.error(f"data_dir not found: {args.data_dir}")
    if not os.path.exists(args.config):
        parser.error(f"config not found: {args.config}")
    if args.resume and args.all_backbones:
        parser.error("--resume only supports a single backbone run")

    backbone_runs = resolve_requested_backbones(args.backbone, args.all_backbones)
    cli_cfg = cli_args_to_config(args)
    full_config = load_config(args.config, cli_overrides=cli_cfg)
    output_root = _resolve_output_root(args.config, args.output_dir)

    for backbone in backbone_runs:
        if args.strict_plain_reid:
            from tools.run_atrw_main_ablation import derive_plain_baseline_config

            baseline_config = derive_plain_baseline_config(
                full_config,
                backbone_override=backbone,
                total_epochs=args.total_epochs,
                baseline_head="plain_global",
            )
        else:
            baseline_config = derive_baseline_config(
                full_config,
                backbone_override=backbone,
                total_epochs=args.total_epochs,
            )

        run_output_dir = os.path.join(output_root, backbone)
        _set_output_dir(baseline_config, run_output_dir)
        summary = summarize_baseline_derivation(full_config, baseline_config)
        summary["output_dir"] = run_output_dir
        summary["config_path"] = args.config
        summary["data_dir"] = args.data_dir

        save_baseline_artifacts(run_output_dir, baseline_config, summary)
        _print_run_header(args.config, [backbone], summary)

        if args.dry_run:
            print(f"[Dry Run] Saved derived config and summary to: {run_output_dir}")
            continue

        _require_training_dependencies()
        dist_ctx = init_distributed_mode(args.device, baseline_config, cli_local_rank=args.local_rank)
        logger = setup_logging(
            run_output_dir,
            is_main_process=dist_ctx["rank"] == 0,
            rank=dist_ctx["rank"],
        )

        try:
            runtime = _extract_runtime_params(baseline_config)
            trainer = BaselineTrainer(
                data_dir=args.data_dir,
                output_dir=run_output_dir,
                config=baseline_config,
                config_path=args.config,
                backbone=runtime["backbone"],
                batch_size=runtime["batch_size"],
                phase1_epochs=runtime["phase1_epochs"],
                phase2_epochs=runtime["phase2_epochs"],
                phase3_epochs=runtime["phase3_epochs"],
                learning_rate=runtime["learning_rate"],
                num_stripes=runtime["num_stripes"],
                device=args.device,
                logger=logger,
                resume_checkpoint=args.resume,
                eval_interval=runtime["eval_interval"],
                p_size=runtime["p_size"],
                k_size=runtime["k_size"],
                circle_gamma=runtime["circle_gamma"],
                img_height=runtime["img_height"],
                img_width=runtime["img_width"],
                use_iicl=False,
                iicl_weight=0.0,
                iicl_num_variants=0,
                num_workers=runtime["num_workers"],
                rank=dist_ctx["rank"],
                local_rank=dist_ctx["local_rank"],
                world_size=dist_ctx["world_size"],
                is_distributed=dist_ctx["is_distributed"],
                ddp_find_unused_parameters=dist_ctx["find_unused_parameters"],
            )
            trainer.train()
        finally:
            cleanup_distributed()


if __name__ == "__main__":
    main()
