#!/usr/bin/env python3
"""Run fine-grained ATRW component ablations for the TMM extension."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import yaml


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import load_config
from tools.run_atrw_main_ablation import (
    AblationJob,
    _apply_backbone_override,
    _as_dict,
    _build_joint_command,
    _evaluate_job,
    _find_checkpoint,
    _run_logged_command,
    _set_output_dir,
)


DEFAULT_VARIANT_ORDER = (
    "full_model",
    "no_task_aware_rollback",
    "no_model_aware_residual",
    "no_feature_fusion",
    "no_branch_attention_fusion",
    "no_iicl",
    "no_softap",
    "no_teacher_manifold",
    "teacher_tube_only",
    "teacher_separation_only",
    "no_nuisance_decoupling",
    "no_identity_image_preserving",
    "no_photo_prior",
)


def _module_params(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    top_params = config.setdefault("illumination_module", {}).setdefault("module_params", {})
    model_params = (
        config.setdefault("model", {})
        .setdefault("illumination_module", {})
        .setdefault("module_params", {})
    )
    return top_params, model_params


def _set_module_param(config: Dict[str, Any], key: str, value: Any) -> None:
    top_params, model_params = _module_params(config)
    top_params[key] = value
    model_params[key] = value


def _set_section_enabled(
    config: Dict[str, Any],
    section_name: str,
    enabled: bool,
    zero_keys: Iterable[str] = ("weight",),
) -> None:
    section = config.setdefault("training", {}).setdefault(section_name, {})
    section["enabled"] = bool(enabled)
    if not enabled:
        for key in zero_keys:
            if key in section:
                section[key] = 0.0


def _mark_tmm_ablation(config: Dict[str, Any], variant_key: str, disabled_components: Iterable[str]) -> None:
    eval_cfg = _as_dict(config.get("evaluation"))
    model_cfg = _as_dict(config.get("model"))
    config["tmm_ablation"] = {
        "enabled": True,
        "variant": variant_key,
        "source_backbone": model_cfg.get("backbone"),
        "disabled_components": list(disabled_components),
        "evaluation_protocol": eval_cfg.get("protocol"),
        "best_metric": eval_cfg.get("best_metric"),
    }


def derive_full_model_config(full_config: Dict[str, Any], backbone_override: Optional[str] = None) -> Dict[str, Any]:
    derived = copy.deepcopy(_as_dict(full_config))
    _apply_backbone_override(derived, backbone_override)
    _mark_tmm_ablation(derived, "full_model", [])
    return derived


def derive_no_task_aware_rollback_config(
    full_config: Dict[str, Any],
    backbone_override: Optional[str] = None,
) -> Dict[str, Any]:
    derived = derive_full_model_config(full_config, backbone_override)
    _set_module_param(derived, "enable_task_aware_rollback", False)
    _mark_tmm_ablation(derived, "no_task_aware_rollback", ["task_aware_rollback"])
    return derived


def derive_no_model_aware_residual_config(
    full_config: Dict[str, Any],
    backbone_override: Optional[str] = None,
) -> Dict[str, Any]:
    derived = derive_full_model_config(full_config, backbone_override)
    _set_module_param(derived, "use_model_aware_residual", False)
    _set_module_param(derived, "enable_coarse_task_grad", False)
    _mark_tmm_ablation(
        derived,
        "no_model_aware_residual",
        ["model_aware_residual", "coarse_task_grad"],
    )
    return derived


def derive_no_feature_fusion_config(
    full_config: Dict[str, Any],
    backbone_override: Optional[str] = None,
) -> Dict[str, Any]:
    derived = derive_full_model_config(full_config, backbone_override)
    derived.setdefault("model", {}).setdefault("feature_fusion", {})["enabled"] = False
    _mark_tmm_ablation(derived, "no_feature_fusion", ["feature_fusion"])
    return derived


def derive_no_branch_attention_fusion_config(
    full_config: Dict[str, Any],
    backbone_override: Optional[str] = None,
) -> Dict[str, Any]:
    derived = derive_full_model_config(full_config, backbone_override)
    derived.setdefault("model", {}).setdefault("branch_attention_fusion", {})["enabled"] = False
    _mark_tmm_ablation(derived, "no_branch_attention_fusion", ["branch_attention_fusion"])
    return derived


def derive_no_iicl_config(full_config: Dict[str, Any], backbone_override: Optional[str] = None) -> Dict[str, Any]:
    derived = derive_full_model_config(full_config, backbone_override)
    iicl_cfg = derived.setdefault("training", {}).setdefault("iicl", {})
    iicl_cfg.update({"enabled": False, "weight": 0.0, "num_variants": 0, "num_grad_variants": 0})
    phase3_aux = (
        derived.setdefault("training", {})
        .setdefault("phases", {})
        .setdefault("phase3", {})
        .setdefault("aux_ramp", {})
    )
    phase3_aux["iicl_start"] = 0.0
    phase3_aux["iicl_end"] = 0.0
    _mark_tmm_ablation(derived, "no_iicl", ["iicl"])
    return derived


def derive_no_softap_config(full_config: Dict[str, Any], backbone_override: Optional[str] = None) -> Dict[str, Any]:
    derived = derive_full_model_config(full_config, backbone_override)
    _set_section_enabled(derived, "cross_light_softap", False)
    phase3_aux = (
        derived.setdefault("training", {})
        .setdefault("phases", {})
        .setdefault("phase3", {})
        .setdefault("aux_ramp", {})
    )
    phase3_aux["cross_light_start"] = 0.0
    phase3_aux["cross_light_end"] = 0.0
    _mark_tmm_ablation(derived, "no_softap", ["cross_light_softap"])
    return derived


def derive_no_teacher_manifold_config(
    full_config: Dict[str, Any],
    backbone_override: Optional[str] = None,
) -> Dict[str, Any]:
    derived = derive_full_model_config(full_config, backbone_override)
    teacher_cfg = derived.setdefault("training", {}).setdefault("teacher_manifold", {})
    teacher_cfg.update({"enabled": False, "tube_weight": 0.0, "separation_weight": 0.0})
    _mark_tmm_ablation(derived, "no_teacher_manifold", ["teacher_manifold_tube", "teacher_manifold_separation"])
    return derived


def derive_teacher_tube_only_config(
    full_config: Dict[str, Any],
    backbone_override: Optional[str] = None,
) -> Dict[str, Any]:
    derived = derive_full_model_config(full_config, backbone_override)
    teacher_cfg = derived.setdefault("training", {}).setdefault("teacher_manifold", {})
    teacher_cfg["enabled"] = True
    teacher_cfg["separation_weight"] = 0.0
    _mark_tmm_ablation(derived, "teacher_tube_only", ["teacher_manifold_separation"])
    return derived


def derive_teacher_separation_only_config(
    full_config: Dict[str, Any],
    backbone_override: Optional[str] = None,
) -> Dict[str, Any]:
    derived = derive_full_model_config(full_config, backbone_override)
    teacher_cfg = derived.setdefault("training", {}).setdefault("teacher_manifold", {})
    teacher_cfg["enabled"] = True
    teacher_cfg["tube_weight"] = 0.0
    _mark_tmm_ablation(derived, "teacher_separation_only", ["teacher_manifold_tube"])
    return derived


def derive_no_nuisance_decoupling_config(
    full_config: Dict[str, Any],
    backbone_override: Optional[str] = None,
) -> Dict[str, Any]:
    derived = derive_full_model_config(full_config, backbone_override)
    derived.setdefault("model", {}).setdefault("nuisance_head", {})["enabled"] = False
    nuisance_cfg = derived.setdefault("training", {}).setdefault("nuisance_decoupling", {})
    nuisance_cfg.update({"enabled": False, "weight": 0.0, "regression_weight": 0.0, "decorrelation_weight": 0.0})
    _mark_tmm_ablation(derived, "no_nuisance_decoupling", ["nuisance_head", "nuisance_decoupling"])
    return derived


def derive_no_identity_image_preserving_config(
    full_config: Dict[str, Any],
    backbone_override: Optional[str] = None,
) -> Dict[str, Any]:
    derived = derive_full_model_config(full_config, backbone_override)
    _set_section_enabled(derived, "identity_image_preserving", False)
    _mark_tmm_ablation(derived, "no_identity_image_preserving", ["identity_image_preserving"])
    return derived


def derive_no_photo_prior_config(full_config: Dict[str, Any], backbone_override: Optional[str] = None) -> Dict[str, Any]:
    derived = derive_full_model_config(full_config, backbone_override)
    photo_cfg = derived.setdefault("training", {}).setdefault("photo_prior", {})
    photo_cfg["initial_weight"] = 0.0
    photo_cfg["min_weight"] = 0.0
    _mark_tmm_ablation(derived, "no_photo_prior", ["photo_prior"])
    return derived


VARIANT_BUILDERS: Dict[str, Tuple[str, Callable[[Dict[str, Any], Optional[str]], Dict[str, Any]]]] = {
    "full_model": ("Full TMM Candidate", derive_full_model_config),
    "no_task_aware_rollback": ("No Task-Aware Rollback", derive_no_task_aware_rollback_config),
    "no_model_aware_residual": ("No Model-Aware Residual", derive_no_model_aware_residual_config),
    "no_feature_fusion": ("No Feature Fusion", derive_no_feature_fusion_config),
    "no_branch_attention_fusion": ("No Branch-Attention Fusion", derive_no_branch_attention_fusion_config),
    "no_iicl": ("No IICL", derive_no_iicl_config),
    "no_softap": ("No SoftAP Cross-Light Ranking", derive_no_softap_config),
    "no_teacher_manifold": ("No Teacher Manifold", derive_no_teacher_manifold_config),
    "teacher_tube_only": ("Teacher Tube Only", derive_teacher_tube_only_config),
    "teacher_separation_only": ("Teacher Separation Only", derive_teacher_separation_only_config),
    "no_nuisance_decoupling": ("No Nuisance Decoupling", derive_no_nuisance_decoupling_config),
    "no_identity_image_preserving": ("No Identity Image Preserving", derive_no_identity_image_preserving_config),
    "no_photo_prior": ("No Photo Prior", derive_no_photo_prior_config),
}


def _materialize_job(job: AblationJob) -> None:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    with open(job.config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(job.config, handle, sort_keys=False, allow_unicode=True)


def build_tmm_component_ablation_jobs(
    config_path: str,
    data_dir: str,
    output_root: Path,
    device: str,
    backbone_override: Optional[str] = None,
    num_workers: int = 4,
) -> List[AblationJob]:
    full_cfg = load_config(config_path)
    jobs: List[AblationJob] = []

    for variant_key in DEFAULT_VARIANT_ORDER:
        display_name, builder = VARIANT_BUILDERS[variant_key]
        config = builder(full_cfg, backbone_override)
        output_dir = output_root / variant_key
        _set_output_dir(config, output_dir)
        job = AblationJob(
            variant_key=variant_key,
            display_name=display_name,
            train_entrypoint=(PROJECT_ROOT / "tools" / "train_joint.py").as_posix(),
            eval_mode="atrw_openset",
            config=config,
            output_dir=output_dir,
            config_path=output_dir / "derived_config.yaml",
            train_log_path=output_dir / "train.log",
            eval_log_path=output_dir / "eval.log",
            result_path=output_dir / "result.json",
            checkpoint_candidates=["joint_best.pth", "joint_best_reid_best.pth"],
        )
        job.train_command = _build_joint_command(job, data_dir=data_dir, device=device, num_workers=num_workers)
        jobs.append(job)

    return jobs


def _select_variants(requested: str) -> List[str]:
    if requested == "all":
        return list(DEFAULT_VARIANT_ORDER)
    variants = [item.strip() for item in requested.split(",") if item.strip()]
    unknown = [item for item in variants if item not in DEFAULT_VARIANT_ORDER]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Expected subset of {DEFAULT_VARIANT_ORDER}")
    return variants


def _write_summary(output_root: Path, results: List[Dict[str, Any]]) -> None:
    summary_json = output_root / "tmm_component_ablation_results.json"
    summary_csv = output_root / "tmm_component_ablation_table.csv"
    summary_md = output_root / "tmm_component_ablation_table.md"

    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    csv_rows: List[Dict[str, Any]] = []
    for result in results:
        metrics = _as_dict(result.get("metrics"))
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

    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        return f"{float(value):.2f}"

    md_lines = [
        "| Variant | Rank-1 Single | mAP Single | Rank-1 Cross | mAP Cross | mmAP | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in csv_rows:
        md_lines.append(
            "| {display_name} | {rank1_single} | {mAP_single} | {rank1_cross} | {mAP_cross} | {mmAP} | {status} |".format(
                display_name=row["display_name"],
                rank1_single=fmt(row["rank1_single"]),
                mAP_single=fmt(row["mAP_single"]),
                rank1_cross=fmt(row["rank1_cross"]),
                mAP_cross=fmt(row["mAP_cross"]),
                mmAP=fmt(row["mmAP"]),
                status=row["status"],
            )
        )
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ATRW fine-grained TMM component ablations")
    parser.add_argument("--config", type=str, default="config/illumination_config_atrw.yaml", help="Source ATRW config")
    parser.add_argument("--data_dir", type=str, required=True, help="ATRW train directory")
    parser.add_argument(
        "--data_root",
        type=str,
        default="orignal_data/Amur Tiger Re-identification",
        help="ATRW original data root for official open-set evaluation",
    )
    parser.add_argument("--eval_script_dir", type=str, default="ATRWEvalScript-main", help="ATRW official eval script dir")
    parser.add_argument("--output_dir", type=str, default="checkpoints/ablation/tmm_components", help="Output root")
    parser.add_argument("--device", type=str, default="cuda", help="Training device")
    parser.add_argument("--num_workers", type=int, default=4, help="Training dataloader workers")
    parser.add_argument("--backbone", type=str, default=None, help="Optional backbone override")
    parser.add_argument("--variants", type=str, default="all", help="Comma-separated variant keys or 'all'")
    parser.add_argument("--dry_run", action="store_true", help="Only materialize configs and print commands")
    args = parser.parse_args()

    output_root = (PROJECT_ROOT / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = build_tmm_component_ablation_jobs(
        config_path=args.config,
        data_dir=args.data_dir,
        output_root=output_root,
        device=args.device,
        backbone_override=args.backbone,
        num_workers=args.num_workers,
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
