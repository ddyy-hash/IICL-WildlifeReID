#!/usr/bin/env python3
"""Run the paper-oriented cross-species query-gallery ablation."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import load_config
from tools.selection_protocols import CROSS_SPECIES_SELECTION_PROTOCOL
from tools.run_atrw_main_ablation import (
    derive_naive_illumination_config,
    derive_plain_baseline_config,
)


DEFAULT_BACKBONE = "osnet_ain_x1_0"
DEFAULT_BASELINE_HEAD = "local_stripe"
DEFAULT_DATASET_ORDER = (
    "stripespotter",
    "gzgc_zebra",
    "gzgc_giraffe",
)
DEFAULT_VARIANT_ORDER = (
    "white_box_baseline",
    "generic_illumination",
    "full_model",
)
DEFAULT_SELECTION_PROTOCOL = CROSS_SPECIES_SELECTION_PROTOCOL
DEFAULT_FINAL_REPORT_PROTOCOL = "query_gallery"
DEFAULT_SELECTION_METRIC = "mAP"
DEFAULT_JOINT_PHASE_EPOCHS = {
    "phase1": 15,
    "phase2": 15,
    "phase3": 100,
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    config_path: str
    source_protocol: str = "query_gallery"
    use_official_query_gallery_for_training: bool = False


@dataclass
class CrossSpeciesJob:
    dataset_key: str
    dataset_display_name: str
    variant_key: str
    display_name: str
    train_entrypoint: str
    eval_mode: str
    train_data_dir: str
    query_dir: str
    gallery_dir: str
    config: Dict[str, Any]
    output_dir: Path
    config_path: Path
    train_log_path: Path
    eval_log_path: Path
    result_path: Path
    train_command: List[str] = field(default_factory=list)
    checkpoint_candidates: List[str] = field(default_factory=list)


DATASETS: Dict[str, DatasetSpec] = {
    "stripespotter": DatasetSpec(
        key="stripespotter",
        display_name="StripeSpotter",
        config_path="config/illumination_config_stripespotter_actual.yaml",
    ),
    "gzgc_zebra": DatasetSpec(
        key="gzgc_zebra",
        display_name="GZGC Zebra",
        config_path="config/illumination_config_gzgc_zebra_actual.yaml",
    ),
    "gzgc_giraffe": DatasetSpec(
        key="gzgc_giraffe",
        display_name="GZGC Giraffe",
        config_path="config/illumination_config_gzgc_giraffe_actual.yaml",
    ),
    "czechlynx": DatasetSpec(
        key="czechlynx",
        display_name="CzechLynx",
        config_path="config/illumination_config_czechlynx_actual.yaml",
        source_protocol="official_time_closed_split",
        use_official_query_gallery_for_training=True,
    ),
}


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _set_output_dir(config: Dict[str, Any], output_dir: Path) -> None:
    output_text = output_dir.as_posix()
    config["output_dir"] = output_text
    config.setdefault("training", {})["output_dir"] = output_text


def _apply_runtime_dirs(config: Dict[str, Any], runtime: Dict[str, str]) -> None:
    training_cfg = config.setdefault("training", {})
    training_cfg["data_dir"] = runtime["train_data_dir"]
    training_cfg["query_dir"] = runtime["query_dir"]
    training_cfg["gallery_dir"] = runtime["gallery_dir"]


def _force_bf16_amp(config: Dict[str, Any]) -> None:
    hardware_cfg = config.setdefault("hardware", {})
    hardware_cfg["use_amp"] = True
    hardware_cfg["amp_dtype"] = "bfloat16"


def _materialize_joint_phase_defaults(config: Dict[str, Any]) -> None:
    training_cfg = config.setdefault("training", {})
    phases_cfg = training_cfg.setdefault("phases", {})
    for phase_name, default_epochs in DEFAULT_JOINT_PHASE_EPOCHS.items():
        phase_cfg = phases_cfg.setdefault(phase_name, {})
        phase_cfg.setdefault("epochs", default_epochs)


def _apply_baseline_head(config: Dict[str, Any], baseline_head: str) -> None:
    model_cfg = config.setdefault("model", {})
    model_cfg.setdefault("reid_head", {})["type"] = baseline_head
    local_extractor_cfg = model_cfg.setdefault("local_extractor", {})
    if baseline_head == "plain_global":
        local_extractor_cfg["num_parts"] = 1


def _materialize_job(job: CrossSpeciesJob) -> None:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    with open(job.config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(job.config, handle, sort_keys=False, allow_unicode=True)


def _build_baseline_command(job: CrossSpeciesJob, device: str) -> List[str]:
    backbone = str(_as_dict(job.config.get("model")).get("backbone", DEFAULT_BACKBONE))
    return [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "train_baselines.py"),
        "--config",
        str(job.config_path),
        "--data_dir",
        job.train_data_dir,
        "--output_dir",
        str(job.output_dir),
        "--backbone",
        backbone,
        "--device",
        device,
    ]


def _build_joint_command(job: CrossSpeciesJob, device: str, num_workers: int) -> List[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "train_joint.py"),
        "--config",
        str(job.config_path),
        "--data_dir",
        job.train_data_dir,
        "--output_dir",
        str(job.output_dir),
        "--device",
        device,
        "--num_workers",
        str(num_workers),
    ]


def _dataset_runtime_from_config(config: Dict[str, Any], dataset_key: str) -> Dict[str, str]:
    training_cfg = _as_dict(config.get("training"))
    train_data_dir = str(training_cfg.get("data_dir", "")).strip()
    query_dir = str(training_cfg.get("query_dir", "")).strip()
    gallery_dir = str(training_cfg.get("gallery_dir", "")).strip()
    protocol = str(_as_dict(config.get("evaluation")).get("protocol", "")).strip().lower()

    if not train_data_dir or not query_dir or not gallery_dir:
        raise ValueError(f"Dataset {dataset_key} is missing train/query/gallery paths in config.")
    if protocol != "query_gallery":
        raise ValueError(f"Dataset {dataset_key} must use query_gallery protocol, got: {protocol!r}")

    return {
        "train_data_dir": train_data_dir,
        "query_dir": query_dir,
        "gallery_dir": gallery_dir,
        "protocol": protocol,
    }


def _derive_standardized_query_gallery_runtime(base_runtime: Dict[str, str], dataset_key: str) -> Dict[str, str]:
    dataset_root = Path(base_runtime["train_data_dir"]).parent
    selection_runtime = _ensure_selection_query_gallery_split(dataset_root, dataset_key=dataset_key)
    return {
        "train_data_dir": selection_runtime["selection_train_dir"],
        "query_dir": base_runtime["query_dir"],
        "gallery_dir": base_runtime["gallery_dir"],
        "selection_query_dir": selection_runtime["selection_query_dir"],
        "selection_gallery_dir": selection_runtime["selection_gallery_dir"],
        "selection_info": selection_runtime["selection_info"],
        "protocol": base_runtime["protocol"],
    }


def _list_identity_images(identity_dir: Path) -> List[str]:
    return sorted(
        path.name
        for path in identity_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_identity_files(src_identity_dir: Path, dst_root: Path, identity: str, filenames: Sequence[str]) -> None:
    if not filenames:
        return
    dst_identity_dir = dst_root / identity
    dst_identity_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        shutil.copy2(src_identity_dir / filename, dst_identity_dir / filename)


def _ensure_selection_query_gallery_split(dataset_root: Path, dataset_key: str, seed: int = 42) -> Dict[str, str]:
    train_root = dataset_root / "train"
    selection_train_root = dataset_root / "selection_train"
    selection_query_root = dataset_root / "selection_query"
    selection_gallery_root = dataset_root / "selection_gallery"
    selection_info_path = dataset_root / "selection_info.json"

    if (
        selection_info_path.exists()
        and selection_train_root.exists()
        and selection_query_root.exists()
        and selection_gallery_root.exists()
    ):
        return {
            "selection_train_dir": selection_train_root.as_posix(),
            "selection_query_dir": selection_query_root.as_posix(),
            "selection_gallery_dir": selection_gallery_root.as_posix(),
            "selection_info": selection_info_path.as_posix(),
        }

    if not train_root.exists():
        return {
            "selection_train_dir": selection_train_root.as_posix(),
            "selection_query_dir": selection_query_root.as_posix(),
            "selection_gallery_dir": selection_gallery_root.as_posix(),
            "selection_info": selection_info_path.as_posix(),
        }

    _reset_dir(selection_train_root)
    _reset_dir(selection_query_root)
    _reset_dir(selection_gallery_root)

    eligible_ids = 0
    total_train_images = 0
    total_query_images = 0
    total_gallery_images = 0

    for identity_dir in sorted(path for path in train_root.iterdir() if path.is_dir()):
        filenames = _list_identity_images(identity_dir)
        ordered = list(filenames)

        if len(ordered) >= 8:
            eligible_ids += 1
            query_files = ordered[:2]
            gallery_files = ordered[-3:]
            train_files = ordered[2:-3]
        elif len(ordered) >= 5:
            eligible_ids += 1
            query_files = ordered[:1]
            gallery_files = ordered[-2:]
            train_files = ordered[1:-2]
        else:
            query_files = []
            gallery_files = []
            train_files = ordered

        _copy_identity_files(identity_dir, selection_train_root, identity_dir.name, train_files)
        _copy_identity_files(identity_dir, selection_query_root, identity_dir.name, query_files)
        _copy_identity_files(identity_dir, selection_gallery_root, identity_dir.name, gallery_files)

        total_train_images += len(train_files)
        total_query_images += len(query_files)
        total_gallery_images += len(gallery_files)

    selection_info = {
        "dataset": dataset_key,
        "protocol": DEFAULT_SELECTION_PROTOCOL,
        "seed": seed,
        "source_train_dir": train_root.as_posix(),
        "selection_train_dir": selection_train_root.as_posix(),
        "selection_query_dir": selection_query_root.as_posix(),
        "selection_gallery_dir": selection_gallery_root.as_posix(),
        "eligible_ids": eligible_ids,
        "policy": {
            "ordering": "sequence_aware_filename_order",
            "small_identity_rule": "ids_with_fewer_than_5_images_stay_train_only",
            "medium_identity_rule": "5_to_7_images -> 1 query, 2 gallery, rest train",
            "large_identity_rule": "8+ images -> 2 query, 3 gallery, rest train",
        },
        "stats": {
            "train_imgs": total_train_images,
            "query_imgs": total_query_images,
            "gallery_imgs": total_gallery_images,
        },
    }
    with open(selection_info_path, "w", encoding="utf-8") as handle:
        json.dump(selection_info, handle, indent=2, ensure_ascii=False)

    return {
        "selection_train_dir": selection_train_root.as_posix(),
        "selection_query_dir": selection_query_root.as_posix(),
        "selection_gallery_dir": selection_gallery_root.as_posix(),
        "selection_info": selection_info_path.as_posix(),
    }


def _normalize_training_protocol(config: Dict[str, Any], runtime: Dict[str, str]) -> None:
    evaluation_cfg = config.setdefault("evaluation", {})
    if str(evaluation_cfg.get("protocol", "")).strip().lower() == "query_gallery":
        evaluation_cfg["protocol"] = DEFAULT_SELECTION_PROTOCOL
    evaluation_cfg["selection_query_dir"] = runtime["selection_query_dir"]
    evaluation_cfg["selection_gallery_dir"] = runtime["selection_gallery_dir"]
    evaluation_cfg["selection_info"] = runtime["selection_info"]
    evaluation_cfg["best_metric"] = DEFAULT_SELECTION_METRIC
    evaluation_cfg["strict_protocol_check"] = True


def _stamp_paper_protocol_metadata(
    config: Dict[str, Any],
    *,
    source_dataset_protocol: str,
    training_selection_protocol: str,
    final_report_protocol: str,
    selection_metric: str,
    official_protocol: bool = False,
    note: Optional[str] = None,
) -> None:
    config["paper_protocol"] = {
        "source_dataset_protocol": source_dataset_protocol,
        "training_selection_protocol": training_selection_protocol,
        "final_report_protocol": final_report_protocol,
        "selection_metric": selection_metric,
        "official_protocol": bool(official_protocol),
        "note": note
        or (
            "Training uses an internal held-out train query/gallery protocol for checkpoint choice; "
            "final reported numbers come from the dataset's standardized fixed query/gallery split."
        ),
    }


def derive_generic_illumination_config(
    full_config: Dict[str, Any],
    backbone_override: Optional[str] = None,
    baseline_head: str = DEFAULT_BASELINE_HEAD,
) -> Dict[str, Any]:
    derived = derive_naive_illumination_config(full_config, backbone_override=backbone_override)
    _materialize_joint_phase_defaults(derived)
    _apply_baseline_head(derived, baseline_head)
    derived.setdefault("baseline", {})["type"] = "generic_illumination"
    derived["baseline"]["display_name"] = "Generic Illumination"
    return derived


def _joint_total_epochs(config: Dict[str, Any]) -> int:
    training_cfg = config.setdefault("training", {})
    phases_cfg = training_cfg.setdefault("phases", {})
    return sum(
        int(_as_dict(phases_cfg.get(phase_name)).get("epochs", 0))
        for phase_name in ("phase1", "phase2", "phase3")
    )


def derive_cross_species_simplified_baseline_config(
    full_config: Dict[str, Any],
    backbone_override: Optional[str] = None,
    baseline_head: str = DEFAULT_BASELINE_HEAD,
) -> Dict[str, Any]:
    source_config = copy.deepcopy(_as_dict(full_config))
    _materialize_joint_phase_defaults(source_config)
    effective_total_epochs = _joint_total_epochs(source_config)
    derived = derive_plain_baseline_config(
        source_config,
        backbone_override=backbone_override,
        total_epochs=effective_total_epochs,
        baseline_head=baseline_head,
    )
    derived.setdefault("baseline", {})["display_name"] = "Simplified Baseline"
    return derived


def _derive_full_model_config(
    full_config: Dict[str, Any],
    backbone_override: Optional[str] = None,
    baseline_head: str = DEFAULT_BASELINE_HEAD,
) -> Dict[str, Any]:
    derived = copy.deepcopy(_as_dict(full_config))
    if backbone_override:
        derived.setdefault("model", {})["backbone"] = backbone_override
    _materialize_joint_phase_defaults(derived)
    _apply_baseline_head(derived, baseline_head)
    return derived


def build_cross_species_dataset_jobs(
    dataset_key: str,
    output_root: Path,
    device: str,
    backbone_override: str = DEFAULT_BACKBONE,
    num_workers: int = 4,
    baseline_head: str = DEFAULT_BASELINE_HEAD,
) -> List[CrossSpeciesJob]:
    if dataset_key not in DATASETS:
        raise KeyError(f"Unknown dataset: {dataset_key}")

    dataset = DATASETS[dataset_key]
    full_cfg = load_config(dataset.config_path)
    _materialize_joint_phase_defaults(full_cfg)
    base_runtime = _dataset_runtime_from_config(full_cfg, dataset_key)
    if dataset.use_official_query_gallery_for_training:
        runtime = dict(base_runtime)
    else:
        runtime = _derive_standardized_query_gallery_runtime(base_runtime, dataset_key)
    jobs: List[CrossSpeciesJob] = []

    variant_defs = [
        (
            "white_box_baseline",
            "Simplified Baseline",
            derive_cross_species_simplified_baseline_config(
                full_cfg,
                backbone_override=backbone_override,
                baseline_head=baseline_head,
            ),
            (PROJECT_ROOT / "tools" / "train_baselines.py").as_posix(),
            ["baseline_best.pth", "baseline_best_reid_best.pth"],
        ),
        (
            "generic_illumination",
            "Generic Illumination",
            derive_generic_illumination_config(
                full_cfg,
                backbone_override=backbone_override,
                baseline_head=baseline_head,
            ),
            (PROJECT_ROOT / "tools" / "train_joint.py").as_posix(),
            ["joint_best.pth", "joint_best_reid_best.pth"],
        ),
        (
            "full_model",
            "Full Model",
            _derive_full_model_config(
                full_cfg,
                backbone_override=backbone_override,
                baseline_head=baseline_head,
            ),
            (PROJECT_ROOT / "tools" / "train_joint.py").as_posix(),
            ["joint_best.pth", "joint_best_reid_best.pth"],
        ),
    ]

    for variant_key, display_name, config, entrypoint, checkpoint_candidates in variant_defs:
        if dataset.use_official_query_gallery_for_training:
            evaluation_cfg = config.setdefault("evaluation", {})
            evaluation_cfg["protocol"] = DEFAULT_FINAL_REPORT_PROTOCOL
            evaluation_cfg["best_metric"] = DEFAULT_SELECTION_METRIC
            evaluation_cfg["strict_protocol_check"] = True
        else:
            _normalize_training_protocol(config, runtime)
        _force_bf16_amp(config)
        _apply_runtime_dirs(config, runtime)
        _stamp_paper_protocol_metadata(
            config,
            source_dataset_protocol=dataset.source_protocol,
            training_selection_protocol=str(_as_dict(config.get("evaluation")).get("protocol", "")),
            final_report_protocol=DEFAULT_FINAL_REPORT_PROTOCOL,
            selection_metric=str(_as_dict(config.get("evaluation")).get("best_metric", DEFAULT_SELECTION_METRIC)),
            official_protocol=dataset.use_official_query_gallery_for_training,
            note=(
                "Training and final evaluation both follow the official CzechLynx time-closed split; "
                "query/gallery are loaded from the repository's fixed conversion of the official test partition."
                if dataset.use_official_query_gallery_for_training
                else None
            ),
        )
        if "selection_info" in runtime:
            config["paper_protocol"]["selection_info"] = runtime["selection_info"]
        output_dir = output_root / dataset_key / variant_key
        _set_output_dir(config, output_dir)
        job = CrossSpeciesJob(
            dataset_key=dataset_key,
            dataset_display_name=dataset.display_name,
            variant_key=variant_key,
            display_name=display_name,
            train_entrypoint=entrypoint,
            eval_mode="query_gallery",
            train_data_dir=runtime["train_data_dir"],
            query_dir=runtime["query_dir"],
            gallery_dir=runtime["gallery_dir"],
            config=config,
            output_dir=output_dir,
            config_path=output_dir / "derived_config.yaml",
            train_log_path=output_dir / "train.log",
            eval_log_path=output_dir / "eval.log",
            result_path=output_dir / "result.json",
            checkpoint_candidates=checkpoint_candidates,
        )
        if entrypoint.endswith("train_baselines.py"):
            job.train_command = _build_baseline_command(job, device=device)
        else:
            job.train_command = _build_joint_command(job, device=device, num_workers=num_workers)
        jobs.append(job)

    return jobs


def _parse_datasets(requested: str) -> List[str]:
    if requested.strip().lower() == "all":
        return list(DEFAULT_DATASET_ORDER)
    selected = [item.strip() for item in requested.split(",") if item.strip()]
    unknown = [item for item in selected if item not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Expected subset of {DEFAULT_DATASET_ORDER}")
    return selected


def _parse_variants(requested: str) -> List[str]:
    if requested.strip().lower() == "all":
        return list(DEFAULT_VARIANT_ORDER)
    selected = [item.strip() for item in requested.split(",") if item.strip()]
    unknown = [item for item in selected if item not in DEFAULT_VARIANT_ORDER]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Expected subset of {DEFAULT_VARIANT_ORDER}")
    return selected


def _run_logged_command(command: Sequence[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        process = subprocess.run(
            list(command),
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(process.returncode)


def _find_checkpoint(job: CrossSpeciesJob) -> Optional[Path]:
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


def _cleanup_extra_checkpoints(output_dir: Path, keep_paths: Sequence[Path]) -> int:
    keep_resolved = {path.resolve() for path in keep_paths if path.exists()}
    removed = 0
    for checkpoint_path in output_dir.rglob("*.pth"):
        if checkpoint_path.resolve() in keep_resolved:
            continue
        checkpoint_path.unlink(missing_ok=True)
        removed += 1
    return removed


def _parse_eval_metrics(eval_text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    patterns = {
        "rank1": r"Rank-1\s*:\s*([0-9.]+)%",
        "rank5": r"Rank-5\s*:\s*([0-9.]+)%",
        "rank10": r"Rank-10\s*:\s*([0-9.]+)%",
        "mAP": r"mAP\s*:\s*([0-9.]+)%",
        "rank1_seen": r"Rank-1 Seen\s*:\s*([0-9.]+)%",
        "rank1_unseen": r"Rank-1 Unseen\s*:\s*([0-9.]+)%",
        "mAP_seen": r"mAP Seen\s*:\s*([0-9.]+)%",
        "mAP_unseen": r"mAP Unseen\s*:\s*([0-9.]+)%",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, eval_text)
        if match:
            metrics[key] = float(match.group(1))
    if "rank1" not in metrics or "mAP" not in metrics:
        return {}
    metrics.setdefault("rank5", 0.0)
    metrics.setdefault("rank10", 0.0)
    return metrics


def _evaluate_job(
    job: CrossSpeciesJob,
    checkpoint_path: Path,
    device: str,
) -> Dict[str, Any]:
    eval_command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "evaluate_reid.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--query_dir",
        job.query_dir,
        "--gallery_dir",
        job.gallery_dir,
        "--device",
        device,
    ]
    if job.variant_key == "white_box_baseline":
        eval_command.append("--baseline")

    return_code = _run_logged_command(eval_command, log_path=job.eval_log_path, cwd=PROJECT_ROOT)
    eval_text = job.eval_log_path.read_text(encoding="utf-8", errors="replace")
    metrics = _parse_eval_metrics(eval_text)
    paper_protocol = _as_dict(job.config.get("paper_protocol"))
    eval_cfg = _as_dict(job.config.get("evaluation"))
    feature_cfg = _as_dict(eval_cfg.get("feature_extraction"))
    return {
        "return_code": return_code,
        "command": " ".join(eval_command),
        "metrics": metrics,
        "protocol": {
            "source_dataset_protocol": paper_protocol.get("source_dataset_protocol", DEFAULT_FINAL_REPORT_PROTOCOL),
            "derived_from_dataset_protocol": paper_protocol.get("derived_from_dataset_protocol"),
            "training_selection_protocol": paper_protocol.get("training_selection_protocol", eval_cfg.get("protocol")),
            "final_report_protocol": paper_protocol.get("final_report_protocol", DEFAULT_FINAL_REPORT_PROTOCOL),
            "selection_metric": paper_protocol.get("selection_metric", eval_cfg.get("best_metric", DEFAULT_SELECTION_METRIC)),
            "flip_test": bool(feature_cfg.get("flip_test", eval_cfg.get("flip_test", True))),
            "rerank": False,
            "official_protocol": bool(paper_protocol.get("official_protocol", False)),
            "note": paper_protocol.get("note"),
        },
    }


def _fmt_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return "-"


def _write_summary(output_root: Path, results: List[Dict[str, Any]]) -> None:
    summary_json = output_root / "cross_species_paper_ablation_results.json"
    summary_csv = output_root / "cross_species_paper_ablation_table.csv"
    summary_md = output_root / "cross_species_paper_ablation_table.md"

    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    csv_rows: List[Dict[str, Any]] = []
    for result in results:
        metrics = result.get("metrics", {})
        csv_rows.append(
            {
                "dataset": result.get("dataset_key"),
                "dataset_display_name": result.get("dataset_display_name"),
                "variant": result.get("variant_key"),
                "display_name": result.get("display_name"),
                "rank1": metrics.get("rank1"),
                "mAP": metrics.get("mAP"),
                "rank1_seen": metrics.get("rank1_seen"),
                "mAP_seen": metrics.get("mAP_seen"),
                "rank1_unseen": metrics.get("rank1_unseen"),
                "mAP_unseen": metrics.get("mAP_unseen"),
                "selection_protocol": _as_dict(result.get("protocol")).get("training_selection_protocol"),
                "final_protocol": _as_dict(result.get("protocol")).get("final_report_protocol"),
                "selection_metric": _as_dict(result.get("protocol")).get("selection_metric"),
                "official_protocol": _as_dict(result.get("protocol")).get("official_protocol"),
                "status": result.get("status"),
            }
        )

    with open(summary_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "dataset_display_name",
                "variant",
                "display_name",
                "rank1",
                "mAP",
                "rank1_seen",
                "mAP_seen",
                "rank1_unseen",
                "mAP_unseen",
                "selection_protocol",
                "final_protocol",
                "selection_metric",
                "official_protocol",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    md_lines = [
        "# Cross-species Paper Ablation",
        "",
        "| Dataset | Variant | Rank-1 | mAP | Seen R1 | Seen mAP | Unseen R1 | Unseen mAP | Protocol | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in csv_rows:
        md_lines.append(
            "| {dataset} | {variant} | {rank1} | {mAP} | {rank1_seen} | {mAP_seen} | {rank1_unseen} | {mAP_unseen} | {protocol} | {status} |".format(
                dataset=row["dataset_display_name"],
                variant=row["display_name"],
                rank1=_fmt_metric(row["rank1"]),
                mAP=_fmt_metric(row["mAP"]),
                rank1_seen=_fmt_metric(row["rank1_seen"]),
                mAP_seen=_fmt_metric(row["mAP_seen"]),
                rank1_unseen=_fmt_metric(row["rank1_unseen"]),
                mAP_unseen=_fmt_metric(row["mAP_unseen"]),
                protocol="fixed query/gallery" if row["final_protocol"] == "query_gallery" else str(row["final_protocol"]),
                status=row["status"],
            )
        )
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper-oriented cross-species fixed query-gallery ablation")
    parser.add_argument("--datasets", type=str, default="all", help="Comma-separated dataset keys or 'all'")
    parser.add_argument("--variants", type=str, default="all", help="Comma-separated variant keys or 'all'")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/ablation/cross_species_paper",
        help="Ablation output root",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Training / eval device")
    parser.add_argument("--num_workers", type=int, default=4, help="Training dataloader workers")
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE, help="Fixed paper backbone")
    parser.add_argument(
        "--baseline_head",
        type=str,
        default=DEFAULT_BASELINE_HEAD,
        help="White-box baseline head type",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only materialize configs and print commands")
    args = parser.parse_args()

    selected_datasets = _parse_datasets(args.datasets)
    selected_variants = set(_parse_variants(args.variants))
    output_root = (PROJECT_ROOT / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for dataset_key in selected_datasets:
        jobs = build_cross_species_dataset_jobs(
            dataset_key=dataset_key,
            output_root=output_root,
            device=args.device,
            backbone_override=args.backbone,
            num_workers=args.num_workers,
            baseline_head=args.baseline_head,
        )
        jobs = [job for job in jobs if job.variant_key in selected_variants]

        for job in jobs:
            _materialize_job(job)
            if args.dry_run:
                print(f"[Dry Run] [{job.dataset_key}] {job.display_name}")
                print("  Train:", " ".join(job.train_command))
                continue

            train_code = _run_logged_command(job.train_command, log_path=job.train_log_path, cwd=PROJECT_ROOT)
            result_record: Dict[str, Any] = {
                "dataset_key": job.dataset_key,
                "dataset_display_name": job.dataset_display_name,
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
            eval_result = _evaluate_job(job, checkpoint_path=checkpoint_path, device=args.device)
            result_record["eval_log"] = str(job.eval_log_path)
            result_record["eval_command"] = eval_result["command"]
            result_record["metrics"] = eval_result["metrics"]
            result_record["protocol"] = eval_result["protocol"]
            result_record["eval_return_code"] = eval_result["return_code"]
            if eval_result["return_code"] != 0:
                result_record["status"] = "eval_failed"
            elif not eval_result["metrics"]:
                result_record["status"] = "metrics_missing"
            else:
                result_record["status"] = "ok"

            result_record["removed_checkpoints"] = _cleanup_extra_checkpoints(
                job.output_dir,
                keep_paths=[checkpoint_path],
            )
            with open(job.result_path, "w", encoding="utf-8") as handle:
                json.dump(result_record, handle, ensure_ascii=False, indent=2)
            results.append(result_record)

    if not args.dry_run:
        _write_summary(output_root, results)


if __name__ == "__main__":
    main()
