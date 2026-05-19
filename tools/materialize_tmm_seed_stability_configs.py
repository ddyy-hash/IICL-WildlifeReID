#!/usr/bin/env python3
"""Materialize fixed-config seed-stability runs for TMM evidence."""

from __future__ import annotations

import argparse
import copy
import csv
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

from tools.run_atrw_main_ablation import derive_plain_baseline_config


RUN_ID = "tmm_missing_evidence_20260517"


DATASETS: Dict[str, Dict[str, str]] = {
    "atrw": {
        "config": "config/illumination_config_atrw.yaml",
        "data_dir": "data/processed/atrw/train",
        "query_dir": "data/processed/atrw/query",
        "gallery_dir": "data/processed/atrw/gallery",
    },
    "gzgc_zebra": {
        "config": "config/tmm_formal48g/illumination_config_gzgc_zebra_tmm48g_primary.yaml",
        "data_dir": "data/processed/gzgc_zebra/train",
        "query_dir": "data/processed/gzgc_zebra/query",
        "gallery_dir": "data/processed/gzgc_zebra/gallery",
    },
    "leopard": {
        "config": "config/tmm_finalbestopt_20260512/illumination_config_leopard_low_wd_smooth.yaml",
        "data_dir": "data/processed/leopard/train",
        "query_dir": "data/processed/leopard/query",
        "gallery_dir": "data/processed/leopard/gallery",
    },
    "whaleshark": {
        "config": "config/tmm_formal48g/illumination_config_whaleshark_tmm48g_primary.yaml",
        "data_dir": "data/processed/whaleshark/train",
        "query_dir": "data/processed/whaleshark/query",
        "gallery_dir": "data/processed/whaleshark/gallery",
    },
}


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def set_nested(cfg: Dict[str, Any], keys: Sequence[str], value: Any) -> None:
    cursor = cfg
    for key in keys[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value
    cursor[keys[-1]] = value


def set_seed_and_paths(
    cfg: Dict[str, Any],
    dataset: str,
    variant: str,
    seed: int,
    spec: Dict[str, str],
    run_root: str,
) -> Dict[str, Any]:
    output_dir = f"{run_root}/{dataset}/{variant}_seed{seed}"
    set_nested(cfg, ["seed"], int(seed))
    set_nested(cfg, ["experiment", "seed"], int(seed))
    set_nested(cfg, ["training", "seed"], int(seed))
    set_nested(cfg, ["training", "data_dir"], spec["data_dir"])
    set_nested(cfg, ["training", "query_dir"], spec["query_dir"])
    set_nested(cfg, ["training", "gallery_dir"], spec["gallery_dir"])
    set_nested(cfg, ["training", "output_dir"], output_dir)
    set_nested(cfg, ["output_dir"], output_dir)
    set_nested(cfg, ["output", "checkpoint_dir"], output_dir)
    set_nested(cfg, ["output", "log_dir"], output_dir)
    set_nested(cfg, ["hardware", "num_workers"], 4)
    set_nested(cfg, ["hardware", "use_amp"], True)
    set_nested(cfg, ["hardware", "amp_dtype"], "bfloat16")
    set_nested(cfg, ["checkpointing", "save_best_only"], False)
    set_nested(cfg, ["tmm_seed_stability_20260517"], {
        "dataset": dataset,
        "variant": variant,
        "seed": int(seed),
        "fixed_config": True,
    })

    if dataset != "atrw":
        set_nested(cfg, ["evaluation", "protocol"], "query_gallery")
        set_nested(cfg, ["evaluation", "query_dir"], spec["query_dir"])
        set_nested(cfg, ["evaluation", "gallery_dir"], spec["gallery_dir"])
        set_nested(cfg, ["evaluation", "best_metric"], "mAP")
        set_nested(cfg, ["evaluation", "strict_protocol_check"], True)
        set_nested(cfg, ["evaluation", "additional_protocols"], [])
        set_nested(cfg, ["evaluation", "feature_extraction", "exclude_same_camera"], False)

    return cfg


def save_yaml(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def materialize(
    out_dir: Path,
    run_root: str,
    seeds: Sequence[int],
    datasets: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset key: {dataset}")
        spec = DATASETS[dataset]
        full_cfg = load_yaml(PROJECT_ROOT / spec["config"])
        variants = {
            "plain": derive_plain_baseline_config(full_cfg, baseline_head="plain_global"),
            "full": copy.deepcopy(full_cfg),
        }
        for variant, cfg in variants.items():
            for seed in seeds:
                cfg_seed = set_seed_and_paths(
                    copy.deepcopy(cfg),
                    dataset=dataset,
                    variant=variant,
                    seed=int(seed),
                    spec=spec,
                    run_root=run_root,
                )
                config_path = out_dir / dataset / variant / f"seed{seed}.yaml"
                save_yaml(config_path, cfg_seed)
                rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "seed": int(seed),
                        "config": project_relative(config_path),
                        "data_dir": spec["data_dir"],
                        "query_dir": spec["query_dir"],
                        "gallery_dir": spec["gallery_dir"],
                        "output_dir": cfg_seed["output_dir"],
                    }
                )
    return rows


def write_manifest(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "variant", "seed", "config", "data_dir", "query_dir", "gallery_dir", "output_dir"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=str, default="config/tmm_seed_stability_20260517")
    parser.add_argument("--run_root", type=str, default="checkpoints/tmm_seed_stability_20260517")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--datasets", type=str, default="atrw,gzgc_zebra,leopard,whaleshark")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    out_dir = PROJECT_ROOT / args.out_dir
    rows = materialize(out_dir=out_dir, run_root=args.run_root, seeds=seeds, datasets=datasets)
    manifest_path = out_dir / "seed_stability_manifest.tsv"
    write_manifest(manifest_path, rows)
    summary = {
        "run_id": RUN_ID,
        "seeds": seeds,
        "datasets": datasets,
        "num_configs": len(rows),
        "manifest": project_relative(manifest_path),
    }
    (out_dir / "seed_stability_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
