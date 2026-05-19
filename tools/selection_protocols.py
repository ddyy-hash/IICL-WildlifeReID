#!/usr/bin/env python3
"""Lightweight helpers for custom selection protocols."""

from __future__ import annotations

from typing import Any, Dict


CROSS_SPECIES_SELECTION_PROTOCOL = "self_defined_train_qg"
QUERY_GALLERY_PROTOCOL = "query_gallery"


def resolve_selection_query_gallery_eval_spec(eval_cfg: Dict[str, Any]) -> Dict[str, str]:
    query_dir = str(eval_cfg.get("selection_query_dir", "")).strip()
    gallery_dir = str(eval_cfg.get("selection_gallery_dir", "")).strip()
    info_path = str(eval_cfg.get("selection_info", "")).strip()

    if not query_dir or not gallery_dir:
        raise ValueError(
            "selection_query_dir and selection_gallery_dir must be provided "
            "for self_defined_train_qg evaluation."
        )

    return {
        "query_dir": query_dir,
        "gallery_dir": gallery_dir,
        "info_path": info_path,
        "exclude_same_camera": False,
        "force_standard_eval": True,
    }


def resolve_official_query_gallery_eval_spec(eval_cfg: Dict[str, Any]) -> Dict[str, str]:
    query_dir = str(eval_cfg.get("query_dir", "")).strip()
    gallery_dir = str(eval_cfg.get("gallery_dir", "")).strip()
    feature_cfg = eval_cfg.get("feature_extraction", {}) if isinstance(eval_cfg, dict) else {}
    exclude_same_camera = bool(
        feature_cfg.get("exclude_same_camera", eval_cfg.get("exclude_same_camera", True))
    )

    if not query_dir or not gallery_dir:
        raise ValueError(
            "evaluation.query_dir and evaluation.gallery_dir must be provided "
            "for query_gallery evaluation."
        )

    return {
        "query_dir": query_dir,
        "gallery_dir": gallery_dir,
        "info_path": "",
        "exclude_same_camera": exclude_same_camera,
        "force_standard_eval": True,
    }
