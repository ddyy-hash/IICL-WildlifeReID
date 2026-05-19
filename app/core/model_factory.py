"""Helpers for reconstructing JointReIDModel from config/checkpoints."""

from __future__ import annotations

import copy
from typing import Any, Dict, Tuple


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def extract_config_from_checkpoint(checkpoint: Any) -> Dict[str, Any]:
    """Return embedded config dict from a checkpoint-like object."""
    if isinstance(checkpoint, dict):
        return _as_dict(checkpoint.get("config"))
    return {}


def resolve_joint_model_init(
    config: Dict[str, Any],
    num_classes: int,
    backbone_override: str | None = None,
    pretrained_backbone: bool = False,
) -> Dict[str, Any]:
    """Resolve JointReIDModel init kwargs from merged config."""
    config = _as_dict(config)
    model_cfg = _as_dict(config.get("model"))
    aug_cfg = _as_dict(config.get("data_augmentation"))
    illum_top_cfg = _as_dict(config.get("illumination_module"))
    illum_model_cfg = _as_dict(model_cfg.get("illumination_module"))
    local_cfg = _as_dict(model_cfg.get("local_extractor"))
    feature_fusion_cfg = _as_dict(model_cfg.get("feature_fusion"))
    branch_attention_cfg = _as_dict(model_cfg.get("branch_attention_fusion"))
    nuisance_head_cfg = _as_dict(model_cfg.get("nuisance_head"))
    reid_head_cfg = _as_dict(model_cfg.get("reid_head"))
    hardware_cfg = _as_dict(config.get("hardware"))

    if "enabled" in illum_model_cfg:
        use_ipaid = bool(illum_model_cfg.get("enabled", True))
    else:
        module_type = str(illum_top_cfg.get("module_type", "IPAIDModule")).lower()
        use_ipaid = module_type not in {"none", "disabled", "null"}

    module_params: Dict[str, Any] = {}
    module_params.update(copy.deepcopy(_as_dict(illum_top_cfg.get("module_params"))))
    module_params.update(copy.deepcopy(_as_dict(illum_model_cfg.get("module_params"))))

    module_params["_feature_fusion"] = copy.deepcopy(feature_fusion_cfg)
    module_params["_branch_attention_fusion"] = copy.deepcopy(branch_attention_cfg)
    module_params["_nuisance_head"] = copy.deepcopy(nuisance_head_cfg)
    module_params["_reid_head"] = copy.deepcopy(reid_head_cfg)
    module_params["_backbone_random_erasing"] = copy.deepcopy(_as_dict(aug_cfg.get("random_erasing")))

    return {
        "num_classes": int(num_classes),
        "backbone_name": backbone_override or model_cfg.get("backbone", "osnet_ain_x1_0"),
        "num_stripes": int(local_cfg.get("num_parts", 6)),
        "pretrained_backbone": bool(pretrained_backbone),
        "soft_mask_temperature": float(model_cfg.get("soft_mask_temperature", 10.0)),
        "soft_mask_type": str(model_cfg.get("soft_mask_type", "sigmoid")),
        "use_ipaid": use_ipaid,
        "dropout": float(local_cfg.get("dropout", 0.0)),
        "use_backbone_checkpointing": bool(hardware_cfg.get("use_backbone_checkpointing", True)),
        "ipaid_params": module_params,
    }


def resolve_eval_input_size(config: Dict[str, Any]) -> Tuple[int, int]:
    """Resolve evaluation input size from config."""
    config = _as_dict(config)
    training_cfg = _as_dict(config.get("training"))
    default_size = int(training_cfg.get("image_size", 256))
    img_h = int(training_cfg.get("image_height", default_size))
    img_w = int(training_cfg.get("image_width", default_size))
    return img_h, img_w
