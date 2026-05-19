#!/usr/bin/env python3
"""Inspect parameter-name and shape compatibility between a model and a checkpoint."""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, ".")

from app.core.joint_model import JointReIDModel


def check_match(ckpt_path: str, backbone: str = "osnet_ain_x1_0", num_classes: int = 107) -> tuple[int, int]:
    """Print a compact compatibility report for the provided checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_state = checkpoint.get("model_state_dict", checkpoint)

    model = JointReIDModel(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained_backbone=False,
    )
    model_state = model.state_dict()

    print(f"Checkpoint parameters: {len(ckpt_state)}")
    print(f"Model parameters:      {len(model_state)}")

    matched: list[str] = []
    ckpt_only: list[str] = []
    model_only: list[str] = []
    shape_mismatch: list[tuple[str, torch.Size, torch.Size]] = []

    for key in ckpt_state:
        if key in model_state:
            if ckpt_state[key].shape == model_state[key].shape:
                matched.append(key)
            else:
                shape_mismatch.append((key, ckpt_state[key].shape, model_state[key].shape))
        else:
            ckpt_only.append(key)

    for key in model_state:
        if key not in ckpt_state:
            model_only.append(key)

    print("\n=== Compatibility summary ===")
    print(f"Exact matches:         {len(matched)}")
    print(f"Checkpoint-only keys:  {len(ckpt_only)}")
    print(f"Model-only keys:       {len(model_only)}")
    print(f"Shape mismatches:      {len(shape_mismatch)}")

    if ckpt_only:
        print("\n=== Keys present only in the checkpoint (first 10) ===")
        for key in ckpt_only[:10]:
            print(f"  {key}")

    if model_only:
        print("\n=== Keys present only in the model (first 10) ===")
        for key in model_only[:10]:
            print(f"  {key}")

    if shape_mismatch:
        print("\n=== Shape mismatches ===")
        for key, ckpt_shape, model_shape in shape_mismatch:
            print(f"  {key}: checkpoint {ckpt_shape} vs model {model_shape}")

    return len(matched), len(ckpt_state)


if __name__ == "__main__":
    checkpoint_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "checkpoints/joint_atrw_optimized/joint_best_reid_best.pth"
    )
    check_match(checkpoint_path)
