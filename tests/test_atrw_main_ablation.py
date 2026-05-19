from pathlib import Path

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import load_config
from tools.run_atrw_main_ablation import (
    build_atrw_main_ablation_jobs,
    derive_illumination_only_config,
    derive_naive_illumination_config,
    derive_plain_baseline_config,
)


ATRW_CONFIG = "config/illumination_config_atrw.yaml"


def test_derive_plain_baseline_config_builds_lowerplain_baseline() -> None:
    full_cfg = load_config(ATRW_CONFIG)

    plain_cfg = derive_plain_baseline_config(full_cfg)

    assert plain_cfg["model"]["illumination_module"]["enabled"] is False
    assert plain_cfg["model"]["feature_fusion"]["enabled"] is False
    assert plain_cfg["model"]["branch_attention_fusion"]["enabled"] is False
    assert plain_cfg["model"]["nuisance_head"]["enabled"] is False
    assert plain_cfg["model"]["reid_head"]["type"] == "plain_global"
    assert plain_cfg["model"]["local_extractor"]["num_parts"] == 1
    assert plain_cfg["training"]["pk_sampler"]["enabled"] is False
    assert plain_cfg["training"]["center_loss"]["enabled"] is False
    assert plain_cfg["training"]["center_loss"]["weight"] == 0.0
    assert plain_cfg["training"]["metric_learning"]["ce_loss"]["weight"] == 1.0
    assert plain_cfg["training"]["metric_learning"]["ce_loss"]["label_smoothing"] == 0.0
    assert plain_cfg["training"]["metric_learning"]["triplet_loss"]["weight"] == 0.0
    assert plain_cfg["training"]["metric_learning"]["arcface_loss"]["weight"] == 0.0
    assert plain_cfg["training"]["metric_learning"]["circle_loss"]["weight"] == 0.0
    assert plain_cfg["evaluation"]["flip_test"] is False
    assert plain_cfg["evaluation"]["feature_extraction"]["flip_test"] is False
    assert plain_cfg["training"]["iicl"]["enabled"] is False
    assert plain_cfg["training"]["cross_light_softap"]["enabled"] is False
    assert plain_cfg["training"]["teacher_manifold"]["enabled"] is False
    assert plain_cfg["training"]["ranking_topology"]["enabled"] is False
    assert plain_cfg["training"]["anisotropic_identity_protection"]["enabled"] is False
    assert plain_cfg["data_augmentation"]["random_erasing"]["enabled"] is False
    assert plain_cfg["data_augmentation"]["color_jitter"]["enabled"] is False
    assert plain_cfg["data_augmentation"]["color_jitter"]["brightness"] == 0.0
    assert plain_cfg["data_augmentation"]["color_jitter"]["contrast"] == 0.0
    assert plain_cfg["data_augmentation"]["color_jitter"]["saturation"] == 0.0
    assert plain_cfg["data_augmentation"]["color_jitter"]["hue"] == 0.0
    assert plain_cfg["training"]["phases"]["phase1"]["epochs"] == 81
    assert plain_cfg["training"]["phases"]["phase2"]["epochs"] == 0
    assert plain_cfg["training"]["phases"]["phase3"]["epochs"] == 0


def test_derive_plain_baseline_config_can_preserve_stripe_head_when_requested() -> None:
    full_cfg = load_config(ATRW_CONFIG)

    stripe_cfg = derive_plain_baseline_config(full_cfg, baseline_head="local_stripe")

    assert stripe_cfg["model"]["illumination_module"]["enabled"] is False
    assert stripe_cfg["model"]["feature_fusion"]["enabled"] is False
    assert stripe_cfg["model"]["branch_attention_fusion"]["enabled"] is False
    assert stripe_cfg["model"]["nuisance_head"]["enabled"] is False
    assert stripe_cfg["model"]["reid_head"]["type"] == "local_stripe"
    assert stripe_cfg["model"]["local_extractor"]["num_parts"] == 6
    assert stripe_cfg["model"]["local_extractor"]["dropout"] == 0.10
    assert stripe_cfg["training"]["pk_sampler"]["enabled"] is False
    assert stripe_cfg["training"]["center_loss"]["enabled"] is False
    assert stripe_cfg["training"]["metric_learning"]["triplet_loss"]["enabled"] is False
    assert stripe_cfg["training"]["metric_learning"]["arcface_loss"]["enabled"] is False
    assert stripe_cfg["training"]["metric_learning"]["circle_loss"]["enabled"] is False
    assert stripe_cfg["evaluation"]["flip_test"] is False
    assert stripe_cfg["training"]["phases"]["phase1"]["epochs"] == 81


def test_derive_illumination_only_config_keeps_photometric_branch_but_disables_high_level_route_b_geometry() -> None:
    full_cfg = load_config(ATRW_CONFIG)

    illum_cfg = derive_illumination_only_config(full_cfg)

    assert illum_cfg["model"]["illumination_module"]["enabled"] is True
    assert illum_cfg["model"]["feature_fusion"]["enabled"] is True
    assert illum_cfg["model"]["branch_attention_fusion"]["enabled"] is False
    assert illum_cfg["model"]["nuisance_head"]["enabled"] is False
    assert illum_cfg["training"]["iicl"]["enabled"] is False
    assert illum_cfg["training"]["cross_light_softap"]["enabled"] is False
    assert illum_cfg["training"]["teacher_manifold"]["enabled"] is False
    assert illum_cfg["training"]["ranking_topology"]["enabled"] is False
    assert illum_cfg["training"]["anisotropic_identity_protection"]["enabled"] is False
    assert illum_cfg["training"]["semantic_non_confusion"]["enabled"] is False
    assert illum_cfg["training"]["nuisance_decoupling"]["enabled"] is False
    assert illum_cfg["training"]["identity_image_preserving"]["enabled"] is False
    assert illum_cfg["training"]["phases"]["phase1"]["epochs"] == 15
    assert illum_cfg["training"]["phases"]["phase3"]["epochs"] == 66


def test_derive_naive_illumination_config_keeps_generic_photometric_branch_but_disables_model_aware_guidance() -> None:
    full_cfg = load_config(ATRW_CONFIG)

    naive_cfg = derive_naive_illumination_config(full_cfg)

    assert naive_cfg["model"]["illumination_module"]["enabled"] is True
    assert naive_cfg["model"]["feature_fusion"]["enabled"] is False
    assert naive_cfg["model"]["branch_attention_fusion"]["enabled"] is False
    assert naive_cfg["model"]["nuisance_head"]["enabled"] is False
    assert naive_cfg["illumination_module"]["module_params"]["use_feature_guided"] is False
    assert naive_cfg["illumination_module"]["module_params"]["enable_task_aware_rollback"] is False
    assert naive_cfg["illumination_module"]["module_params"]["use_model_aware_residual"] is False
    assert naive_cfg["illumination_module"]["module_params"]["enable_coarse_task_grad"] is False
    assert naive_cfg["training"]["iicl"]["enabled"] is False
    assert naive_cfg["training"]["teacher_manifold"]["enabled"] is False
    assert naive_cfg["training"]["identity_image_preserving"]["enabled"] is False
    assert naive_cfg["training"]["phases"]["phase1"]["epochs"] == 15
    assert naive_cfg["training"]["phases"]["phase3"]["epochs"] == 66


def test_build_atrw_main_ablation_jobs_uses_paper_friendly_variant_order_and_trainers(tmp_path: Path) -> None:
    jobs = build_atrw_main_ablation_jobs(
        config_path=ATRW_CONFIG,
        data_dir="data/processed/atrw/train",
        output_root=tmp_path,
        device="cuda",
    )

    assert [job.variant_key for job in jobs] == [
        "plain_baseline",
        "naive_illumination",
        "illumination_only",
        "full_model",
    ]
    assert jobs[0].train_entrypoint.endswith("tools/train_baselines.py")
    assert jobs[1].train_entrypoint.endswith("tools/train_joint.py")
    assert jobs[2].train_entrypoint.endswith("tools/train_joint.py")
    assert jobs[3].train_entrypoint.endswith("tools/train_joint.py")
    assert jobs[0].eval_mode == "atrw_openset"
    assert jobs[-1].eval_mode == "atrw_openset"


def test_build_atrw_main_ablation_jobs_can_use_stripe_head_baseline(tmp_path: Path) -> None:
    jobs = build_atrw_main_ablation_jobs(
        config_path=ATRW_CONFIG,
        data_dir="data/processed/atrw/train",
        output_root=tmp_path,
        device="cuda",
        baseline_head="local_stripe",
    )

    plain_job = jobs[0]

    assert plain_job.variant_key == "plain_baseline"
    assert plain_job.config["model"]["reid_head"]["type"] == "local_stripe"
    assert plain_job.config["model"]["local_extractor"]["num_parts"] == 6
    assert plain_job.config["model"]["local_extractor"]["dropout"] == 0.10


def test_backbone_override_syncs_illumination_mid_channels_for_joint_variants(tmp_path: Path) -> None:
    jobs = build_atrw_main_ablation_jobs(
        config_path=ATRW_CONFIG,
        data_dir="data/processed/atrw/train",
        output_root=tmp_path,
        device="cuda",
        backbone_override="resnet50",
        baseline_head="local_stripe",
    )

    jobs_by_variant = {job.variant_key: job for job in jobs}

    for variant_key in ("naive_illumination", "illumination_only", "full_model"):
        config = jobs_by_variant[variant_key].config
        assert config["model"]["backbone"] == "resnet50"
        assert config["illumination_module"]["module_params"]["backbone_mid_channels"] == 512


def test_resnet_backbone_override_preserves_joint_schedule(tmp_path: Path) -> None:
    jobs = build_atrw_main_ablation_jobs(
        config_path=ATRW_CONFIG,
        data_dir="data/processed/atrw/train",
        output_root=tmp_path,
        device="cuda",
        backbone_override="resnet50",
        baseline_head="local_stripe",
    )

    jobs_by_variant = {job.variant_key: job for job in jobs}

    for variant_key in ("naive_illumination", "illumination_only", "full_model"):
        phases = jobs_by_variant[variant_key].config["training"]["phases"]
        assert phases["phase1"]["epochs"] == 15
        assert phases["phase2"]["epochs"] == 0
        assert phases["phase3"]["epochs"] == 66
