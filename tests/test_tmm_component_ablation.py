from pathlib import Path

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import load_config
from tools.run_tmm_component_ablation import (
    DEFAULT_VARIANT_ORDER,
    build_tmm_component_ablation_jobs,
    derive_no_branch_attention_fusion_config,
    derive_no_feature_fusion_config,
    derive_no_iicl_config,
    derive_no_model_aware_residual_config,
    derive_no_nuisance_decoupling_config,
    derive_no_photo_prior_config,
    derive_no_softap_config,
    derive_no_task_aware_rollback_config,
    derive_no_teacher_manifold_config,
    derive_teacher_separation_only_config,
    derive_teacher_tube_only_config,
)


ATRW_CONFIG = "config/illumination_config_atrw.yaml"


def test_tmm_component_jobs_materialize_expected_variant_order(tmp_path: Path) -> None:
    jobs = build_tmm_component_ablation_jobs(
        config_path=ATRW_CONFIG,
        data_dir="data/processed/atrw/train",
        output_root=tmp_path,
        device="cuda",
    )

    assert [job.variant_key for job in jobs] == list(DEFAULT_VARIANT_ORDER)
    assert all(job.train_entrypoint.endswith("tools/train_joint.py") for job in jobs)
    assert all(job.eval_mode == "atrw_openset" for job in jobs)
    assert jobs[0].config["tmm_ablation"]["variant"] == "full_model"


def test_no_task_aware_rollback_disables_top_and_model_module_params() -> None:
    cfg = derive_no_task_aware_rollback_config(load_config(ATRW_CONFIG))

    assert cfg["illumination_module"]["module_params"]["enable_task_aware_rollback"] is False
    assert cfg["model"]["illumination_module"]["module_params"]["enable_task_aware_rollback"] is False
    assert "task_aware_rollback" in cfg["tmm_ablation"]["disabled_components"]


def test_no_model_aware_residual_also_disables_coarse_task_grad() -> None:
    cfg = derive_no_model_aware_residual_config(load_config(ATRW_CONFIG))

    top_params = cfg["illumination_module"]["module_params"]
    model_params = cfg["model"]["illumination_module"]["module_params"]
    assert top_params["use_model_aware_residual"] is False
    assert top_params["enable_coarse_task_grad"] is False
    assert model_params["use_model_aware_residual"] is False
    assert model_params["enable_coarse_task_grad"] is False


def test_fusion_ablation_keeps_branch_attention_isolated_from_two_branch_fusion() -> None:
    full_cfg = load_config(ATRW_CONFIG)

    no_feature = derive_no_feature_fusion_config(full_cfg)
    no_branch_attention = derive_no_branch_attention_fusion_config(full_cfg)

    assert no_feature["model"]["feature_fusion"]["enabled"] is False
    assert no_feature["model"]["branch_attention_fusion"]["enabled"] is True
    assert no_branch_attention["model"]["feature_fusion"]["enabled"] is True
    assert no_branch_attention["model"]["branch_attention_fusion"]["enabled"] is False


def test_no_iicl_zeroes_consistency_variants_and_phase3_ramp() -> None:
    cfg = derive_no_iicl_config(load_config(ATRW_CONFIG))
    iicl_cfg = cfg["training"]["iicl"]
    aux_ramp = cfg["training"]["phases"]["phase3"]["aux_ramp"]

    assert iicl_cfg["enabled"] is False
    assert iicl_cfg["weight"] == 0.0
    assert iicl_cfg["num_variants"] == 0
    assert iicl_cfg["num_grad_variants"] == 0
    assert aux_ramp["iicl_start"] == 0.0
    assert aux_ramp["iicl_end"] == 0.0


def test_no_softap_zeroes_cross_light_ramp_but_keeps_teacher_manifold() -> None:
    cfg = derive_no_softap_config(load_config(ATRW_CONFIG))
    softap_cfg = cfg["training"]["cross_light_softap"]
    teacher_cfg = cfg["training"]["teacher_manifold"]
    aux_ramp = cfg["training"]["phases"]["phase3"]["aux_ramp"]

    assert softap_cfg["enabled"] is False
    assert softap_cfg["weight"] == 0.0
    assert teacher_cfg["enabled"] is True
    assert teacher_cfg["tube_weight"] > 0
    assert teacher_cfg["separation_weight"] > 0
    assert aux_ramp["cross_light_start"] == 0.0
    assert aux_ramp["cross_light_end"] == 0.0


def test_teacher_manifold_variants_isolate_tube_and_separation_terms() -> None:
    full_cfg = load_config(ATRW_CONFIG)

    no_teacher = derive_no_teacher_manifold_config(full_cfg)
    tube_only = derive_teacher_tube_only_config(full_cfg)
    separation_only = derive_teacher_separation_only_config(full_cfg)

    assert no_teacher["training"]["teacher_manifold"]["enabled"] is False
    assert no_teacher["training"]["teacher_manifold"]["tube_weight"] == 0.0
    assert no_teacher["training"]["teacher_manifold"]["separation_weight"] == 0.0
    assert tube_only["training"]["teacher_manifold"]["tube_weight"] > 0
    assert tube_only["training"]["teacher_manifold"]["separation_weight"] == 0.0
    assert separation_only["training"]["teacher_manifold"]["tube_weight"] == 0.0
    assert separation_only["training"]["teacher_manifold"]["separation_weight"] > 0


def test_nuisance_and_photo_prior_ablations_zero_their_runtime_weights() -> None:
    full_cfg = load_config(ATRW_CONFIG)

    no_nuisance = derive_no_nuisance_decoupling_config(full_cfg)
    no_photo = derive_no_photo_prior_config(full_cfg)

    assert no_nuisance["model"]["nuisance_head"]["enabled"] is False
    assert no_nuisance["training"]["nuisance_decoupling"]["enabled"] is False
    assert no_nuisance["training"]["nuisance_decoupling"]["weight"] == 0.0
    assert no_photo["training"]["photo_prior"]["initial_weight"] == 0.0
    assert no_photo["training"]["photo_prior"]["min_weight"] == 0.0
