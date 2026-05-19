from argparse import Namespace
from pathlib import Path

from app.core.config import DEFAULT_CONFIG, cli_args_to_config, load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_repo_file(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_default_config_exposes_route_b_controls() -> None:
    loss_params = DEFAULT_CONFIG["illumination_module"]["loss_params"]
    module_params = DEFAULT_CONFIG["illumination_module"]["module_params"]
    feature_fusion_cfg = DEFAULT_CONFIG["model"]["feature_fusion"]
    branch_attention_cfg = DEFAULT_CONFIG["model"]["branch_attention_fusion"]
    training_cfg = DEFAULT_CONFIG["training"]
    identity_cfg = training_cfg["identity_preserving"]

    assert "lambda_structure" in loss_params
    assert "chroma_mode" in loss_params

    assert "enable_task_aware_rollback" in module_params
    assert "rollback_hidden_dim" in module_params
    assert "rollback_min_alpha" in module_params
    assert "rollback_max_alpha" in module_params
    assert "rollback_granularity" in module_params
    assert "rollback_num_stripes" in module_params
    assert "use_model_aware_residual" in module_params
    assert "model_residual_hidden_dim" in module_params
    assert "model_residual_scale" in module_params
    assert "enable_coarse_task_grad" in module_params
    assert "coarse_guidance_mode" in module_params
    assert "num_grad_variants" in module_params

    assert feature_fusion_cfg["enabled"] is True
    assert feature_fusion_cfg["include_illum_stats"] is True
    assert branch_attention_cfg["enabled"] is False
    assert branch_attention_cfg["num_branches"] == 3
    assert branch_attention_cfg["temperature"] == 1.0

    assert "photo_prior" in training_cfg
    assert "identity_preserving" in training_cfg
    assert "num_grad_variants" in training_cfg["iicl"]
    assert training_cfg["iicl"]["loss_type"] == "cosine"
    assert identity_cfg["mode"] == "geometry"
    assert "anchor_weight" in identity_cfg
    assert "geometry_weight" in identity_cfg
    assert "logit_weight" in identity_cfg
    assert "detach_reference" in identity_cfg


def test_default_config_exposes_retrieval_geometry_preserving_controls() -> None:
    fusion_cfg = DEFAULT_CONFIG["model"]["feature_fusion"]
    training_cfg = DEFAULT_CONFIG["training"]

    assert "max_residual_scale" in fusion_cfg
    assert 0.0 < fusion_cfg["max_residual_scale"] <= 1.0

    assert "feature_trust_region" in training_cfg
    assert "relative_class_structure" in training_cfg
    assert "local_rank_preserving" in training_cfg
    assert "identity_image_preserving" in training_cfg
    assert "cross_light_margin_preserving" in training_cfg
    assert "ranking_topology" in training_cfg
    assert "anisotropic_identity_protection" in training_cfg

    trust_cfg = training_cfg["feature_trust_region"]
    rel_cfg = training_cfg["relative_class_structure"]
    rank_cfg = training_cfg["local_rank_preserving"]
    image_cfg = training_cfg["identity_image_preserving"]
    cmp_cfg = training_cfg["cross_light_margin_preserving"]
    topo_cfg = training_cfg["ranking_topology"]
    aniso_cfg = training_cfg["anisotropic_identity_protection"]

    assert "enabled" in trust_cfg
    assert "weight" in trust_cfg
    assert "base_radius" in trust_cfg
    assert "adaptive_scale" in trust_cfg
    assert "class_spread_scale" in trust_cfg

    assert "enabled" in rel_cfg
    assert "weight" in rel_cfg
    assert "metric" in rel_cfg
    assert "radial_weight" in rel_cfg

    assert "enabled" in rank_cfg
    assert "weight" in rank_cfg
    assert "alpha" in rank_cfg
    assert "k_positive" in rank_cfg
    assert "k_negative" in rank_cfg

    assert "enabled" in image_cfg
    assert "weight" in image_cfg

    assert "enabled" in cmp_cfg
    assert "weight" in cmp_cfg
    assert "topk_positive" in cmp_cfg
    assert "topk_negative" in cmp_cfg
    assert "margin_delta" in cmp_cfg
    assert "beta" in cmp_cfg

    assert "enabled" in topo_cfg
    assert "weight" in topo_cfg
    assert "topk_positive" in topo_cfg
    assert "topk_negative" in topo_cfg
    assert "margin_slack" in topo_cfg
    assert "beta" in topo_cfg
    assert "queue_size" in topo_cfg

    assert "enabled" in aniso_cfg
    assert "weight" in aniso_cfg
    assert "topk_positive" in aniso_cfg
    assert "topk_negative" in aniso_cfg
    assert "subspace_rank" in aniso_cfg
    assert "identity_weight" in aniso_cfg
    assert "nuisance_weight" in aniso_cfg
    assert "nuisance_radius" in aniso_cfg


def test_joint_model_route_b_guidance_helper_has_no_recursive_fallback() -> None:
    source = _read_repo_file("app/core/joint_model.py")
    start = source.index("def _compute_illumination_guidance(")
    end = source.index("def extract_backbone_features(", start)
    helper_source = source[start:end]
    assert "self._compute_illumination_guidance(images)" not in helper_source
    assert "if False:" not in helper_source


def test_trainer_source_contains_route_b_weight_schedules() -> None:
    source = _read_repo_file("tools/train_joint.py")
    assert "def _get_photo_prior_weight" in source
    assert "photo_prior_weight" in source
    assert "phase2_scale" in source
    assert "phase3_scale" in source


def test_trainer_source_contains_phase3_teacher_and_retrieval_geometry_losses() -> None:
    source = _read_repo_file("tools/train_joint.py")

    assert "def _refresh_phase3_teacher(" in source
    assert "self.phase3_teacher_model" in source
    assert "self.feature_trust_region_loss" in source
    assert "self.local_rank_preserving_loss" in source
    assert "self.identity_image_preserving_weight" in source
    assert "self.ranking_topology_loss" in source
    assert "self.anisotropic_identity_protection_loss" in source


def test_default_config_exposes_additional_eval_protocols() -> None:
    evaluation_cfg = DEFAULT_CONFIG["evaluation"]

    assert "additional_protocols" in evaluation_cfg
    assert evaluation_cfg["additional_protocols"] == []


def test_atrw_config_defaults_to_openset_and_tracks_closedset() -> None:
    source = _read_repo_file("config/illumination_config_atrw.yaml")

    assert 'protocol: "atrw_openset"' in source
    assert "additional_protocols:" in source
    assert '- "atrw_closedset_train70_val30"' in source
    assert 'best_metric: "mmAP"' in source


def test_atrw_config_records_paper_reranking_recipe_without_enabling_it_by_default() -> None:
    source = _read_repo_file("config/illumination_config_atrw.yaml")
    cfg = load_config("config/illumination_config_atrw.yaml")

    assert "reranking:" in source
    assert "enabled: false" in source
    assert "k1: 25" in source
    assert "k2: 6" in source
    assert "lambda_value: 0.2" in source

    evaluation_cfg = cfg["evaluation"]
    assert evaluation_cfg["rerank"] is False
    assert evaluation_cfg["rerank_params"]["k1"] == 25
    assert evaluation_cfg["rerank_params"]["k2"] == 6
    assert evaluation_cfg["rerank_params"]["lambda_value"] == 0.2


def test_atrw_openset_eval_script_supports_configured_reranking() -> None:
    source = _read_repo_file("tools/eval_atrw_openset.py")

    assert '--rerank' in source
    assert '--rerank_k1' in source
    assert '--rerank_k2' in source
    assert '--rerank_lambda' in source
    assert 'build_submission_from_distance' in source
    assert (
        'feature_extraction_cfg.get("rerank"' in source
        or 'eval_cfg.get("rerank"' in source
        or "eval_cfg.get('rerank'" in source
    )
    assert 'rerank_params.get("k1", 25)' in source
    assert 'rerank_params.get("k2", 6)' in source
    assert 'rerank_params.get("lambda_value", 0.2)' in source


def test_atrw_config_uses_shorter_a1_phase_schedule() -> None:
    source = _read_repo_file("config/illumination_config_atrw.yaml")

    assert "eval_interval: 2" in source
    assert "phase2:" in source
    assert "phase3:" in source
    assert "epochs: 0" in source
    assert "epochs: 66" in source


def test_atrw_config_uses_geometry_preserving_phase2_weights() -> None:
    source = _read_repo_file("config/illumination_config_atrw.yaml")

    assert "illumination_weight: 1.0" in source
    assert "reid_weight: 0.0" in source


def test_atrw_config_uses_phase3_auxiliary_ramp_for_two_stage_joint_training() -> None:
    source = _read_repo_file("config/illumination_config_atrw.yaml")
    cfg = load_config("config/illumination_config_atrw.yaml")

    assert "aux_ramp:" in source
    assert "enabled: true" in source
    assert "illumination_start: 0.05" in source
    assert "illumination_end: 0.35" in source
    assert "iicl_start: 0.00" in source
    assert "iicl_end: 0.10" in source
    assert "cross_light_start: 0.00" in source
    assert "cross_light_end: 0.12" in source
    assert cfg["training"]["phases"]["phase3"]["aux_ramp"]["epochs"] == 6
    assert cfg["training"]["phases"]["phase3"]["aux_ramp"]["illumination_start"] == 0.05
    assert cfg["training"]["phases"]["phase3"]["aux_ramp"]["illumination_end"] == 0.35
    assert cfg["training"]["phases"]["phase3"]["aux_ramp"]["iicl_end"] == 0.10
    assert cfg["training"]["phases"]["phase3"]["aux_ramp"]["cross_light_end"] == 0.12


def test_atrw_config_enables_stripe_level_rollback_protection() -> None:
    source = _read_repo_file("config/illumination_config_atrw.yaml")

    assert 'rollback_granularity: "stripe"' in source


def test_atrw_config_enables_model_aware_residual_and_branch_attention() -> None:
    source = _read_repo_file("config/illumination_config_atrw.yaml")

    assert "use_model_aware_residual: true" in source
    assert "branch_attention_fusion:" in source
    assert "num_branches: 3" in source
    assert "temperature: 1.0" in source


def test_atrw_config_uses_softap_identity_defaults() -> None:
    source = _read_repo_file("config/illumination_config_atrw.yaml")

    assert 'loss_type: "cosine"' in source
    assert 'mode: "softap"' in source
    assert "phase3_scale: 1.00" in source


def test_atrw_config_exposes_cross_light_metric_alignment_controls() -> None:
    source = _read_repo_file("config/illumination_config_atrw.yaml")

    assert "cross_light_margin_preserving:" in source
    assert "teacher_prototype_anchor:" in source
    assert "relative_class_structure:" in source
    assert "aux_gradient_gate:" in source
    assert "photometric_scale:" in source
    assert "photometric_offset:" in source
    assert "topk_positive:" in source
    assert "topk_negative:" in source
    assert "margin_delta:" in source
    assert "beta:" in source


def test_atrw_config_exposes_geometry_preserving_retrieval_controls() -> None:
    source = _read_repo_file("config/illumination_config_atrw.yaml")
    cfg = load_config("config/illumination_config_atrw.yaml")

    assert "max_residual_scale:" in source
    assert "feature_trust_region:" in source
    assert "relative_class_structure:" in source
    assert "local_rank_preserving:" in source
    assert "identity_image_preserving:" in source
    assert "cross_light_softap:" in source
    assert "teacher_manifold:" in source
    assert "ranking_topology:" in source
    assert "anisotropic_identity_protection:" in source
    assert "neighborhood_consistency:" in source
    assert "semantic_non_confusion:" in source
    assert "nuisance_decoupling:" in source

    assert cfg["model"]["feature_fusion"]["max_residual_scale"] < 1.0
    assert cfg["training"]["cross_light_softap"]["enabled"] is True
    assert cfg["training"]["teacher_manifold"]["enabled"] is True
    assert cfg["training"]["ranking_topology"]["enabled"] is False
    assert cfg["training"]["anisotropic_identity_protection"]["enabled"] is False
    assert cfg["training"]["neighborhood_consistency"]["enabled"] is False
    assert cfg["training"]["semantic_non_confusion"]["enabled"] is False
    assert cfg["training"]["nuisance_decoupling"]["enabled"] is True
    assert cfg["training"]["feature_trust_region"]["enabled"] is False
    assert cfg["training"]["relative_class_structure"]["enabled"] is False
    assert cfg["training"]["local_rank_preserving"]["enabled"] is False
    assert cfg["training"]["identity_image_preserving"]["enabled"] is True


def test_atrw_config_rebalances_photo_iicl_and_phase3_illumination() -> None:
    source = _read_repo_file("config/illumination_config_atrw.yaml")
    cfg = load_config("config/illumination_config_atrw.yaml")

    assert "initial_weight: 0.45" in source
    assert "min_weight: 0.05" in source
    assert "weight: 0.10" in source
    assert "weight: 0.12" in source
    assert "tube_weight: 0.30" in source
    assert "separation_weight: 0.10" in source
    assert "margin_slack: 0.01" in source
    assert "nuisance_radius: 0.12" in source
    assert "weight: 0.00" in source
    assert "weight: 0.04" in source
    assert "illumination_weight: 0.35" in source
    assert cfg["training"]["cross_light_softap"]["weight"] == 0.12
    assert cfg["training"]["teacher_manifold"]["tube_weight"] == 0.30
    assert cfg["training"]["teacher_manifold"]["separation_weight"] == 0.10
    assert cfg["training"]["ranking_topology"]["weight"] == 0.18
    assert cfg["training"]["anisotropic_identity_protection"]["weight"] == 0.10
    assert cfg["training"]["semantic_non_confusion"]["weight"] == 0.0
    assert cfg["training"]["nuisance_decoupling"]["weight"] == 0.04
    assert cfg["training"]["photo_prior"]["min_weight"] == 0.05
    assert cfg["training"]["cross_light_margin_preserving"]["enabled"] is False
    assert cfg["training"]["teacher_prototype_anchor"]["enabled"] is False
    assert cfg["training"]["relative_class_structure"]["weight"] == 0.0
    assert cfg["training"]["feature_trust_region"]["weight"] == 0.0
    assert cfg["training"]["phases"]["phase3"]["batch_size"] == 32


def test_atrw_config_uses_global_photometric_stats_and_fixed_semantic_margin() -> None:
    source = _read_repo_file("config/illumination_config_atrw.yaml")
    cfg = load_config("config/illumination_config_atrw.yaml")
    trainer_source = _read_repo_file("tools/train_joint.py")
    loss_source = _read_repo_file("app/core/illumination_module_v2.py")

    assert "photometric_dim: 4" in source
    assert "stats_num_stripes:" not in source
    assert "stats_protection_floor:" not in source
    assert "severity_relaxation:" not in source
    assert "severity_scale:" not in source
    assert "severity_offset:" not in source
    assert "max_margin_drop:" not in source
    assert cfg["model"]["nuisance_head"]["photometric_dim"] == 4
    assert "stats_num_stripes" not in cfg["training"]["cross_light_softap"]
    assert "stats_protection_floor" not in cfg["training"]["cross_light_softap"]
    assert "severity_relaxation" not in cfg["training"]["semantic_non_confusion"]
    assert "severity_scale" not in cfg["training"]["semantic_non_confusion"]
    assert "severity_offset" not in cfg["training"]["semantic_non_confusion"]
    assert "max_margin_drop" not in cfg["training"]["semantic_non_confusion"]
    assert "severity=semantic_severity" not in trainer_source
    assert "stats_num_stripes" not in trainer_source
    assert "stats_protection_floor" not in trainer_source
    assert "severity_relaxation" not in loss_source


def test_atrw_config_restores_teacher_manifold_phase3_geometry() -> None:
    source = _read_repo_file("config/illumination_config_atrw.yaml")
    cfg = load_config("config/illumination_config_atrw.yaml")
    trainer_source = _read_repo_file("tools/train_joint.py")
    loss_source = _read_repo_file("app/core/illumination_module_v2.py")

    assert "teacher_manifold:" in source
    assert "tube_weight:" in source
    assert "separation_weight:" in source
    assert "margin:" in source
    assert "ranking_topology:" in source
    assert "margin_slack:" in source
    assert "beta:" in source
    assert "anisotropic_identity_protection:" in source
    assert "subspace_rank:" in source
    assert "identity_weight:" in source
    assert "nuisance_weight:" in source
    assert "queue_size:" in source
    assert "weight: 0.12" in source
    assert "enabled: false" in source
    assert "regression_weight: 0.0" in source
    assert cfg["training"]["teacher_manifold"]["enabled"] is True
    assert cfg["training"]["teacher_manifold"]["tube_weight"] > 0.0
    assert cfg["training"]["teacher_manifold"]["separation_weight"] > 0.0
    assert cfg["training"]["ranking_topology"]["enabled"] is False
    assert cfg["training"]["anisotropic_identity_protection"]["enabled"] is False
    assert cfg["training"]["semantic_non_confusion"]["enabled"] is False
    assert cfg["training"]["nuisance_decoupling"]["regression_weight"] == 0.0
    assert "TeacherManifoldTubeLoss" in trainer_source
    assert "TeacherManifoldSeparationLoss" in trainer_source
    assert "self.teacher_manifold_tube_loss" in trainer_source
    assert "self.teacher_manifold_separation_loss" in trainer_source
    assert "loss_teacher_manifold_tube" in trainer_source
    assert "loss_teacher_manifold_separation" in trainer_source
    assert "RankingTopologyPreservingLoss" in trainer_source
    assert "AnisotropicIdentityProtectionLoss" in trainer_source
    assert "NeighborhoodConsistencyLoss" in trainer_source
    assert "self.ranking_topology_loss" in trainer_source
    assert "self.anisotropic_identity_protection_loss" in trainer_source
    assert "self.neighborhood_consistency_loss" in trainer_source
    assert "class TeacherManifoldTubeLoss" in loss_source
    assert "class TeacherManifoldSeparationLoss" in loss_source
    assert "class RankingTopologyPreservingLoss" in loss_source
    assert "class AnisotropicIdentityProtectionLoss" in loss_source
    assert "class NeighborhoodConsistencyLoss" in loss_source


def test_trainer_source_contains_cmp_regularizer_integration() -> None:
    source = _read_repo_file("tools/train_joint.py")

    assert "CrossLightMarginPreservingLoss" in source
    assert "self.cross_light_margin_preserving_loss" in source
    assert "self.identity_preserving_mode == 'margin_preserving'" in source
    assert "loss_cross_light = self.cross_light_margin_preserving_loss(" in source


def test_atrw_config_disables_iicl_variant_backprop_to_fit_phase3_memory_budget() -> None:
    source = _read_repo_file("config/illumination_config_atrw.yaml")
    cfg = load_config("config/illumination_config_atrw.yaml")

    assert "num_grad_variants: 0" in source
    assert cfg["training"]["iicl"]["num_grad_variants"] == 0


def test_atrw_config_uses_phase3_checkpointing_override() -> None:
    source = _read_repo_file("config/illumination_config_atrw.yaml")

    assert "use_backbone_checkpointing: false" in source
    assert "phase3:" in source
    assert "use_backbone_checkpointing: true" in source


def test_atrw_config_enables_amp_with_bf16() -> None:
    source = _read_repo_file("config/illumination_config_atrw.yaml")

    assert "use_amp: true" in source
    assert 'amp_dtype: "bfloat16"' in source


def test_phase2_source_uses_configured_loss_weights() -> None:
    source = _read_repo_file("tools/train_joint.py")
    start = source.index("def _setup_optimizer_phase2_fgid(")
    end = source.index("def _setup_optimizer_phase3(", start)
    phase2_source = source[start:end]

    assert "phase_config = self.config['training']['phases'].get('phase2', {})" in phase2_source
    assert "phase_config.get('illumination_weight', 1.0)" in phase2_source
    assert "phase_config.get('reid_weight', 0.0)" in phase2_source
    assert "'illumination': 1.0" not in phase2_source
    assert "'reid': 0.0" not in phase2_source


def test_trainer_source_toggles_backbone_checkpointing_per_phase() -> None:
    source = _read_repo_file("tools/train_joint.py")

    assert "def _set_phase_backbone_checkpointing(" in source
    assert "phase_config.get(" in source
    assert "'use_backbone_checkpointing'" in source
    assert "self.model.use_backbone_checkpointing = bool(enabled)" in source
    assert "self._set_phase_backbone_checkpointing(phase=1)" in source
    assert "self._set_phase_backbone_checkpointing(phase=2)" in source
    assert "self._set_phase_backbone_checkpointing(phase=3)" in source


def test_phase2_train_epoch_source_does_not_force_reid_loss_off() -> None:
    source = _read_repo_file("tools/train_joint.py")
    start = source.index("def train_epoch(")
    end = source.index("# Backward", start)
    train_epoch_source = source[start:end]

    assert (
        "if phase == 2:\n"
        "                    loss = self.loss_weights['illumination'] * loss_illum\n"
        "                    loss_reid = torch.tensor(0.0, device=self.device)\n"
        "                    loss_iicl = torch.tensor(0.0, device=self.device)"
    ) not in train_epoch_source


def test_train_epoch_source_uses_raw_reference_geometry_preservation() -> None:
    source = _read_repo_file("tools/train_joint.py")
    start = source.index("def train_epoch(")
    end = source.index("# Backward", start)
    train_epoch_source = source[start:end]

    assert "forward_mode='raw_reference'" in train_epoch_source
    assert "self.teacher_anchor_loss(" in train_epoch_source
    assert "self.geometry_preserving_loss(" in train_epoch_source
    assert "self.teacher_logit_consistency_loss(" in train_epoch_source
    assert "self.identity_preserving_loss(features, labels)" not in train_epoch_source


def test_nested_eval_feature_extraction_flip_test_overrides_top_level(tmp_path: Path) -> None:
    cfg_path = tmp_path / "atrw_eval.yaml"
    cfg_path.write_text(
        "evaluation:\n"
        "  flip_test: true\n"
        "  feature_extraction:\n"
        "    flip_test: false\n",
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path))

    assert cfg["evaluation"]["flip_test"] is False


def test_trainer_source_supports_additional_eval_protocols() -> None:
    source = _read_repo_file("tools/train_joint.py")

    assert "self.additional_eval_protocols" in source
    assert "def _evaluate_single_protocol(" in source
    assert "for protocol in self.additional_eval_protocols" in source


def test_trainer_source_uses_configured_flip_test_for_internal_feature_extraction() -> None:
    source = _read_repo_file("tools/train_joint.py")

    assert "self.eval_flip_test" in source
    assert source.count("flip_test=self.eval_flip_test") >= 2
    assert "flip_test=False" not in source


def test_openset_eval_script_uses_configurable_flip_test() -> None:
    source = _read_repo_file("tools/eval_atrw_openset.py")

    assert "args.flip_test" in source
    assert "flip_test=args.flip_test" in source


def test_evaluation_source_contains_oom_resilient_feature_extraction() -> None:
    source = _read_repo_file("app/core/evaluation.py")

    assert "def _release_cuda_memory()" in source
    assert "def _forward_feature_batch_with_adaptive_split(" in source
    assert "torch.OutOfMemoryError" in source
    assert "retrying with chunk_size=" in source


def test_trainer_source_evaluates_phase2_and_forces_eval_on_phase_end() -> None:
    source = _read_repo_file("tools/train_joint.py")

    assert "should_eval = ((epoch + 1) % self.eval_interval == 0) or ((epoch + 1) == total_epochs)" in source
    assert "maybe_run_eval(epoch, phase=2, total_epochs=self.phase2_fgid_epochs, metrics=metrics)" in source


def test_joint_model_source_contains_memory_saving_helpers() -> None:
    source = _read_repo_file("app/core/joint_model.py")

    assert "def _illumination_trainable(" in source
    assert "def _backbone_trainable(" in source
    assert "def _extract_raw_feature_map_for_fusion(" in source
    assert "use_guidance_grad = self.enable_coarse_task_grad and self._illumination_trainable()" in source
    assert "raw_feature_map = self._extract_raw_feature_map_for_fusion(images)" in source


def test_joint_model_source_contains_raw_reference_teacher_path() -> None:
    source = _read_repo_file("app/core/joint_model.py")

    assert "def forward_raw_reference(" in source
    assert "raw_normalized = self._prepare_backbone_input(images)" in source
    assert "feature_map = self.extract_backbone_features(raw_normalized)" in source
    assert "features, logits = self.local_extractor(feature_map)" in source


def test_forward_adapter_source_supports_raw_reference_mode() -> None:
    source = _read_repo_file("tools/train_joint.py")
    start = source.index("class JointModelForwardAdapter")
    end = source.index("# ============================================================================", start)
    adapter_source = source[start:end]

    assert "if forward_mode == 'raw_reference':" in adapter_source
    assert "forward_raw_reference(images" in adapter_source


def test_forward_adapter_source_supports_illumination_only_mode() -> None:
    source = _read_repo_file("tools/train_joint.py")
    start = source.index("class JointModelForwardAdapter")
    end = source.index("# ============================================================================", start)
    adapter_source = source[start:end]

    assert "if forward_mode == 'illumination_only':" in adapter_source
    assert "forward_illumination_only(images" in adapter_source


def test_phase2_source_freezes_backbone_but_keeps_feature_fusion_trainable() -> None:
    source = _read_repo_file("app/core/joint_model.py")
    trainer_source = _read_repo_file("tools/train_joint.py")

    assert "def freeze_local_extractor(" in source
    assert "def freeze_feature_fusion(" in source
    assert "self.model.freeze_local_extractor(True)" in trainer_source
    assert "self.model.freeze_feature_fusion(False)" in trainer_source


def test_default_config_exposes_cross_light_alignment_defaults() -> None:
    training_cfg = DEFAULT_CONFIG["training"]

    assert "cross_light_prototype" in training_cfg
    assert "cross_light_margin_preserving" in training_cfg
    assert "cross_light_softap" in training_cfg
    assert "semantic_non_confusion" in training_cfg
    assert "nuisance_decoupling" in training_cfg
    assert "teacher_prototype_anchor" in training_cfg
    assert "relative_class_structure" in training_cfg
    assert "aux_gradient_gate" in training_cfg
    assert training_cfg["cross_light_prototype"]["enabled"] is True
    assert training_cfg["cross_light_prototype"]["weight"] == 0.12
    assert training_cfg["cross_light_margin_preserving"]["enabled"] is False
    assert training_cfg["cross_light_margin_preserving"]["weight"] == 0.15
    assert training_cfg["cross_light_softap"]["enabled"] is False
    assert training_cfg["semantic_non_confusion"]["enabled"] is True
    assert training_cfg["nuisance_decoupling"]["enabled"] is True
    assert training_cfg["teacher_prototype_anchor"]["enabled"] is False
    assert training_cfg["teacher_prototype_anchor"]["weight"] == 0.0
    assert training_cfg["relative_class_structure"]["enabled"] is True
    assert training_cfg["relative_class_structure"]["weight"] == 0.08
    assert training_cfg["aux_gradient_gate"]["enabled"] is True


def test_trainer_source_contains_cross_light_metric_alignment_helpers() -> None:
    source = _read_repo_file("tools/train_joint.py")

    assert "CrossLightPrototypeLoss" in source
    assert "CrossLightMarginPreservingLoss" in source
    assert "SoftAPCrossLightLoss" in source
    assert "SemanticNonConfusionLoss" in source
    assert "CrossCovarianceDecorrelationLoss" in source
    assert "RelativeClassStructureLoss" in source
    assert "def _compute_nonnegative_gradient_alignment(" in source
    assert "def _linear_warmup_value(" in source
    assert "def _get_phase3_aux_weight(" in source
    assert "def _reset_cross_light_queue(" in source
    assert "def _update_cross_light_queue(" in source
    assert "if self.use_iicl and phase == 3 and phase_iicl_weight > 0:" in source
    assert "self.cross_light_prototype_loss" in source
    assert "self.cross_light_margin_preserving_loss" in source
    assert "self.cross_light_softap_loss" in source
    assert "self.semantic_non_confusion_loss" in source
    assert "self.relative_class_structure_loss" in source
    assert "identity_preserving_mode" in source


def test_joint_model_source_contains_nuisance_head_outputs() -> None:
    source = _read_repo_file("app/core/joint_model.py")

    assert '"_nuisance_head"' in source
    assert "self.nuisance_head_enabled" in source
    assert '"nuisance_features"' in source
    assert '"photometric_prediction"' in source


def test_model_factory_passes_nuisance_head_cfg_to_joint_model() -> None:
    source = _read_repo_file("app/core/model_factory.py")

    assert 'nuisance_head_cfg = _as_dict(model_cfg.get("nuisance_head"))' in source
    assert 'module_params["_nuisance_head"]' in source


def test_cli_output_dir_overrides_top_level_output_dir() -> None:
    args = Namespace(
        output_dir="./checkpoints/custom_run",
        backbone=None,
        batch_size=None,
        phase1_epochs=None,
        phase2_epochs=None,
        learning_rate=None,
        num_stripes=None,
        eval_interval=None,
        p_size=None,
        k_size=None,
        circle_gamma=None,
        img_height=None,
        img_width=None,
        num_workers=None,
        eval_protocol=None,
        best_metric=None,
        strict_protocol_check=None,
        use_iicl=None,
        iicl_weight=None,
        iicl_variants=None,
    )

    cfg = cli_args_to_config(args)

    assert cfg["output_dir"] == "./checkpoints/custom_run"


def test_default_config_exposes_amp_controls() -> None:
    hardware_cfg = DEFAULT_CONFIG["hardware"]

    assert hardware_cfg["use_amp"] is True
    assert hardware_cfg["amp_dtype"] == "float16"


def test_trainer_source_contains_amp_training_path() -> None:
    source = _read_repo_file("tools/train_joint.py")

    assert "from contextlib import nullcontext" in source
    assert "torch.amp.autocast" in source
    assert "torch.amp.GradScaler" in source
    assert "self.grad_scaler.scale(loss).backward()" in source
    assert "self.grad_scaler.unscale_(self.optimizer)" in source


def test_trainer_source_contains_nonfinite_guardrails() -> None:
    source = _read_repo_file("tools/train_joint.py")

    assert "def _ensure_finite_tensor(" in source
    assert "self._ensure_finite_tensor(features, 'output.features'" in source
    assert "self._ensure_finite_tensor(logits, 'output.logits'" in source
    assert "self._ensure_finite_tensor(loss, 'loss.total'" in source
    assert "Non-finite grad norm detected under AMP" in source
    assert "self._ensure_finite_tensor(grad_norm, 'grad.norm'" in source


def test_joint_model_source_runs_illumination_branch_outside_amp() -> None:
    source = _read_repo_file("app/core/joint_model.py")

    assert "def _autocast_disabled_context(" in source
    assert "with self._autocast_disabled_context(images):" in source
    assert "with self._autocast_disabled_context(images):" in source


def test_trainer_source_keeps_random_erasing_out_of_physical_rgb_dataloader() -> None:
    source = _read_repo_file("tools/train_joint.py")
    start = source.index("transform = transforms.Compose([")
    end = source.index("self.dataset = FullImageDataset", start)
    transform_source = source[start:end]

    assert "transforms.RandomErasing(" not in transform_source
    assert "Backbone-space Gaussian erasing is applied inside JointReIDModel" in source


def test_model_factory_passes_backbone_random_erasing_cfg_to_joint_model() -> None:
    source = _read_repo_file("app/core/model_factory.py")

    assert 'aug_cfg = _as_dict(config.get("data_augmentation"))' in source
    assert 'module_params["_backbone_random_erasing"]' in source


def test_model_factory_passes_branch_attention_cfg_to_joint_model() -> None:
    source = _read_repo_file("app/core/model_factory.py")

    assert 'branch_attention_cfg = _as_dict(model_cfg.get("branch_attention_fusion"))' in source
    assert 'module_params["_branch_attention_fusion"]' in source


def test_joint_model_source_applies_random_erasing_after_normalization() -> None:
    source = _read_repo_file("app/core/joint_model.py")

    assert "def _apply_backbone_random_erasing(" in source
    assert "normalized = self._apply_backbone_random_erasing(normalized)" in source
    assert "self.backbone_random_erasing_enabled = bool(backbone_erasing_cfg.get(\"enabled\", False))" in source


def test_joint_model_source_contains_branch_attention_fusion_path() -> None:
    source = _read_repo_file("app/core/joint_model.py")

    assert "class StripeAwareBranchAttentionFusion" in source
    assert "self.branch_attention_fusion_enabled" in source
    assert "def _maybe_fuse_branch_feature_maps(" in source
    assert "branch_attention_weights" in source


def test_illumination_module_source_contains_model_aware_reflectance_residual() -> None:
    source = _read_repo_file("app/core/illumination_module_v2.py")

    assert "class ModelAwareReflectanceResidual" in source
    assert "self.use_model_aware_residual" in source
    assert "reflectance_base" in source
    assert "reflectance_att" in source


def test_default_config_exposes_ddp_controls() -> None:
    hardware_cfg = DEFAULT_CONFIG["hardware"]

    assert hardware_cfg["use_ddp"] is True
    assert hardware_cfg["ddp_backend"] == "nccl"
    assert hardware_cfg["ddp_find_unused_parameters"] is False
    assert hardware_cfg["ddp_timeout_minutes"] == 30


def test_default_config_exposes_backbone_checkpointing_control() -> None:
    hardware_cfg = DEFAULT_CONFIG["hardware"]

    assert hardware_cfg["use_backbone_checkpointing"] is True


def test_default_config_keeps_phase3_config_extensible() -> None:
    phase3_cfg = DEFAULT_CONFIG["training"]["phases"]["phase3"]

    assert isinstance(phase3_cfg, dict)


def test_data_source_contains_distributed_pk_sampler() -> None:
    source = _read_repo_file("app/core/data.py")

    assert "class DistributedPKSampler(" in source
    assert "self.local_p = p // num_replicas" in source
    assert "def set_epoch(" in source
    assert "selected_ids = rng.choice(" in source


def test_trainer_source_contains_ddp_training_scaffold() -> None:
    source = _read_repo_file("tools/train_joint.py")

    assert "from torch.nn.parallel import DistributedDataParallel" in source
    assert "from torch.utils.data.distributed import DistributedSampler" in source
    assert "self.is_distributed" in source
    assert "self.is_main_process" in source
    assert "self.model_ddp" in source
    assert "dist.barrier()" in source
    assert "LOCAL_RANK" in source


def test_trainer_source_has_rank_aware_checkpoint_and_resume_support() -> None:
    source = _read_repo_file("tools/train_joint.py")

    assert "def _load_model_state_dict_compat(" in source
    assert "if not self.is_main_process:" in source
    assert "self.model.state_dict()" in source


def test_joint_model_source_contains_backbone_checkpointing_path() -> None:
    source = _read_repo_file("app/core/joint_model.py")

    assert "from torch.utils.checkpoint import checkpoint" in source
    assert "self.use_backbone_checkpointing" in source
    assert "def _extract_backbone_features_impl(" in source
    assert "checkpoint(self._extract_backbone_features_impl, x, use_reentrant=False)" in source


def test_joint_model_checkpoint_gate_does_not_require_grad_on_input() -> None:
    source = _read_repo_file("app/core/joint_model.py")
    start = source.index("def extract_backbone_features(")
    end = source.index("def forward(", start)
    helper_source = source[start:end]

    assert "self.use_backbone_checkpointing and self.training" in helper_source
    assert "x.requires_grad" not in helper_source


def test_joint_model_source_reapplies_eval_mode_to_frozen_modules() -> None:
    source = _read_repo_file("app/core/joint_model.py")

    assert "self._freeze_backbone = False" in source
    assert "self._freeze_illumination = False" in source
    assert "self._freeze_local_extractor = False" in source
    assert "self._freeze_feature_fusion = False" in source
    assert "def train(self, mode: bool = True):" in source
    assert "if mode and self._freeze_backbone:" in source
    assert "self.backbone.eval()" in source
    assert "self.local_extractor.eval()" in source
    assert "self.feature_fusion.eval()" in source


def test_joint_model_source_prepares_backbone_inputs_in_physical_range() -> None:
    source = _read_repo_file("app/core/joint_model.py")

    assert "def _prepare_backbone_input(" in source
    assert "images = torch.clamp(images, 0.0, 1.0)" in source
    assert "normalized = self._prepare_backbone_input(images)" in source
    assert "feat_mid = self.extract_early_backbone_features(self._prepare_backbone_input(guidance))" in source
    assert "illuminated_normalized = self._prepare_backbone_input(illuminated)" in source
    assert "reflectance_normalized = self._prepare_backbone_input(reflectance)" in source
    assert "variant_normalized = self._prepare_backbone_input(variant)" in source


def test_joint_model_source_uses_variant_specific_raw_features_for_iicl() -> None:
    source = _read_repo_file("app/core/joint_model.py")

    assert "variant_raw_feature_map = self._extract_raw_feature_map_for_fusion(variant)" in source
    assert "feature_map_var = self._maybe_fuse_feature_maps(variant_raw_feature_map, feature_map_var" in source
    assert "detached_variant_raw = variant_raw_feature_map.detach()" in source
    assert "feature_map_var = self._maybe_fuse_feature_maps(detached_variant_raw, feature_map_var" in source


def test_joint_model_source_keeps_raw_feature_fusion_branch_on_graph() -> None:
    source = _read_repo_file("app/core/joint_model.py")

    start = source.index("def _extract_raw_feature_map_for_fusion(")
    end = source.index("def _maybe_fuse_feature_maps(", start)
    helper_source = source[start:end]

    assert "with torch.no_grad()" not in helper_source
    assert ".detach()" not in helper_source
    assert "return self.extract_backbone_features(normalized)" in helper_source


def test_evaluation_source_restores_model_state_and_fails_loud_on_bad_images() -> None:
    source = _read_repo_file("app/core/evaluation.py")

    assert 'raise FileNotFoundError(f"Failed to read evaluation image: {img_path}")' in source
    assert source.count("was_training = self.model.training") >= 2
    assert source.count("self.model.train(was_training)") >= 2


def test_joint_model_source_has_no_legacy_duplicate_freeze_helpers() -> None:
    source = _read_repo_file("app/core/joint_model.py")

    assert source.count("def freeze_backbone(") == 1
    assert source.count("def freeze_illumination(") == 1
    assert source.count("def freeze_local_extractor(") == 1
    assert source.count("def freeze_feature_fusion(") == 1
    assert source.count("def _imagenet_normalize(") == 1


def test_joint_model_source_is_ascii_only() -> None:
    source = _read_repo_file("app/core/joint_model.py")

    source.lstrip("\ufeff").encode("ascii")
