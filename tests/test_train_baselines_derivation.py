import sys
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import load_config
from tools.train_baselines import (
    DEFAULT_BASELINE_BACKBONES,
    _extract_runtime_params,
    build_parser,
    derive_baseline_config,
    format_baseline_summary_lines,
    main,
    resolve_requested_backbones,
    summarize_baseline_derivation,
)


ATRW_CONFIG = "config/illumination_config_atrw.yaml"


def test_derive_baseline_config_disables_illumination_specific_mechanisms() -> None:
    full_cfg = load_config(ATRW_CONFIG)
    original = deepcopy(full_cfg)

    baseline_cfg = derive_baseline_config(full_cfg, backbone_override="osnet_x1_0")

    assert full_cfg == original

    assert baseline_cfg["model"]["backbone"] == "osnet_x1_0"
    assert baseline_cfg["model"]["illumination_module"]["enabled"] is False
    assert baseline_cfg["model"]["feature_fusion"]["enabled"] is False

    module_params = baseline_cfg["illumination_module"]["module_params"]
    assert module_params["use_sensitivity"] is False
    assert module_params["use_refinement"] is False
    assert module_params["use_feature_guided"] is False
    assert module_params["use_color_illumination"] is False
    assert module_params["enable_task_aware_rollback"] is False
    assert module_params["enable_coarse_task_grad"] is False

    loss_params = baseline_cfg["illumination_module"]["loss_params"]
    assert loss_params["lambda_recon"] == 0.0
    assert loss_params["lambda_smooth"] == 0.0
    assert loss_params["lambda_structure"] == 0.0
    assert loss_params["lambda_identity"] == 0.0
    assert loss_params["lambda_lab_chroma"] == 0.0
    assert loss_params["lambda_high_freq"] == 0.0
    assert loss_params["lambda_log_chroma"] == 0.0

    assert baseline_cfg["training"]["iicl"]["enabled"] is False
    assert baseline_cfg["training"]["iicl"]["weight"] == 0.0
    assert baseline_cfg["training"]["photo_prior"]["initial_weight"] == 0.0
    assert baseline_cfg["training"]["photo_prior"]["min_weight"] == 0.0
    assert baseline_cfg["training"]["identity_preserving"]["phase2_scale"] == 0.0
    assert baseline_cfg["training"]["identity_preserving"]["phase3_scale"] == 0.0
    assert baseline_cfg["training"]["identity_preserving"]["anchor_weight"] == 0.0
    assert baseline_cfg["training"]["identity_preserving"]["geometry_weight"] == 0.0
    assert baseline_cfg["training"]["identity_preserving"]["logit_weight"] == 0.0

    assert baseline_cfg["evaluation"] == full_cfg["evaluation"]
    assert baseline_cfg["hardware"]["use_amp"] == full_cfg["hardware"]["use_amp"]
    assert baseline_cfg["hardware"]["amp_dtype"] == full_cfg["hardware"]["amp_dtype"]
    assert baseline_cfg["training"]["batch_size"] == full_cfg["training"]["batch_size"]
    assert baseline_cfg["training"]["pk_sampler"] == full_cfg["training"]["pk_sampler"]


def test_derive_baseline_config_collapses_three_phase_schedule_into_single_reid_stage() -> None:
    full_cfg = load_config(ATRW_CONFIG)

    baseline_cfg = derive_baseline_config(full_cfg)
    phases = baseline_cfg["training"]["phases"]

    assert phases["phase1"]["name"] == "baseline_reid"
    assert phases["phase1"]["epochs"] == 81
    assert phases["phase1"]["freeze_backbone"] is False
    assert phases["phase1"]["freeze_illumination"] is True
    assert phases["phase1"]["illumination_weight"] == 0.0
    assert phases["phase1"]["reid_weight"] == 1.0
    assert phases["phase2"]["epochs"] == 0
    assert phases["phase3"]["epochs"] == 0

    overridden = derive_baseline_config(full_cfg, total_epochs=12)
    overridden_phases = overridden["training"]["phases"]
    assert overridden_phases["phase1"]["epochs"] == 12
    assert overridden_phases["phase2"]["epochs"] == 0
    assert overridden_phases["phase3"]["epochs"] == 0


def test_extract_runtime_params_follow_atrw_config_values() -> None:
    full_cfg = load_config(ATRW_CONFIG)

    baseline_cfg = derive_baseline_config(full_cfg)
    runtime = _extract_runtime_params(baseline_cfg)

    assert runtime["backbone"] == "osnet_ain_x1_0"
    assert runtime["batch_size"] == 32
    assert runtime["phase1_epochs"] == 81
    assert runtime["phase2_epochs"] == 0
    assert runtime["phase3_epochs"] == 0
    assert runtime["learning_rate"] == 0.00035
    assert runtime["eval_interval"] == 2
    assert runtime["p_size"] == 8
    assert runtime["k_size"] == 4
    assert runtime["img_height"] == 256
    assert runtime["img_width"] == 384


def test_summary_lines_include_learning_rate_and_phase_schedule() -> None:
    full_cfg = load_config(ATRW_CONFIG)

    baseline_cfg = derive_baseline_config(full_cfg)
    summary = summarize_baseline_derivation(full_cfg, baseline_cfg)

    lines = format_baseline_summary_lines(summary)

    assert any("learning_rate: 0.00035" in line for line in lines)
    assert any("source_phase_epochs: phase1=15, phase2=0, phase3=66" in line for line in lines)
    assert any("baseline_phase_epochs: phase1=81, phase2=0, phase3=0" in line for line in lines)


def test_parser_does_not_accept_legacy_query_or_gallery_args() -> None:
    parser = build_parser()

    query_actions = [action for action in parser._actions if "--query_dir" in action.option_strings]
    gallery_actions = [action for action in parser._actions if "--gallery_dir" in action.option_strings]

    assert query_actions == []
    assert gallery_actions == []


def test_main_derives_runtime_and_avoids_legacy_eval_kwargs(monkeypatch, tmp_path) -> None:
    data_dir = tmp_path / "train"
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "outputs"
    data_dir.mkdir()
    config_path.write_text("model: {}\ntraining: {}\n", encoding="utf-8")

    trainer_calls = []

    class DummyTrainer:
        def __init__(self, **kwargs):
            trainer_calls.append(kwargs)

        def train(self) -> None:
            return None

    def fake_load_config(_config_path, cli_overrides=None):
        full_cfg = load_config(ATRW_CONFIG)
        if cli_overrides:
            from app.core.config import deep_merge

            deep_merge(full_cfg, cli_overrides)
        return full_cfg

    monkeypatch.setattr("tools.train_baselines.load_config", fake_load_config)
    monkeypatch.setattr("tools.train_baselines.BaselineTrainer", DummyTrainer)
    monkeypatch.setattr("tools.train_baselines._require_training_dependencies", lambda: None)
    monkeypatch.setattr(
        "tools.train_baselines.init_distributed_mode",
        lambda *_args, **_kwargs: {
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "is_distributed": False,
            "find_unused_parameters": False,
        },
    )
    monkeypatch.setattr("tools.train_baselines.setup_logging", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("tools.train_baselines.cleanup_distributed", lambda: None)
    monkeypatch.setattr("tools.train_baselines.save_baseline_artifacts", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("tools.train_baselines._print_run_header", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_baselines.py",
            "--config",
            str(config_path),
            "--data_dir",
            str(data_dir),
            "--output_dir",
            str(output_dir),
            "--backbone",
            "osnet_ain_x1_0",
        ],
    )

    main()

    assert len(trainer_calls) == 1
    kwargs = trainer_calls[0]
    assert kwargs["batch_size"] == 32
    assert kwargs["phase1_epochs"] == 81
    assert kwargs["phase2_epochs"] == 0
    assert kwargs["phase3_epochs"] == 0
    assert kwargs["learning_rate"] == 0.00035
    assert kwargs["eval_interval"] == 2
    assert kwargs["p_size"] == 8
    assert kwargs["k_size"] == 4
    assert kwargs["img_height"] == 256
    assert kwargs["img_width"] == 384
    assert "query_dir" not in kwargs
    assert "gallery_dir" not in kwargs


def test_resolve_requested_backbones_supports_single_and_all_backbones() -> None:
    assert resolve_requested_backbones(backbone="resnet50", all_backbones=False) == ["resnet50"]
    assert resolve_requested_backbones(backbone=None, all_backbones=False) == [DEFAULT_BASELINE_BACKBONES[0]]
    assert resolve_requested_backbones(backbone=None, all_backbones=True) == DEFAULT_BASELINE_BACKBONES
