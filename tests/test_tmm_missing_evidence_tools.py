from pathlib import Path
import sys

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.materialize_tmm_seed_stability_configs import materialize
from tools.run_external_feature_baselines import DATASETS as EXTERNAL_DATASETS
from tools.run_selection_locked_rerank import select_best_grid_row


def test_selection_locked_tie_break_prefers_simpler_k1():
    rows = [
        {"mAP": 50.0, "rank1": 70.0, "k1": 14, "k2": 3, "lambda": 0.5},
        {"mAP": 50.0, "rank1": 70.0, "k1": 6, "k2": 3, "lambda": 0.5},
        {"mAP": 49.9, "rank1": 80.0, "k1": 6, "k2": 1, "lambda": 0.3},
    ]
    best = select_best_grid_row(rows)
    assert best["k1"] == 6
    assert best["mAP"] == 50.0


def test_external_baseline_dataset_set_contains_required_tmm_datasets():
    assert {"atrw", "gzgc_zebra", "leopard", "whaleshark", "stripespotter"}.issubset(
        set(EXTERNAL_DATASETS)
    )


def test_seed_stability_materializer_writes_plain_and_full_configs(tmp_path):
    out_dir = tmp_path / "seed_configs"
    rows = materialize(
        out_dir=out_dir,
        run_root="checkpoints/test_seed_stability",
        seeds=[42],
        datasets=["leopard"],
    )
    assert len(rows) == 2
    assert {row["variant"] for row in rows} == {"plain", "full"}
    for row in rows:
        config_path = Path(row["config"])
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        assert config_path.exists()
        text = config_path.read_text(encoding="utf-8")
        assert "seed: 42" in text
        assert "query_gallery" in text


def test_seed_stability_plain_uses_acmmm_strict_plain_reid_baseline(tmp_path):
    out_dir = tmp_path / "seed_configs"
    rows = materialize(
        out_dir=out_dir,
        run_root="checkpoints/test_seed_stability",
        seeds=[42],
        datasets=["atrw"],
    )
    plain_row = next(row for row in rows if row["variant"] == "plain")
    config_path = Path(plain_row["config"])
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert cfg["baseline"]["type"] == "plain_global_reid"
    assert cfg["model"]["reid_head"]["type"] == "plain_global"
    assert cfg["model"]["local_extractor"]["num_parts"] == 1
    assert cfg["model"]["illumination_module"]["enabled"] is False
    assert cfg["model"]["feature_fusion"]["enabled"] is False
    assert cfg["model"]["branch_attention_fusion"]["enabled"] is False
    assert cfg["model"]["nuisance_head"]["enabled"] is False
    assert cfg["training"]["pk_sampler"]["enabled"] is False
    assert cfg["training"]["center_loss"]["enabled"] is False
    assert cfg["training"]["metric_learning"]["triplet_loss"]["enabled"] is False
    assert cfg["training"]["metric_learning"]["arcface_loss"]["enabled"] is False
    assert cfg["training"]["metric_learning"]["ce_loss"]["label_smoothing"] == 0.0
    assert cfg["training"]["cross_light_softap"]["enabled"] is False
    assert cfg["training"]["cross_light_softap"]["weight"] == 0.0
    assert cfg["training"]["teacher_manifold"]["enabled"] is False
    assert cfg["training"]["teacher_manifold"]["tube_weight"] == 0.0
    assert cfg["training"]["teacher_manifold"]["separation_weight"] == 0.0
    assert cfg["training"]["nuisance_decoupling"]["enabled"] is False
    assert cfg["training"]["nuisance_decoupling"]["weight"] == 0.0
    assert cfg["data_augmentation"]["random_erasing"]["enabled"] is False
    assert cfg["data_augmentation"]["color_jitter"]["enabled"] is False


def test_remote_queue_routes_plain_to_strict_baseline_trainer():
    script_path = (
        REPO_ROOT
        / ".autonomous"
        / "riic-acmmm-tmm-experiments"
        / "remote_tmm_missing_evidence_20260517.sh"
    )
    if not script_path.exists():
        pytest.skip("publish branch does not include autonomous task logs")

    script = script_path.read_text(encoding="utf-8")

    assert 'if [ "${variant}" = "plain" ]; then' in script
    assert "tools/train_baselines.py" in script
    assert "--strict_plain_reid" in script
    assert "tools/train_joint.py" in script
