import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_cross_species_paper_ablation import (
    DEFAULT_BACKBONE,
    DEFAULT_BASELINE_HEAD,
    DEFAULT_DATASET_ORDER,
    DEFAULT_SELECTION_PROTOCOL,
    DEFAULT_VARIANT_ORDER,
    DEFAULT_FINAL_REPORT_PROTOCOL,
    _cleanup_extra_checkpoints,
    _ensure_selection_query_gallery_split,
    _parse_eval_metrics,
    DATASETS,
    build_cross_species_dataset_jobs,
    derive_generic_illumination_config,
)
from app.core.config import load_config


def test_dataset_registry_uses_actual_query_gallery_species_configs() -> None:
    assert DEFAULT_DATASET_ORDER == (
        "stripespotter",
        "gzgc_zebra",
        "gzgc_giraffe",
    )

    assert "czechlynx" in DATASETS
    assert DATASETS["czechlynx"].config_path.endswith("illumination_config_czechlynx_actual.yaml")
    assert DATASETS["czechlynx"].source_protocol == "official_time_closed_split"
    assert DATASETS["czechlynx"].use_official_query_gallery_for_training is True

    for dataset_key in DEFAULT_DATASET_ORDER:
        dataset = DATASETS[dataset_key]
        assert dataset.config_path.endswith("_actual.yaml")
        assert dataset.display_name
        cfg = load_config(dataset.config_path)
        assert cfg["model"]["backbone"] == DEFAULT_BACKBONE
        assert cfg["evaluation"]["protocol"] == "query_gallery"


def test_build_cross_species_dataset_jobs_keeps_czechlynx_on_official_query_gallery_protocol(
    tmp_path: Path,
) -> None:
    jobs = build_cross_species_dataset_jobs(
        dataset_key="czechlynx",
        output_root=tmp_path,
        device="cuda",
    )

    assert [job.variant_key for job in jobs] == list(DEFAULT_VARIANT_ORDER)

    for job in jobs:
        assert job.train_data_dir == "data/processed/czechlynx/train"
        assert job.query_dir == "data/processed/czechlynx/query"
        assert job.gallery_dir == "data/processed/czechlynx/gallery"
        assert job.config["training"]["data_dir"] == "data/processed/czechlynx/train"
        assert job.config["evaluation"]["protocol"] == DEFAULT_FINAL_REPORT_PROTOCOL
        assert job.config["evaluation"]["best_metric"] == "mAP"
        assert job.config["evaluation"]["strict_protocol_check"] is True
        assert job.config["paper_protocol"]["source_dataset_protocol"] == "official_time_closed_split"
        assert job.config["paper_protocol"]["training_selection_protocol"] == DEFAULT_FINAL_REPORT_PROTOCOL
        assert job.config["paper_protocol"]["final_report_protocol"] == DEFAULT_FINAL_REPORT_PROTOCOL
        assert job.config["paper_protocol"]["official_protocol"] is True
        assert "official CzechLynx time-closed split" in job.config["paper_protocol"]["note"]
        assert "selection_info" not in job.config["paper_protocol"]


def test_build_cross_species_dataset_jobs_uses_stripe_head_white_box_baseline(tmp_path: Path) -> None:
    jobs = build_cross_species_dataset_jobs(
        dataset_key="stripespotter",
        output_root=tmp_path,
        device="cuda",
    )

    assert DEFAULT_BASELINE_HEAD == "local_stripe"
    assert [job.variant_key for job in jobs] == list(DEFAULT_VARIANT_ORDER)

    white_box_job = jobs[0]
    assert white_box_job.dataset_key == "stripespotter"
    assert white_box_job.display_name == "Simplified Baseline"
    assert white_box_job.eval_mode == "query_gallery"
    assert white_box_job.config["evaluation"]["protocol"] == DEFAULT_SELECTION_PROTOCOL
    assert white_box_job.config["paper_protocol"]["source_dataset_protocol"] == "query_gallery"
    assert white_box_job.config["paper_protocol"]["training_selection_protocol"] == DEFAULT_SELECTION_PROTOCOL
    assert white_box_job.config["paper_protocol"]["final_report_protocol"] == "query_gallery"
    assert white_box_job.config["paper_protocol"]["selection_metric"] == "mAP"
    assert white_box_job.config["paper_protocol"]["official_protocol"] is False
    assert "fixed query/gallery" in white_box_job.config["paper_protocol"]["note"]
    assert white_box_job.train_entrypoint.endswith("tools/train_baselines.py")
    assert white_box_job.train_data_dir == "data/processed/stripespotter/selection_train"
    assert white_box_job.config["training"]["data_dir"] == "data/processed/stripespotter/selection_train"
    assert white_box_job.query_dir == "data/processed/stripespotter/query"
    assert white_box_job.gallery_dir == "data/processed/stripespotter/gallery"
    assert white_box_job.config["evaluation"]["selection_query_dir"] == "data/processed/stripespotter/selection_query"
    assert white_box_job.config["evaluation"]["selection_gallery_dir"] == "data/processed/stripespotter/selection_gallery"
    assert white_box_job.config["model"]["backbone"] == DEFAULT_BACKBONE
    assert white_box_job.config["model"]["reid_head"]["type"] == "local_stripe"
    assert white_box_job.config["model"]["local_extractor"]["num_parts"] == 6
    assert white_box_job.config["model"]["illumination_module"]["enabled"] is False
    assert white_box_job.config["model"]["feature_fusion"]["enabled"] is False
    assert white_box_job.config["training"]["iicl"]["enabled"] is False
    assert white_box_job.config["training"]["pk_sampler"]["enabled"] is False
    assert white_box_job.config["training"]["center_loss"]["enabled"] is False
    assert white_box_job.config["training"]["metric_learning"]["arcface_loss"]["enabled"] is False
    assert white_box_job.config["training"]["metric_learning"]["triplet_loss"]["enabled"] is False
    assert white_box_job.config["training"]["metric_learning"]["ce_loss"]["label_smoothing"] == 0.0
    assert white_box_job.config["evaluation"]["feature_extraction"]["flip_test"] is False


def test_build_cross_species_dataset_jobs_makes_joint_phase3_budget_explicit_and_matches_baseline_budget(
    tmp_path: Path,
) -> None:
    jobs = build_cross_species_dataset_jobs(
        dataset_key="stripespotter",
        output_root=tmp_path,
        device="cuda",
    )

    by_variant = {job.variant_key: job for job in jobs}
    baseline_job = by_variant["white_box_baseline"]
    generic_job = by_variant["generic_illumination"]
    full_job = by_variant["full_model"]

    generic_phases = generic_job.config["training"]["phases"]
    full_phases = full_job.config["training"]["phases"]

    assert generic_phases["phase3"]["epochs"] == 100
    assert full_phases["phase3"]["epochs"] == 100

    expected_total_epochs = (
        generic_phases["phase1"]["epochs"]
        + generic_phases["phase2"]["epochs"]
        + generic_phases["phase3"]["epochs"]
    )
    assert expected_total_epochs == (
        full_phases["phase1"]["epochs"]
        + full_phases["phase2"]["epochs"]
        + full_phases["phase3"]["epochs"]
    )
    assert baseline_job.config["training"]["phases"]["phase1"]["epochs"] == expected_total_epochs
    assert baseline_job.config["training"]["phases"]["phase2"]["epochs"] == 0
    assert baseline_job.config["training"]["phases"]["phase3"]["epochs"] == 0


def test_build_cross_species_dataset_jobs_forces_bf16_amp(tmp_path: Path) -> None:
    jobs = build_cross_species_dataset_jobs(
        dataset_key="stripespotter",
        output_root=tmp_path,
        device="cuda",
    )

    for job in jobs:
        assert job.config["hardware"]["use_amp"] is True
        assert job.config["hardware"]["amp_dtype"] == "bfloat16"


def test_ensure_selection_query_gallery_split_creates_harder_multi_query_multi_gallery_split(tmp_path: Path) -> None:
    dataset_root = tmp_path / "toy"
    train_root = dataset_root / "train"
    (train_root / "id_a").mkdir(parents=True)
    (train_root / "id_b").mkdir(parents=True)
    (train_root / "id_c").mkdir(parents=True)

    for idx in range(8):
        (train_root / "id_a" / f"a_{idx}.jpg").write_bytes(b"a")
    for idx in range(6):
        (train_root / "id_b" / f"b_{idx}.jpg").write_bytes(b"b")
    for idx in range(4):
        (train_root / "id_c" / f"c_{idx}.jpg").write_bytes(b"c")

    runtime = _ensure_selection_query_gallery_split(dataset_root, dataset_key="toy")

    assert Path(runtime["selection_train_dir"]).exists()
    assert Path(runtime["selection_query_dir"]).exists()
    assert Path(runtime["selection_gallery_dir"]).exists()

    train_a = sorted(p.name for p in (Path(runtime["selection_train_dir"]) / "id_a").iterdir())
    query_a = sorted(p.name for p in (Path(runtime["selection_query_dir"]) / "id_a").iterdir())
    gallery_a = sorted(p.name for p in (Path(runtime["selection_gallery_dir"]) / "id_a").iterdir())
    assert train_a == ["a_2.jpg", "a_3.jpg", "a_4.jpg"]
    assert query_a == ["a_0.jpg", "a_1.jpg"]
    assert gallery_a == ["a_5.jpg", "a_6.jpg", "a_7.jpg"]

    train_b = sorted(p.name for p in (Path(runtime["selection_train_dir"]) / "id_b").iterdir())
    query_b = sorted(p.name for p in (Path(runtime["selection_query_dir"]) / "id_b").iterdir())
    gallery_b = sorted(p.name for p in (Path(runtime["selection_gallery_dir"]) / "id_b").iterdir())
    assert train_b == ["b_1.jpg", "b_2.jpg", "b_3.jpg"]
    assert query_b == ["b_0.jpg"]
    assert gallery_b == ["b_4.jpg", "b_5.jpg"]

    train_c = sorted(p.name for p in (Path(runtime["selection_train_dir"]) / "id_c").iterdir())
    assert train_c == ["c_0.jpg", "c_1.jpg", "c_2.jpg", "c_3.jpg"]
    assert not (Path(runtime["selection_query_dir"]) / "id_c").exists()
    assert not (Path(runtime["selection_gallery_dir"]) / "id_c").exists()

    selection_info = json.loads(Path(runtime["selection_info"]).read_text(encoding="utf-8"))
    assert selection_info["eligible_ids"] == 2
    assert selection_info["stats"] == {"train_imgs": 10, "query_imgs": 3, "gallery_imgs": 5}


def test_parse_eval_metrics_extracts_open_set_breakdown() -> None:
    eval_text = """
    ===== Open-Set ReID Evaluation Results =====
    Rank-1  : 74.12%
    Rank-5  : 88.55%
    Rank-10 : 92.01%
    mAP     : 63.44%
    Rank-1 Seen   : 81.20%
    Rank-1 Unseen : 57.91%
    mAP Seen      : 70.33%
    mAP Unseen    : 49.18%
    """

    metrics = _parse_eval_metrics(eval_text)

    assert metrics == {
        "rank1": 74.12,
        "rank5": 88.55,
        "rank10": 92.01,
        "mAP": 63.44,
        "rank1_seen": 81.20,
        "rank1_unseen": 57.91,
        "mAP_seen": 70.33,
        "mAP_unseen": 49.18,
    }


def test_derive_generic_illumination_config_keeps_photometric_branch_but_disables_model_aware_guidance() -> None:
    full_cfg = load_config(DATASETS["gzgc_zebra"].config_path)

    generic_cfg = derive_generic_illumination_config(full_cfg, backbone_override=DEFAULT_BACKBONE)

    assert generic_cfg["baseline"]["type"] == "generic_illumination"
    assert generic_cfg["model"]["backbone"] == DEFAULT_BACKBONE
    assert generic_cfg["model"]["reid_head"]["type"] == "local_stripe"
    assert generic_cfg["model"]["illumination_module"]["enabled"] is True
    assert generic_cfg["model"]["feature_fusion"]["enabled"] is False
    assert generic_cfg["model"]["branch_attention_fusion"]["enabled"] is False
    assert generic_cfg["model"]["nuisance_head"]["enabled"] is False
    assert generic_cfg["illumination_module"]["module_params"]["use_feature_guided"] is False
    assert generic_cfg["illumination_module"]["module_params"]["enable_task_aware_rollback"] is False
    assert generic_cfg["illumination_module"]["module_params"]["use_model_aware_residual"] is False
    assert generic_cfg["illumination_module"]["module_params"]["enable_coarse_task_grad"] is False
    assert generic_cfg["training"]["iicl"]["enabled"] is False


def test_cleanup_extra_checkpoints_removes_non_best_artifacts(tmp_path: Path) -> None:
    keep_file = tmp_path / "joint_best.pth"
    extra_a = tmp_path / "joint_phase1_epoch10.pth"
    extra_b = tmp_path / "joint_phase2_epoch20.pth"
    non_checkpoint = tmp_path / "train.log"

    keep_file.write_bytes(b"best")
    extra_a.write_bytes(b"a")
    extra_b.write_bytes(b"b")
    non_checkpoint.write_text("log", encoding="utf-8")

    removed = _cleanup_extra_checkpoints(tmp_path, keep_paths=[keep_file])

    assert keep_file.exists()
    assert non_checkpoint.exists()
    assert not extra_a.exists()
    assert not extra_b.exists()
    assert removed == 2
