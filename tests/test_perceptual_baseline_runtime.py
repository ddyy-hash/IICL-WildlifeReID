import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.run_perceptual_baseline_ablation as perceptual_module
from tools.train_baselines import BaselineTrainer


def test_enhance_tree_rebuilds_corrupted_existing_outputs(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    src_path = source_root / "id0" / "sample.jpg"
    dst_path = target_root / "id0" / "sample.jpg"
    src_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(src_path)
    dst_path.write_bytes(b"not-a-valid-image")

    class DummyEnhancer:
        def enhance_image(self, image: Image.Image) -> Image.Image:
            return Image.new("RGB", image.size, color=(200, 100, 50))

    count = perceptual_module._enhance_tree(DummyEnhancer(), source_root, target_root)

    assert count == 1
    with Image.open(dst_path) as rebuilt:
        assert rebuilt.size == (8, 8)
        assert rebuilt.getpixel((0, 0)) == (200, 100, 50)


def test_prepare_atrw_enhanced_dataset_releases_accelerator_memory(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "project"
    enhanced_root = tmp_path / "enhanced"
    train_source = project_root / "data" / "processed" / "atrw" / "train"
    test_source = project_root / "atrw_test"
    train_source.mkdir(parents=True, exist_ok=True)
    test_source.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(perceptual_module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(perceptual_module, "ENHANCED_DATA_ROOT", enhanced_root)
    monkeypatch.setattr(perceptual_module, "_atrw_test_dir_from_data_root", lambda _root: test_source)
    monkeypatch.setattr(perceptual_module, "_build_enhancer", lambda _method_key, _device: object())
    monkeypatch.setattr(perceptual_module, "_enhance_tree", lambda _enhancer, _src, _dst: 0)

    release_calls = []
    monkeypatch.setattr(
        perceptual_module,
        "_release_accelerator_memory",
        lambda device: release_calls.append(str(device)),
        raising=False,
    )

    perceptual_module._prepare_atrw_enhanced_dataset(
        "zerodcepp",
        device="cuda",
        atrw_data_root="ignored",
    )

    assert release_calls == ["cuda"]


def test_baseline_trainer_uses_configured_periodic_checkpoint_interval() -> None:
    saved_epochs = []

    class DummyLogger:
        def info(self, *_args, **_kwargs) -> None:
            return None

        def warning(self, *_args, **_kwargs) -> None:
            return None

    class DummyScheduler:
        def step(self) -> None:
            return None

        def state_dict(self):
            return {}

    class DummyOptimizer:
        param_groups = [{"lr": 1e-4}]

        def state_dict(self):
            return {}

    trainer = BaselineTrainer.__new__(BaselineTrainer)
    trainer.logger = DummyLogger()
    trainer.phase1_epochs = 6
    trainer.eval_interval = 100
    trainer.best_acc = 0.0
    trainer.best_rank1 = 0.0
    trainer.best_map = 0.0
    trainer.best_metric_name = "rank1"
    trainer.best_metric_value = 0.0
    trainer.resume_phase = None
    trainer.resume_epoch = 0
    trainer.resume_optimizer_state = None
    trainer.resume_scheduler_state = None
    trainer.current_phase = 0
    trainer.current_epoch = -1
    trainer.device = torch.device("cpu")
    trainer.is_main_process = False
    trainer.is_distributed = False
    trainer.optimizer = DummyOptimizer()
    trainer.scheduler = DummyScheduler()
    trainer.checkpoint_save_interval = 3
    trainer._setup_optimizer_phase1 = lambda: None
    trainer._set_sampler_epoch = lambda _epoch: None
    trainer.train_epoch = lambda _epoch, phase=1: {"total_loss": 0.0, "accuracy": 100.0}
    trainer._barrier = lambda: None
    trainer.evaluate_for_model_selection = lambda: {}
    trainer._format_eval_metrics = lambda _metrics: ""
    trainer._log_additional_eval_results = lambda: None
    trainer._extract_primary_metric = lambda _metrics: 0.0
    trainer._extract_reid_selection_metrics = lambda _metrics: (0.0, 0.0)
    trainer._save_emergency_checkpoint = lambda _reason: None
    trainer.save_checkpoint = (
        lambda epoch, metrics, phase, is_best=False, suffix="": saved_epochs.append(epoch + 1)
    )

    trainer.train()

    assert saved_epochs == [3, 6]
