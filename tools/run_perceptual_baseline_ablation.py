#!/usr/bin/env python3
"""Run frozen perceptual-enhancement baselines on top of the white-box ReID baseline."""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import gc
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import load_config
from tools.run_atrw_main_ablation import _as_dict, _set_output_dir, derive_plain_baseline_config
from tools.run_cross_species_paper_ablation import (
    DEFAULT_BACKBONE,
    DEFAULT_BASELINE_HEAD,
    DEFAULT_FINAL_REPORT_PROTOCOL,
    DEFAULT_SELECTION_METRIC,
    _apply_runtime_dirs,
    _dataset_runtime_from_config,
    _derive_standardized_query_gallery_runtime,
    _force_bf16_amp,
    _materialize_joint_phase_defaults,
    _normalize_training_protocol,
    _stamp_paper_protocol_metadata,
)


DEFAULT_DATASETS = ("atrw", "gzgc_zebra")
DEFAULT_METHODS = ("zerodcepp", "retinexnet")

THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party" / "perceptual_baselines"
ENHANCED_DATA_ROOT = PROJECT_ROOT / "data" / "perceptual_baselines"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "checkpoints" / "perceptual_baselines"


@dataclass(frozen=True)
class MethodSpec:
    key: str
    display_name: str
    repo_url: str
    commit: str
    repo_dirname: str


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    config_path: str
    training_mode: str


@dataclass
class PerceptualJob:
    dataset_key: str
    dataset_display_name: str
    method_key: str
    method_display_name: str
    output_dir: Path
    config_path: Path
    train_log_path: Path
    eval_log_path: Path
    result_path: Path
    enhanced_root: Path
    train_data_dir: str
    query_dir: Optional[str]
    gallery_dir: Optional[str]
    test_dir: Optional[str]
    config: Dict[str, Any]
    train_command: List[str] = field(default_factory=list)
    checkpoint_candidates: List[str] = field(default_factory=list)


METHODS: Dict[str, MethodSpec] = {
    "zerodcepp": MethodSpec(
        key="zerodcepp",
        display_name="Zero-DCE++",
        repo_url="https://github.com/Li-Chongyi/Zero-DCE_extension.git",
        commit="09f202b690f82da939b8e6ec8535960ae97ad8bd",
        repo_dirname="Zero-DCE_extension",
    ),
    "retinexnet": MethodSpec(
        key="retinexnet",
        display_name="RetinexNet",
        repo_url="https://github.com/aasharma90/RetinexNet_PyTorch.git",
        commit="22675105f52432715a7935db16562f5117c9d369",
        repo_dirname="RetinexNet_PyTorch",
    ),
}


DATASETS: Dict[str, DatasetSpec] = {
    "atrw": DatasetSpec(
        key="atrw",
        display_name="ATRW",
        config_path="config/illumination_config_atrw.yaml",
        training_mode="atrw_openset",
    ),
    "gzgc_zebra": DatasetSpec(
        key="gzgc_zebra",
        display_name="GZGC Zebra",
        config_path="config/illumination_config_gzgc_zebra_actual.yaml",
        training_mode="cross_species_query_gallery",
    ),
}


def _image_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            yield path


def _run_logged_command(command: Sequence[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        process = subprocess.run(
            list(command),
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(process.returncode)


def _fmt_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return "-"


def _parse_requested(requested: str, valid: Sequence[str]) -> List[str]:
    if requested.strip().lower() == "all":
        return list(valid)
    selected = [item.strip() for item in requested.split(",") if item.strip()]
    unknown = [item for item in selected if item not in valid]
    if unknown:
        raise ValueError(f"Unknown values: {unknown}. Expected subset of {valid}")
    return selected


def _parse_csv_metrics(eval_text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    patterns = {
        "rank1": r"Rank-1\s*:\s*([0-9.]+)%",
        "rank5": r"Rank-5\s*:\s*([0-9.]+)%",
        "rank10": r"Rank-10\s*:\s*([0-9.]+)%",
        "mAP": r"mAP\s*:\s*([0-9.]+)%",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, eval_text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def _parse_atrw_metrics(eval_text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    single_match = re.search(
        r"Single-camera\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%",
        eval_text,
        flags=re.MULTILINE,
    )
    cross_match = re.search(
        r"Cross-camera\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%",
        eval_text,
        flags=re.MULTILINE,
    )
    mmap_match = re.search(r"mmAP\s+.*?([0-9.]+)%", eval_text, flags=re.MULTILINE)

    if single_match:
        metrics["rank1_single"] = float(single_match.group(1))
        metrics["rank5_single"] = float(single_match.group(2))
        metrics["mAP_single"] = float(single_match.group(3))
    if cross_match:
        metrics["rank1_cross"] = float(cross_match.group(1))
        metrics["rank5_cross"] = float(cross_match.group(2))
        metrics["mAP_cross"] = float(cross_match.group(3))
    if mmap_match:
        metrics["mmAP"] = float(mmap_match.group(1))
    return metrics


def _find_checkpoint(output_dir: Path, checkpoint_candidates: Sequence[str]) -> Optional[Path]:
    for filename in checkpoint_candidates:
        direct_candidate = output_dir / filename
        if direct_candidate.exists():
            return direct_candidate
        recursive_candidates = sorted(
            output_dir.rglob(filename),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if recursive_candidates:
            return recursive_candidates[0]

    for pattern in ("baseline_epoch*.pth", "joint_phase*_epoch*.pth", "joint_epoch*.pth"):
        matches = sorted(output_dir.rglob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    return None


def _clone_or_checkout_repo(spec: MethodSpec) -> Path:
    THIRD_PARTY_ROOT.mkdir(parents=True, exist_ok=True)
    repo_dir = THIRD_PARTY_ROOT / spec.repo_dirname
    git_dir = repo_dir / ".git"
    marker_path = repo_dir / ".pinned_commit"

    if repo_dir.exists() and not git_dir.exists():
        if marker_path.exists() and marker_path.read_text(encoding="utf-8").strip() == spec.commit:
            return repo_dir
        shutil.rmtree(repo_dir)

    def _download_archive() -> Path:
        archive_base = spec.repo_url[:-4] if spec.repo_url.endswith(".git") else spec.repo_url
        archive_url = archive_base.replace("https://github.com/", "https://codeload.github.com/") + f"/zip/{spec.commit}"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            zip_path = tmpdir_path / "repo.zip"
            urllib.request.urlretrieve(archive_url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(tmpdir_path)
            extracted_roots = [path for path in tmpdir_path.iterdir() if path.is_dir()]
            if not extracted_roots:
                raise RuntimeError(f"Archive extraction failed for {archive_url}")
            extracted_root = extracted_roots[0]
            if repo_dir.exists():
                shutil.rmtree(repo_dir)
            shutil.move(str(extracted_root), str(repo_dir))
        marker_path.write_text(spec.commit, encoding="utf-8")
        return repo_dir

    if not repo_dir.exists():
        try:
            subprocess.run(
                ["git", "-c", "http.version=HTTP/1.1", "clone", "--depth", "1", spec.repo_url, str(repo_dir)],
                check=True,
            )
        except subprocess.CalledProcessError:
            return _download_archive()

    try:
        current_commit = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        current_commit = marker_path.read_text(encoding="utf-8").strip() if marker_path.exists() else ""

    if current_commit == spec.commit:
        return repo_dir

    if git_dir.exists():
        subprocess.run(
            ["git", "-C", str(repo_dir), "-c", "http.version=HTTP/1.1", "fetch", "--depth", "1", "origin", spec.commit],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "checkout", "--force", spec.commit],
            check=True,
        )
        return repo_dir

    return _download_archive()


def _load_module_from_path(module_name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, int, int]:
    orig_h, orig_w = x.shape[-2:]
    target_h = ((orig_h + multiple - 1) // multiple) * multiple
    target_w = ((orig_w + multiple - 1) // multiple) * multiple
    pad_h = target_h - orig_h
    pad_w = target_w - orig_w
    if pad_h == 0 and pad_w == 0:
        return x, orig_h, orig_w
    padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, orig_h, orig_w


class FrozenEnhancer:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def enhance(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def enhance_image(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        tensor = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        enhanced = self.enhance(tensor).clamp(0.0, 1.0)[0].cpu()
        array = (enhanced.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8")
        return Image.fromarray(array)


class ZeroDCEPPEnhancer(FrozenEnhancer):
    def __init__(self, repo_dir: Path, device: torch.device) -> None:
        super().__init__(device)
        module = _load_module_from_path(
            "zero_dcepp_model",
            repo_dir / "Zero-DCE++" / "model.py",
        )
        self.net = module.enhance_net_nopool(12).to(device)
        ckpt_path = repo_dir / "Zero-DCE++" / "snapshots_Zero_DCE++" / "Epoch99.pth"
        state_dict = torch.load(ckpt_path, map_location=device)
        self.net.load_state_dict(state_dict, strict=True)
        self.net.eval()

    @torch.inference_mode()
    def enhance(self, x: torch.Tensor) -> torch.Tensor:
        padded, orig_h, orig_w = _pad_to_multiple(x, 12)
        enhanced, _ = self.net(padded)
        return enhanced[..., :orig_h, :orig_w]


class RetinexNetEnhancer(FrozenEnhancer):
    def __init__(self, repo_dir: Path, device: torch.device) -> None:
        super().__init__(device)
        module = _load_module_from_path("retinexnet_model", repo_dir / "model.py")
        self.decom = module.DecomNet().to(device)
        self.relight = module.RelightNet().to(device)

        decom_state = torch.load(repo_dir / "ckpts" / "Decom" / "9200.tar", map_location=device)
        relight_state = torch.load(repo_dir / "ckpts" / "Relight" / "9200.tar", map_location=device)
        self.decom.load_state_dict(decom_state, strict=True)
        self.relight.load_state_dict(relight_state, strict=True)
        self.decom.eval()
        self.relight.eval()

    @torch.inference_mode()
    def enhance(self, x: torch.Tensor) -> torch.Tensor:
        reflectance, illum = self.decom(x)
        illum_delta = self.relight(illum, reflectance)
        return reflectance * illum_delta.repeat(1, 3, 1, 1)


def _build_enhancer(method_key: str, device: torch.device) -> FrozenEnhancer:
    spec = METHODS[method_key]
    repo_dir = _clone_or_checkout_repo(spec)
    if method_key == "zerodcepp":
        return ZeroDCEPPEnhancer(repo_dir, device)
    if method_key == "retinexnet":
        return RetinexNetEnhancer(repo_dir, device)
    raise KeyError(f"Unknown method: {method_key}")


def _atrw_test_dir_from_data_root(data_root: str) -> Path:
    candidate_1 = Path(data_root) / "test"
    candidate_2 = Path(data_root) / "atrw_reid_test" / "test"
    if candidate_1.exists():
        return candidate_1
    if candidate_2.exists():
        return candidate_2
    raise FileNotFoundError(
        f"ATRW test directory not found under {data_root}: {candidate_1} | {candidate_2}"
    )


def _enhance_tree(
    enhancer: FrozenEnhancer,
    source_root: Path,
    target_root: Path,
) -> int:
    if not source_root.exists():
        raise FileNotFoundError(f"Source directory not found: {source_root}")

    count = 0
    for src_path in _image_files(source_root):
        rel_path = src_path.relative_to(source_root)
        dst_path = target_root / rel_path
        if dst_path.exists() and _is_valid_image_file(dst_path):
            count += 1
            continue
        if dst_path.exists():
            dst_path.unlink(missing_ok=True)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src_path) as image:
            enhanced = enhancer.enhance_image(image)
        save_kwargs = {"quality": 95} if dst_path.suffix.lower() in {".jpg", ".jpeg"} else {}
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{dst_path.stem}_",
            suffix=dst_path.suffix,
            dir=str(dst_path.parent),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            enhanced.save(tmp_path, **save_kwargs)
            tmp_path.replace(dst_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        count += 1
    return count


def _device_for_preprocess(device: str) -> torch.device:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


def _is_valid_image_file(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, SyntaxError, ValueError):
        return False


def _release_accelerator_memory(device: torch.device | str) -> None:
    resolved_device = torch.device(device) if isinstance(device, str) else device
    gc.collect()
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        return
    with contextlib.suppress(Exception):
        torch.cuda.synchronize(resolved_device)
    torch.cuda.empty_cache()
    with contextlib.suppress(Exception):
        torch.cuda.ipc_collect()


def _configure_perceptual_checkpointing(config: Dict[str, Any]) -> None:
    checkpoint_cfg = config.setdefault("checkpointing", {})
    checkpoint_cfg["save_interval"] = 0
    checkpoint_cfg["max_keep"] = 2


def _cleanup_stale_baseline_checkpoints(output_dir: Path) -> int:
    removed = 0
    for pattern in ("baseline_epoch*.pth", "baseline_exception_epoch*.pth"):
        for checkpoint_path in output_dir.rglob(pattern):
            checkpoint_path.unlink(missing_ok=True)
            removed += 1
    return removed


def _prepare_atrw_enhanced_dataset(
    method_key: str,
    *,
    device: str,
    atrw_data_root: str,
) -> Path:
    target_root = ENHANCED_DATA_ROOT / "atrw" / method_key
    train_source = PROJECT_ROOT / "data" / "processed" / "atrw" / "train"
    test_source = _atrw_test_dir_from_data_root(atrw_data_root)
    train_target = target_root / "train"
    test_target = target_root / "test"
    preprocess_device = _device_for_preprocess(device)
    enhancer = _build_enhancer(method_key, preprocess_device)
    try:
        print(f"[Preprocess] ATRW / {method_key}: train -> {train_target}")
        _enhance_tree(enhancer, train_source, train_target)
        print(f"[Preprocess] ATRW / {method_key}: test -> {test_target}")
        _enhance_tree(enhancer, test_source, test_target)
    finally:
        del enhancer
        _release_accelerator_memory(preprocess_device)
    return target_root


def _prepare_gzgc_zebra_enhanced_dataset(
    method_key: str,
    *,
    device: str,
) -> Path:
    target_root = ENHANCED_DATA_ROOT / "gzgc_zebra" / method_key
    preprocess_device = _device_for_preprocess(device)
    enhancer = _build_enhancer(method_key, preprocess_device)
    source_root = PROJECT_ROOT / "data" / "processed" / "gzgc_zebra"
    try:
        for subdir in ("train", "query", "gallery"):
            src = source_root / subdir
            dst = target_root / subdir
            print(f"[Preprocess] GZGC Zebra / {method_key}: {subdir} -> {dst}")
            _enhance_tree(enhancer, src, dst)
    finally:
        del enhancer
        _release_accelerator_memory(preprocess_device)
    return target_root


def _build_baseline_command(job: PerceptualJob, backbone: str, device: str) -> List[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "train_baselines.py"),
        "--config",
        str(job.config_path),
        "--data_dir",
        job.train_data_dir,
        "--output_dir",
        str(job.output_dir),
        "--backbone",
        backbone,
        "--device",
        device,
    ]


def _sum_effective_phase_epochs(config: Dict[str, Any]) -> int:
    phases_cfg = _as_dict(_as_dict(config.get("training")).get("phases"))
    return sum(int(_as_dict(phases_cfg.get(phase_name)).get("epochs", 0)) for phase_name in ("phase1", "phase2", "phase3"))


def _build_atrw_job(
    method_key: str,
    output_root: Path,
    device: str,
    *,
    atrw_data_root: str,
    atrw_eval_script_dir: str,
    skip_preprocess: bool,
) -> PerceptualJob:
    dataset = DATASETS["atrw"]
    method = METHODS[method_key]
    enhanced_root = ENHANCED_DATA_ROOT / "atrw" / method_key
    if not skip_preprocess:
        enhanced_root = _prepare_atrw_enhanced_dataset(
            method_key,
            device=device,
            atrw_data_root=atrw_data_root,
        )

    full_cfg = load_config(dataset.config_path)
    derived = derive_plain_baseline_config(
        full_cfg,
        backbone_override=DEFAULT_BACKBONE,
        baseline_head=DEFAULT_BASELINE_HEAD,
    )
    train_data_dir = (enhanced_root / "train").as_posix()
    test_dir = (enhanced_root / "test").as_posix()

    derived.setdefault("training", {})["data_dir"] = train_data_dir
    eval_cfg = derived.setdefault("evaluation", {})
    eval_cfg.setdefault("atrw", {})
    eval_cfg["atrw"]["data_root"] = atrw_data_root
    eval_cfg["atrw"]["eval_script_dir"] = atrw_eval_script_dir
    eval_cfg["atrw"]["test_dir"] = test_dir
    _configure_perceptual_checkpointing(derived)
    derived["perceptual_baseline"] = {
        "enabled": True,
        "method": method_key,
        "display_name": method.display_name,
        "mode": "offline_frozen_enhancement",
        "train_data_dir": train_data_dir,
        "test_dir": test_dir,
        "backbone": DEFAULT_BACKBONE,
        "baseline_head": DEFAULT_BASELINE_HEAD,
        "external_repo": {"url": method.repo_url, "commit": method.commit},
    }

    output_dir = output_root / "atrw" / method_key
    _set_output_dir(derived, output_dir)
    job = PerceptualJob(
        dataset_key="atrw",
        dataset_display_name=dataset.display_name,
        method_key=method_key,
        method_display_name=method.display_name,
        output_dir=output_dir,
        config_path=output_dir / "derived_config.yaml",
        train_log_path=output_dir / "train.log",
        eval_log_path=output_dir / "eval.log",
        result_path=output_dir / "result.json",
        enhanced_root=enhanced_root,
        train_data_dir=train_data_dir,
        query_dir=None,
        gallery_dir=None,
        test_dir=test_dir,
        config=derived,
        checkpoint_candidates=["baseline_best.pth", "baseline_best_reid_best.pth"],
    )
    job.train_command = _build_baseline_command(job, backbone=DEFAULT_BACKBONE, device=device)
    return job


def _build_gzgc_zebra_job(method_key: str, output_root: Path, device: str, *, skip_preprocess: bool) -> PerceptualJob:
    dataset = DATASETS["gzgc_zebra"]
    method = METHODS[method_key]
    enhanced_root = ENHANCED_DATA_ROOT / "gzgc_zebra" / method_key
    if not skip_preprocess:
        enhanced_root = _prepare_gzgc_zebra_enhanced_dataset(method_key, device=device)

    full_cfg = load_config(dataset.config_path)
    _materialize_joint_phase_defaults(full_cfg)
    base_runtime = _dataset_runtime_from_config(full_cfg, dataset.key)
    enhanced_runtime = {
        "train_data_dir": (enhanced_root / "train").as_posix(),
        "query_dir": (enhanced_root / "query").as_posix(),
        "gallery_dir": (enhanced_root / "gallery").as_posix(),
        "protocol": base_runtime["protocol"],
    }
    runtime = _derive_standardized_query_gallery_runtime(enhanced_runtime, dataset.key)

    derived = derive_plain_baseline_config(
        copy.deepcopy(full_cfg),
        backbone_override=DEFAULT_BACKBONE,
        total_epochs=_sum_effective_phase_epochs(full_cfg),
        baseline_head=DEFAULT_BASELINE_HEAD,
    )
    _force_bf16_amp(derived)
    _normalize_training_protocol(derived, runtime)
    _apply_runtime_dirs(derived, runtime)
    _stamp_paper_protocol_metadata(
        derived,
        source_dataset_protocol="query_gallery",
        training_selection_protocol=str(_as_dict(derived.get("evaluation")).get("protocol", "")),
        final_report_protocol=DEFAULT_FINAL_REPORT_PROTOCOL,
        selection_metric=DEFAULT_SELECTION_METRIC,
        official_protocol=False,
        note=(
            "Frozen perceptual enhancer baseline: images are enhanced offline before training and evaluation. "
            "Checkpoint choice keeps the same internal held-out query/gallery split as the standard-protocol cross-species runner."
        ),
    )
    derived.setdefault("paper_protocol", {})["selection_info"] = runtime["selection_info"]
    _configure_perceptual_checkpointing(derived)
    derived["perceptual_baseline"] = {
        "enabled": True,
        "method": method_key,
        "display_name": method.display_name,
        "mode": "offline_frozen_enhancement",
        "train_data_dir": runtime["train_data_dir"],
        "query_dir": runtime["query_dir"],
        "gallery_dir": runtime["gallery_dir"],
        "selection_query_dir": runtime["selection_query_dir"],
        "selection_gallery_dir": runtime["selection_gallery_dir"],
        "backbone": DEFAULT_BACKBONE,
        "baseline_head": DEFAULT_BASELINE_HEAD,
        "external_repo": {"url": method.repo_url, "commit": method.commit},
    }

    output_dir = output_root / "gzgc_zebra" / method_key
    _set_output_dir(derived, output_dir)
    job = PerceptualJob(
        dataset_key="gzgc_zebra",
        dataset_display_name=dataset.display_name,
        method_key=method_key,
        method_display_name=method.display_name,
        output_dir=output_dir,
        config_path=output_dir / "derived_config.yaml",
        train_log_path=output_dir / "train.log",
        eval_log_path=output_dir / "eval.log",
        result_path=output_dir / "result.json",
        enhanced_root=enhanced_root,
        train_data_dir=runtime["train_data_dir"],
        query_dir=runtime["query_dir"],
        gallery_dir=runtime["gallery_dir"],
        test_dir=None,
        config=derived,
        checkpoint_candidates=["baseline_best.pth", "baseline_best_reid_best.pth"],
    )
    job.train_command = _build_baseline_command(job, backbone=DEFAULT_BACKBONE, device=device)
    return job


def _materialize_job(job: PerceptualJob) -> None:
    import yaml

    job.output_dir.mkdir(parents=True, exist_ok=True)
    with open(job.config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(job.config, handle, sort_keys=False, allow_unicode=True)


def _evaluate_job(
    job: PerceptualJob,
    checkpoint_path: Path,
    *,
    device: str,
    atrw_data_root: str,
    atrw_eval_script_dir: str,
) -> Dict[str, Any]:
    if job.dataset_key == "atrw":
        eval_command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "eval_atrw_openset.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--data_root",
            atrw_data_root,
            "--eval_script_dir",
            atrw_eval_script_dir,
            "--test_dir",
            str(job.test_dir),
            "--output",
            str(job.output_dir / "submission_openset.json"),
            "--backbone",
            DEFAULT_BACKBONE,
        ]
        eval_code = _run_logged_command(eval_command, job.eval_log_path, PROJECT_ROOT)
        eval_text = job.eval_log_path.read_text(encoding="utf-8", errors="replace")
        return {"return_code": eval_code, "command": " ".join(eval_command), "metrics": _parse_atrw_metrics(eval_text)}

    eval_command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "evaluate_reid.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--query_dir",
        str(job.query_dir),
        "--gallery_dir",
        str(job.gallery_dir),
        "--device",
        device,
        "--baseline",
    ]
    eval_code = _run_logged_command(eval_command, job.eval_log_path, PROJECT_ROOT)
    eval_text = job.eval_log_path.read_text(encoding="utf-8", errors="replace")
    return {"return_code": eval_code, "command": " ".join(eval_command), "metrics": _parse_csv_metrics(eval_text)}


def _write_summary(output_root: Path, results: List[Dict[str, Any]]) -> None:
    summary_json = output_root / "perceptual_baseline_results.json"
    summary_csv = output_root / "perceptual_baseline_table.csv"
    summary_md = output_root / "perceptual_baseline_table.md"

    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    rows: List[Dict[str, Any]] = []
    for result in results:
        metrics = result.get("metrics", {})
        rows.append(
            {
                "dataset": result.get("dataset_key"),
                "method": result.get("method_key"),
                "display_name": result.get("display_name"),
                "rank1": metrics.get("rank1"),
                "mAP": metrics.get("mAP"),
                "rank1_single": metrics.get("rank1_single"),
                "mAP_single": metrics.get("mAP_single"),
                "rank1_cross": metrics.get("rank1_cross"),
                "mAP_cross": metrics.get("mAP_cross"),
                "mmAP": metrics.get("mmAP"),
                "status": result.get("status"),
            }
        )

    with open(summary_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "method",
                "display_name",
                "rank1",
                "mAP",
                "rank1_single",
                "mAP_single",
                "rank1_cross",
                "mAP_cross",
                "mmAP",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Perceptual Baseline Ablation",
        "",
        "| Dataset | Method | Rank-1 | mAP | ATRW Single R1 | ATRW Single mAP | ATRW Cross R1 | ATRW Cross mAP | mmAP | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {display_name} | {rank1} | {mAP} | {rank1_single} | {mAP_single} | {rank1_cross} | {mAP_cross} | {mmAP} | {status} |".format(
                dataset=row["dataset"],
                display_name=row["display_name"],
                rank1=_fmt_metric(row["rank1"]),
                mAP=_fmt_metric(row["mAP"]),
                rank1_single=_fmt_metric(row["rank1_single"]),
                mAP_single=_fmt_metric(row["mAP_single"]),
                rank1_cross=_fmt_metric(row["rank1_cross"]),
                mAP_cross=_fmt_metric(row["mAP_cross"]),
                mmAP=_fmt_metric(row["mmAP"]),
                status=row["status"],
            )
        )
    summary_md.write_text("\n".join(lines), encoding="utf-8")


def build_jobs(
    datasets: Sequence[str],
    methods: Sequence[str],
    *,
    output_root: Path,
    device: str,
    atrw_data_root: str,
    atrw_eval_script_dir: str,
    skip_preprocess: bool,
) -> List[PerceptualJob]:
    jobs: List[PerceptualJob] = []
    for dataset_key in datasets:
        for method_key in methods:
            if dataset_key == "atrw":
                jobs.append(
                    _build_atrw_job(
                        method_key,
                        output_root,
                        device,
                        atrw_data_root=atrw_data_root,
                        atrw_eval_script_dir=atrw_eval_script_dir,
                        skip_preprocess=skip_preprocess,
                    )
                )
            elif dataset_key == "gzgc_zebra":
                jobs.append(_build_gzgc_zebra_job(method_key, output_root, device, skip_preprocess=skip_preprocess))
            else:
                raise KeyError(f"Unsupported dataset for perceptual baseline: {dataset_key}")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen perceptual enhancement baselines")
    parser.add_argument("--datasets", type=str, default="all", help="Comma-separated datasets or 'all'")
    parser.add_argument("--methods", type=str, default="all", help="Comma-separated methods or 'all'")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        help="Output root relative to the project root",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Training / evaluation device")
    parser.add_argument(
        "--atrw_data_root",
        type=str,
        default="orignal_data/Amur Tiger Re-identification",
        help="ATRW original data root used by official open-set evaluation",
    )
    parser.add_argument(
        "--atrw_eval_script_dir",
        type=str,
        default="ATRWEvalScript-main",
        help="ATRW official evaluation script directory",
    )
    parser.add_argument("--skip_preprocess", action="store_true", help="Skip offline enhancement generation")
    parser.add_argument("--dry_run", action="store_true", help="Materialize configs and print commands only")
    args = parser.parse_args()

    output_root = (PROJECT_ROOT / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = _parse_requested(args.datasets, DEFAULT_DATASETS)
    methods = _parse_requested(args.methods, DEFAULT_METHODS)
    jobs = build_jobs(
        datasets,
        methods,
        output_root=output_root,
        device=args.device,
        atrw_data_root=args.atrw_data_root,
        atrw_eval_script_dir=args.atrw_eval_script_dir,
        skip_preprocess=args.skip_preprocess,
    )
    _release_accelerator_memory(_device_for_preprocess(args.device))

    results: List[Dict[str, Any]] = []
    for job in jobs:
        _materialize_job(job)
        if args.dry_run:
            print(f"[Dry Run] {job.dataset_display_name} / {job.method_display_name}")
            print(f"  Enhanced root: {job.enhanced_root}")
            print(f"  Train: {' '.join(job.train_command)}")
            if job.dataset_key == "atrw":
                print(f"  Eval test dir: {job.test_dir}")
            else:
                print(f"  Final q/g: {job.query_dir} | {job.gallery_dir}")
            continue

        removed_before_train = _cleanup_stale_baseline_checkpoints(job.output_dir)
        _release_accelerator_memory(_device_for_preprocess(args.device))
        train_code = _run_logged_command(job.train_command, job.train_log_path, PROJECT_ROOT)
        _release_accelerator_memory(_device_for_preprocess(args.device))
        record: Dict[str, Any] = {
            "dataset_key": job.dataset_key,
            "dataset_display_name": job.dataset_display_name,
            "method_key": job.method_key,
            "display_name": job.method_display_name,
            "train_command": " ".join(job.train_command),
            "train_log": str(job.train_log_path),
            "status": "trained" if train_code == 0 else "train_failed",
            "enhanced_root": str(job.enhanced_root),
            "removed_checkpoints_before_train": removed_before_train,
        }

        if train_code != 0:
            record["train_return_code"] = train_code
            with open(job.result_path, "w", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2)
            results.append(record)
            continue

        checkpoint_path = _find_checkpoint(job.output_dir, job.checkpoint_candidates)
        if checkpoint_path is None:
            record["status"] = "checkpoint_missing"
            with open(job.result_path, "w", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2)
            results.append(record)
            continue

        record["checkpoint"] = str(checkpoint_path)
        eval_result = _evaluate_job(
            job,
            checkpoint_path,
            device=args.device,
            atrw_data_root=args.atrw_data_root,
            atrw_eval_script_dir=args.atrw_eval_script_dir,
        )
        record["eval_command"] = eval_result["command"]
        record["eval_log"] = str(job.eval_log_path)
        record["eval_return_code"] = eval_result["return_code"]
        record["metrics"] = eval_result["metrics"]
        record["status"] = "done" if eval_result["return_code"] == 0 and eval_result["metrics"] else "eval_failed"
        record["removed_checkpoints_after_eval"] = _cleanup_stale_baseline_checkpoints(job.output_dir)

        with open(job.result_path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
        results.append(record)

    if not args.dry_run:
        _write_summary(output_root, results)


if __name__ == "__main__":
    main()
