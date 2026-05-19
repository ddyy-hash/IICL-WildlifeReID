#!/usr/bin/env python3
"""
Joint training script for wildlife ReID with IPAID and IICL-style consistency regularization.

This script implements a three-phase training strategy:
- Phase 1: ReID warmup with the illumination module frozen but still active in forward.
- Phase 2: Illumination optimization with the backbone frozen and raw-branch geometry preserved.
- Phase 3: Joint optimization with optional feature-consistency regularization on illumination variants.

Key Features:
- IPAID (task-oriented illumination correction for ReID)
- IICL-style variant consistency regularization
- Three-phase training strategy
- PK sampler for balanced batch sampling
- Configurable model-selection protocol; default is the in-training 70/30 split unless overridden

Usage:
    python tools/train_joint.py \
        --data_dir ./data/processed/atrw/train \
        --output_dir ./checkpoints/atrw_joint \
        --config ./config/illumination_config.yaml
"""
#config documnets must be in yaml format, and the config file must contain the following sections:
# config.yaml name rules: "illumination_config_{dataset_name}_{version}.yaml", e.g. "illumination_config_atrw_v1.yaml    "
import os
import sys
import copy
import gc
import random
from contextlib import nullcontext
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import argparse
import logging
from torchvision import transforms
from typing import Any, Optional, List, Tuple, Dict
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_HELP_MODE = any(arg in {"-h", "--help"} for arg in sys.argv)
try:
    from app.core.illumination_module_v2 import (
        CrossCovarianceDecorrelationLoss,
        CrossLightMarginPreservingLoss,
        CrossLightPrototypeLoss,
        FeatureTrustRegionLoss,
        IPAIDModule,
        IPAIDLoss,
        GeometryPreservingLoss,
        IlluminationFeatureConsistencyLoss,
        LocalRankPreservingLoss,
        NeighborhoodConsistencyLoss,
        RankingTopologyPreservingLoss,
        RelativeClassStructureLoss,
        AnisotropicIdentityProtectionLoss,
        SemanticNonConfusionLoss,
        SoftAPCrossLightLoss,
        TeacherPrototypeAnchorLoss,
        TeacherAnchorLoss,
        TeacherManifoldSeparationLoss,
        TeacherManifoldTubeLoss,
        TeacherLogitConsistencyLoss,
    )
    from app.core.metric_losses import TripletLoss, ArcFaceLoss, CircleLoss, CenterLoss
    from app.core.joint_model import JointReIDModel, SUPPORTED_BACKBONES, get_backbone_dim
    from app.core.data import FullImageDataset, PKSampler, DistributedPKSampler
    from app.core.evaluation import (
        ReIDDataset,
        ReIDEvaluator,
        build_submission_from_distance,
        build_submission_from_features,
        compute_cmc_map,
        compute_distance_matrix,
        evaluate_atrw_official,
        extract_features,
        load_atrw_gt,
    )
    from app.core.config import load_config, cli_args_to_config
    from app.core.model_factory import resolve_joint_model_init
    from tools.selection_protocols import (
        CROSS_SPECIES_SELECTION_PROTOCOL,
        QUERY_GALLERY_PROTOCOL,
        resolve_official_query_gallery_eval_spec,
        resolve_selection_query_gallery_eval_spec,
    )
    from tools.reranking import re_ranking
except ModuleNotFoundError as exc:
    if not _HELP_MODE:
        raise

    _IMPORT_ERROR = exc
    SUPPORTED_BACKBONES = ["osnet_ain_x1_0", "osnet_x1_0", "resnet50"]

    def _missing_dependency(*_args, **_kwargs):
        raise ModuleNotFoundError(
            f"Training dependencies missing: {_IMPORT_ERROR}. Please install dependencies first."
        ) from _IMPORT_ERROR

    IPAIDModule = _missing_dependency
    IPAIDLoss = _missing_dependency
    CrossCovarianceDecorrelationLoss = _missing_dependency
    CrossLightMarginPreservingLoss = _missing_dependency
    CrossLightPrototypeLoss = _missing_dependency
    FeatureTrustRegionLoss = _missing_dependency
    GeometryPreservingLoss = _missing_dependency
    IlluminationFeatureConsistencyLoss = _missing_dependency
    LocalRankPreservingLoss = _missing_dependency
    NeighborhoodConsistencyLoss = _missing_dependency
    RankingTopologyPreservingLoss = _missing_dependency
    RelativeClassStructureLoss = _missing_dependency
    AnisotropicIdentityProtectionLoss = _missing_dependency
    SemanticNonConfusionLoss = _missing_dependency
    SoftAPCrossLightLoss = _missing_dependency
    TeacherPrototypeAnchorLoss = _missing_dependency
    TeacherAnchorLoss = _missing_dependency
    TeacherManifoldSeparationLoss = _missing_dependency
    TeacherManifoldTubeLoss = _missing_dependency
    TeacherLogitConsistencyLoss = _missing_dependency
    TripletLoss = _missing_dependency
    ArcFaceLoss = _missing_dependency
    CircleLoss = _missing_dependency
    CenterLoss = _missing_dependency
    JointReIDModel = _missing_dependency
    FullImageDataset = _missing_dependency
    PKSampler = _missing_dependency
    DistributedPKSampler = _missing_dependency
    ReIDDataset = _missing_dependency
    ReIDEvaluator = _missing_dependency

    def get_backbone_dim(*_args, **_kwargs):
        return 256

    def load_config(*_args, **_kwargs):
        return {}

    def cli_args_to_config(*_args, **_kwargs):
        return {}

    def resolve_joint_model_init(*_args, **_kwargs):
        return {}

    CROSS_SPECIES_SELECTION_PROTOCOL = "self_defined_train_qg"
    QUERY_GALLERY_PROTOCOL = "query_gallery"

    def resolve_selection_query_gallery_eval_spec(*_args, **_kwargs):
        return {}

    def resolve_official_query_gallery_eval_spec(*_args, **_kwargs):
        return {}

    def compute_cmc_map(*_args, **_kwargs):
        return None, None

    def compute_distance_matrix(*_args, **_kwargs):
        return None

    def build_submission_from_distance(*_args, **_kwargs):
        return None

    def build_submission_from_features(*_args, **_kwargs):
        return None

    def evaluate_atrw_official(*_args, **_kwargs):
        return None

    def extract_features(*_args, **_kwargs):
        return None

    def load_atrw_gt(*_args, **_kwargs):
        return None

# ============================================================================
#                           Helper Functions
# ============================================================================

def split_train_val(samples: List[Tuple], id_to_samples: Dict, train_ratio: float = 0.7, seed: int = 42):
    """
    Split dataset into train/val sets with 70/30 ratio within each identity.

    This is the in-training split protocol used by this script.
    Each identity's images are split 70% for training and 30% for validation.

    Args:
        samples: List of (img_path, label) tuples
        id_to_samples: Dict mapping label -> list of sample indices
        train_ratio: Ratio of training samples (default 0.7)
        seed: Random seed for reproducibility

    Returns:
        train_samples: List of training samples
        val_samples: List of validation samples
        num_classes: Number of unique identities
    """
    rng = np.random.RandomState(seed)
    train_samples = []
    val_samples = []

    for label, indices in id_to_samples.items():
        indices = list(indices)
        rng.shuffle(indices)

        split_point = int(len(indices) * train_ratio)
        train_indices = indices[:split_point]
        val_indices = indices[split_point:]

        for idx in train_indices:
            train_samples.append(samples[idx])
        for idx in val_indices:
            val_samples.append(samples[idx])

    num_classes = len(id_to_samples)
    return train_samples, val_samples, num_classes


def _release_cuda_eval_memory() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _compute_nonnegative_gradient_alignment(
    primary_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    anchor: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Keep auxiliary objectives only when they agree with the ReID gradient."""
    if not isinstance(primary_loss, torch.Tensor) or not isinstance(auxiliary_loss, torch.Tensor):
        return anchor.new_tensor(1.0)
    if not primary_loss.requires_grad or not auxiliary_loss.requires_grad or not anchor.requires_grad:
        return anchor.new_tensor(1.0)

    grad_primary = torch.autograd.grad(
        primary_loss,
        anchor,
        retain_graph=True,
        allow_unused=True,
    )[0]
    grad_auxiliary = torch.autograd.grad(
        auxiliary_loss,
        anchor,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad_primary is None or grad_auxiliary is None:
        return anchor.new_tensor(1.0)

    grad_primary = grad_primary.detach().reshape(grad_primary.size(0), -1)
    grad_auxiliary = grad_auxiliary.detach().reshape(grad_auxiliary.size(0), -1)
    cosine = torch.nn.functional.cosine_similarity(
        grad_primary,
        grad_auxiliary,
        dim=1,
        eps=eps,
    )
    return cosine.clamp_min(0.0).mean()


def _linear_warmup_value(
    start: float,
    end: float,
    epoch: int,
    warmup_epochs: int,
) -> float:
    """Linearly interpolate an auxiliary weight during the early joint stage."""
    if warmup_epochs <= 0:
        return float(end)
    if epoch <= 0:
        return float(start)
    if warmup_epochs <= 1 or epoch >= warmup_epochs - 1:
        return float(end)
    progress = epoch / float(max(warmup_epochs - 1, 1))
    return float(start + (end - start) * progress)


SUPPORTED_EVAL_PROTOCOLS = {
    'val_split_70_30',
    QUERY_GALLERY_PROTOCOL,
    CROSS_SPECIES_SELECTION_PROTOCOL,
    'atrw_openset',
    'atrw_closedset_train70_val30',
    'atrw_closedset_animals_701',
}

ATRW_EVAL_PROTOCOLS = {
    'atrw_openset',
    'atrw_closedset_train70_val30',
    'atrw_closedset_animals_701',
}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def resolve_training_seed(config: Optional[Dict[str, Any]], default: int = 42) -> int:
    """Resolve the experiment seed from a config snapshot."""
    if not isinstance(config, dict):
        return int(default)
    candidates = (
        config.get("seed"),
        (config.get("experiment", {}) or {}).get("seed")
        if isinstance(config.get("experiment", {}), dict)
        else None,
        (config.get("training", {}) or {}).get("seed")
        if isinstance(config.get("training", {}), dict)
        else None,
    )
    for value in candidates:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return int(default)


def set_global_random_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable seed-stability runs."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_distributed_mode(
    requested_device: str,
    config: Dict[str, Any],
    cli_local_rank: int = 0,
) -> Dict[str, Any]:
    hardware_cfg = config.get('hardware', {}) or {}
    use_ddp = bool(hardware_cfg.get('use_ddp', True))
    world_size = _env_int('WORLD_SIZE', 1)
    rank = _env_int('RANK', 0)
    local_rank = _env_int('LOCAL_RANK', cli_local_rank)
    is_distributed = use_ddp and world_size > 1

    if is_distributed:
        if requested_device == 'cpu':
            backend = 'gloo'
        else:
            backend = str(hardware_cfg.get('ddp_backend', 'nccl')).strip().lower() or 'nccl'

        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable, but DDP launch was requested")

        if not dist.is_initialized():
            timeout_minutes = int(hardware_cfg.get('ddp_timeout_minutes', 30))
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                timeout=timedelta(minutes=timeout_minutes),
            )

        if torch.cuda.is_available() and requested_device != 'cpu':
            torch.cuda.set_device(local_rank)
    else:
        backend = None

    return {
        'is_distributed': is_distributed,
        'rank': rank,
        'local_rank': local_rank,
        'world_size': world_size,
        'backend': backend,
        'find_unused_parameters': bool(hardware_cfg.get('ddp_find_unused_parameters', False)),
    }


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class JointModelForwardAdapter(nn.Module):
    """Expose all train-time forward paths through a single DDP-safe forward API."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        images: torch.Tensor,
        forward_mode: str = 'standard',
        return_illuminated: bool = False,
        return_local_features: bool = False,
        num_variants: int = 2,
        detach_reference: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if forward_mode == 'consistency':
            return self.model.forward_with_consistency_variants(
                images,
                num_variants=num_variants,
                return_local_features=return_local_features,
            )
        if forward_mode == 'illumination_only':
            return self.model.forward_illumination_only(images, return_illuminated=return_illuminated)
        if forward_mode == 'raw_reference':
            return self.model.forward_raw_reference(images, detach=detach_reference, return_local_features=return_local_features)
        return self.model(
            images,
            None,
            return_illuminated=return_illuminated,
            return_local_features=return_local_features,
        )


# ============================================================================
#                           Joint Trainer
# ============================================================================

class JointTrainer:
    """
    Three-phase joint training for wildlife ReID with illumination handling.

    Training Phases:
    - Phase 1: train ReID modules while the illumination module is frozen but still used in forward
    - Phase 2: optimize the illumination module while the backbone stays frozen
    - Phase 3: jointly optimize ReID and illumination with optional IICL consistency
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        config: Optional[dict] = None,
        config_path: Optional[str] = None,
        backbone: str = "osnet_ain_x1_0",
        batch_size: int = 28,
        phase1_epochs: int = 50,
        phase2_epochs: int = 150,
        phase3_epochs: int = 100,
        learning_rate: float = 3e-4,
        num_stripes: int = 6,
        device: str = 'auto',
        logger: Optional[logging.Logger] = None,
        resume_checkpoint: Optional[str] = None,
        eval_interval: int = 5,
        p_size: Optional[int] = None,
        k_size: int = 4,
        circle_gamma: int = 256,
        img_height: int = 256,
        img_width: int = 256,
        use_iicl: bool = True,
        iicl_weight: float = 1.0,
        iicl_num_variants: int = 2,
        num_workers: int = 4,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
        is_distributed: bool = False,
        ddp_find_unused_parameters: bool = False,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.backbone = backbone
        self.batch_size = batch_size
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs
        self.phase2_fgid_epochs = 10  # Will be overridden from config
        self.phase3_epochs = phase3_epochs
        self.learning_rate = learning_rate
        self.num_stripes = num_stripes
        self.logger = logger or logging.getLogger('JointTraining')
        self.resume_checkpoint = resume_checkpoint
        self.resume_phase: Optional[int] = None
        self.resume_epoch: int = 0
        self.resume_optimizer_state: Optional[dict] = None
        self.resume_scheduler_state: Optional[dict] = None
        self.best_acc: float = 0.0
        self.best_rank1: float = 0.0
        self.best_map: float = 0.0
        self.best_metric_value: float = 0.0
        self.current_phase: int = 0
        self.current_epoch: int = -1

        # Evaluation interval for in-training validation split.
        self.eval_interval = eval_interval

        # PK sampler parameters
        self.p_size = p_size
        self.k_size = k_size
        self.circle_gamma = circle_gamma

        # Image dimensions
        self.img_height = img_height
        self.img_width = img_width

        # IICL parameters
        self.use_iicl_arg = use_iicl
        self.iicl_weight_arg = iicl_weight
        self.iicl_num_variants_arg = iicl_num_variants

        # Number of workers
        self.num_workers = num_workers
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_distributed = is_distributed
        self.is_main_process = self.rank == 0
        self.ddp_find_unused_parameters = ddp_find_unused_parameters
        self.model_ddp: Optional[nn.Module] = None
        self.model_forward_adapter: Optional[nn.Module] = None
        self.phase3_teacher_model: Optional[nn.Module] = None
        self.train_sampler = None
        self.pk_sampler = None

        # Unified config object (defaults <- yaml <- cli)
        self.config = config if config is not None else load_config(config_path)
        self.seed = resolve_training_seed(self.config, default=42)

        eval_cfg = self.config.get('evaluation', {}) or {}
        self.eval_cfg = eval_cfg
        self.atrw_eval_cfg = eval_cfg.get('atrw', {}) or {}
        default_protocol = 'val_split_70_30'
        self.eval_protocol = str(eval_cfg.get('protocol', default_protocol)).strip().lower()
        additional_protocols = eval_cfg.get('additional_protocols', []) or []
        if isinstance(additional_protocols, str):
            additional_protocols = [additional_protocols]
        self.additional_eval_protocols = []
        for protocol in additional_protocols:
            protocol_name = str(protocol).strip().lower()
            if not protocol_name or protocol_name == self.eval_protocol:
                continue
            if protocol_name not in self.additional_eval_protocols:
                self.additional_eval_protocols.append(protocol_name)
        self.last_additional_eval_results: Dict[str, Dict[str, float]] = {}
        self.best_metric_name = str(eval_cfg.get('best_metric', 'rank1')).strip()
        self.strict_protocol_check = bool(eval_cfg.get('strict_protocol_check', False))
        self.selection_eval_spec: Optional[Dict[str, str]] = None
        requested_protocols = [self.eval_protocol] + self.additional_eval_protocols
        if QUERY_GALLERY_PROTOCOL in requested_protocols:
            self.selection_eval_spec = resolve_official_query_gallery_eval_spec(eval_cfg)
        elif CROSS_SPECIES_SELECTION_PROTOCOL in requested_protocols:
            self.selection_eval_spec = resolve_selection_query_gallery_eval_spec(eval_cfg)
        feature_extraction_cfg = eval_cfg.get('feature_extraction', {}) or {}
        self.eval_flip_test = bool(feature_extraction_cfg.get('flip_test', eval_cfg.get('flip_test', False)))
        if self.eval_protocol == 'atrw_openset' and self.best_metric_name == 'rank1':
            self.best_metric_name = 'mmAP'

        # Device setup
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device(f'cuda:{self.local_rank}' if self.is_distributed else 'cuda')
            else:
                self.device = torch.device('cpu')
        else:
            if device == 'cuda' and self.is_distributed:
                self.device = torch.device(f'cuda:{self.local_rank}')
            else:
                self.device = torch.device(device)

        set_global_random_seed(self.seed)

        hardware_cfg = self.config.get('hardware', {}) or {}
        amp_dtype_name = str(hardware_cfg.get('amp_dtype', 'float16')).strip().lower()
        amp_dtype_map = {
            'float16': torch.float16,
            'fp16': torch.float16,
            'half': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
        }
        self.amp_dtype = amp_dtype_map.get(amp_dtype_name, torch.float16)
        self.use_amp = bool(hardware_cfg.get('use_amp', True)) and self.device.type == 'cuda'
        self.grad_scaler = torch.amp.GradScaler(
            'cuda',
            enabled=self.use_amp and self.amp_dtype == torch.float16,
        )

        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Backbone: {self.backbone}")
        self.logger.info(
            "Distributed training: enabled=%s, rank=%d, local_rank=%d, world_size=%d, find_unused=%s",
            self.is_distributed,
            self.rank,
            self.local_rank,
            self.world_size,
            self.ddp_find_unused_parameters,
        )
        self.logger.info(
            "Mixed precision: enabled=%s, dtype=%s, grad_scaler=%s",
            self.use_amp,
            'bf16' if self.amp_dtype == torch.bfloat16 else 'fp16',
            self.grad_scaler.is_enabled(),
        )
        self.logger.info("Experiment seed: %d", self.seed)
        self.logger.info(
            f"Evaluation setup: protocol={self.eval_protocol}, additional={self.additional_eval_protocols or ['none']}, "
            f"best_metric={self.best_metric_name}, "
            f"flip_test={self.eval_flip_test}, "
            f"strict={self.strict_protocol_check}"
        )
        if any(protocol == 'atrw_openset' for protocol in [self.eval_protocol] + self.additional_eval_protocols):
            self.logger.warning(
                "ATRW open-set model selection is enabled during training. "
                "This is convenient for engineering selection, but it is not a strict train/val protocol."
            )
        if self.selection_eval_spec is not None:
            self.logger.info(
                "Cross-species selection query/gallery: query=%s gallery=%s",
                self.selection_eval_spec.get("query_dir"),
                self.selection_eval_spec.get("gallery_dir"),
            )

        os.makedirs(output_dir, exist_ok=True)

        # Initialize components
        self._init_dataloader()
        self._init_model()
        self._init_losses()
        self._maybe_resume()

        # Prepare in-training validation split used for model selection.
        self._prepare_validation_split()

    def _autocast_context(self):
        if self.use_amp and self.device.type == 'cuda':
            return torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype)
        return nullcontext()

    def _autocast_disabled_context(self):
        if self.device.type in {'cuda', 'cpu'}:
            return torch.amp.autocast(device_type=self.device.type, enabled=False)
        return nullcontext()

    def _model_for_training(self) -> nn.Module:
        return self.model_ddp if self.model_ddp is not None else self.model_forward_adapter

    def _barrier(self) -> None:
        if self.is_distributed and dist.is_initialized():
            dist.barrier()

    def _set_sampler_epoch(self, epoch: int) -> None:
        if self.train_sampler is not None and hasattr(self.train_sampler, 'set_epoch'):
            self.train_sampler.set_epoch(epoch)
        elif hasattr(self, 'pk_sampler') and hasattr(self.pk_sampler, 'set_epoch'):
            self.pk_sampler.set_epoch(epoch)

    def _seed_worker(self, worker_id: int) -> None:
        worker_seed = (int(self.seed) + int(self.rank) * 1000 + int(worker_id)) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def _reduce_epoch_stats(
        self,
        total_loss: float,
        illum_loss_sum: float,
        reid_loss_sum: float,
        correct: int,
        total: int,
        num_batches: int,
    ) -> Tuple[float, float, float, int, int, int]:
        stats = torch.tensor(
            [
                total_loss,
                illum_loss_sum,
                reid_loss_sum,
                float(correct),
                float(total),
                float(num_batches),
            ],
            device=self.device if self.device.type == 'cuda' else torch.device('cpu'),
            dtype=torch.float64,
        )
        if self.is_distributed:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        reduced = stats.tolist()
        return (
            float(reduced[0]),
            float(reduced[1]),
            float(reduced[2]),
            int(reduced[3]),
            int(reduced[4]),
            int(reduced[5]),
        )

    @staticmethod
    def _summarize_tensor(tensor: torch.Tensor) -> str:
        detached = tensor.detach()
        shape = tuple(detached.shape)
        dtype = str(detached.dtype)
        numel = detached.numel()
        finite_mask = torch.isfinite(detached)
        finite_count = int(finite_mask.sum().item())
        if finite_count == 0:
            return f"shape={shape}, dtype={dtype}, finite=0/{numel}"

        finite_values = detached[finite_mask]
        return (
            f"shape={shape}, dtype={dtype}, finite={finite_count}/{numel}, "
            f"min={finite_values.min().item():.6f}, max={finite_values.max().item():.6f}, "
            f"mean={finite_values.mean().item():.6f}"
        )

    def _ensure_finite_tensor(
        self,
        tensor: Optional[torch.Tensor],
        name: str,
        phase: int,
        epoch: int,
        batch_idx: int,
    ) -> None:
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return
        if torch.isfinite(tensor).all():
            return

        message = (
            f"Non-finite tensor detected: {name} "
            f"(phase={phase}, epoch={epoch + 1}, batch={batch_idx + 1}) | "
            f"{self._summarize_tensor(tensor)}"
        )
        self.logger.error(message)
        raise RuntimeError(message)

    @staticmethod
    def _strip_state_dict_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        return {
            (key[len(prefix):] if key.startswith(prefix) else key): value
            for key, value in state_dict.items()
        }

    def _load_model_state_dict_compat(self, model_state: Dict[str, torch.Tensor]):
        candidates = [
            model_state,
            self._strip_state_dict_prefix(model_state, 'module.model.'),
            self._strip_state_dict_prefix(model_state, 'module.'),
            self._strip_state_dict_prefix(model_state, 'model.'),
        ]

        tried = set()
        last_result = None
        for candidate in candidates:
            signature = tuple(candidate.keys())
            if signature in tried:
                continue
            tried.add(signature)
            last_result = self.model.load_state_dict(candidate, strict=False)
            missing_keys = getattr(last_result, 'missing_keys', [])
            unexpected_keys = getattr(last_result, 'unexpected_keys', [])
            if not unexpected_keys:
                return last_result

        if last_result is None:
            raise RuntimeError("No compatible checkpoint state_dict candidate was produced")
        return last_result

    def _save_emergency_checkpoint(self, reason: str) -> None:
        if not self.is_main_process:
            return
        if not hasattr(self, 'optimizer') or not hasattr(self, 'scheduler'):
            return

        path = os.path.join(
            self.output_dir,
            f'joint_{reason}_phase{self.current_phase}_epoch{max(self.current_epoch + 1, 0)}.pth',
        )
        checkpoint = {
            'epoch': self.current_epoch,
            'phase': self.current_phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'num_classes': self.num_classes,
            'config': self.config,
            'best_acc': self.best_acc,
            'best_rank1': self.best_rank1,
            'best_map': self.best_map,
            'best_metric_name': self.best_metric_name,
            'best_metric_value': self.best_metric_value,
            'reason': reason,
        }
        torch.save(checkpoint, path)
        self.logger.warning("Emergency checkpoint saved: %s", path)

    def _prepare_validation_split(self):
        """
        Prepare a 70/30 validation split for in-training model selection.

        Note:
            This does not rebuild the training dataloader; training still uses the
            dataset initialized in _init_dataloader().
        """
        requested_protocols = [self.eval_protocol] + self.additional_eval_protocols
        unsupported_protocols = [p for p in requested_protocols if p not in SUPPORTED_EVAL_PROTOCOLS]
        if unsupported_protocols:
            message = (
                f"Unsupported in-training eval protocol(s): {unsupported_protocols}. "
                f"Supported protocols: val_split_70_30, {QUERY_GALLERY_PROTOCOL}, "
                f"{CROSS_SPECIES_SELECTION_PROTOCOL}, atrw_openset, "
                "atrw_closedset_train70_val30, atrw_closedset_animals_701."
            )
            if self.strict_protocol_check:
                raise RuntimeError(message)
            self.logger.warning("%s", message)
            if self.eval_protocol in unsupported_protocols:
                self.eval_protocol = 'val_split_70_30'
            self.additional_eval_protocols = [
                p for p in self.additional_eval_protocols if p in SUPPORTED_EVAL_PROTOCOLS and p != self.eval_protocol
            ]
            requested_protocols = [self.eval_protocol] + self.additional_eval_protocols

        atrw_protocols = [p for p in requested_protocols if p in ATRW_EVAL_PROTOCOLS]
        for protocol in atrw_protocols:
            self.logger.warning(
                "In-training model selection uses ATRW protocol '%s'. "
                "If this protocol touches official test data, treat results as engineering selection, not strict academic validation.",
                protocol,
            )

        if 'val_split_70_30' not in requested_protocols:
            self.val_samples = []
        else:
            # Build validation subset from the loaded training data.
            self.logger.info("Preparing 70/30 train/val split for evaluation...")

            # Get all samples and build id_to_samples mapping
            samples = []
            id_to_samples = defaultdict(list)

            for idx in range(len(self.dataset)):
                img_path, label = self.dataset.samples[idx]
                samples.append((img_path, label))
                id_to_samples[label].append(idx)

            # Split into train/val
            _, self.val_samples, _ = split_train_val(samples, id_to_samples, train_ratio=0.7, seed=self.seed)

            self.logger.info(f"Validation set: {len(self.val_samples)} images from {len(id_to_samples)} identities")
            self.logger.info("Evaluation will use all-vs-all protocol on validation subset")

        if QUERY_GALLERY_PROTOCOL in requested_protocols or CROSS_SPECIES_SELECTION_PROTOCOL in requested_protocols:
            if self.selection_eval_spec is None:
                raise RuntimeError("Query/gallery evaluation protocol requested but query/gallery paths are missing.")
            missing_paths = [
                path
                for path in (
                    self.selection_eval_spec.get("query_dir"),
                    self.selection_eval_spec.get("gallery_dir"),
                )
                if not path or not os.path.exists(path)
            ]
            if missing_paths:
                raise FileNotFoundError(
                    f"Selection query/gallery protocol paths are missing: {missing_paths}"
                )

    def _init_dataloader(self):
        """Initialize dataloader and transforms."""
        train_cfg = self.config.get('training', {})
        aug_cfg = self.config.get('data_augmentation', {})

        # Determine image size
        if self.img_height != 256 or self.img_width != 256:
            img_h, img_w = self.img_height, self.img_width
            self.logger.info(f"Using custom image size: {img_h}x{img_w}")
        else:
            img_h = train_cfg.get('image_height', train_cfg.get('image_size', 256))
            # Default width fallback keeps ATRW's common aspect ratio when width is not explicitly set.
            img_w = train_cfg.get('image_width', train_cfg.get('image_size', 512))

        self.img_height = img_h
        self.img_width = img_w

        # Calculate crop size for random crop augmentation
        crop_cfg = aug_cfg.get('random_crop', {})
        if crop_cfg.get('enabled', True):
            crop_scale = crop_cfg.get('scale', [0.85, 1.0])[1] / crop_cfg.get('scale', [0.85, 1.0])[0]
            crop_scale = min(1.2, 1.0 / crop_cfg.get('scale', [0.85, 1.0])[0])
        else:
            crop_scale = 1.125
        crop_h = int(img_h * crop_scale)
        crop_w = int(img_w * crop_scale)

        color_cfg = aug_cfg.get('color_jitter', {})

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((crop_h, crop_w)),
            transforms.RandomCrop((img_h, img_w)),
            transforms.RandomHorizontalFlip(p=aug_cfg.get('random_horizontal_flip', 0.5)),
            transforms.ColorJitter(
                brightness=color_cfg.get('brightness', 0.2),
                contrast=color_cfg.get('contrast', 0.15),
                saturation=color_cfg.get('saturation', 0.15),
                hue=color_cfg.get('hue', 0.03)
            ),
            transforms.ToTensor(),
            # Explicit normalization is deferred to joint_model.py.
            # Backbone-space Gaussian erasing is applied inside JointReIDModel
            # after ImageNet normalization so the illumination branch keeps
            # seeing physically valid RGB images in [0, 1].
        ])

        self.dataset = FullImageDataset(self.data_dir, transform=transform)
        self.num_classes = self.dataset.num_classes
        self._configure_train_dataloader(
            batch_size=self.batch_size,
            p_size_override=self.p_size,
            phase=0,
            log_prefix="Initial",
        )

    def _configure_train_dataloader(
        self,
        batch_size: int,
        p_size_override: Optional[int],
        phase: int,
        log_prefix: str,
    ) -> None:
        """Build the train dataloader for the requested phase-specific batch shape."""
        requested_batch_size = max(int(batch_size), 1)
        pk_cfg = self.config.get('training', {}).get('pk_sampler', {})
        use_pk_sampler = pk_cfg.get('enabled', True)
        num_workers = self.num_workers
        self.train_sampler = None
        self.pk_sampler = None

        if use_pk_sampler and len(self.dataset.idx_to_samples) >= 4:
            k = self.k_size if self.k_size else pk_cfg.get('k', 4)
            p = p_size_override if p_size_override is not None else requested_batch_size // k

            # Ensure P >= 2 for metric-learning batches.
            if p < 2:
                p = 2
                k = max(requested_batch_size // p, 2)

            # Ensure P doesn't exceed number of identities.
            p = min(p, len(self.dataset.idx_to_samples))

            actual_batch_size = p * k

            if self.is_distributed:
                if actual_batch_size % self.world_size != 0:
                    raise RuntimeError(
                        f"Global batch_size={actual_batch_size} must be divisible by world_size={self.world_size}"
                    )
                if p % self.world_size != 0:
                    raise RuntimeError(
                        f"Global P={p} must be divisible by world_size={self.world_size} for distributed PK sampling"
                    )

                self.pk_sampler = DistributedPKSampler(
                    self.dataset,
                    p=p,
                    k=k,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    seed=self.seed,
                )
                local_batch_size = self.pk_sampler.local_batch_size
                self.train_sampler = self.pk_sampler
                self.dataloader = DataLoader(
                    self.dataset,
                    batch_size=local_batch_size,
                    sampler=self.pk_sampler,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=False,
                    worker_init_fn=self._seed_worker,
                )
                self.logger.info(
                    "%s [Distributed PK Sampler] phase=%d global_P=%d, local_P=%d, K=%d, global_batch=%d, local_batch=%d",
                    log_prefix,
                    phase,
                    p,
                    self.pk_sampler.local_p,
                    k,
                    actual_batch_size,
                    local_batch_size,
                )
            else:
                self.pk_sampler = PKSampler(self.dataset, p=p, k=k, seed=self.seed)
                self.train_sampler = self.pk_sampler
                self.dataloader = DataLoader(
                    self.dataset,
                    batch_size=actual_batch_size,
                    sampler=self.pk_sampler,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=False,
                    worker_init_fn=self._seed_worker,
                )
                self.logger.info(
                    "%s [PK Sampler] phase=%d P=%d, K=%d, actual_batch_size=%d (requested: %d)",
                    log_prefix,
                    phase,
                    p,
                    k,
                    actual_batch_size,
                    requested_batch_size,
                )
        else:
            if self.is_distributed:
                if requested_batch_size % self.world_size != 0:
                    raise RuntimeError(
                        f"Global batch_size={requested_batch_size} must be divisible by world_size={self.world_size}"
                    )
                local_batch_size = requested_batch_size // self.world_size
                self.train_sampler = DistributedSampler(
                    self.dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True,
                    drop_last=True,
                )
                self.dataloader = DataLoader(
                    self.dataset,
                    batch_size=local_batch_size,
                    sampler=self.train_sampler,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True,
                    worker_init_fn=self._seed_worker,
                )
                self.logger.info(
                    "%s Using DistributedSampler: phase=%d global_batch_size=%d, local_batch_size=%d",
                    log_prefix,
                    phase,
                    requested_batch_size,
                    local_batch_size,
                )
            else:
                self.dataloader = DataLoader(
                    self.dataset,
                    batch_size=requested_batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True,
                    worker_init_fn=self._seed_worker,
                )
                self.logger.info(
                    "%s Using standard random sampler: phase=%d batch_size=%d",
                    log_prefix,
                    phase,
                    requested_batch_size,
                )

        self.logger.info(f"Dataset: {len(self.dataset)} images, {self.num_classes} identities")

    def _get_phase_config(self, phase: int) -> Dict[str, Any]:
        phases_cfg = self.config.get('training', {}).get('phases', {})
        if phase == 1:
            return phases_cfg.get('phase1', {})
        if phase == 2:
            return phases_cfg.get('phase2', {})
        if phase == 3:
            return phases_cfg.get('phase3', phases_cfg.get('phase2', {}))
        return {}

    def _get_phase_batch_settings(self, phase: int) -> Tuple[int, Optional[int]]:
        """Resolve phase-specific batch settings.

        When a phase overrides ``batch_size`` without an explicit ``p``/``p_size``,
        the PK sampler should derive ``P`` from the new batch size instead of
        reusing the global ``self.p_size``.
        """
        phase_config = self._get_phase_config(phase)
        batch_size = max(int(phase_config.get('batch_size', self.batch_size)), 1)

        if 'p_size' in phase_config:
            p_size_override = int(phase_config['p_size'])
        elif 'p' in phase_config:
            p_size_override = int(phase_config['p'])
        elif 'batch_size' in phase_config:
            p_size_override = None
        else:
            p_size_override = self.p_size

        return batch_size, p_size_override

    def _configure_phase_dataloader(self, phase: int, log_prefix: str) -> None:
        batch_size, p_size_override = self._get_phase_batch_settings(phase)
        self._configure_train_dataloader(
            batch_size=batch_size,
            p_size_override=p_size_override,
            phase=phase,
            log_prefix=log_prefix,
        )

    def _set_phase_backbone_checkpointing(self, phase: int) -> None:
        phase_config = self._get_phase_config(phase)
        hardware_cfg = self.config.get('hardware', {}) or {}
        enabled = phase_config.get(
            'use_backbone_checkpointing',
            hardware_cfg.get('use_backbone_checkpointing', True),
        )
        self.model.use_backbone_checkpointing = bool(enabled)
        logger = getattr(self, "logger", None)
        if logger is not None:
            logger.info(
                "Phase %d backbone checkpointing: %s",
                phase,
                self.model.use_backbone_checkpointing,
            )

    def _init_model(self):
        """Initialize the joint ReID model."""
        model_kwargs = resolve_joint_model_init(
            self.config,
            num_classes=self.num_classes,
            backbone_override=self.backbone,
            pretrained_backbone=True,
        )
        model_kwargs['num_stripes'] = self.num_stripes
        self.model = JointReIDModel(**model_kwargs).to(self.device)
        self.model_forward_adapter = JointModelForwardAdapter(self.model).to(self.device)

        self.use_ipaid = bool(model_kwargs.get('use_ipaid', True))
        iicl_cfg = self.config.get('training', {}).get('iicl', {})
        self.model.num_grad_variants = int(
            iicl_cfg.get(
                'num_grad_variants',
                model_kwargs.get('ipaid_params', {}).get('num_grad_variants', 1),
            )
        )
        if getattr(self.model, 'variant_generator', None) is not None:
            self.model.variant_generator.num_variants = max(
                int(iicl_cfg.get('num_variants', iicl_cfg.get('variants', 2))),
                self.model.num_grad_variants,
            )

        if self.is_distributed:
            ddp_kwargs = {
                'find_unused_parameters': self.ddp_find_unused_parameters,
            }
            if self.device.type == 'cuda':
                ddp_kwargs['device_ids'] = [self.local_rank]
                ddp_kwargs['output_device'] = self.local_rank
            self.model_ddp = DistributedDataParallel(
                self.model_forward_adapter,
                **ddp_kwargs,
            )

        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model initialized: {total_params/1e6:.2f}M parameters")
        self.logger.info(f"IPAID module: {'enabled' if self.use_ipaid else 'disabled'}")

    def _init_losses(self):
        """Initialize all loss functions."""
        loss_params = self.config['illumination_module']['loss_params']
        train_cfg = self.config.get('training', {})
        metric_cfg = train_cfg.get('metric_learning', {})
        center_cfg = train_cfg.get('center_loss', {})
        photo_prior_cfg = train_cfg.get('photo_prior', {})
        cross_light_cfg = train_cfg.get('cross_light_prototype', {})
        cross_light_margin_cfg = train_cfg.get('cross_light_margin_preserving', {})
        cross_light_softap_cfg = train_cfg.get('cross_light_softap', {})
        teacher_manifold_cfg = train_cfg.get('teacher_manifold', {})
        ranking_topology_cfg = train_cfg.get('ranking_topology', {})
        anisotropic_identity_cfg = train_cfg.get('anisotropic_identity_protection', {})
        semantic_non_confusion_cfg = train_cfg.get('semantic_non_confusion', {})
        nuisance_decoupling_cfg = train_cfg.get('nuisance_decoupling', {})
        teacher_prototype_anchor_cfg = train_cfg.get('teacher_prototype_anchor', {})
        feature_trust_cfg = train_cfg.get('feature_trust_region', {})
        local_rank_cfg = train_cfg.get('local_rank_preserving', {})
        neighborhood_cfg = train_cfg.get('neighborhood_consistency', {})
        identity_image_cfg = train_cfg.get('identity_image_preserving', {})
        aux_gradient_gate_cfg = train_cfg.get('aux_gradient_gate', {})
        identity_cfg = train_cfg.get('identity_preserving', {})

        # IPAID Loss
        self.ipaid_loss = IPAIDLoss(
            lambda_recon=loss_params.get('lambda_recon', 1.0),
            lambda_smooth=loss_params.get('lambda_smooth', 0.15),
            lambda_edge=loss_params.get('lambda_edge', 0.08),
            lambda_structure=loss_params.get('lambda_structure'),
            lambda_sensitivity=loss_params.get('lambda_sensitivity', 0.02),
            lambda_lab_chroma=loss_params.get('lambda_lab_chroma', 0.1),
            lambda_high_freq=loss_params.get('lambda_high_freq', 0.05),
            lambda_log_chroma=loss_params.get('lambda_log_chroma', 0.0),
            chroma_mode=loss_params.get('chroma_mode', 'dual'),
        ).to(self.device)

        # Geometry-preserving teacher losses
        triplet_cfg = metric_cfg.get('triplet_loss', {})
        self.identity_preserving_mode = str(identity_cfg.get('mode', 'geometry')).lower()
        if self.identity_preserving_mode in {'margin', 'cmp', 'ranking'}:
            self.identity_preserving_mode = 'margin_preserving'
        if self.identity_preserving_mode in {'softap', 'listwise', 'listwise_semantic', 'softap_semantic'}:
            self.identity_preserving_mode = 'softap'
        self.identity_similarity = str(identity_cfg.get('similarity', 'cosine')).lower()
        self.identity_geometry_loss_type = str(identity_cfg.get('geometry_loss', 'mse')).lower()
        self.teacher_anchor_loss = TeacherAnchorLoss(
            metric=self.identity_similarity,
        ).to(self.device)
        self.geometry_preserving_loss = GeometryPreservingLoss(
            metric=self.identity_similarity,
            loss_type=self.identity_geometry_loss_type,
        ).to(self.device)
        self.identity_teacher_temperature = float(identity_cfg.get('teacher_temperature', 2.0))
        self.teacher_logit_consistency_loss = TeacherLogitConsistencyLoss(
            temperature=self.identity_teacher_temperature,
        ).to(self.device)
        self.identity_loss_weight = loss_params.get('lambda_identity', 0.1)
        self.identity_anchor_weight = float(identity_cfg.get('anchor_weight', 1.0))
        self.identity_geometry_weight = float(identity_cfg.get('geometry_weight', 0.5))
        self.identity_logit_weight = float(identity_cfg.get('logit_weight', 0.0))
        self.identity_detach_reference = bool(identity_cfg.get('detach_reference', True))
        self.photo_prior_initial_weight = float(photo_prior_cfg.get('initial_weight', 1.0))
        self.photo_prior_min_weight = float(photo_prior_cfg.get('min_weight', 0.35))
        self.photo_prior_decay_power = float(photo_prior_cfg.get('decay_power', 1.0))
        self.identity_phase2_scale = float(identity_cfg.get('phase2_scale', 1.0))
        self.identity_phase3_scale = float(identity_cfg.get('phase3_scale', 0.35))
        self.use_cross_light_prototype = bool(
            cross_light_cfg.get('enabled', self.identity_preserving_mode == 'prototype')
        )
        self.cross_light_prototype_weight = float(cross_light_cfg.get('weight', self.identity_loss_weight))
        self.cross_light_prototype_loss = CrossLightPrototypeLoss(
            similarity=cross_light_cfg.get('similarity', self.identity_similarity),
            photometric_scale=cross_light_cfg.get('photometric_scale', 8.0),
            photometric_offset=cross_light_cfg.get('photometric_offset', 0.1),
            min_gap_weight=cross_light_cfg.get('min_gap_weight', 0.1),
        ).to(self.device)
        self.use_cross_light_margin_preserving = bool(
            cross_light_margin_cfg.get('enabled', self.identity_preserving_mode == 'margin_preserving')
        )
        self.cross_light_margin_preserving_weight = float(
            cross_light_margin_cfg.get('weight', self.identity_loss_weight)
        )
        self.cross_light_margin_preserving_loss = CrossLightMarginPreservingLoss(
            similarity=cross_light_margin_cfg.get('similarity', self.identity_similarity),
            photometric_scale=cross_light_margin_cfg.get('photometric_scale', 8.0),
            photometric_offset=cross_light_margin_cfg.get('photometric_offset', 0.1),
            topk_positive=cross_light_margin_cfg.get('topk_positive', 2),
            topk_negative=cross_light_margin_cfg.get('topk_negative', 4),
            margin_delta=cross_light_margin_cfg.get('margin_delta', 0.02),
            beta=cross_light_margin_cfg.get('beta', 12.0),
        ).to(self.device)
        self.use_cross_light_softap = bool(
            cross_light_softap_cfg.get('enabled', self.identity_preserving_mode == 'softap')
        )
        self.cross_light_softap_weight = float(
            cross_light_softap_cfg.get('weight', self.identity_loss_weight)
        )
        self.cross_light_softap_queue_size = max(int(cross_light_softap_cfg.get('queue_size', 192)), 0)
        self.cross_light_softap_loss = SoftAPCrossLightLoss(
            similarity=cross_light_softap_cfg.get('similarity', self.identity_similarity),
            photometric_scale=cross_light_softap_cfg.get('photometric_scale', 8.0),
            photometric_offset=cross_light_softap_cfg.get('photometric_offset', 0.1),
            min_positive_weight=cross_light_softap_cfg.get('min_positive_weight', 0.05),
            rank_temperature=cross_light_softap_cfg.get('rank_temperature', 0.07),
        ).to(self.device)
        self.use_teacher_manifold = bool(teacher_manifold_cfg.get('enabled', False))
        self.teacher_manifold_tube_weight = float(teacher_manifold_cfg.get('tube_weight', 0.0))
        self.teacher_manifold_separation_weight = float(
            teacher_manifold_cfg.get('separation_weight', 0.0)
        )
        self.teacher_manifold_queue_size = max(int(teacher_manifold_cfg.get('queue_size', 192)), 0)
        self.ranking_topology_queue_size = max(
            int(ranking_topology_cfg.get('queue_size', self.teacher_manifold_queue_size or 192)),
            0,
        )
        self.teacher_gallery_queue_size = max(
            self.teacher_manifold_queue_size,
            self.ranking_topology_queue_size,
        )
        self.teacher_manifold_tube_loss = TeacherManifoldTubeLoss(
            similarity=teacher_manifold_cfg.get('similarity', self.identity_similarity),
            photometric_scale=teacher_manifold_cfg.get('photometric_scale', 8.0),
            photometric_offset=teacher_manifold_cfg.get('photometric_offset', 0.1),
            min_positive_weight=teacher_manifold_cfg.get('min_positive_weight', 0.05),
            shrinkage=teacher_manifold_cfg.get('shrinkage', 0.8),
            orthogonal_weight=teacher_manifold_cfg.get('orthogonal_weight', 1.0),
            subspace_rank=teacher_manifold_cfg.get('subspace_rank', 1),
            min_radius=teacher_manifold_cfg.get('min_radius', 0.02),
        ).to(self.device)
        self.teacher_manifold_separation_loss = TeacherManifoldSeparationLoss(
            similarity=teacher_manifold_cfg.get('similarity', self.identity_similarity),
            photometric_scale=teacher_manifold_cfg.get('photometric_scale', 8.0),
            photometric_offset=teacher_manifold_cfg.get('photometric_offset', 0.1),
            min_positive_weight=teacher_manifold_cfg.get('min_positive_weight', 0.05),
            margin=teacher_manifold_cfg.get('margin', 0.08),
        ).to(self.device)
        self.use_neighborhood_consistency = bool(neighborhood_cfg.get('enabled', False))
        self.neighborhood_consistency_weight = float(neighborhood_cfg.get('weight', 0.0))
        self.neighborhood_consistency_loss = NeighborhoodConsistencyLoss(
            temperature=neighborhood_cfg.get('temperature', 0.07),
            topk=neighborhood_cfg.get('topk', 6),
            positive_weight=neighborhood_cfg.get('positive_weight', 1.0),
            negative_weight=neighborhood_cfg.get('negative_weight', 0.25),
            local_weight=neighborhood_cfg.get('local_weight', 0.35),
            use_global=neighborhood_cfg.get('use_global', True),
            use_local=neighborhood_cfg.get('use_local', True),
            use_hard_negatives=neighborhood_cfg.get('use_hard_negatives', True),
            teacher_target=neighborhood_cfg.get('teacher_target', 'soft'),
        ).to(self.device)
        self.use_ranking_topology = bool(ranking_topology_cfg.get('enabled', False))
        self.ranking_topology_weight = float(ranking_topology_cfg.get('weight', 0.0))
        self.ranking_topology_loss = RankingTopologyPreservingLoss(
            similarity=ranking_topology_cfg.get('similarity', self.identity_similarity),
            photometric_scale=ranking_topology_cfg.get('photometric_scale', 8.0),
            photometric_offset=ranking_topology_cfg.get('photometric_offset', 0.1),
            min_positive_weight=ranking_topology_cfg.get('min_positive_weight', 0.05),
            topk_positive=ranking_topology_cfg.get('topk_positive', 2),
            topk_negative=ranking_topology_cfg.get('topk_negative', 4),
            margin_slack=ranking_topology_cfg.get('margin_slack', 0.01),
            beta=ranking_topology_cfg.get('beta', 12.0),
        ).to(self.device)
        self.use_anisotropic_identity_protection = bool(
            anisotropic_identity_cfg.get('enabled', False)
        )
        self.anisotropic_identity_protection_weight = float(
            anisotropic_identity_cfg.get('weight', 0.0)
        )
        self.anisotropic_identity_protection_loss = AnisotropicIdentityProtectionLoss(
            similarity=anisotropic_identity_cfg.get('similarity', self.identity_similarity),
            photometric_scale=anisotropic_identity_cfg.get('photometric_scale', 8.0),
            photometric_offset=anisotropic_identity_cfg.get('photometric_offset', 0.1),
            min_positive_weight=anisotropic_identity_cfg.get('min_positive_weight', 0.05),
            topk_positive=anisotropic_identity_cfg.get('topk_positive', 2),
            topk_negative=anisotropic_identity_cfg.get('topk_negative', 4),
            subspace_rank=anisotropic_identity_cfg.get('subspace_rank', 1),
            identity_weight=anisotropic_identity_cfg.get('identity_weight', 1.0),
            nuisance_weight=anisotropic_identity_cfg.get('nuisance_weight', 0.5),
            nuisance_radius=anisotropic_identity_cfg.get('nuisance_radius', 0.12),
        ).to(self.device)
        self.use_teacher_gallery_alignment = bool(
            (
                self.use_teacher_manifold
                and (
                    self.teacher_manifold_tube_weight > 0
                    or self.teacher_manifold_separation_weight > 0
                )
            )
            or (self.use_ranking_topology and self.ranking_topology_weight > 0)
            or (self.use_neighborhood_consistency and self.neighborhood_consistency_weight > 0)
            or (
                self.use_anisotropic_identity_protection
                and self.anisotropic_identity_protection_weight > 0
            )
        )
        self.use_semantic_non_confusion = bool(semantic_non_confusion_cfg.get('enabled', False))
        self.semantic_non_confusion_weight = float(
            semantic_non_confusion_cfg.get('weight', 0.0)
        )
        self.semantic_non_confusion_loss = SemanticNonConfusionLoss(
            margin_delta=semantic_non_confusion_cfg.get('margin_delta', 0.02),
            squared=semantic_non_confusion_cfg.get('squared', True),
        ).to(self.device)
        self.use_nuisance_decoupling = bool(nuisance_decoupling_cfg.get('enabled', False))
        self.nuisance_decoupling_weight = float(nuisance_decoupling_cfg.get('weight', 0.0))
        self.nuisance_regression_weight = float(
            nuisance_decoupling_cfg.get('regression_weight', 1.0)
        )
        self.nuisance_decorrelation_weight = float(
            nuisance_decoupling_cfg.get('decorrelation_weight', 1.0)
        )
        self.photometric_regression_loss = nn.SmoothL1Loss().to(self.device)
        self.nuisance_decorrelation_loss = CrossCovarianceDecorrelationLoss().to(self.device)
        self._reset_cross_light_queue()
        self._reset_teacher_manifold_queue()
        self.use_teacher_prototype_anchor = bool(
            teacher_prototype_anchor_cfg.get('enabled', self.identity_preserving_mode == 'margin_preserving')
        )
        self.teacher_prototype_anchor_weight = float(
            teacher_prototype_anchor_cfg.get('weight', 0.0)
        )
        self.teacher_prototype_anchor_loss = TeacherPrototypeAnchorLoss(
            metric=teacher_prototype_anchor_cfg.get('metric', self.identity_similarity),
        ).to(self.device)
        relative_class_structure_cfg = train_cfg.get('relative_class_structure', {})
        self.use_relative_class_structure = bool(
            relative_class_structure_cfg.get('enabled', self.identity_preserving_mode == 'margin_preserving')
        )
        self.relative_class_structure_weight = float(
            relative_class_structure_cfg.get('weight', 0.0)
        )
        self.relative_class_structure_loss = RelativeClassStructureLoss(
            metric=relative_class_structure_cfg.get('metric', self.identity_similarity),
            radial_weight=relative_class_structure_cfg.get('radial_weight', 0.5),
        ).to(self.device)
        self.use_aux_gradient_gate = bool(aux_gradient_gate_cfg.get('enabled', False))
        self.aux_gradient_gate_eps = float(aux_gradient_gate_cfg.get('eps', 1e-8))
        self.use_feature_trust_region = bool(feature_trust_cfg.get('enabled', False))
        self.feature_trust_region_weight = float(feature_trust_cfg.get('weight', 0.0))
        self.feature_trust_region_loss = FeatureTrustRegionLoss(
            base_radius=feature_trust_cfg.get('base_radius', 0.12),
            adaptive_scale=feature_trust_cfg.get('adaptive_scale', 0.0),
            class_spread_scale=feature_trust_cfg.get('class_spread_scale', 0.0),
        ).to(self.device)
        self.use_local_rank_preserving = bool(local_rank_cfg.get('enabled', False))
        self.local_rank_preserving_weight = float(local_rank_cfg.get('weight', 0.0))
        self.local_rank_preserving_loss = LocalRankPreservingLoss(
            alpha=local_rank_cfg.get('alpha', 0.9),
            k_positive=local_rank_cfg.get('k_positive', 1),
            k_negative=local_rank_cfg.get('k_negative', 1),
        ).to(self.device)
        self.use_identity_image_preserving = bool(identity_image_cfg.get('enabled', False))
        self.identity_image_preserving_weight = float(identity_image_cfg.get('weight', 0.0))

        # ReID Losses
        ce_cfg = metric_cfg.get('ce_loss', {})
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=ce_cfg.get('label_smoothing', 0.1)
        )
        self.ce_weight = ce_cfg.get('weight', 1.0)

        self.triplet_loss = TripletLoss(
            margin=triplet_cfg.get('margin', 0.3),
            mining_type=triplet_cfg.get('mining_type', 'soft')
        )
        self.triplet_weight = triplet_cfg.get('weight', 1.0)

        circle_cfg = metric_cfg.get('circle_loss', {})
        circle_gamma_value = self.circle_gamma if self.circle_gamma else circle_cfg.get('gamma', 256)
        self.circle_loss = CircleLoss(
            m=circle_cfg.get('margin', 0.25),
            gamma=circle_gamma_value
        )
        self.circle_weight = circle_cfg.get('weight', 0.5)

        # ArcFace Loss
        arcface_cfg = metric_cfg.get('arcface_loss', {})
        self.arcface_weight = arcface_cfg.get('weight', 1.0)
        if self.arcface_weight > 0:
            feat_dim = center_cfg.get('feat_dim', 256)
            self.arcface_loss = ArcFaceLoss(
                in_features=feat_dim,
                out_features=self.num_classes,
                s=arcface_cfg.get('s', 30.0),
                m=arcface_cfg.get('m', 0.35)
            ).to(self.device)
        else:
            self.arcface_loss = None

        # Center Loss
        if center_cfg.get('enabled', True):
            self.center_loss = CenterLoss(
                num_classes=self.num_classes,
                feat_dim=center_cfg.get('feat_dim', 256),
            ).to(self.device)
            self.center_loss_weight = center_cfg.get('weight', 0.0005)
            self.center_lr_scale = center_cfg.get('lr_scale', 0.5)
        else:
            self.center_loss = None
            self.center_loss_weight = 0

        # IICL: feature consistency across illumination variants
        iicl_cfg = train_cfg.get('iicl', {})
        self.use_iicl = iicl_cfg.get('enabled', True) if self.use_iicl_arg is None else bool(self.use_iicl_arg)
        self.iicl_consistency_loss = IlluminationFeatureConsistencyLoss(
            temperature=iicl_cfg.get('temperature', 0.1),
            loss_type=iicl_cfg.get('loss_type', 'cosine'),
        )
        self.iicl_weight = iicl_cfg.get('weight', 0.5) if self.iicl_weight_arg is None else float(self.iicl_weight_arg)
        cfg_iicl_variants = iicl_cfg.get('num_variants', iicl_cfg.get('variants', 2))
        self.iicl_num_variants = cfg_iicl_variants if self.iicl_num_variants_arg is None else int(self.iicl_num_variants_arg)
        self.iicl_num_grad_variants = int(iicl_cfg.get('num_grad_variants', 1))
        if hasattr(self.model, 'num_grad_variants'):
            self.model.num_grad_variants = self.iicl_num_grad_variants

        phase3_cfg = train_cfg.get('phases', {}).get('phase3', {})
        aux_ramp_cfg = phase3_cfg.get('aux_ramp', {}) or {}
        phase3_illum_end = float(phase3_cfg.get('illumination_weight', 0.2))
        if self.identity_preserving_mode == 'softap':
            phase3_cross_light_end = self.cross_light_softap_weight
        elif self.identity_preserving_mode == 'margin_preserving':
            phase3_cross_light_end = self.cross_light_margin_preserving_weight
        else:
            phase3_cross_light_end = self.cross_light_prototype_weight
        self.phase3_aux_ramp_enabled = bool(aux_ramp_cfg.get('enabled', False))
        self.phase3_aux_ramp_epochs = int(aux_ramp_cfg.get('epochs', 0))
        self.phase3_aux_ramp = {
            'illumination': (
                float(aux_ramp_cfg.get('illumination_start', phase3_illum_end)),
                float(aux_ramp_cfg.get('illumination_end', phase3_illum_end)),
            ),
            'iicl': (
                float(aux_ramp_cfg.get('iicl_start', self.iicl_weight)),
                float(aux_ramp_cfg.get('iicl_end', self.iicl_weight)),
            ),
            'cross_light': (
                float(aux_ramp_cfg.get('cross_light_start', phase3_cross_light_end)),
                float(aux_ramp_cfg.get('cross_light_end', phase3_cross_light_end)),
            ),
        }

        # Phase 2 FGID warmup config
        phase2_cfg = train_cfg.get('phases', {}).get('phase2', {})
        self.phase2_fgid_epochs = int(phase2_cfg.get('epochs', 10))
        self.phase2_fgid_lr = float(phase2_cfg.get('illumination_lr', phase2_cfg.get('lr', 1e-4)))

        self.logger.info("=" * 50)
        self.logger.info("Loss Configuration:")
        self.logger.info(f"  IPAID Loss: recon={loss_params.get('lambda_recon', 1.0)}, "
                        f"smooth={loss_params.get('lambda_smooth', 0.15)}, "
                        f"structure={loss_params.get('lambda_structure', loss_params.get('lambda_edge', 0.08))}, "
                        f"sens={loss_params.get('lambda_sensitivity', 0.02)}, "
                        f"lab={loss_params.get('lambda_lab_chroma', 0.1)}, "
                        f"hf={loss_params.get('lambda_high_freq', 0.05)}, "
                        f"log_chroma={loss_params.get('lambda_log_chroma', 0.0)}, "
                        f"chroma_mode={loss_params.get('chroma_mode', 'dual')}")
        self.logger.info(
            f"  Identity Geometry: mode={self.identity_preserving_mode}, "
            f"weight={self.identity_loss_weight}, anchor={self.identity_anchor_weight}, "
            f"geometry={self.identity_geometry_weight}, logit={self.identity_logit_weight}, "
            f"detach_ref={self.identity_detach_reference}, temp={self.identity_teacher_temperature}"
        )
        self.logger.info(
            f"  Cross-Light Prototype: enabled={self.use_cross_light_prototype}, "
            f"weight={self.cross_light_prototype_weight}, "
            f"scale={cross_light_cfg.get('photometric_scale', 8.0)}, "
            f"offset={cross_light_cfg.get('photometric_offset', 0.1)}, "
            f"min_gap={cross_light_cfg.get('min_gap_weight', 0.1)}"
        )
        self.logger.info(
            f"  Cross-Light Margin: enabled={self.use_cross_light_margin_preserving}, "
            f"weight={self.cross_light_margin_preserving_weight}, "
            f"scale={cross_light_margin_cfg.get('photometric_scale', 8.0)}, "
            f"offset={cross_light_margin_cfg.get('photometric_offset', 0.1)}, "
            f"k+={cross_light_margin_cfg.get('topk_positive', 2)}, "
            f"k-={cross_light_margin_cfg.get('topk_negative', 4)}, "
            f"delta={cross_light_margin_cfg.get('margin_delta', 0.02)}, "
            f"beta={cross_light_margin_cfg.get('beta', 12.0)}"
        )
        self.logger.info(
            f"  Cross-Light SoftAP: enabled={self.use_cross_light_softap}, "
            f"weight={self.cross_light_softap_weight}, "
            f"scale={cross_light_softap_cfg.get('photometric_scale', 8.0)}, "
            f"offset={cross_light_softap_cfg.get('photometric_offset', 0.1)}, "
            f"min_pos={cross_light_softap_cfg.get('min_positive_weight', 0.05)}, "
            f"rank_temp={cross_light_softap_cfg.get('rank_temperature', 0.07)}, "
            f"queue={self.cross_light_softap_queue_size}"
        )
        self.logger.info(
            f"  Teacher Manifold: enabled={self.use_teacher_manifold}, "
            f"tube={self.teacher_manifold_tube_weight}, "
            f"sep={self.teacher_manifold_separation_weight}, "
            f"shrink={teacher_manifold_cfg.get('shrinkage', 0.8)}, "
            f"orth={teacher_manifold_cfg.get('orthogonal_weight', 1.0)}, "
            f"rank={teacher_manifold_cfg.get('subspace_rank', 1)}, "
            f"margin={teacher_manifold_cfg.get('margin', 0.08)}, "
            f"queue={self.teacher_manifold_queue_size}"
        )
        self.logger.info(
            f"  Ranking Topology: enabled={self.use_ranking_topology}, "
            f"weight={self.ranking_topology_weight}, "
            f"k+={ranking_topology_cfg.get('topk_positive', 2)}, "
            f"k-={ranking_topology_cfg.get('topk_negative', 4)}, "
            f"slack={ranking_topology_cfg.get('margin_slack', 0.01)}, "
            f"beta={ranking_topology_cfg.get('beta', 12.0)}, "
            f"queue={self.ranking_topology_queue_size}"
        )
        self.logger.info(
            f"  Anisotropic Identity Protection: enabled={self.use_anisotropic_identity_protection}, "
            f"weight={self.anisotropic_identity_protection_weight}, "
            f"k+={anisotropic_identity_cfg.get('topk_positive', 2)}, "
            f"k-={anisotropic_identity_cfg.get('topk_negative', 4)}, "
            f"rank={anisotropic_identity_cfg.get('subspace_rank', 1)}, "
            f"id={anisotropic_identity_cfg.get('identity_weight', 1.0)}, "
            f"nuisance={anisotropic_identity_cfg.get('nuisance_weight', 0.5)}, "
            f"radius={anisotropic_identity_cfg.get('nuisance_radius', 0.12)}"
        )
        self.logger.info(
            f"  Semantic Non-Confusion: enabled={self.use_semantic_non_confusion}, "
            f"weight={self.semantic_non_confusion_weight}, "
            f"delta={semantic_non_confusion_cfg.get('margin_delta', 0.02)}, "
            f"squared={semantic_non_confusion_cfg.get('squared', True)}"
        )
        self.logger.info(
            f"  Nuisance Decoupling: enabled={self.use_nuisance_decoupling}, "
            f"weight={self.nuisance_decoupling_weight}, "
            f"reg={self.nuisance_regression_weight}, "
            f"decorr={self.nuisance_decorrelation_weight}"
        )
        self.logger.info(
            f"  Teacher Prototype Anchor: enabled={self.use_teacher_prototype_anchor}, "
            f"weight={self.teacher_prototype_anchor_weight}, "
            f"metric={teacher_prototype_anchor_cfg.get('metric', self.identity_similarity)}"
        )
        self.logger.info(
            f"  Relative Class Structure: enabled={self.use_relative_class_structure}, "
            f"weight={self.relative_class_structure_weight}, "
            f"metric={relative_class_structure_cfg.get('metric', self.identity_similarity)}, "
            f"radial={relative_class_structure_cfg.get('radial_weight', 0.5)}"
        )
        self.logger.info(
            f"  Feature Trust Region: enabled={self.use_feature_trust_region}, "
            f"weight={self.feature_trust_region_weight}, "
            f"radius={feature_trust_cfg.get('base_radius', 0.12)}, "
            f"adaptive={feature_trust_cfg.get('adaptive_scale', 0.0)}, "
            f"class_spread={feature_trust_cfg.get('class_spread_scale', 0.0)}"
        )
        self.logger.info(
            f"  Local Rank Preserving: enabled={self.use_local_rank_preserving}, "
            f"weight={self.local_rank_preserving_weight}, "
            f"alpha={local_rank_cfg.get('alpha', 0.9)}, "
            f"k+={local_rank_cfg.get('k_positive', 1)}, "
            f"k-={local_rank_cfg.get('k_negative', 1)}"
        )
        self.logger.info(
            f"  Neighborhood Consistency: enabled={self.use_neighborhood_consistency}, "
            f"weight={self.neighborhood_consistency_weight}, "
            f"topk={neighborhood_cfg.get('topk', 6)}, "
            f"temp={neighborhood_cfg.get('temperature', 0.07)}, "
            f"pos={neighborhood_cfg.get('positive_weight', 1.0)}, "
            f"neg={neighborhood_cfg.get('negative_weight', 0.25)}, "
            f"local={neighborhood_cfg.get('local_weight', 0.35)}, "
            f"use_global={neighborhood_cfg.get('use_global', True)}, "
            f"use_local={neighborhood_cfg.get('use_local', True)}, "
            f"hard_neg={neighborhood_cfg.get('use_hard_negatives', True)}, "
            f"target={neighborhood_cfg.get('teacher_target', 'soft')}"
        )
        self.logger.info(
            f"  Identity Image Preserving: enabled={self.use_identity_image_preserving}, "
            f"weight={self.identity_image_preserving_weight}"
        )
        self.logger.info(
            f"  Photo Prior: initial={self.photo_prior_initial_weight}, "
            f"min={self.photo_prior_min_weight}, decay_power={self.photo_prior_decay_power}"
        )
        self.logger.info(
            f"  Identity Phase Scale: phase2={self.identity_phase2_scale}, "
            f"phase3={self.identity_phase3_scale}"
        )
        self.logger.info(f"  CE Loss: weight={self.ce_weight}, label_smooth={ce_cfg.get('label_smoothing', 0.1)}")
        self.logger.info(f"  Triplet Loss: weight={self.triplet_weight}, margin={triplet_cfg.get('margin', 0.3)}")
        self.logger.info(f"  Circle Loss: weight={self.circle_weight}, m={circle_cfg.get('margin', 0.25)}, gamma={circle_gamma_value}")
        self.logger.info(f"  ArcFace Loss: weight={self.arcface_weight}, s={arcface_cfg.get('s', 30.0)}, m={arcface_cfg.get('m', 0.35)}")
        self.logger.info(f"  Center Loss: weight={self.center_loss_weight}, enabled={center_cfg.get('enabled', True)}")
        self.logger.info(
            f"  IICL Consistency: weight={self.iicl_weight}, enabled={self.use_iicl}, "
            f"variants={self.iicl_num_variants}, grad_variants={self.iicl_num_grad_variants}, "
            f"type={iicl_cfg.get('loss_type', 'cosine')}"
        )
        if self.phase3_aux_ramp_enabled:
            self.logger.info(
                "  Phase 3 Aux Ramp: epochs=%d, illum=%.3f->%.3f, iicl=%.3f->%.3f, %s=%.3f->%.3f",
                self.phase3_aux_ramp_epochs,
                self.phase3_aux_ramp['illumination'][0],
                self.phase3_aux_ramp['illumination'][1],
                self.phase3_aux_ramp['iicl'][0],
                self.phase3_aux_ramp['iicl'][1],
                self._cross_light_objective_name(),
                self.phase3_aux_ramp['cross_light'][0],
                self.phase3_aux_ramp['cross_light'][1],
            )
        self.logger.info(
            f"  Aux Gradient Gate: enabled={self.use_aux_gradient_gate}, eps={self.aux_gradient_gate_eps}"
        )
        self.logger.info(f"  Phase 2 Illumination Optimization: {self.phase2_fgid_epochs} epochs, lr={self.phase2_fgid_lr}")
        self.logger.info("=" * 50)

    def _maybe_resume(self):
        """Resume training from checkpoint if specified."""
        if not self.resume_checkpoint:
            return

        if not os.path.exists(self.resume_checkpoint):
            self.logger.warning(f"Checkpoint not found: {self.resume_checkpoint}")
            self.resume_checkpoint = None
            return

        self.logger.info(f"Resuming from checkpoint: {self.resume_checkpoint}")
        checkpoint = torch.load(self.resume_checkpoint, map_location=self.device, weights_only=False)

        # Load model state
        model_state = checkpoint.get('model_state_dict')
        if model_state is None and isinstance(checkpoint, dict):
            model_state = checkpoint.get('state_dict')
        if model_state is None and isinstance(checkpoint, dict):
            model_state = checkpoint
        if model_state is None or not isinstance(model_state, dict):
            self.logger.warning("No valid model weights found in checkpoint")
            self.resume_checkpoint = None
            return

        load_result = self._load_model_state_dict_compat(model_state)
        missing_keys = getattr(load_result, 'missing_keys', [])
        unexpected_keys = getattr(load_result, 'unexpected_keys', [])
        if missing_keys:
            self.logger.warning(f"Missing keys when resuming: {len(missing_keys)}")
        if unexpected_keys:
            self.logger.warning(f"Unexpected keys when resuming: {len(unexpected_keys)}")

        self.resume_phase = checkpoint.get('phase', 1)
        checkpoint_epoch = int(checkpoint.get('epoch', -1))
        checkpoint_reason = str(checkpoint.get('reason', '') or '').strip().lower()
        if checkpoint_reason in {'exception', 'interrupt'}:
            # Re-run the interrupted epoch on exception checkpoints because the
            # previous epoch may have failed before completing a single step.
            self.resume_epoch = max(0, checkpoint_epoch)
        else:
            self.resume_epoch = max(0, checkpoint_epoch + 1)
        self.best_acc = checkpoint.get('best_acc', checkpoint.get('metrics', {}).get('accuracy', 0.0))
        self.best_rank1 = float(checkpoint.get('best_rank1', self.best_rank1))
        self.best_map = float(checkpoint.get('best_map', self.best_map))
        self.best_metric_name = str(checkpoint.get('best_metric_name', self.best_metric_name))
        default_best_metric = self.best_rank1 if self.best_rank1 > 0 else self.best_metric_value
        self.best_metric_value = float(checkpoint.get('best_metric_value', default_best_metric))
        self.resume_optimizer_state = checkpoint.get('optimizer_state_dict')
        self.resume_scheduler_state = checkpoint.get('scheduler_state_dict')

        self.logger.info(
            f"Resumed: phase={self.resume_phase}, epoch={self.resume_epoch}, "
            f"best_acc={self.best_acc:.2f}%, best_metric={self.best_metric_name}:{self.best_metric_value:.2f}"
        )

    def _create_scheduler(self, optimizer, total_epochs: int):
        """Create learning rate scheduler."""
        train_cfg = self.config.get('training', {})
        scheduler_cfg = train_cfg.get('scheduler', {})

        scheduler_type = scheduler_cfg.get('type', 'CosineAnnealingLR')
        eta_min = float(scheduler_cfg.get('eta_min', 1e-6))

        self.logger.info(f"  Scheduler: {scheduler_type}")

        if scheduler_type == 'CosineAnnealingWarmRestarts':
            T_0 = int(scheduler_cfg.get('T_0', 50))
            T_mult = int(scheduler_cfg.get('T_mult', 2))
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
            )
            self.logger.info(f"  CosineAnnealingWarmRestarts: T_0={T_0}, T_mult={T_mult}, eta_min={eta_min}")
        elif scheduler_type == 'StepLR':
            step_size = int(scheduler_cfg.get('step_size', 30))
            gamma = float(scheduler_cfg.get('gamma', 0.1))
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
            self.logger.info(f"  StepLR: step_size={step_size}, gamma={gamma}")
        elif scheduler_type == 'MultiStepLR':
            milestones = scheduler_cfg.get('milestones', [30, 60, 90])
            gamma = float(scheduler_cfg.get('gamma', 0.1))
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=gamma
            )
            self.logger.info(f"  MultiStepLR: milestones={milestones}, gamma={gamma}")
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs, eta_min=eta_min
            )
            self.logger.info(f"  CosineAnnealingLR: T_max={total_epochs}, eta_min={eta_min}")

        return scheduler

    def _setup_optimizer_phase1(self):
        """
        Phase 1: ReID warmup with frozen illumination parameters.

        The illumination module still runs in forward, so the backbone learns on
        the current corrected image distribution; only illumination parameters are frozen.
        """
        self._configure_phase_dataloader(phase=1, log_prefix="Phase 1")
        self._set_phase_backbone_checkpointing(phase=1)
        self.model.freeze_illumination(True)
        self.model.freeze_backbone(False)
        self.model.freeze_local_extractor(False)
        self.model.freeze_feature_fusion(False)
        self._set_aux_metric_modules_trainable(True)

        backbone_params = list(self.model.backbone.parameters())
        extractor_params = list(self.model.local_extractor.parameters())
        fusion_params = []
        if getattr(self.model, "feature_fusion", None) is not None:
            fusion_params.extend(list(self.model.feature_fusion.parameters()))
        if getattr(self.model, "branch_attention_fusion", None) is not None:
            fusion_params.extend(list(self.model.branch_attention_fusion.parameters()))
        if getattr(self.model, "nuisance_projection", None) is not None:
            fusion_params.extend(list(self.model.nuisance_projection.parameters()))
        if getattr(self.model, "photometric_regressor", None) is not None:
            fusion_params.extend(list(self.model.photometric_regressor.parameters()))
        center_params = list(self.center_loss.parameters()) if self.center_loss is not None else []

        phase_config = self.config['training']['phases'].get('phase1', {})
        backbone_lr = float(phase_config.get('backbone_lr', self.learning_rate))

        train_cfg = self.config.get('training', {})
        weight_decay = float(train_cfg.get('weight_decay', 5e-4))

        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': extractor_params, 'lr': backbone_lr},
        ]
        if fusion_params:
            param_groups.append({'params': fusion_params, 'lr': backbone_lr})
        if center_params:
            param_groups.append({'params': center_params, 'lr': backbone_lr * 0.5})

        if self.arcface_loss is not None and self.arcface_weight > 0:
            arcface_params = list(self.arcface_loss.parameters())
            param_groups.append({'params': arcface_params, 'lr': backbone_lr})

        self.optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
        self.scheduler = self._create_scheduler(self.optimizer, self.phase1_epochs)

        self.loss_weights = {
            'illumination': float(phase_config.get('illumination_weight', 0.0)),
            'reid': float(phase_config.get('reid_weight', 1.0))
        }

        total_params = (
            sum(p.numel() for p in backbone_params)
            + sum(p.numel() for p in extractor_params)
            + (sum(p.numel() for p in fusion_params) if fusion_params else 0)
            + (sum(p.numel() for p in center_params) if center_params else 0)
        )
        self.logger.info("Phase 1 Optimizer Setup (ReID Warmup - Illumination Frozen but Active)")
        self.logger.info(f"  Frozen: illumination module")
        self.logger.info(f"  Training: backbone + extractor + fusion/aux ({total_params/1e6:.2f}M params)")
        self.logger.info(f"  backbone_lr: {backbone_lr}, weight_decay: {weight_decay}")
        self.logger.info(f"  Loss weights: illum={self.loss_weights['illumination']}, reid={self.loss_weights['reid']}")

    def _set_aux_metric_modules_trainable(self, trainable: bool) -> None:
        if self.arcface_loss is not None:
            for param in self.arcface_loss.parameters():
                param.requires_grad = trainable
        if self.center_loss is not None:
            for param in self.center_loss.parameters():
                param.requires_grad = trainable

    def _setup_optimizer_phase2_fgid(self):
        """
        Phase 2: geometry-preserving illumination optimization with frozen ReID modules.
        """
        self._configure_phase_dataloader(phase=2, log_prefix="Phase 2")
        self._set_phase_backbone_checkpointing(phase=2)
        self.model.freeze_backbone(True)
        self.model.freeze_illumination(False)
        self.model.freeze_local_extractor(True)
        self.model.freeze_feature_fusion(False)
        self._set_aux_metric_modules_trainable(False)

        param_groups = []
        if self.model.illumination is not None:
            illum_params = list(self.model.illumination.parameters())
            param_groups.append({'params': illum_params, 'lr': self.phase2_fgid_lr})
        if getattr(self.model, "feature_fusion", None) is not None:
            fusion_params = list(self.model.feature_fusion.parameters())
            param_groups.append({'params': fusion_params, 'lr': self.phase2_fgid_lr})

        train_cfg = self.config.get('training', {})
        phase_config = self.config['training']['phases'].get('phase2', {})
        weight_decay = float(train_cfg.get('weight_decay', 5e-4))

        self.optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
        self.scheduler = self._create_scheduler(self.optimizer, self.phase2_fgid_epochs)

        self.loss_weights = {
            'illumination': float(phase_config.get('illumination_weight', 1.0)),
            'reid': float(phase_config.get('reid_weight', 0.0)),
        }

        total_params = sum(p.numel() for g in param_groups for p in g['params'])
        self.logger.info("Phase 2 Optimizer Setup (Illumination Optimization)")
        self.logger.info(f"  Frozen: backbone")
        self.logger.info(f"  Training: illumination + feature_fusion ({total_params/1e6:.2f}M params)")
        self.logger.info(f"  lr: {self.phase2_fgid_lr}, weight_decay: {weight_decay}")
        self.logger.info(f"  Loss weights: illum={self.loss_weights['illumination']}, reid={self.loss_weights['reid']}")

    def _setup_optimizer_phase3(self):
        """
        Phase 3: joint optimization.
        """
        self._configure_phase_dataloader(phase=3, log_prefix="Phase 3")
        self._set_phase_backbone_checkpointing(phase=3)
        self.model.freeze_backbone(False)
        self.model.freeze_illumination(False)
        self.model.freeze_local_extractor(False)
        self.model.freeze_feature_fusion(False)
        self._set_aux_metric_modules_trainable(True)

        phase_config = self.config['training']['phases'].get('phase3',
                       self.config['training']['phases'].get('phase2', {}))
        illum_lr = float(phase_config.get('illumination_lr', 1e-4))
        backbone_lr = float(phase_config.get('backbone_lr', self.learning_rate * 0.5))

        param_groups = []

        if self.model.illumination is not None:
            illum_params = list(self.model.illumination.parameters())
            param_groups.append({'params': illum_params, 'lr': illum_lr})

        backbone_params = list(self.model.backbone.parameters())
        param_groups.append({'params': backbone_params, 'lr': backbone_lr})

        extractor_params = list(self.model.local_extractor.parameters())
        param_groups.append({'params': extractor_params, 'lr': backbone_lr})

        fusion_params = []
        if getattr(self.model, "feature_fusion", None) is not None:
            fusion_params.extend(list(self.model.feature_fusion.parameters()))
        if getattr(self.model, "branch_attention_fusion", None) is not None:
            fusion_params.extend(list(self.model.branch_attention_fusion.parameters()))
        if getattr(self.model, "nuisance_projection", None) is not None:
            fusion_params.extend(list(self.model.nuisance_projection.parameters()))
        if getattr(self.model, "photometric_regressor", None) is not None:
            fusion_params.extend(list(self.model.photometric_regressor.parameters()))
        if fusion_params:
            param_groups.append({'params': fusion_params, 'lr': backbone_lr})

        if self.center_loss is not None:
            center_params = list(self.center_loss.parameters())
            param_groups.append({'params': center_params, 'lr': backbone_lr * 0.5})

        if self.arcface_loss is not None and self.arcface_weight > 0:
            arcface_params = list(self.arcface_loss.parameters())
            param_groups.append({'params': arcface_params, 'lr': backbone_lr})

        train_cfg = self.config.get('training', {})
        weight_decay = float(train_cfg.get('weight_decay', 5e-4))

        self.optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
        self.scheduler = self._create_scheduler(self.optimizer, self.phase3_epochs)

        self.loss_weights = {
            'illumination': float(phase_config.get('illumination_weight', 0.2)),
            'reid': float(phase_config.get('reid_weight', 1.0))
        }

        self.logger.info("Phase 3 Optimizer Setup (Joint Optimization)")
        self.logger.info(f"  backbone_lr: {backbone_lr}, illum_lr: {illum_lr}")
        self.logger.info(f"  Loss weights: illum={self.loss_weights['illumination']}, reid={self.loss_weights['reid']}")

    def _refresh_phase3_teacher(self):
        """Snapshot a frozen raw-reference teacher at the start of phase 3."""
        self.phase3_teacher_model = copy.deepcopy(self.model).to(self.device)
        if hasattr(self.phase3_teacher_model, 'use_backbone_checkpointing'):
            self.phase3_teacher_model.use_backbone_checkpointing = False
        self.phase3_teacher_model.eval()
        for param in self.phase3_teacher_model.parameters():
            param.requires_grad_(False)
        self.logger.info("Phase 3 teacher refreshed from the current model snapshot")

    def _get_photo_prior_weight(self, phase: int, epoch: int) -> float:
        """Anneal photometric priors in phase 3 while keeping earlier phases stable."""
        if phase != 3:
            return self.photo_prior_initial_weight

        if self.phase3_epochs <= 1:
            return self.photo_prior_min_weight

        progress = min(max(epoch / max(self.phase3_epochs - 1, 1), 0.0), 1.0)
        decay = progress ** max(self.photo_prior_decay_power, 1e-6)
        weight = self.photo_prior_initial_weight + (
            self.photo_prior_min_weight - self.photo_prior_initial_weight
        ) * decay
        lower = min(self.photo_prior_initial_weight, self.photo_prior_min_weight)
        upper = max(self.photo_prior_initial_weight, self.photo_prior_min_weight)
        return float(min(upper, max(lower, weight)))

    def _get_identity_preserving_scale(self, phase: int) -> float:
        """Keep strong identity anchoring in phase 2 and relax it in phase 3."""
        if phase == 2:
            return self.identity_phase2_scale
        if phase == 3:
            return self.identity_phase3_scale
        return 0.0

    def _get_phase3_aux_weight(self, name: str, phase: int, epoch: int) -> float:
        """Warm auxiliary weights into the joint stage instead of switching them on abruptly."""
        start, end = self.phase3_aux_ramp.get(name, (0.0, 0.0))
        if phase != 3:
            return float(end)
        if not self.phase3_aux_ramp_enabled:
            return float(end)
        return _linear_warmup_value(start, end, epoch=epoch, warmup_epochs=self.phase3_aux_ramp_epochs)

    def _cross_light_objective_name(self) -> str:
        if self.identity_preserving_mode == 'softap':
            return 'softap'
        if self.identity_preserving_mode == 'margin_preserving':
            return 'cmp'
        return 'xproto'

    def _reset_cross_light_queue(self) -> None:
        self.cross_light_queue_features = None
        self.cross_light_queue_labels = None
        self.cross_light_queue_stats = None

    def _reset_teacher_manifold_queue(self) -> None:
        self.teacher_manifold_queue_features = None
        self.teacher_manifold_queue_labels = None
        self.teacher_manifold_queue_stats = None

    @torch.no_grad()
    def _update_cross_light_queue(
        self,
        features: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        photometric_stats: Optional[torch.Tensor],
    ) -> None:
        if (
            not self.use_cross_light_softap
            or self.cross_light_softap_queue_size <= 0
            or not isinstance(features, torch.Tensor)
            or not isinstance(labels, torch.Tensor)
            or not isinstance(photometric_stats, torch.Tensor)
        ):
            return

        features_detached = features.detach()
        labels_detached = labels.detach().view(-1)
        stats_detached = photometric_stats.detach()
        if features_detached.size(0) == 0:
            return

        if self.cross_light_queue_features is None:
            self.cross_light_queue_features = features_detached
            self.cross_light_queue_labels = labels_detached
            self.cross_light_queue_stats = stats_detached
            return

        self.cross_light_queue_features = torch.cat(
            [self.cross_light_queue_features, features_detached],
            dim=0,
        )[-self.cross_light_softap_queue_size :]
        self.cross_light_queue_labels = torch.cat(
            [self.cross_light_queue_labels, labels_detached],
            dim=0,
        )[-self.cross_light_softap_queue_size :]
        self.cross_light_queue_stats = torch.cat(
            [self.cross_light_queue_stats, stats_detached],
            dim=0,
        )[-self.cross_light_softap_queue_size :]

    @torch.no_grad()
    def _update_teacher_manifold_queue(
        self,
        teacher_features: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        photometric_stats: Optional[torch.Tensor],
    ) -> None:
        if (
            not self.use_teacher_gallery_alignment
            or self.teacher_gallery_queue_size <= 0
            or not isinstance(teacher_features, torch.Tensor)
            or not isinstance(labels, torch.Tensor)
            or not isinstance(photometric_stats, torch.Tensor)
        ):
            return

        features_detached = teacher_features.detach()
        labels_detached = labels.detach().view(-1)
        stats_detached = photometric_stats.detach()
        if features_detached.size(0) == 0:
            return

        if self.teacher_manifold_queue_features is None:
            self.teacher_manifold_queue_features = features_detached
            self.teacher_manifold_queue_labels = labels_detached
            self.teacher_manifold_queue_stats = stats_detached
            return

        self.teacher_manifold_queue_features = torch.cat(
            [self.teacher_manifold_queue_features, features_detached],
            dim=0,
        )[-self.teacher_gallery_queue_size :]
        self.teacher_manifold_queue_labels = torch.cat(
            [self.teacher_manifold_queue_labels, labels_detached],
            dim=0,
        )[-self.teacher_gallery_queue_size :]
        self.teacher_manifold_queue_stats = torch.cat(
            [self.teacher_manifold_queue_stats, stats_detached],
            dim=0,
        )[-self.teacher_gallery_queue_size :]

    def _build_cross_light_stats(
        self,
        ipaid_details: Optional[Dict[str, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """Summarize each sample's photometric state for cross-light pairing."""
        if not ipaid_details:
            return None

        illumination = ipaid_details.get('effective_illumination', ipaid_details.get('illumination'))
        correction_gap = ipaid_details.get('correction_gap')
        rollback_alpha = ipaid_details.get('rollback_alpha')
        lambda_color = ipaid_details.get('lambda_color')
        if illumination is None:
            return None

        stats = [
            illumination.float().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1),
        ]
        if correction_gap is None:
            stats.append(stats[0].new_zeros(stats[0].shape))
        else:
            stats.append(correction_gap.float().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1))
        if rollback_alpha is None:
            stats.append(stats[0].new_zeros(stats[0].shape))
        else:
            stats.append(rollback_alpha.float().view(rollback_alpha.size(0), -1).mean(dim=1, keepdim=True))
        if lambda_color is None:
            stats.append(stats[0].new_zeros(stats[0].shape))
        else:
            stats.append(lambda_color.float().view(lambda_color.size(0), -1).mean(dim=1, keepdim=True))
        return torch.cat(stats, dim=1)

    def train_epoch(self, epoch: int, phase: int) -> Dict[str, float]:
        """
        Train one epoch.

        Phase 1: ReID warmup with frozen illumination parameters
        Phase 2: illumination-only optimization with frozen backbone
        Phase 3: joint optimization with optional IICL consistency
        """
        self.model.train()
        if self.model_forward_adapter is not None:
            self.model_forward_adapter.train()
        if self.model_ddp is not None:
            self.model_ddp.train()

        total_loss = 0.0
        illum_loss_sum = 0.0
        reid_loss_sum = 0.0
        correct = 0
        total = 0
        num_batches = 0

        input_range_logged = False
        photo_prior_weight = self._get_photo_prior_weight(phase, epoch)
        identity_phase_scale = self._get_identity_preserving_scale(phase)
        phase_illumination_weight = self.loss_weights['illumination']
        phase_reid_weight = self.loss_weights['reid']
        phase_iicl_weight = self.iicl_weight
        if (
            phase == 2
            and phase_reid_weight <= 0
            and identity_phase_scale > 0
            and self.rank == 0
        ):
            self.logger.warning(
                "Phase 2 has reid_weight=0 but identity_phase_scale=%.3f, "
                "so training will use the full standard forward instead of "
                "the lightweight illumination_only path.",
                identity_phase_scale,
            )
        if self.identity_preserving_mode == 'softap':
            phase_cross_light_weight = self.cross_light_softap_weight
        elif self.identity_preserving_mode == 'margin_preserving':
            phase_cross_light_weight = self.cross_light_margin_preserving_weight
        else:
            phase_cross_light_weight = self.cross_light_prototype_weight
        if phase == 3:
            phase_illumination_weight = self._get_phase3_aux_weight('illumination', phase=phase, epoch=epoch)
            phase_iicl_weight = self._get_phase3_aux_weight('iicl', phase=phase, epoch=epoch)
            phase_cross_light_weight = self._get_phase3_aux_weight('cross_light', phase=phase, epoch=epoch)
        training_model = self._model_for_training()
        use_phase3_teacher = (
            phase == 3
            and self.phase3_teacher_model is not None
            and (
                (self.use_relative_class_structure and self.relative_class_structure_weight > 0)
                or
                (self.use_feature_trust_region and self.feature_trust_region_weight > 0)
                or (self.use_local_rank_preserving and self.local_rank_preserving_weight > 0)
                or (self.use_neighborhood_consistency and self.neighborhood_consistency_weight > 0)
                or (
                    self.use_teacher_manifold
                    and (
                        self.teacher_manifold_tube_weight > 0
                        or self.teacher_manifold_separation_weight > 0
                    )
                )
                or (self.use_ranking_topology and self.ranking_topology_weight > 0)
                or (
                    self.use_anisotropic_identity_protection
                    and self.anisotropic_identity_protection_weight > 0
                )
                or (self.use_semantic_non_confusion and self.semantic_non_confusion_weight > 0)
                or (
                    self.identity_preserving_mode == 'margin_preserving'
                    and self.use_cross_light_margin_preserving
                    and phase_cross_light_weight > 0
                )
            )
        )
        use_local_feature_supervision = (
            phase == 3
            and self.use_neighborhood_consistency
            and self.neighborhood_consistency_weight > 0
        )

        for batch_idx, (images, labels, _) in enumerate(self.dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # The physical RGB branch should remain in [0, 1].
            # Backbone-space Gaussian erasing is applied later inside JointReIDModel.
            images_for_model = images
            if not input_range_logged:
                self.logger.info(
                    "Input range check (first batch): min=%.4f max=%.4f mean=%.4f",
                    images_for_model.min().item(),
                    images_for_model.max().item(),
                    images_for_model.mean().item(),
                )
                input_range_logged = True

            with self._autocast_context():
                # Forward pass
                if self.use_iicl and phase == 3 and phase_iicl_weight > 0:
                    output = training_model(
                        images_for_model,
                        forward_mode='consistency',
                        num_variants=self.iicl_num_variants,
                        return_local_features=use_local_feature_supervision,
                    )
                    features_variants = output.get('features_variants', [])
                elif phase == 2 and self.loss_weights['reid'] <= 0 and identity_phase_scale <= 0:
                    output = training_model(
                        images_for_model,
                        forward_mode='illumination_only',
                        return_illuminated=True,
                    )
                    features_variants = []
                else:
                    output = training_model(
                        images_for_model,
                        forward_mode='standard',
                        return_illuminated=True,
                        return_local_features=use_local_feature_supervision,
                    )
                    features_variants = []

                features = output.get('features')
                logits = output.get('logits')
                nuisance_features = output.get('nuisance_features')
                photometric_prediction = output.get('photometric_prediction')
                ipaid_details = output.get('ipaid_details')
                illum_loss_dict = {}
                loss_feature_trust = torch.tensor(0.0, device=self.device)
                loss_local_rank = torch.tensor(0.0, device=self.device)
                loss_neighborhood_consistency = torch.tensor(0.0, device=self.device)
                loss_identity_image = torch.tensor(0.0, device=self.device)
                loss_teacher_prototype_anchor = torch.tensor(0.0, device=self.device)
                loss_relative_class_structure = torch.tensor(0.0, device=self.device)
                loss_teacher_manifold_tube = torch.tensor(0.0, device=self.device)
                loss_teacher_manifold_separation = torch.tensor(0.0, device=self.device)
                loss_ranking_topology = torch.tensor(0.0, device=self.device)
                loss_anisotropic_identity = torch.tensor(0.0, device=self.device)
                loss_semantic_non_confusion = torch.tensor(0.0, device=self.device)
                loss_nuisance_regression = torch.tensor(0.0, device=self.device)
                loss_nuisance_decorrelation = torch.tensor(0.0, device=self.device)
                loss_feature_trust_scaled = torch.tensor(0.0, device=self.device)
                loss_local_rank_scaled = torch.tensor(0.0, device=self.device)
                loss_neighborhood_consistency_scaled = torch.tensor(0.0, device=self.device)
                loss_identity_image_scaled = torch.tensor(0.0, device=self.device)
                loss_teacher_prototype_anchor_scaled = torch.tensor(0.0, device=self.device)
                loss_relative_class_structure_scaled = torch.tensor(0.0, device=self.device)
                loss_teacher_manifold_tube_scaled = torch.tensor(0.0, device=self.device)
                loss_teacher_manifold_separation_scaled = torch.tensor(0.0, device=self.device)
                loss_ranking_topology_scaled = torch.tensor(0.0, device=self.device)
                loss_anisotropic_identity_scaled = torch.tensor(0.0, device=self.device)
                loss_semantic_non_confusion_scaled = torch.tensor(0.0, device=self.device)
                loss_nuisance_regression_scaled = torch.tensor(0.0, device=self.device)
                loss_nuisance_decorrelation_scaled = torch.tensor(0.0, device=self.device)

                self._ensure_finite_tensor(features, 'output.features', phase, epoch, batch_idx)
                self._ensure_finite_tensor(logits, 'output.logits', phase, epoch, batch_idx)
                self._ensure_finite_tensor(
                    nuisance_features,
                    'output.nuisance_features',
                    phase,
                    epoch,
                    batch_idx,
                )
                self._ensure_finite_tensor(
                    photometric_prediction,
                    'output.photometric_prediction',
                    phase,
                    epoch,
                    batch_idx,
                )
                if ipaid_details is not None:
                    for key in (
                        'reflectance',
                        'illumination',
                        'effective_illumination',
                        'rollback_alpha',
                        'lambda_color',
                        'correction_gap',
                    ):
                        self._ensure_finite_tensor(
                            ipaid_details.get(key),
                            f'output.ipaid_details.{key}',
                            phase,
                            epoch,
                            batch_idx,
                        )
                for variant_idx, variant_features in enumerate(features_variants):
                    self._ensure_finite_tensor(
                        variant_features,
                        f'output.features_variants[{variant_idx}]',
                        phase,
                        epoch,
                        batch_idx,
                    )

                # === Loss Computation ===

                # 1. IPAID illumination loss
                if ipaid_details is not None:
                    with self._autocast_disabled_context():
                        loss_photo, illum_loss_dict = self.ipaid_loss(
                            ipaid_details, ipaid_module=self.model.illumination,
                        )
                        loss_anchor = torch.tensor(0.0, device=self.device)
                        loss_geometry = torch.tensor(0.0, device=self.device)
                        loss_logit = torch.tensor(0.0, device=self.device)
                        loss_cross_light = torch.tensor(0.0, device=self.device)
                        if (
                            identity_phase_scale > 0
                            and self.identity_loss_weight > 0
                            and self.identity_preserving_mode == 'geometry'
                        ):
                            raw_reference = training_model(
                                images_for_model,
                                forward_mode='raw_reference',
                                detach_reference=self.identity_detach_reference,
                            )
                            self._ensure_finite_tensor(
                                raw_reference.get('features'),
                                'output.raw_reference.features',
                                phase,
                                epoch,
                                batch_idx,
                            )
                            self._ensure_finite_tensor(
                                raw_reference.get('logits'),
                                'output.raw_reference.logits',
                                phase,
                                epoch,
                                batch_idx,
                            )
                            loss_anchor = self.teacher_anchor_loss(features, raw_reference['features'])
                            loss_geometry = self.geometry_preserving_loss(features, raw_reference['features'])
                            if self.identity_logit_weight > 0:
                                loss_logit = self.teacher_logit_consistency_loss(logits, raw_reference['logits'])
                        elif (
                            phase == 3
                            and identity_phase_scale > 0
                            and self.use_cross_light_prototype
                            and self.identity_preserving_mode == 'prototype'
                        ):
                            photometric_stats = self._build_cross_light_stats(ipaid_details)
                            loss_cross_light = self.cross_light_prototype_loss(
                                features,
                                labels,
                                photometric_stats,
                            )
                        loss_identity_preserve = (
                            self.identity_anchor_weight * loss_anchor +
                            self.identity_geometry_weight * loss_geometry +
                            self.identity_logit_weight * loss_logit
                        )
                        loss_illum = (
                            photo_prior_weight * loss_photo +
                            self.identity_loss_weight * identity_phase_scale * loss_identity_preserve
                        )

                    self._ensure_finite_tensor(loss_photo, 'loss.photo', phase, epoch, batch_idx)
                    self._ensure_finite_tensor(loss_anchor, 'loss.anchor', phase, epoch, batch_idx)
                    self._ensure_finite_tensor(loss_geometry, 'loss.geometry', phase, epoch, batch_idx)
                    self._ensure_finite_tensor(loss_logit, 'loss.logit', phase, epoch, batch_idx)
                    self._ensure_finite_tensor(loss_identity_preserve, 'loss.identity_preserve', phase, epoch, batch_idx)
                    self._ensure_finite_tensor(loss_illum, 'loss.illumination', phase, epoch, batch_idx)
                    illum_loss_dict['photo_prior_weight'] = photo_prior_weight
                    illum_loss_dict['identity_scale'] = identity_phase_scale
                    illum_loss_dict['loss_photo'] = loss_photo.item()
                    illum_loss_dict['loss_anchor'] = loss_anchor.item()
                    illum_loss_dict['loss_geometry'] = loss_geometry.item()
                    illum_loss_dict['loss_logit'] = loss_logit.item()
                    illum_loss_dict['loss_identity_preserve'] = loss_identity_preserve.item()
                    illum_loss_dict['loss_cross_light'] = loss_cross_light.item()
                else:
                    loss_photo = torch.tensor(0.0, device=self.device)
                    loss_anchor = torch.tensor(0.0, device=self.device)
                    loss_geometry = torch.tensor(0.0, device=self.device)
                    loss_logit = torch.tensor(0.0, device=self.device)
                    loss_cross_light = torch.tensor(0.0, device=self.device)
                    loss_identity_preserve = torch.tensor(0.0, device=self.device)
                    loss_illum = torch.tensor(0.0, device=self.device)
                photometric_stats = None
                teacher_features = None

                if phase == 3 and isinstance(features, torch.Tensor):
                    with self._autocast_disabled_context():
                        teacher_logits = None
                        photometric_stats = self._build_cross_light_stats(ipaid_details)
                        if use_phase3_teacher:
                            teacher_reference = self.phase3_teacher_model.forward_raw_reference(
                                images_for_model,
                                detach=True,
                                return_local_features=use_local_feature_supervision,
                            )
                            teacher_features = teacher_reference.get('features')
                            teacher_logits = teacher_reference.get('logits')
                            self._ensure_finite_tensor(
                                teacher_features,
                                'output.phase3_teacher.features',
                                phase,
                                epoch,
                                batch_idx,
                            )
                            self._ensure_finite_tensor(
                                teacher_logits,
                                'output.phase3_teacher.logits',
                                phase,
                                epoch,
                                batch_idx,
                            )
                        if (
                            teacher_features is not None
                            and self.use_relative_class_structure
                            and self.relative_class_structure_weight > 0
                            and self.identity_preserving_mode == 'margin_preserving'
                        ):
                            loss_relative_class_structure = self.relative_class_structure_loss(
                                features,
                                teacher_features,
                                labels,
                            )
                            loss_relative_class_structure_scaled = (
                                self.relative_class_structure_weight
                                * identity_phase_scale
                                * loss_relative_class_structure
                            )
                        if (
                            teacher_features is not None
                            and self.use_teacher_prototype_anchor
                            and self.teacher_prototype_anchor_weight > 0
                            and self.identity_preserving_mode == 'margin_preserving'
                        ):
                            loss_teacher_prototype_anchor = self.teacher_prototype_anchor_loss(
                                features,
                                teacher_features,
                                labels,
                            )
                            loss_teacher_prototype_anchor_scaled = (
                                self.teacher_prototype_anchor_weight
                                * identity_phase_scale
                                * loss_teacher_prototype_anchor
                            )
                        if (
                            teacher_features is not None
                            and self.use_feature_trust_region
                            and self.feature_trust_region_weight > 0
                        ):
                            trust_severity = None
                            if ipaid_details is not None:
                                trust_severity = ipaid_details.get('correction_gap')
                            loss_feature_trust = self.feature_trust_region_loss(
                                features,
                                teacher_features,
                                severity=trust_severity,
                                labels=labels,
                            )
                            loss_feature_trust_scaled = (
                                self.feature_trust_region_weight * loss_feature_trust
                            )
                        if (
                            teacher_features is not None
                            and self.use_local_rank_preserving
                            and self.local_rank_preserving_weight > 0
                        ):
                            loss_local_rank = self.local_rank_preserving_loss(
                                features,
                                teacher_features,
                                labels,
                            )
                            loss_local_rank_scaled = (
                                self.local_rank_preserving_weight * loss_local_rank
                            )
                        if (
                            teacher_features is not None
                            and self.use_neighborhood_consistency
                            and self.neighborhood_consistency_weight > 0
                        ):
                            teacher_gallery_features_for_neighborhood = teacher_features.detach()
                            teacher_gallery_labels_for_neighborhood = labels.detach()
                            if (
                                isinstance(self.teacher_manifold_queue_features, torch.Tensor)
                                and isinstance(self.teacher_manifold_queue_labels, torch.Tensor)
                            ):
                                teacher_gallery_features_for_neighborhood = torch.cat(
                                    [
                                        teacher_gallery_features_for_neighborhood,
                                        self.teacher_manifold_queue_features,
                                    ],
                                    dim=0,
                                )
                                teacher_gallery_labels_for_neighborhood = torch.cat(
                                    [
                                        teacher_gallery_labels_for_neighborhood,
                                        self.teacher_manifold_queue_labels,
                                    ],
                                    dim=0,
                                )
                            loss_neighborhood_consistency = self.neighborhood_consistency_loss(
                                features,
                                teacher_features,
                                labels,
                                teacher_gallery_features=teacher_gallery_features_for_neighborhood,
                                teacher_gallery_labels=teacher_gallery_labels_for_neighborhood,
                                student_local_features=output.get('local_features'),
                                teacher_local_features=teacher_reference.get('local_features'),
                                same_source_size=teacher_features.size(0),
                            )
                            loss_neighborhood_consistency_scaled = (
                                self.neighborhood_consistency_weight
                                * identity_phase_scale
                                * loss_neighborhood_consistency
                            )
                        if (
                            teacher_features is not None
                            and photometric_stats is not None
                            and identity_phase_scale > 0
                            and self.use_teacher_gallery_alignment
                        ):
                            teacher_gallery_features = teacher_features.detach()
                            teacher_gallery_labels = labels.detach()
                            teacher_gallery_stats = photometric_stats.detach()
                            if (
                                isinstance(self.teacher_manifold_queue_features, torch.Tensor)
                                and isinstance(self.teacher_manifold_queue_labels, torch.Tensor)
                                and isinstance(self.teacher_manifold_queue_stats, torch.Tensor)
                            ):
                                teacher_gallery_features = torch.cat(
                                    [teacher_gallery_features, self.teacher_manifold_queue_features],
                                    dim=0,
                                )
                                teacher_gallery_labels = torch.cat(
                                    [teacher_gallery_labels, self.teacher_manifold_queue_labels],
                                    dim=0,
                                )
                                teacher_gallery_stats = torch.cat(
                                    [teacher_gallery_stats, self.teacher_manifold_queue_stats],
                                    dim=0,
                                )
                            if self.use_teacher_manifold and self.teacher_manifold_tube_weight > 0:
                                loss_teacher_manifold_tube = self.teacher_manifold_tube_loss(
                                    features,
                                    labels,
                                    photometric_stats,
                                    teacher_gallery_features,
                                    teacher_gallery_labels,
                                    teacher_gallery_stats,
                                    same_source_size=teacher_features.size(0),
                                )
                                loss_teacher_manifold_tube_scaled = (
                                    self.teacher_manifold_tube_weight
                                    * identity_phase_scale
                                    * loss_teacher_manifold_tube
                                )
                            if self.use_teacher_manifold and self.teacher_manifold_separation_weight > 0:
                                loss_teacher_manifold_separation = self.teacher_manifold_separation_loss(
                                    features,
                                    labels,
                                    photometric_stats,
                                    teacher_gallery_features,
                                    teacher_gallery_labels,
                                    teacher_gallery_stats,
                                    same_source_size=teacher_features.size(0),
                                )
                                loss_teacher_manifold_separation_scaled = (
                                    self.teacher_manifold_separation_weight
                                    * identity_phase_scale
                                    * loss_teacher_manifold_separation
                                )
                            if self.use_ranking_topology and self.ranking_topology_weight > 0:
                                loss_ranking_topology = self.ranking_topology_loss(
                                    features,
                                    labels,
                                    photometric_stats,
                                    teacher_gallery_features,
                                    teacher_gallery_labels,
                                    teacher_gallery_stats,
                                    same_source_size=teacher_features.size(0),
                                )
                                loss_ranking_topology_scaled = (
                                    self.ranking_topology_weight
                                    * identity_phase_scale
                                    * loss_ranking_topology
                                )
                            if (
                                self.use_anisotropic_identity_protection
                                and self.anisotropic_identity_protection_weight > 0
                            ):
                                loss_anisotropic_identity = self.anisotropic_identity_protection_loss(
                                    features,
                                    teacher_features.detach(),
                                    labels,
                                    photometric_stats,
                                    teacher_gallery_features,
                                    teacher_gallery_labels,
                                    teacher_gallery_stats,
                                    same_source_size=teacher_features.size(0),
                                )
                                loss_anisotropic_identity_scaled = (
                                    self.anisotropic_identity_protection_weight
                                    * identity_phase_scale
                                    * loss_anisotropic_identity
                                )
                        if (
                            teacher_features is not None
                            and ipaid_details is not None
                            and identity_phase_scale > 0
                            and self.use_cross_light_margin_preserving
                            and phase_cross_light_weight > 0
                            and self.identity_preserving_mode == 'margin_preserving'
                            and photometric_stats is not None
                        ):
                            loss_cross_light = self.cross_light_margin_preserving_loss(
                                features,
                                teacher_features,
                                labels,
                                photometric_stats,
                            )
                        if (
                            photometric_stats is not None
                            and identity_phase_scale > 0
                            and self.use_cross_light_softap
                            and phase_cross_light_weight > 0
                            and self.identity_preserving_mode == 'softap'
                        ):
                            gallery_features = features.detach()
                            gallery_labels = labels.detach()
                            gallery_stats = photometric_stats.detach()
                            if (
                                isinstance(self.cross_light_queue_features, torch.Tensor)
                                and isinstance(self.cross_light_queue_labels, torch.Tensor)
                                and isinstance(self.cross_light_queue_stats, torch.Tensor)
                            ):
                                gallery_features = torch.cat(
                                    [gallery_features, self.cross_light_queue_features],
                                    dim=0,
                                )
                                gallery_labels = torch.cat(
                                    [gallery_labels, self.cross_light_queue_labels],
                                    dim=0,
                                )
                                gallery_stats = torch.cat(
                                    [gallery_stats, self.cross_light_queue_stats],
                                    dim=0,
                                )
                            loss_cross_light = self.cross_light_softap_loss(
                                features,
                                labels,
                                photometric_stats,
                                gallery_features,
                                gallery_labels,
                                gallery_stats,
                                same_source_size=features.size(0),
                            )
                        if (
                            teacher_logits is not None
                            and isinstance(logits, torch.Tensor)
                            and self.use_semantic_non_confusion
                            and self.semantic_non_confusion_weight > 0
                        ):
                            loss_semantic_non_confusion = self.semantic_non_confusion_loss(
                                logits,
                                teacher_logits,
                                labels,
                            )
                            loss_semantic_non_confusion_scaled = (
                                self.semantic_non_confusion_weight
                                * identity_phase_scale
                                * loss_semantic_non_confusion
                            )
                        if (
                            photometric_stats is not None
                            and isinstance(nuisance_features, torch.Tensor)
                            and isinstance(photometric_prediction, torch.Tensor)
                            and self.use_nuisance_decoupling
                            and self.nuisance_decoupling_weight > 0
                        ):
                            pred_photometric = photometric_prediction.float()
                            target_photometric = photometric_stats.float()
                            if pred_photometric.size(1) != target_photometric.size(1):
                                shared_dim = min(
                                    pred_photometric.size(1),
                                    target_photometric.size(1),
                                )
                                pred_photometric = pred_photometric[:, :shared_dim]
                                target_photometric = target_photometric[:, :shared_dim]
                            loss_nuisance_regression = self.photometric_regression_loss(
                                pred_photometric,
                                target_photometric,
                            )
                            loss_nuisance_decorrelation = self.nuisance_decorrelation_loss(
                                features,
                                nuisance_features,
                            )
                            nuisance_weight = self.nuisance_decoupling_weight * identity_phase_scale
                            loss_nuisance_regression_scaled = (
                                nuisance_weight
                                * self.nuisance_regression_weight
                                * loss_nuisance_regression
                            )
                            loss_nuisance_decorrelation_scaled = (
                                nuisance_weight
                                * self.nuisance_decorrelation_weight
                                * loss_nuisance_decorrelation
                            )
                        if (
                            ipaid_details is not None
                            and self.use_identity_image_preserving
                            and self.identity_image_preserving_weight > 0
                        ):
                            identity_protection_map = ipaid_details.get('identity_protection_map')
                            reflectance = ipaid_details.get('reflectance')
                            original = ipaid_details.get('original')
                            if (
                                isinstance(identity_protection_map, torch.Tensor)
                                and isinstance(reflectance, torch.Tensor)
                                and isinstance(original, torch.Tensor)
                            ):
                                loss_identity_image = (
                                    identity_protection_map.float()
                                    * torch.abs(reflectance.float() - original.float())
                                ).mean()
                                loss_identity_image_scaled = (
                                    self.identity_image_preserving_weight * loss_identity_image
                                )

                    self._ensure_finite_tensor(
                        loss_relative_class_structure,
                        'loss.relative_class_structure',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_teacher_prototype_anchor,
                        'loss.teacher_prototype_anchor',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_feature_trust,
                        'loss.feature_trust_region',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_local_rank,
                        'loss.local_rank_preserving',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_neighborhood_consistency,
                        'loss.neighborhood_consistency',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_identity_image,
                        'loss.identity_image_preserving',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_teacher_manifold_tube,
                        'loss.teacher_manifold_tube',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_teacher_manifold_separation,
                        'loss.teacher_manifold_separation',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_ranking_topology,
                        'loss.ranking_topology',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_anisotropic_identity,
                        'loss.anisotropic_identity',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_semantic_non_confusion,
                        'loss.semantic_non_confusion',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_nuisance_regression,
                        'loss.nuisance_regression',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_nuisance_decorrelation,
                        'loss.nuisance_decorrelation',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_relative_class_structure_scaled,
                        'loss.relative_class_structure_scaled',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_teacher_prototype_anchor_scaled,
                        'loss.teacher_prototype_anchor_scaled',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_feature_trust_scaled,
                        'loss.feature_trust_region_scaled',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_local_rank_scaled,
                        'loss.local_rank_preserving_scaled',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_neighborhood_consistency_scaled,
                        'loss.neighborhood_consistency_scaled',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_identity_image_scaled,
                        'loss.identity_image_preserving_scaled',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_teacher_manifold_tube_scaled,
                        'loss.teacher_manifold_tube_scaled',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_teacher_manifold_separation_scaled,
                        'loss.teacher_manifold_separation_scaled',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_ranking_topology_scaled,
                        'loss.ranking_topology_scaled',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_anisotropic_identity_scaled,
                        'loss.anisotropic_identity_scaled',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_semantic_non_confusion_scaled,
                        'loss.semantic_non_confusion_scaled',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_nuisance_regression_scaled,
                        'loss.nuisance_regression_scaled',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    self._ensure_finite_tensor(
                        loss_nuisance_decorrelation_scaled,
                        'loss.nuisance_decorrelation_scaled',
                        phase,
                        epoch,
                        batch_idx,
                    )
                    illum_loss_dict['loss_relative_class_structure'] = loss_relative_class_structure.item()
                    illum_loss_dict['loss_teacher_prototype_anchor'] = loss_teacher_prototype_anchor.item()
                    illum_loss_dict['loss_feature_trust'] = loss_feature_trust.item()
                    illum_loss_dict['loss_local_rank'] = loss_local_rank.item()
                    illum_loss_dict['loss_neighborhood_consistency'] = loss_neighborhood_consistency.item()
                    illum_loss_dict['loss_identity_image'] = loss_identity_image.item()
                    illum_loss_dict['loss_teacher_manifold_tube'] = loss_teacher_manifold_tube.item()
                    illum_loss_dict['loss_teacher_manifold_separation'] = loss_teacher_manifold_separation.item()
                    illum_loss_dict['loss_ranking_topology'] = loss_ranking_topology.item()
                    illum_loss_dict['loss_anisotropic_identity'] = loss_anisotropic_identity.item()
                    illum_loss_dict['loss_semantic_non_confusion'] = loss_semantic_non_confusion.item()
                    illum_loss_dict['loss_nuisance_regression'] = loss_nuisance_regression.item()
                    illum_loss_dict['loss_nuisance_decorrelation'] = loss_nuisance_decorrelation.item()
                    illum_loss_dict['loss_cross_light'] = loss_cross_light.item()

                self._ensure_finite_tensor(loss_cross_light, 'loss.cross_light', phase, epoch, batch_idx)

                compute_reid_loss = phase != 2 or self.loss_weights['reid'] > 0
                if compute_reid_loss:
                    # 2. ReID losses
                    loss_ce = self.ce_loss(logits, labels)
                    loss_triplet = self.triplet_loss(features, labels)
                    loss_circle = self.circle_loss(features, labels)

                    if self.arcface_loss is not None and self.arcface_weight > 0:
                        loss_arcface = self.arcface_loss(features, labels)
                    else:
                        loss_arcface = torch.tensor(0.0, device=self.device)

                    if self.center_loss is not None:
                        loss_center = self.center_loss(features, labels)
                    else:
                        loss_center = torch.tensor(0.0, device=self.device)

                    loss_reid = (
                        self.ce_weight * loss_ce +
                        self.triplet_weight * loss_triplet +
                        self.circle_weight * loss_circle +
                        self.arcface_weight * loss_arcface +
                        self.center_loss_weight * loss_center
                    )

                    # 3. IICL consistency loss
                    if self.use_iicl and len(features_variants) > 0:
                        loss_iicl = self.iicl_consistency_loss(features, features_variants)
                    else:
                        loss_iicl = torch.tensor(0.0, device=self.device)

                else:
                    loss_reid = torch.tensor(0.0, device=self.device)
                    loss_iicl = torch.tensor(0.0, device=self.device)

                if (
                    phase == 3
                    and self.use_aux_gradient_gate
                    and compute_reid_loss
                    and self.use_iicl
                    and len(features_variants) > 0
                ):
                    iicl_alignment = _compute_nonnegative_gradient_alignment(
                        loss_reid,
                        loss_iicl,
                        features,
                        eps=self.aux_gradient_gate_eps,
                    )
                else:
                    iicl_alignment = torch.tensor(1.0, device=self.device)

                loss_cross_light_scaled = (
                    phase_cross_light_weight * identity_phase_scale * loss_cross_light
                )
                loss = (
                    phase_reid_weight * loss_reid +
                    phase_illumination_weight * loss_illum +
                    phase_iicl_weight * iicl_alignment * loss_iicl +
                    loss_cross_light_scaled +
                    loss_teacher_manifold_tube_scaled +
                    loss_teacher_manifold_separation_scaled +
                    loss_ranking_topology_scaled +
                    loss_anisotropic_identity_scaled +
                    loss_semantic_non_confusion_scaled +
                    loss_nuisance_regression_scaled +
                    loss_nuisance_decorrelation_scaled +
                    loss_relative_class_structure_scaled +
                    loss_teacher_prototype_anchor_scaled +
                    loss_feature_trust_scaled +
                    loss_local_rank_scaled +
                    loss_neighborhood_consistency_scaled +
                    loss_identity_image_scaled
                )

                self._ensure_finite_tensor(loss_reid, 'loss.reid', phase, epoch, batch_idx)
                self._ensure_finite_tensor(loss_iicl, 'loss.iicl', phase, epoch, batch_idx)
                self._ensure_finite_tensor(loss_cross_light_scaled, 'loss.cross_light_scaled', phase, epoch, batch_idx)
                self._ensure_finite_tensor(
                    loss_semantic_non_confusion_scaled,
                    'loss.semantic_non_confusion_scaled',
                    phase,
                    epoch,
                    batch_idx,
                )
                self._ensure_finite_tensor(
                    loss_teacher_manifold_tube_scaled,
                    'loss.teacher_manifold_tube_scaled',
                    phase,
                    epoch,
                    batch_idx,
                )
                self._ensure_finite_tensor(
                    loss_teacher_manifold_separation_scaled,
                    'loss.teacher_manifold_separation_scaled',
                    phase,
                    epoch,
                    batch_idx,
                )
                self._ensure_finite_tensor(
                    loss_nuisance_regression_scaled,
                    'loss.nuisance_regression_scaled',
                    phase,
                    epoch,
                    batch_idx,
                )
                self._ensure_finite_tensor(
                    loss_nuisance_decorrelation_scaled,
                    'loss.nuisance_decorrelation_scaled',
                    phase,
                    epoch,
                    batch_idx,
                )
                self._ensure_finite_tensor(
                    loss_relative_class_structure_scaled,
                    'loss.relative_class_structure_scaled',
                    phase,
                    epoch,
                    batch_idx,
                )
                self._ensure_finite_tensor(
                    loss_teacher_prototype_anchor_scaled,
                    'loss.teacher_prototype_anchor_scaled',
                    phase,
                    epoch,
                    batch_idx,
                )
                self._ensure_finite_tensor(loss_feature_trust_scaled, 'loss.feature_trust_region_scaled', phase, epoch, batch_idx)
                self._ensure_finite_tensor(loss_local_rank_scaled, 'loss.local_rank_preserving_scaled', phase, epoch, batch_idx)
                self._ensure_finite_tensor(
                    loss_neighborhood_consistency_scaled,
                    'loss.neighborhood_consistency_scaled',
                    phase,
                    epoch,
                    batch_idx,
                )
                self._ensure_finite_tensor(loss_identity_image_scaled, 'loss.identity_image_preserving_scaled', phase, epoch, batch_idx)
                self._ensure_finite_tensor(iicl_alignment, 'loss.iicl_alignment', phase, epoch, batch_idx)
                self._ensure_finite_tensor(loss, 'loss.total', phase, epoch, batch_idx)

            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            if self.grad_scaler.is_enabled():
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if self.center_loss is not None and self.center_loss_weight > 0:
                for param in self.center_loss.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1. / self.center_loss_weight)

            if self.grad_scaler.is_enabled():
                self.grad_scaler.unscale_(self.optimizer)

            grad_clip = self.config.get('training', {}).get('gradient_clip', 1.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
            if self.grad_scaler.is_enabled():
                if not torch.isfinite(grad_norm):
                    self.logger.warning(
                        "Non-finite grad norm detected under AMP "
                        "(phase=%d, epoch=%d, batch=%d, scaler=%.1f); "
                        "letting GradScaler skip/update this step.",
                        phase,
                        epoch + 1,
                        batch_idx + 1,
                        float(self.grad_scaler.get_scale()),
                    )
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self._ensure_finite_tensor(grad_norm, 'grad.norm', phase, epoch, batch_idx)
                self.optimizer.step()

            if phase == 3:
                self._update_cross_light_queue(features, labels, photometric_stats)
                self._update_teacher_manifold_queue(teacher_features, labels, photometric_stats)

            # Statistics
            total_loss += loss.item()
            illum_loss_sum += loss_illum.item()
            reid_loss_sum += loss_reid.item()
            num_batches += 1

            if isinstance(logits, torch.Tensor):
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 20 == 0:
                extra = (
                    f", illum_w: {phase_illumination_weight:.3f}"
                    f", photo_w: {photo_prior_weight:.3f}"
                    f", id_scale: {identity_phase_scale:.3f}"
                    f", photo: {loss_photo.item():.4f}"
                    f", idp: {loss_identity_preserve.item():.4f}"
                )
                if 'loss_anchor' in illum_loss_dict:
                    extra += f", anchor: {illum_loss_dict['loss_anchor']:.4f}"
                if 'loss_geometry' in illum_loss_dict:
                    extra += f", geom: {illum_loss_dict['loss_geometry']:.4f}"
                if 'loss_logit' in illum_loss_dict and self.identity_logit_weight > 0:
                    extra += f", logit: {illum_loss_dict['loss_logit']:.4f}"
                if self.use_iicl and len(features_variants) > 0:
                    extra += f", iicl_w: {phase_iicl_weight:.3f}"
                    extra += f", iicl: {loss_iicl.item():.4f}"
                    extra += f", iicl_align: {iicl_alignment.item():.4f}"
                if self.identity_preserving_mode == 'prototype' and self.use_cross_light_prototype:
                    extra += f", xproto_w: {phase_cross_light_weight:.3f}"
                    extra += f", xproto: {loss_cross_light.item():.4f}"
                if self.identity_preserving_mode == 'margin_preserving' and self.use_cross_light_margin_preserving:
                    extra += f", cmp_w: {phase_cross_light_weight:.3f}"
                    extra += f", cmp: {loss_cross_light.item():.4f}"
                if self.identity_preserving_mode == 'softap' and self.use_cross_light_softap:
                    extra += f", softap_w: {phase_cross_light_weight:.3f}"
                    extra += f", softap: {loss_cross_light.item():.4f}"
                if self.use_teacher_manifold and phase == 3:
                    extra += f", tm_tube: {loss_teacher_manifold_tube.item():.4f}"
                    extra += f", tm_sep: {loss_teacher_manifold_separation.item():.4f}"
                if self.use_ranking_topology and phase == 3:
                    extra += f", topo: {loss_ranking_topology.item():.4f}"
                if self.use_anisotropic_identity_protection and phase == 3:
                    extra += f", aniso: {loss_anisotropic_identity.item():.4f}"
                if self.use_semantic_non_confusion and phase == 3:
                    extra += f", sem: {loss_semantic_non_confusion.item():.4f}"
                if self.use_nuisance_decoupling and phase == 3:
                    extra += f", nu: {loss_nuisance_regression.item():.4f}"
                    extra += f", dec: {loss_nuisance_decorrelation.item():.4f}"
                if self.use_relative_class_structure and phase == 3:
                    extra += f", rel_w: {self.relative_class_structure_weight:.3f}"
                    extra += f", rel: {loss_relative_class_structure.item():.4f}"
                if (
                    self.use_teacher_prototype_anchor
                    and phase == 3
                    and self.identity_preserving_mode == 'margin_preserving'
                ):
                    extra += f", tproto_w: {self.teacher_prototype_anchor_weight:.3f}"
                    extra += f", tproto: {loss_teacher_prototype_anchor.item():.4f}"
                if self.use_feature_trust_region and phase == 3:
                    extra += f", trust: {loss_feature_trust.item():.4f}"
                if self.use_local_rank_preserving and phase == 3:
                    extra += f", rank: {loss_local_rank.item():.4f}"
                if self.use_neighborhood_consistency and phase == 3:
                    extra += f", neigh: {loss_neighborhood_consistency.item():.4f}"
                if self.use_identity_image_preserving and phase == 3:
                    extra += f", imgp: {loss_identity_image.item():.4f}"
                if 'lambda_color_mean' in illum_loss_dict:
                    extra += f", lambda_color: {illum_loss_dict['lambda_color_mean']:.4f}"
                if ipaid_details is not None and isinstance(ipaid_details.get('rollback_alpha'), torch.Tensor):
                    extra += f", rollback: {ipaid_details['rollback_alpha'].mean().item():.4f}"
                self.logger.info(
                    f"Phase {phase} Epoch [{epoch+1}] "
                    f"Batch [{batch_idx+1}/{len(self.dataloader)}] "
                    f"Loss: {loss.item():.4f} (illum: {loss_illum.item():.4f}, "
                    f"reid: {loss_reid.item():.4f}{extra}) "
                    f"Acc: {100.*correct/max(total, 1):.2f}%"
                )

        total_loss, illum_loss_sum, reid_loss_sum, correct, total, num_batches = self._reduce_epoch_stats(
            total_loss,
            illum_loss_sum,
            reid_loss_sum,
            correct,
            total,
            num_batches,
        )
        num_batches = max(num_batches, 1)
        total = max(total, 1)
        return {
            'total_loss': total_loss / num_batches,
            'illum_loss': illum_loss_sum / num_batches,
            'reid_loss': reid_loss_sum / num_batches,
            'accuracy': 100. * correct / total,
            'photo_prior_weight': photo_prior_weight,
            'identity_scale': identity_phase_scale,
            'illumination_weight': phase_illumination_weight,
            'iicl_weight': phase_iicl_weight,
            'cross_light_weight': phase_cross_light_weight,
        }

    @torch.no_grad()
    def evaluate_on_validation_set(self) -> Optional[Dict[str, float]]:
        """
        Evaluate on the in-training validation split.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            if not hasattr(self, 'val_samples') or len(self.val_samples) == 0:
                return None

            # Extract features from validation samples
            val_paths = [sample[0] for sample in self.val_samples]
            val_labels = np.array([sample[1] for sample in self.val_samples])

            # Create temporary dataset for validation
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.img_height, self.img_width)),
                transforms.ToTensor(),
            ])

            # Prepare samples in the format [(img_path, identity), ...]
            val_samples_formatted = [(path, label) for path, label in zip(val_paths, val_labels)]
            val_dataset = ReIDDataset(samples=val_samples_formatted, transform=transform)
            val_loader = DataLoader(
                val_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )

            # Extract features
            features, ids, cams, paths = extract_features(
                self.model,
                val_loader,
                self.device,
                flip_test=self.eval_flip_test
            )

            # Compute distance matrix (all-vs-all)
            distmat = compute_distance_matrix(features, features, metric="cosine")

            # Exclude self-matching (diagonal)
            np.fill_diagonal(distmat, np.inf)

            # Compute CMC and mAP
            cmc, m_ap = compute_cmc_map(
                distmat,
                val_labels,
                val_labels,
                query_cams=None,
                gallery_cams=None,
                query_paths=paths,
                gallery_paths=paths,
                exclude_same_camera=False,  # No camera info in ATRW
            )

            return {
                'rank1': cmc[0] * 100,
                'rank5': cmc[4] * 100 if len(cmc) > 4 else 0.0,
                'rank10': cmc[9] * 100 if len(cmc) > 9 else 0.0,
                'mAP': m_ap * 100
            }
        finally:
            self.model.train(was_training)

    def _infer_atrw_data_root(self) -> str:
        data_root = self.atrw_eval_cfg.get('data_root')
        if data_root:
            return str(data_root)

        candidate = os.path.abspath(self.data_dir)
        base = os.path.basename(candidate).lower()
        if base == 'train':
            candidate = os.path.dirname(candidate)
            base = os.path.basename(candidate).lower()
        if base == 'atrw_reid_train':
            candidate = os.path.dirname(candidate)
        return candidate

    def _resolve_atrw_test_dir(self, data_root: str) -> str:
        configured = self.atrw_eval_cfg.get('test_dir')
        if configured:
            return str(configured)

        candidate_1 = os.path.join(data_root, "test")
        candidate_2 = os.path.join(data_root, "atrw_reid_test", "test")
        if os.path.exists(candidate_1):
            return candidate_1
        if os.path.exists(candidate_2):
            return candidate_2
        raise FileNotFoundError(
            f"ATRW test directory not found. Tried: {candidate_1} | {candidate_2}"
        )

    def _build_eval_loader_from_samples(
        self,
        samples: List[Tuple[str, int]],
        batch_size: int,
        num_workers: int,
    ) -> DataLoader:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
        ])
        dataset = ReIDDataset(samples=samples, transform=transform)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers if os.name != 'nt' else 0,
            pin_memory=self.device.type == 'cuda',
        )

    @torch.no_grad()
    def evaluate_atrw_openset(self) -> Optional[Dict[str, float]]:
        data_root = self._infer_atrw_data_root()
        eval_script_dir = str(self.atrw_eval_cfg.get('eval_script_dir', 'ATRWEvalScript-main'))
        batch_size = int(self.atrw_eval_cfg.get('batch_size', 64))
        num_workers = int(self.atrw_eval_cfg.get('num_workers', self.num_workers))

        gt_file = os.path.join(eval_script_dir, "annotations", "gt_test_plain.json")
        if not os.path.exists(gt_file):
            raise FileNotFoundError(f"ATRW official GT not found: {gt_file}")

        from tools.eval_atrw_openset import get_test_images

        gt_data = load_atrw_gt(gt_file)
        test_dir = self._resolve_atrw_test_dir(data_root)
        samples = get_test_images(test_dir, set(gt_data["imgids"]))
        if not samples:
            raise RuntimeError(f"No valid ATRW open-set samples found in {test_dir}")

        loader = self._build_eval_loader_from_samples(samples, batch_size=batch_size, num_workers=num_workers)
        features, imgids, _, _ = extract_features(
            self.model,
            loader,
            self.device,
            flip_test=self.eval_flip_test,
        )
        if bool(self.eval_cfg.get('rerank', False)):
            rerank_params = dict(self.eval_cfg.get('rerank_params', {}))
            distmat = re_ranking(
                features,
                features,
                k1=int(rerank_params.get('k1', 20)),
                k2=int(rerank_params.get('k2', 6)),
                lambda_value=float(rerank_params.get('lambda_value', 0.3)),
            )
            submission = build_submission_from_distance(imgids, distmat)
        else:
            submission = build_submission_from_features(imgids, features)
        return evaluate_atrw_official(gt_data, submission)

    @torch.no_grad()
    def evaluate_on_selection_query_gallery(self) -> Optional[Dict[str, float]]:
        if self.selection_eval_spec is None:
            return None

        was_training = self.model.training
        self.model.eval()
        try:
            evaluator = ReIDEvaluator(
                model=self.model,
                device=self.device,
                img_height=self.img_height,
                img_width=self.img_width,
                batch_size=int(self.eval_cfg.get('batch_size', 32)),
                flip_test=self.eval_flip_test,
                rerank=bool(self.eval_cfg.get('rerank', False)),
                rerank_params=dict(self.eval_cfg.get('rerank_params', {})),
                num_workers=self.num_workers,
                exclude_same_camera=bool(
                    self.selection_eval_spec.get('exclude_same_camera', self.eval_cfg.get('exclude_same_camera', True))
                ),
            )
            return evaluator.evaluate(
                self.selection_eval_spec["query_dir"],
                self.selection_eval_spec["gallery_dir"],
                openset_info={} if self.selection_eval_spec.get("force_standard_eval", False) else None,
            )
        finally:
            self.model.train(was_training)

    @torch.no_grad()
    def evaluate_atrw_closedset(self, protocol: str) -> Optional[Dict[str, float]]:
        from tools.eval_atrw_closedset import (
            evaluate_samples,
            prepare_animals701_samples,
            prepare_train70_val30_samples,
        )

        data_root = self._infer_atrw_data_root()
        eval_script_dir = str(self.atrw_eval_cfg.get('eval_script_dir', 'ATRWEvalScript-main'))
        batch_size = int(self.atrw_eval_cfg.get('batch_size', 64))
        train_ratio = float(self.atrw_eval_cfg.get('train_ratio', 0.7))
        seed = int(self.atrw_eval_cfg.get('seed', 42))

        if protocol == 'atrw_closedset_animals_701':
            samples, info = prepare_animals701_samples(data_root, eval_script_dir)
            distance_metric = "euclidean"
            self.logger.info(
                "ATRW closed-set animals_701: images=%d entities=%d missing=%d",
                info.get('eval_images', 0),
                info.get('eval_entities', 0),
                info.get('missing_images', 0),
            )
        else:
            samples, info = prepare_train70_val30_samples(data_root, train_ratio, seed)
            distance_metric = "cosine"
            self.logger.info(
                "ATRW closed-set train70_val30: images=%d entities=%d seed=%d ratio=%.2f",
                info.get('eval_images', 0),
                info.get('eval_entities', 0),
                seed,
                train_ratio,
            )

        return evaluate_samples(
            model=self.model,
            samples=samples,
            device=self.device,
            batch_size=batch_size,
            distance_metric=distance_metric,
            img_height=self.img_height,
            img_width=self.img_width,
            flip_test=self.eval_flip_test,
        )

    def _evaluate_single_protocol(self, protocol: str) -> Optional[Dict[str, float]]:
        if protocol == 'val_split_70_30':
            return self.evaluate_on_validation_set()
        if protocol == QUERY_GALLERY_PROTOCOL:
            return self.evaluate_on_selection_query_gallery()
        if protocol == CROSS_SPECIES_SELECTION_PROTOCOL:
            return self.evaluate_on_selection_query_gallery()
        if protocol == 'atrw_openset':
            return self.evaluate_atrw_openset()
        if protocol in {'atrw_closedset_train70_val30', 'atrw_closedset_animals_701'}:
            return self.evaluate_atrw_closedset(protocol)

        message = f"Unsupported evaluation protocol: {protocol}"
        if self.strict_protocol_check:
            raise RuntimeError(message)
        self.logger.warning(message)
        return None

    def evaluate_for_model_selection(self) -> Optional[Dict[str, float]]:
        if not self.is_main_process:
            self.last_additional_eval_results = {}
            return None
        self.last_additional_eval_results = {}
        try:
            _release_cuda_eval_memory()
            primary_metrics = self._evaluate_single_protocol(self.eval_protocol)
            if not primary_metrics:
                return None

            additional_results: Dict[str, Dict[str, float]] = {}
            for protocol in self.additional_eval_protocols:
                try:
                    metrics = self._evaluate_single_protocol(protocol)
                except Exception as exc:
                    self.logger.warning("Additional evaluation skipped for protocol '%s': %s", protocol, exc)
                    continue
                if metrics:
                    additional_results[protocol] = metrics

            self.last_additional_eval_results = additional_results
            return primary_metrics
        except Exception as exc:
            if self.strict_protocol_check:
                raise
            self.logger.warning("Evaluation skipped for protocol '%s': %s", self.eval_protocol, exc)
            return None
        finally:
            _release_cuda_eval_memory()

    def _log_additional_eval_results(self) -> None:
        for protocol, metrics in self.last_additional_eval_results.items():
            self.logger.info(
                "Additional Evaluation (%s): %s",
                protocol,
                self._format_eval_metrics(metrics),
            )

    def _format_eval_metrics(self, eval_metrics: Dict[str, float]) -> str:
        if 'rank1' in eval_metrics:
            return (
                f"Rank-1: {eval_metrics.get('rank1', 0.0):.2f}%, "
                f"Rank-5: {eval_metrics.get('rank5', 0.0):.2f}%, "
                f"mAP: {eval_metrics.get('mAP', 0.0):.2f}%"
            )
        if 'mmAP' in eval_metrics:
            return (
                f"rank1_single: {eval_metrics.get('rank1_single', 0.0):.2f}%, "
                f"rank1_cross: {eval_metrics.get('rank1_cross', 0.0):.2f}%, "
                f"mAP_single: {eval_metrics.get('mAP_single', 0.0):.2f}%, "
                f"mAP_cross: {eval_metrics.get('mAP_cross', 0.0):.2f}%, "
                f"mmAP: {eval_metrics.get('mmAP', 0.0):.2f}%"
            )
        return ", ".join(f"{k}: {float(v):.4f}" for k, v in eval_metrics.items())

    def _extract_reid_selection_metrics(self, eval_metrics: Dict[str, float]) -> Tuple[float, float]:
        current_rank1 = float(
            eval_metrics.get('rank1', eval_metrics.get('rank1_cross', eval_metrics.get('rank1_single', 0.0)))
        )
        current_map = float(
            eval_metrics.get('mAP', eval_metrics.get('mmAP', eval_metrics.get('mAP_cross', eval_metrics.get('mAP_single', 0.0))))
        )
        return current_rank1, current_map

    def _extract_primary_metric(self, eval_metrics: Dict[str, float]) -> float:
        if self.best_metric_name in eval_metrics:
            return float(eval_metrics[self.best_metric_name])
        current_rank1, current_map = self._extract_reid_selection_metrics(eval_metrics)
        if 'mmAP' in eval_metrics:
            return float(eval_metrics.get('mmAP', current_map))
        return current_rank1

    def train(self):
        """Complete three-phase training."""
        def maybe_run_eval(
            epoch: int,
            phase: int,
            total_epochs: int,
            metrics: Dict[str, float],
        ) -> None:
            should_eval = ((epoch + 1) % self.eval_interval == 0) or ((epoch + 1) == total_epochs)
            if not should_eval:
                return

            eval_error: Optional[Exception] = None
            error_flag = torch.zeros(
                1,
                device=self.device if self.device.type == 'cuda' else torch.device('cpu'),
                dtype=torch.int64,
            )
            self._barrier()
            try:
                eval_metrics = self.evaluate_for_model_selection()
            except Exception as exc:
                eval_error = exc
                error_flag.fill_(1)

            if self.is_distributed:
                dist.broadcast(error_flag, src=0)
            if int(error_flag.item()) > 0:
                if eval_error is not None and not self.is_distributed:
                    raise eval_error
                raise RuntimeError(
                    f"Rank-0 evaluation failed at phase={phase}, epoch={epoch + 1}"
                )

            if self.is_main_process and eval_metrics:
                self.logger.info("Evaluation Results (%s): %s", self.eval_protocol, self._format_eval_metrics(eval_metrics))
                self._log_additional_eval_results()

                metric_value = self._extract_primary_metric(eval_metrics)
                current_rank1, current_map = self._extract_reid_selection_metrics(eval_metrics)
                is_new_best_metric = metric_value > self.best_metric_value
                rank_eps = 1e-12
                rank1_tied = abs(current_rank1 - self.best_rank1) <= rank_eps
                is_new_best_reid = (
                    current_rank1 > self.best_rank1
                    or (rank1_tied and current_map > self.best_map)
                )

                if is_new_best_metric or is_new_best_reid:
                    checkpoint_metrics = dict(metrics)
                    checkpoint_metrics['eval'] = eval_metrics

                    if is_new_best_metric:
                        self.best_metric_value = metric_value
                        self.save_checkpoint(epoch, checkpoint_metrics, phase=phase, is_best=True, suffix='')
                        self.logger.info(
                            f"New best joint_best by {self.best_metric_name}: {self.best_metric_value:.2f}"
                        )

                    if is_new_best_reid:
                        self.best_rank1 = current_rank1
                        self.best_map = current_map
                        self.save_checkpoint(epoch, checkpoint_metrics, phase=phase, is_best=True, suffix='_reid_best')
                        self.logger.info(
                            f"New best joint_best_reid_best: Rank-1={self.best_rank1:.2f}%, "
                            f"mAP={self.best_map:.2f}%"
                        )
            self._barrier()

        schedule_name = "Starting Three-Phase Training"
        if self.phase2_fgid_epochs <= 0:
            schedule_name = "Starting Two-Stage Training"
        self.logger.info("=" * 70)
        self.logger.info(schedule_name)
        self.logger.info(f"Phase 1: {self.phase1_epochs} epochs (ReID Warmup)")
        self.logger.info(f"Phase 2: {self.phase2_fgid_epochs} epochs (Illumination Optimization)")
        self.logger.info(f"Phase 3: {self.phase3_epochs} epochs (Joint Optimization)")
        self.logger.info("=" * 70)

        best_acc = self.best_acc

        if self.resume_phase == 1:
            start_phase1_epoch = min(self.resume_epoch, self.phase1_epochs)
        elif self.resume_phase and self.resume_phase > 1:
            start_phase1_epoch = self.phase1_epochs
        else:
            start_phase1_epoch = 0

        try:
            self.logger.info("\n" + "=" * 30 + " Phase 1: ReID Warmup " + "=" * 30)
            self._setup_optimizer_phase1()

            if self.resume_phase == 1 and self.resume_optimizer_state and self.resume_scheduler_state:
                self.optimizer.load_state_dict(self.resume_optimizer_state)
                self.scheduler.load_state_dict(self.resume_scheduler_state)

            for epoch in range(start_phase1_epoch, self.phase1_epochs):
                self.current_phase = 1
                self.current_epoch = epoch
                self._set_sampler_epoch(epoch)
                metrics = self.train_epoch(epoch, phase=1)
                self.scheduler.step()
                lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f"Phase 1 Epoch [{epoch+1}/{self.phase1_epochs}] | "
                    f"Loss: {metrics['total_loss']:.4f} | Acc: {metrics['accuracy']:.2f}% | LR: {lr:.6f}"
                )
                if metrics['accuracy'] > best_acc:
                    best_acc = metrics['accuracy']
                    self.best_acc = best_acc

                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(epoch, metrics, phase=1, is_best=False)

                maybe_run_eval(epoch, phase=1, total_epochs=self.phase1_epochs, metrics=metrics)

            if self.resume_phase == 1:
                self.resume_phase = None
                self.resume_epoch = 0
                self.resume_optimizer_state = None
                self.resume_scheduler_state = None

            if self.phase2_fgid_epochs > 0:
                self.logger.info("\n" + "=" * 30 + " Phase 2: Illumination Optimization " + "=" * 30)
                self._setup_optimizer_phase2_fgid()

                if self.resume_phase == 2:
                    start_p2 = min(self.resume_epoch, self.phase2_fgid_epochs)
                    if self.resume_optimizer_state and self.resume_scheduler_state:
                        self.optimizer.load_state_dict(self.resume_optimizer_state)
                        self.scheduler.load_state_dict(self.resume_scheduler_state)
                elif self.resume_phase and self.resume_phase > 2:
                    start_p2 = self.phase2_fgid_epochs
                else:
                    start_p2 = 0

                for epoch in range(start_p2, self.phase2_fgid_epochs):
                    self.current_phase = 2
                    self.current_epoch = epoch
                    self._set_sampler_epoch(epoch)
                    metrics = self.train_epoch(epoch, phase=2)
                    self.scheduler.step()
                    lr = self.optimizer.param_groups[0]['lr']
                    self.logger.info(
                        f"Phase 2 Epoch [{epoch+1}/{self.phase2_fgid_epochs}] | "
                        f"Loss: {metrics['total_loss']:.4f} | Illum: {metrics['illum_loss']:.4f} | LR: {lr:.6f}"
                    )
                    if (epoch + 1) % 10 == 0:
                        self.save_checkpoint(epoch, metrics, phase=2, is_best=False)
                    maybe_run_eval(epoch, phase=2, total_epochs=self.phase2_fgid_epochs, metrics=metrics)
            else:
                self.logger.info("\n" + "=" * 30 + " Phase 2: Skipped (0 epochs) " + "=" * 30)

            if self.resume_phase == 2:
                self.resume_phase = None
                self.resume_epoch = 0
                self.resume_optimizer_state = None
                self.resume_scheduler_state = None

            if self.phase3_epochs > 0:
                self.logger.info("\n" + "=" * 30 + " Phase 3: Joint Optimization " + "=" * 30)
                self._setup_optimizer_phase3()

                if self.resume_phase == 3:
                    start_p3 = min(self.resume_epoch, self.phase3_epochs)
                    if self.resume_optimizer_state and self.resume_scheduler_state:
                        self.optimizer.load_state_dict(self.resume_optimizer_state)
                        self.scheduler.load_state_dict(self.resume_scheduler_state)
                else:
                    start_p3 = 0

                self._refresh_phase3_teacher()
                self._reset_cross_light_queue()
                self._reset_teacher_manifold_queue()

                for epoch in range(start_p3, self.phase3_epochs):
                    self.current_phase = 3
                    self.current_epoch = epoch
                    self._set_sampler_epoch(epoch)
                    metrics = self.train_epoch(epoch, phase=3)
                    self.scheduler.step()
                    lr = self.optimizer.param_groups[0]['lr']
                    self.logger.info(
                        f"Phase 3 Epoch [{epoch+1}/{self.phase3_epochs}] | "
                        f"Loss: {metrics['total_loss']:.4f} | Illum: {metrics['illum_loss']:.4f} | "
                        f"ReID: {metrics['reid_loss']:.4f} | Acc: {metrics['accuracy']:.2f}% | LR: {lr:.6f} | "
                        f"AuxW: illum={metrics['illumination_weight']:.3f}, "
                        f"iicl={metrics['iicl_weight']:.3f}, "
                        f"{self._cross_light_objective_name()}={metrics['cross_light_weight']:.3f}"
                    )
                    if metrics['accuracy'] > best_acc:
                        best_acc = metrics['accuracy']
                        self.best_acc = best_acc

                    if (epoch + 1) % 10 == 0:
                        self.save_checkpoint(epoch, metrics, phase=3, is_best=False)

                    maybe_run_eval(epoch, phase=3, total_epochs=self.phase3_epochs, metrics=metrics)
            else:
                self.logger.info("\n" + "=" * 30 + " Phase 3: Skipped (0 epochs) " + "=" * 30)

        except KeyboardInterrupt:
            self._save_emergency_checkpoint('interrupt')
            raise
        except Exception:
            self._save_emergency_checkpoint('exception')
            raise
        finally:
            self.resume_optimizer_state = None
            self.resume_scheduler_state = None

        self.best_acc = best_acc
        self.logger.info("=" * 70)
        self.logger.info(f"Training Complete! Best train accuracy: {best_acc:.2f}%")
        if self.best_metric_value > 0:
            self.logger.info(
                f"Best Metric: {self.best_metric_name}: {self.best_metric_value:.2f} | "
                f"Best ReID Rank-1: {self.best_rank1:.2f}% | mAP: {self.best_map:.2f}%"
            )
        self.logger.info("=" * 70)

    def save_checkpoint(self, epoch: int, metrics: dict, phase: int, is_best: bool = False, suffix: str = ''):
        """Save training checkpoint."""
        if not self.is_main_process:
            return
        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'num_classes': self.num_classes,
            'metrics': metrics,
            'config': self.config,
            'best_acc': self.best_acc,
            'best_rank1': self.best_rank1,
            'best_map': self.best_map,
            'best_metric_name': self.best_metric_name,
            'best_metric_value': self.best_metric_value,
            'eval_protocol': self.eval_protocol,
            'additional_eval_protocols': list(self.additional_eval_protocols),
        }

        if is_best:
            path = os.path.join(self.output_dir, f'joint_best{suffix}.pth')
        else:
            path = os.path.join(self.output_dir, f'joint_phase{phase}_epoch{epoch+1}.pth')

        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")


# ============================================================================
#                           Utility Functions
# ============================================================================

def setup_logging(log_dir: str, is_main_process: bool = True, rank: int = 0) -> logging.Logger:
    """Initialize training logger."""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger('JointTraining')
    logger.setLevel(logging.INFO if is_main_process else logging.WARNING)
    logger.handlers.clear()
    logger.propagate = False

    if not is_main_process:
        logger.addHandler(logging.NullHandler())
        return logger

    fh = logging.FileHandler(os.path.join(log_dir, 'joint_training.log'), encoding='utf-8')
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(f'[%(asctime)s] [rank={rank}] %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ============================================================================
#                           Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Joint training script with three-phase ReID/illumination optimization')
    parser.add_argument('--data_dir', type=str, required=True, help='Training data root directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for checkpoints/logs')
    parser.add_argument('--config', type=str, default='./config/illumination_config.yaml', help='YAML config path')
    parser.add_argument('--backbone', type=str, default=None, choices=SUPPORTED_BACKBONES, help='Backbone model name')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--phase1_epochs', type=int, default=None, help='Phase 1 epochs')
    parser.add_argument('--phase2_epochs', type=int, default=None, help='Phase 2 epochs')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--num_stripes', type=int, default=None, help='Number of local stripes')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto/cpu/cuda')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for torchrun compatibility')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--eval_interval', type=int, default=None, help='Run evaluation every N epochs')
    parser.add_argument('--p_size', type=int, default=None, help='PK sampler P value')
    parser.add_argument('--k_size', type=int, default=None, help='PK sampler K value')
    parser.add_argument('--circle_gamma', type=int, default=None, help='Circle loss gamma')
    parser.add_argument('--img_height', type=int, default=None, help='Input image height')
    parser.add_argument('--img_width', type=int, default=None, help='Input image width')
    parser.add_argument(
        '--eval_protocol',
        type=str,
        choices=[
            'val_split_70_30',
            QUERY_GALLERY_PROTOCOL,
            'atrw_openset',
            'atrw_closedset_train70_val30',
            'atrw_closedset_animals_701',
        ],
        default=None,
        help='In-training evaluation protocol',
    )
    parser.add_argument('--best_metric', type=str, default=None, help='Primary metric name for joint_best selection')
    parser.add_argument('--strict_protocol_check', dest='strict_protocol_check', action='store_true',
                        help='Enable strict protocol checks')
    parser.add_argument('--no_strict_protocol_check', dest='strict_protocol_check', action='store_false',
                        help='Disable strict protocol checks')

    parser.add_argument('--use_iicl', dest='use_iicl', action='store_true', help='Enable IICL')
    parser.add_argument('--no_iicl', dest='use_iicl', action='store_false', help='Disable IICL')
    parser.set_defaults(use_iicl=None, strict_protocol_check=None)
    parser.add_argument('--iicl_weight', type=float, default=None, help='IICL loss weight')
    parser.add_argument('--iicl_variants', type=int, default=None, help='Number of IICL variants')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')

    args = parser.parse_args()

    cli_cfg = cli_args_to_config(args)
    config = load_config(args.config, cli_overrides=cli_cfg)
    rank_hint = _env_int('RANK', 0)

    training_cfg = config.get('training', {})
    model_cfg = config.get('model', {})
    phase1_cfg = training_cfg.get('phases', {}).get('phase1', {})
    phase2_cfg = training_cfg.get('phases', {}).get('phase2', {})
    phase3_cfg = training_cfg.get('phases', {}).get('phase3', {})
    pk_cfg = training_cfg.get('pk_sampler', {})
    eval_cfg = config.get('evaluation', {}) or {}

    # Read output_dir from top-level config first, then fall back to training section
    output_dir = config.get('output_dir', training_cfg.get('output_dir', './checkpoints/joint'))
    backbone = model_cfg.get('backbone', 'osnet_ain_x1_0')
    batch_size = training_cfg.get('batch_size', 32)
    phase1_epochs = phase1_cfg.get('epochs', 15)
    phase2_epochs = phase2_cfg.get('epochs', 15)
    phase3_epochs = phase3_cfg.get('epochs', 100)
    learning_rate = training_cfg.get('optimizer', {}).get('lr', training_cfg.get('learning_rate', 3.5e-4))
    num_stripes = model_cfg.get('local_extractor', {}).get('num_parts', 6)
    eval_interval = training_cfg.get('eval_interval', 5)
    p_size = pk_cfg.get('p')
    k_size = pk_cfg.get('k', 4)
    circle_gamma = training_cfg.get('metric_learning', {}).get('circle_loss', {}).get('gamma', 256)
    img_height = training_cfg.get('image_height', 256)
    img_width = training_cfg.get('image_width', 512)
    num_workers = config.get('hardware', {}).get('num_workers', 4)
    eval_protocol = eval_cfg.get('protocol', 'val_split_70_30')
    additional_eval_protocols = eval_cfg.get('additional_protocols', []) or []
    best_metric = eval_cfg.get('best_metric', 'rank1')
    strict_protocol_check = eval_cfg.get('strict_protocol_check', False)

    if rank_hint == 0:
        print(f"\n{'='*60}")
        print(f"Config file: {args.config}")
        print(f"{'='*60}")
        print(f"  backbone: {backbone}")
        print(f"  batch_size: {batch_size}")
        print(f"  p_size: {p_size}, k_size: {k_size}")
        print(f"  phase1_epochs: {phase1_epochs}, phase2_epochs: {phase2_epochs}, phase3_epochs: {phase3_epochs}")
        print(f"  learning_rate: {learning_rate}")
        print(f"  img_size: {img_height}x{img_width}")
        print(f"  num_workers: {num_workers}")
        print(f"  eval_protocol: {eval_protocol}")
        print(f"  additional_eval_protocols: {additional_eval_protocols}")
        print(f"  best_metric: {best_metric}")
        print(f"  strict_protocol_check: {strict_protocol_check}")
        print(f"{'='*60}\n")

    if not os.path.exists(args.data_dir):
        print(f"[ERROR] data_dir not found: {args.data_dir}")
        sys.exit(1)

    dist_ctx = init_distributed_mode(args.device, config, cli_local_rank=args.local_rank)
    logger = setup_logging(
        output_dir,
        is_main_process=dist_ctx['rank'] == 0,
        rank=dist_ctx['rank'],
    )

    try:
        trainer = JointTrainer(
            data_dir=args.data_dir,
            output_dir=output_dir,
            config=config,
            config_path=args.config,
            backbone=backbone,
            batch_size=batch_size,
            phase1_epochs=phase1_epochs,
            phase2_epochs=phase2_epochs,
            phase3_epochs=phase3_epochs,
            learning_rate=learning_rate,
            num_stripes=num_stripes,
            device=args.device,
            logger=logger,
            resume_checkpoint=args.resume,
            eval_interval=eval_interval,
            p_size=p_size,
            k_size=k_size,
            circle_gamma=circle_gamma,
            img_height=img_height,
            img_width=img_width,
            use_iicl=args.use_iicl,
            iicl_weight=args.iicl_weight,
            iicl_num_variants=args.iicl_variants,
            num_workers=num_workers,
            rank=dist_ctx['rank'],
            local_rank=dist_ctx['local_rank'],
            world_size=dist_ctx['world_size'],
            is_distributed=dist_ctx['is_distributed'],
            ddp_find_unused_parameters=dist_ctx['find_unused_parameters'],
        )

        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()

