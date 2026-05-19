import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    torch = None
    DataLoader = None
    Dataset = object

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    cv2 = None


@unittest.skipIf(torch is None, "torch is unavailable in this interpreter")
class RouteBAlignmentRuntimeTests(unittest.TestCase):
    def _write_image(self, path: Path) -> None:
        image = np.full((16, 16, 3), 127, dtype=np.uint8)
        if cv2 is None:
            raise unittest.SkipTest("cv2 is unavailable in this interpreter")
        ok = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"failed to write test image: {path}")

    def test_extract_features_restores_model_training_state(self):
        from app.core.evaluation import extract_features

        class TinyDataset(Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, index):
                img = torch.rand(3, 8, 8)
                return img, index, -1, f"img_{index}.jpg"

        class TinyModel(torch.nn.Module):
            def forward(self, imgs, boxes_list=None, return_illuminated=False):
                return {"features": imgs.mean(dim=(2, 3))}

        loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False)
        model = TinyModel()

        model.train()
        self.assertTrue(model.training)
        extract_features(model, loader, torch.device("cpu"), flip_test=False)
        self.assertTrue(model.training)

        model.eval()
        self.assertFalse(model.training)
        extract_features(model, loader, torch.device("cpu"), flip_test=False)
        self.assertFalse(model.training)

    def test_extract_features_retries_oom_by_splitting_batches(self):
        from app.core.evaluation import extract_features

        class TinyDataset(Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, index):
                img = torch.full((3, 8, 8), float(index + 1))
                return img, index, -1, f"img_{index}.jpg"

        class OOMOnLargeBatchModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen_batch_sizes = []

            def forward(self, imgs, boxes_list=None, return_illuminated=False):
                self.seen_batch_sizes.append(int(imgs.shape[0]))
                if imgs.shape[0] > 1:
                    raise torch.OutOfMemoryError("synthetic eval oom")
                return {"features": imgs.mean(dim=(2, 3))}

        loader = DataLoader(TinyDataset(), batch_size=4, shuffle=False)
        model = OOMOnLargeBatchModel()

        feats, ids, cams, paths = extract_features(
            model,
            loader,
            torch.device("cpu"),
            flip_test=False,
        )

        self.assertEqual(feats.shape, (4, 3))
        self.assertEqual(ids, [0, 1, 2, 3])
        self.assertEqual(cams, [-1, -1, -1, -1])
        self.assertEqual(paths, [f"img_{i}.jpg" for i in range(4)])
        self.assertIn(4, model.seen_batch_sizes)
        self.assertTrue(all(batch_size <= 2 for batch_size in model.seen_batch_sizes[1:]))
        self.assertEqual(model.seen_batch_sizes.count(1), 4)

    def _build_model(self):
        from app.core.joint_model import JointReIDModel

        model = JointReIDModel(
            num_classes=4,
            backbone_name="osnet_ain_x1_0",
            pretrained_backbone=False,
            use_backbone_checkpointing=False,
            ipaid_params={
                "base_channels": 8,
                "num_scales": 2,
                "refine_iterations": 1,
                "use_sensitivity": True,
                "use_refinement": True,
                "use_feature_guided": True,
                "use_color_illumination": True,
                "color_illumination_mode": "global_white_balance",
                "enable_task_aware_rollback": True,
                "enable_coarse_task_grad": True,
                "_feature_fusion": {
                    "enabled": True,
                    "include_illum_stats": True,
                    "hidden_dim": 16,
                },
            },
        )
        return model

    def test_task_aware_rollback_gate_returns_one_alpha_per_stripe(self):
        from app.core.illumination_module_v2 import TaskAwareRollbackGate

        gate = TaskAwareRollbackGate(
            feat_channels=8,
            hidden_dim=4,
            min_alpha=0.1,
            max_alpha=0.9,
            granularity="stripe",
            num_stripes=4,
        )

        alpha = gate(
            original=torch.rand(2, 3, 16, 8),
            corrected=torch.rand(2, 3, 16, 8),
            illumination=torch.rand(2, 1, 16, 8),
            color_risk=torch.rand(2, 1, 16, 8),
            lambda_color=torch.rand(2, 1, 1, 1),
            correction_gap=torch.rand(2, 1, 16, 8),
            feat_mid=torch.rand(2, 8, 8, 4),
        )

        self.assertEqual(alpha.shape, (2, 4, 1, 1))
        self.assertGreaterEqual(float(alpha.min().item()), 0.1)
        self.assertLessEqual(float(alpha.max().item()), 0.9)

    def test_ipaid_forward_refine_supports_stripe_level_rollback(self):
        from app.core.illumination_module_v2 import IPAIDModule

        module = IPAIDModule(
            base_channels=8,
            num_scales=2,
            refine_iterations=1,
            use_sensitivity=False,
            use_refinement=False,
            use_feature_guided=False,
            use_color_illumination=False,
            enable_task_aware_rollback=True,
            rollback_granularity="stripe",
            rollback_num_stripes=2,
        )

        class FixedStripeGate(torch.nn.Module):
            def forward(self, **kwargs):
                original = kwargs["original"]
                return torch.tensor(
                    [[[[0.95]], [[0.05]]]],
                    dtype=original.dtype,
                    device=original.device,
                )

        module.rollback_gate = FixedStripeGate()

        x = torch.full((1, 3, 8, 4), 0.2, dtype=torch.float32)
        coarse_out = {
            "L_coarse": torch.ones(1, 1, 8, 4, dtype=x.dtype),
            "sensitivity": torch.ones(1, 1, 8, 4, dtype=x.dtype),
            "lambda_color": torch.full((1, 1, 1, 1), 0.25, dtype=x.dtype),
        }
        corrected = torch.full_like(x, 0.8)
        correction_gap = torch.full((1, 1, 8, 4), 0.1, dtype=x.dtype)

        with mock.patch.object(
            module,
            "apply_safe_illumination_correction",
            return_value=(corrected, {"correction_gap": correction_gap}),
        ):
            outputs = module.forward_refine(x, coarse_out, feat_mid=None)

        reflectance = outputs["reflectance"]
        top_mean = float(reflectance[:, :, :2, :].mean().item())
        bottom_mean = float(reflectance[:, :, -2:, :].mean().item())

        self.assertGreater(top_mean, 0.65)
        self.assertLess(bottom_mean, 0.35)
        self.assertGreater(top_mean - bottom_mean, 0.25)

    def test_model_aware_reflectance_residual_returns_gate_and_delta(self):
        from app.core.illumination_module_v2 import ModelAwareReflectanceResidual

        module = ModelAwareReflectanceResidual(
            feat_channels=8,
            hidden_dim=4,
            residual_scale=0.2,
        )

        reflectance_att, gate, delta = module(
            original=torch.rand(2, 3, 16, 8),
            reflectance=torch.rand(2, 3, 16, 8),
            illumination=torch.rand(2, 1, 16, 8),
            feat_mid=torch.rand(2, 8, 8, 4),
        )

        self.assertEqual(reflectance_att.shape, (2, 3, 16, 8))
        self.assertEqual(gate.shape, (2, 1, 16, 8))
        self.assertEqual(delta.shape, (2, 3, 16, 8))
        self.assertGreaterEqual(float(reflectance_att.min().item()), 0.0)
        self.assertLessEqual(float(reflectance_att.max().item()), 1.0)

    def test_stripe_aware_branch_attention_fusion_competes_across_branches(self):
        from app.core.joint_model import StripeAwareBranchAttentionFusion

        fusion = StripeAwareBranchAttentionFusion(
            channels=4,
            num_stripes=2,
            hidden_dim=8,
            num_branches=3,
            aux_dim=0,
            temperature=1.0,
        )

        raw = torch.ones(1, 4, 8, 4)
        base = torch.ones(1, 4, 8, 4) * 2.0
        adapted = torch.ones(1, 4, 8, 4) * 3.0

        fused, weights = fusion([raw, base, adapted], aux_stats=None)

        self.assertEqual(fused.shape, raw.shape)
        self.assertEqual(weights.shape, (1, 2, 3))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(1, 2), atol=1e-5))

    def test_task_adaptive_feature_fusion_caps_corrected_residual_scale(self):
        from app.core.joint_model import TaskAdaptiveFeatureFusion

        fusion = TaskAdaptiveFeatureFusion(
            channels=4,
            hidden_dim=8,
            init_corrected_bias=20.0,
            aux_dim=0,
            max_residual_scale=0.2,
        )

        raw = torch.zeros(1, 4, 4, 4)
        corrected = torch.ones(1, 4, 4, 4)
        fused = fusion(raw, corrected, aux_stats=None)

        self.assertGreaterEqual(float(fused.min().item()), 0.0)
        self.assertLessEqual(float(fused.max().item()), 0.21)

    def test_joint_model_can_switch_plain_baseline_to_global_gap_head(self):
        from app.core.joint_model import JointReIDModel

        model = JointReIDModel(
            num_classes=4,
            backbone_name="osnet_ain_x1_0",
            pretrained_backbone=False,
            use_ipaid=False,
            use_backbone_checkpointing=False,
            ipaid_params={
                "_reid_head": {
                    "type": "plain_global",
                },
            },
        )

        self.assertEqual(model.local_extractor.__class__.__name__, "PlainGlobalExtractor")

        with torch.no_grad():
            outputs = model(torch.rand(2, 3, 64, 64))

        self.assertEqual(outputs["features"].shape, (2, 512))
        self.assertEqual(outputs["logits"].shape, (2, 4))

    def test_ipaid_forward_refine_respects_identity_protection_projection(self):
        from app.core.illumination_module_v2 import IPAIDModule

        module = IPAIDModule(
            base_channels=8,
            num_scales=2,
            refine_iterations=1,
            use_sensitivity=False,
            use_refinement=False,
            use_feature_guided=False,
            use_color_illumination=False,
            enable_task_aware_rollback=False,
        )

        x = torch.full((1, 3, 8, 4), 0.2, dtype=torch.float32)
        coarse_out = {
            "L_coarse": torch.ones(1, 1, 8, 4, dtype=x.dtype),
            "sensitivity": torch.ones(1, 1, 8, 4, dtype=x.dtype),
            "lambda_color": torch.full((1, 1, 1, 1), 0.25, dtype=x.dtype),
        }
        corrected = torch.full_like(x, 0.8)
        correction_gap = torch.full((1, 1, 8, 4), 0.1, dtype=x.dtype)
        protect_mask = torch.ones(1, 1, 8, 4, dtype=x.dtype)

        with mock.patch.object(
            module,
            "apply_safe_illumination_correction",
            return_value=(corrected, {"correction_gap": correction_gap}),
        ):
            outputs = module.forward_refine(
                x,
                coarse_out,
                feat_mid=None,
                identity_protection_map=protect_mask,
            )

        self.assertTrue(torch.allclose(outputs["reflectance"], x, atol=1e-6))

    def test_cross_light_margin_preserving_loss_penalizes_margin_collapse(self):
        from app.core.illumination_module_v2 import CrossLightMarginPreservingLoss

        loss_fn = CrossLightMarginPreservingLoss(
            similarity="cosine",
            photometric_scale=8.0,
            photometric_offset=0.10,
            topk_positive=1,
            topk_negative=1,
            margin_delta=0.02,
            beta=12.0,
        )

        teacher_features = torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.3, 0.95],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        student_good = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.4],
                [0.1, 0.99],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        student_bad = torch.tensor(
            [
                [1.0, 0.0],
                [0.2, 0.98],
                [0.92, 0.38],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        photometric_stats = torch.tensor(
            [
                [0.1, 0.0, 0.9, 0.2],
                [0.8, 0.3, 0.2, 0.1],
                [0.2, 0.1, 0.7, 0.3],
                [0.7, 0.4, 0.1, 0.2],
            ],
            dtype=torch.float32,
        )

        loss_good = loss_fn(student_good, teacher_features, labels, photometric_stats)
        loss_bad = loss_fn(student_bad, teacher_features, labels, photometric_stats)

        self.assertGreater(float(loss_bad.item()), float(loss_good.item()))

    def test_softap_cross_light_loss_penalizes_listwise_retrieval_collapse(self):
        from app.core.illumination_module_v2 import SoftAPCrossLightLoss

        loss_fn = SoftAPCrossLightLoss(
            similarity="cosine",
            photometric_scale=8.0,
            photometric_offset=0.10,
            min_positive_weight=0.05,
            rank_temperature=0.07,
        )

        anchor_features_good = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        anchor_features_bad = torch.tensor(
            [
                [0.2, 0.98],
                [0.98, 0.2],
            ],
            dtype=torch.float32,
        )
        anchor_labels = torch.tensor([0, 1], dtype=torch.long)
        anchor_stats = torch.tensor(
            [
                [0.10, 0.15, 0.20, 0.05],
                [0.75, 0.25, 0.10, 0.20],
            ],
            dtype=torch.float32,
        )

        gallery_features = torch.tensor(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
                [0.05, 0.95],
                [-1.0, 0.0],
                [0.0, -1.0],
            ],
            dtype=torch.float32,
        )
        gallery_labels = torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.long)
        gallery_stats = torch.tensor(
            [
                [0.12, 0.12, 0.18, 0.08],
                [0.82, 0.22, 0.12, 0.15],
                [0.70, 0.20, 0.12, 0.18],
                [0.08, 0.18, 0.22, 0.07],
                [0.45, 0.50, 0.50, 0.50],
                [0.55, 0.60, 0.60, 0.60],
            ],
            dtype=torch.float32,
        )

        loss_good = loss_fn(
            anchor_features_good,
            anchor_labels,
            anchor_stats,
            gallery_features,
            gallery_labels,
            gallery_stats,
        )
        loss_bad = loss_fn(
            anchor_features_bad,
            anchor_labels,
            anchor_stats,
            gallery_features,
            gallery_labels,
            gallery_stats,
        )

        self.assertGreater(float(loss_bad.item()), float(loss_good.item()))

    def test_teacher_manifold_tube_loss_penalizes_off_manifold_drift(self):
        from app.core.illumination_module_v2 import TeacherManifoldTubeLoss

        loss_fn = TeacherManifoldTubeLoss(
            similarity="cosine",
            photometric_scale=8.0,
            photometric_offset=0.10,
            min_positive_weight=0.05,
            shrinkage=0.80,
            orthogonal_weight=1.0,
            subspace_rank=1,
            min_radius=0.02,
        )

        anchor_features_good = torch.tensor(
            [
                [0.98, 0.04],
                [0.04, 0.98],
            ],
            dtype=torch.float32,
        )
        anchor_features_bad = torch.tensor(
            [
                [0.70, 0.65],
                [0.65, 0.70],
            ],
            dtype=torch.float32,
        )
        anchor_labels = torch.tensor([0, 1], dtype=torch.long)
        anchor_stats = torch.tensor(
            [
                [0.12, 0.10, 0.22, 0.08],
                [0.78, 0.18, 0.10, 0.20],
            ],
            dtype=torch.float32,
        )
        teacher_features = torch.tensor(
            [
                [1.00, 0.00],
                [0.96, 0.04],
                [0.91, -0.03],
                [0.00, 1.00],
                [0.05, 0.96],
                [-0.04, 0.92],
                [-1.0, 0.0],
                [0.0, -1.0],
            ],
            dtype=torch.float32,
        )
        teacher_labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 3], dtype=torch.long)
        teacher_stats = torch.tensor(
            [
                [0.10, 0.08, 0.20, 0.06],
                [0.20, 0.12, 0.18, 0.08],
                [0.82, 0.20, 0.12, 0.16],
                [0.80, 0.20, 0.12, 0.18],
                [0.70, 0.22, 0.10, 0.16],
                [0.08, 0.12, 0.22, 0.06],
                [0.50, 0.50, 0.50, 0.50],
                [0.40, 0.45, 0.55, 0.52],
            ],
            dtype=torch.float32,
        )

        loss_good = loss_fn(
            anchor_features_good,
            anchor_labels,
            anchor_stats,
            teacher_features,
            teacher_labels,
            teacher_stats,
        )
        loss_bad = loss_fn(
            anchor_features_bad,
            anchor_labels,
            anchor_stats,
            teacher_features,
            teacher_labels,
            teacher_stats,
        )

        self.assertGreater(float(loss_bad.item()), float(loss_good.item()))

    def test_teacher_manifold_separation_loss_penalizes_negative_boundary_violation(self):
        from app.core.illumination_module_v2 import TeacherManifoldSeparationLoss

        loss_fn = TeacherManifoldSeparationLoss(
            similarity="cosine",
            photometric_scale=8.0,
            photometric_offset=0.10,
            min_positive_weight=0.05,
            margin=0.08,
        )

        anchor_features_good = torch.tensor(
            [
                [0.98, 0.04],
                [0.04, 0.98],
            ],
            dtype=torch.float32,
        )
        anchor_features_bad = torch.tensor(
            [
                [0.60, 0.80],
                [0.80, 0.60],
            ],
            dtype=torch.float32,
        )
        anchor_labels = torch.tensor([0, 1], dtype=torch.long)
        anchor_stats = torch.tensor(
            [
                [0.12, 0.10, 0.22, 0.08],
                [0.78, 0.18, 0.10, 0.20],
            ],
            dtype=torch.float32,
        )
        teacher_features = torch.tensor(
            [
                [1.00, 0.00],
                [0.96, 0.04],
                [0.91, -0.03],
                [0.00, 1.00],
                [0.05, 0.96],
                [-0.04, 0.92],
                [0.72, 0.68],
                [0.68, 0.72],
            ],
            dtype=torch.float32,
        )
        teacher_labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 3], dtype=torch.long)
        teacher_stats = torch.tensor(
            [
                [0.10, 0.08, 0.20, 0.06],
                [0.20, 0.12, 0.18, 0.08],
                [0.82, 0.20, 0.12, 0.16],
                [0.80, 0.20, 0.12, 0.18],
                [0.70, 0.22, 0.10, 0.16],
                [0.08, 0.12, 0.22, 0.06],
                [0.48, 0.45, 0.52, 0.50],
                [0.45, 0.48, 0.50, 0.52],
            ],
            dtype=torch.float32,
        )

        loss_good = loss_fn(
            anchor_features_good,
            anchor_labels,
            anchor_stats,
            teacher_features,
            teacher_labels,
            teacher_stats,
        )
        loss_bad = loss_fn(
            anchor_features_bad,
            anchor_labels,
            anchor_stats,
            teacher_features,
            teacher_labels,
            teacher_stats,
        )

        self.assertGreater(float(loss_bad.item()), float(loss_good.item()))

    def test_ranking_topology_loss_penalizes_teacher_margin_inversion(self):
        from app.core.illumination_module_v2 import RankingTopologyPreservingLoss

        loss_fn = RankingTopologyPreservingLoss(
            similarity="cosine",
            photometric_scale=8.0,
            photometric_offset=0.10,
            min_positive_weight=0.05,
            topk_positive=1,
            topk_negative=2,
            margin_slack=0.01,
            beta=12.0,
        )

        anchor_features_good = torch.tensor(
            [
                [0.99, 0.01, 0.00],
                [0.01, 0.99, 0.00],
            ],
            dtype=torch.float32,
        )
        anchor_features_bad = torch.tensor(
            [
                [0.15, 0.98, 0.00],
                [0.98, 0.15, 0.00],
            ],
            dtype=torch.float32,
        )
        anchor_labels = torch.tensor([0, 1], dtype=torch.long)
        anchor_stats = torch.tensor(
            [
                [0.10, 0.12, 0.22, 0.06],
                [0.78, 0.20, 0.10, 0.18],
            ],
            dtype=torch.float32,
        )
        teacher_features = torch.tensor(
            [
                [1.00, 0.00, 0.00],
                [0.94, 0.06, 0.00],
                [0.00, 1.00, 0.00],
                [0.05, 0.95, 0.00],
                [-1.00, 0.00, 0.00],
                [0.00, -1.00, 0.00],
            ],
            dtype=torch.float32,
        )
        teacher_labels = torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.long)
        teacher_stats = torch.tensor(
            [
                [0.10, 0.08, 0.20, 0.06],
                [0.76, 0.22, 0.10, 0.16],
                [0.80, 0.22, 0.10, 0.18],
                [0.08, 0.12, 0.24, 0.06],
                [0.50, 0.50, 0.50, 0.50],
                [0.40, 0.45, 0.55, 0.52],
            ],
            dtype=torch.float32,
        )

        loss_good = loss_fn(
            anchor_features_good,
            anchor_labels,
            anchor_stats,
            teacher_features,
            teacher_labels,
            teacher_stats,
        )
        loss_bad = loss_fn(
            anchor_features_bad,
            anchor_labels,
            anchor_stats,
            teacher_features,
            teacher_labels,
            teacher_stats,
        )

        self.assertGreater(float(loss_bad.item()), float(loss_good.item()))

    def test_anisotropic_identity_protection_penalizes_identity_sensitive_motion_more_than_nuisance_motion(self):
        from app.core.illumination_module_v2 import AnisotropicIdentityProtectionLoss

        loss_fn = AnisotropicIdentityProtectionLoss(
            similarity="cosine",
            photometric_scale=8.0,
            photometric_offset=0.10,
            min_positive_weight=0.05,
            topk_positive=2,
            topk_negative=1,
            subspace_rank=1,
            identity_weight=1.0,
            nuisance_weight=0.5,
            nuisance_radius=0.12,
        )

        teacher_anchor = torch.tensor(
            [
                [1.00, 0.00, 0.00],
            ],
            dtype=torch.float32,
        )
        anchor_bad_identity = torch.tensor(
            [
                [0.70, 0.00, 0.00],
            ],
            dtype=torch.float32,
        )
        anchor_nuisance = torch.tensor(
            [
                [1.00, 0.00, 0.26],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0], dtype=torch.long)
        anchor_stats = torch.tensor([[0.10, 0.12, 0.22, 0.06]], dtype=torch.float32)
        teacher_features = torch.tensor(
            [
                [1.00, 0.00, 0.00],
                [0.98, 0.02, 0.00],
                [0.96, -0.04, 0.00],
                [-1.00, 0.00, 0.00],
                [-0.98, 0.02, 0.00],
            ],
            dtype=torch.float32,
        )
        teacher_labels = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        teacher_stats = torch.tensor(
            [
                [0.10, 0.08, 0.20, 0.06],
                [0.20, 0.12, 0.18, 0.08],
                [0.82, 0.20, 0.12, 0.16],
                [0.50, 0.50, 0.50, 0.50],
                [0.45, 0.48, 0.52, 0.51],
            ],
            dtype=torch.float32,
        )

        loss_identity = loss_fn(
            anchor_bad_identity,
            teacher_anchor,
            labels,
            anchor_stats,
            teacher_features,
            teacher_labels,
            teacher_stats,
            same_source_size=1,
        )
        loss_nuisance = loss_fn(
            anchor_nuisance,
            teacher_anchor,
            labels,
            anchor_stats,
            teacher_features,
            teacher_labels,
            teacher_stats,
            same_source_size=1,
        )

        self.assertGreater(float(loss_identity.item()), float(loss_nuisance.item()))

    def test_semantic_non_confusion_loss_penalizes_true_class_margin_drop(self):
        from app.core.illumination_module_v2 import SemanticNonConfusionLoss

        loss_fn = SemanticNonConfusionLoss(margin_delta=0.02)
        labels = torch.tensor([0, 1], dtype=torch.long)
        teacher_logits = torch.tensor(
            [
                [4.2, 1.1, -0.5],
                [0.3, 3.6, 0.2],
            ],
            dtype=torch.float32,
        )
        student_good = torch.tensor(
            [
                [4.0, 1.3, -0.4],
                [0.2, 3.4, 0.1],
            ],
            dtype=torch.float32,
        )
        student_bad = torch.tensor(
            [
                [1.4, 2.8, 0.2],
                [2.5, 1.8, 0.4],
            ],
            dtype=torch.float32,
        )

        loss_good = loss_fn(student_good, teacher_logits, labels)
        loss_bad = loss_fn(student_bad, teacher_logits, labels)

        self.assertLess(float(loss_good.item()), float(loss_bad.item()))

    def test_cross_covariance_decorrelation_loss_penalizes_correlated_embeddings(self):
        from app.core.illumination_module_v2 import CrossCovarianceDecorrelationLoss

        loss_fn = CrossCovarianceDecorrelationLoss()
        identity = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        nuisance_good = torch.tensor(
            [
                [0.2, -0.3],
                [-0.3, 0.2],
                [0.1, 0.4],
                [0.4, 0.1],
            ],
            dtype=torch.float32,
        )
        nuisance_bad = identity.clone()

        loss_good = loss_fn(identity, nuisance_good)
        loss_bad = loss_fn(identity, nuisance_bad)

        self.assertGreater(float(loss_bad.item()), float(loss_good.item()))

    def test_joint_model_exposes_nuisance_outputs_when_enabled(self):
        from app.core.joint_model import JointReIDModel

        model = JointReIDModel(
            num_classes=4,
            backbone_name="osnet_ain_x1_0",
            pretrained_backbone=False,
            use_backbone_checkpointing=False,
            ipaid_params={
                "base_channels": 8,
                "num_scales": 2,
                "refine_iterations": 1,
                "use_sensitivity": True,
                "use_refinement": True,
                "use_feature_guided": True,
                "use_color_illumination": True,
                "color_illumination_mode": "global_white_balance",
                "_nuisance_head": {
                    "enabled": True,
                    "hidden_dim": 64,
                    "nuisance_dim": 32,
                    "photometric_dim": 4,
                },
            },
        )
        model.eval()

        outputs = model(torch.rand(2, 3, 128, 128), return_illuminated=True)

        self.assertIn("nuisance_features", outputs)
        self.assertIn("photometric_prediction", outputs)
        self.assertEqual(outputs["nuisance_features"].shape, (2, 32))
        self.assertEqual(outputs["photometric_prediction"].shape, (2, 4))

    def test_joint_trainer_cross_light_stats_use_global_summary_only(self):
        from tools.train_joint import JointTrainer

        trainer = JointTrainer.__new__(JointTrainer)

        illumination = torch.tensor(
            [
                [
                    [
                        [0.10, 0.90],
                        [0.10, 0.90],
                        [0.40, 0.60],
                        [0.40, 0.60],
                        [0.20, 0.30],
                        [0.20, 0.30],
                    ]
                ]
            ],
            dtype=torch.float32,
        )
        correction_gap = torch.tensor(
            [
                [
                    [
                        [0.05, 0.45],
                        [0.05, 0.45],
                        [0.10, 0.20],
                        [0.10, 0.20],
                        [0.30, 0.35],
                        [0.30, 0.35],
                    ]
                ]
            ],
            dtype=torch.float32,
        )
        protected_map = torch.tensor(
            [
                [
                    [
                        [0.0, 1.0],
                        [0.0, 1.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ]
                ]
            ],
            dtype=torch.float32,
        )
        neutral_map = torch.zeros_like(protected_map)
        base_details = {
            "effective_illumination": illumination,
            "correction_gap": correction_gap,
            "rollback_alpha": torch.full((1, 1, 1, 1), 0.8, dtype=torch.float32),
            "lambda_color": torch.full((1, 1, 1, 1), 0.2, dtype=torch.float32),
        }

        protected_stats = trainer._build_cross_light_stats(
            dict(base_details, identity_protection_map=protected_map)
        )
        neutral_stats = trainer._build_cross_light_stats(
            dict(base_details, identity_protection_map=neutral_map)
        )

        expected = torch.tensor(
            [
                [
                    float(illumination.mean().item()),
                    float(correction_gap.mean().item()),
                    0.8,
                    0.2,
                ]
            ],
            dtype=torch.float32,
        )

        self.assertEqual(protected_stats.shape, (1, 4))
        self.assertTrue(torch.allclose(neutral_stats, expected, atol=1e-6))
        self.assertTrue(torch.allclose(protected_stats, neutral_stats, atol=1e-6))

    def test_joint_model_consistency_forward_exposes_nuisance_outputs_when_enabled(self):
        from app.core.joint_model import JointReIDModel

        model = JointReIDModel(
            num_classes=4,
            backbone_name="osnet_ain_x1_0",
            pretrained_backbone=False,
            use_backbone_checkpointing=False,
            ipaid_params={
                "base_channels": 8,
                "num_scales": 2,
                "refine_iterations": 1,
                "use_sensitivity": True,
                "use_refinement": True,
                "use_feature_guided": True,
                "use_color_illumination": True,
                "color_illumination_mode": "global_white_balance",
                "_nuisance_head": {
                    "enabled": True,
                    "hidden_dim": 64,
                    "nuisance_dim": 32,
                    "photometric_dim": 4,
                },
            },
        )
        model.eval()

        outputs = model.forward_with_consistency_variants(torch.rand(2, 3, 128, 128), num_variants=2)

        self.assertIn("nuisance_features", outputs)
        self.assertIn("photometric_prediction", outputs)
        self.assertEqual(outputs["nuisance_features"].shape, (2, 32))
        self.assertEqual(outputs["photometric_prediction"].shape, (2, 4))

    def test_feature_trust_region_loss_is_zero_inside_radius(self):
        from app.core.illumination_module_v2 import FeatureTrustRegionLoss

        loss_fn = FeatureTrustRegionLoss(base_radius=0.25, adaptive_scale=0.0, class_spread_scale=0.0)
        student = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        teacher = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32)

        loss = loss_fn(student, teacher)
        self.assertLess(float(loss.item()), 1e-6)

    def test_feature_trust_region_loss_expands_radius_for_large_class_spread(self):
        from app.core.illumination_module_v2 import FeatureTrustRegionLoss

        student = torch.tensor(
            [
                [1.0, 0.0],
                [0.7, 0.7],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        teacher = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 0, 0], dtype=torch.long)

        plain_loss = FeatureTrustRegionLoss(
            base_radius=0.05,
            adaptive_scale=0.0,
            class_spread_scale=0.0,
        )(student, teacher, labels=labels)
        spread_aware_loss = FeatureTrustRegionLoss(
            base_radius=0.05,
            adaptive_scale=0.0,
            class_spread_scale=1.0,
        )(student, teacher, labels=labels)

        self.assertGreater(float(plain_loss.item()), float(spread_aware_loss.item()))

    def test_local_rank_preserving_loss_penalizes_teacher_margin_inversion(self):
        from app.core.illumination_module_v2 import LocalRankPreservingLoss

        loss_fn = LocalRankPreservingLoss(alpha=0.9, k_positive=1, k_negative=1)
        teacher = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=torch.float32,
        )
        student_good = teacher.clone()
        student_bad = torch.tensor(
            [
                [1.0, 0.0],
                [0.2, 0.8],
                [0.0, 1.0],
                [0.8, 0.2],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        loss_good = loss_fn(student_good, teacher, labels)
        loss_bad = loss_fn(student_bad, teacher, labels)

        self.assertLess(float(loss_good.item()), 1e-6)
        self.assertGreater(float(loss_bad.item()), 1e-4)

    def test_neighborhood_consistency_loss_penalizes_bad_teacher_neighbors(self):
        from app.core.illumination_module_v2 import NeighborhoodConsistencyLoss

        loss_fn = NeighborhoodConsistencyLoss(
            temperature=0.07,
            topk=2,
            positive_weight=1.0,
            negative_weight=1.0,
        )
        teacher = torch.tensor(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
                [0.05, 0.95],
            ],
            dtype=torch.float32,
        )
        student_good = teacher.clone()
        student_bad = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        loss_good = loss_fn(student_good, teacher, labels)
        loss_bad = loss_fn(student_bad, teacher, labels)

        self.assertLess(float(loss_good.item()), 0.02)
        self.assertGreater(float(loss_bad.item()), float(loss_good.item()) + 0.1)

    def test_neighborhood_consistency_loss_ablation_switches_change_terms(self):
        from app.core.illumination_module_v2 import NeighborhoodConsistencyLoss

        teacher = torch.tensor(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
                [0.05, 0.95],
            ],
            dtype=torch.float32,
        )
        student = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        teacher_local = teacher.view(4, 1, 2).repeat(1, 2, 1)
        student_local = student.view(4, 1, 2).repeat(1, 2, 1)
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        global_only = NeighborhoodConsistencyLoss(
            temperature=0.07,
            topk=2,
            use_global=True,
            use_local=False,
        )(student, teacher, labels, student_local_features=student_local, teacher_local_features=teacher_local)
        local_only = NeighborhoodConsistencyLoss(
            temperature=0.07,
            topk=2,
            use_global=False,
            use_local=True,
            local_weight=1.0,
        )(student, teacher, labels, student_local_features=student_local, teacher_local_features=teacher_local)
        no_hard_negative = NeighborhoodConsistencyLoss(
            temperature=0.07,
            topk=2,
            negative_weight=0.0,
            use_hard_negatives=False,
        )(student, teacher, labels)
        full = NeighborhoodConsistencyLoss(
            temperature=0.07,
            topk=2,
            negative_weight=1.0,
            use_hard_negatives=True,
        )(student, teacher, labels)

        self.assertGreater(float(global_only.item()), 0.0)
        self.assertGreater(float(local_only.item()), 0.0)
        self.assertLess(float(no_hard_negative.item()), float(full.item()))

    def test_neighborhood_consistency_loss_exposes_reciprocal_jaccard_targets(self):
        from app.core.illumination_module_v2 import NeighborhoodConsistencyLoss

        loss_fn = NeighborhoodConsistencyLoss(
            temperature=0.2,
            topk=2,
            teacher_target="reciprocal",
            use_local=False,
        )
        teacher = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.96, 0.04, 0.0],
                [0.92, 0.08, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.96, 0.04],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)

        target = loss_fn._teacher_graph_targets(teacher, labels=labels, exclude_self=True)

        self.assertEqual(tuple(target.shape), (5, 5))
        self.assertAlmostEqual(float(target[0].sum().item()), 1.0, places=5)
        self.assertGreater(float(target[0, 1].item()), 0.0)
        self.assertGreater(
            float(target[0, 2].item()),
            float(target[0, 3].item()),
            "Reciprocal graph target should keep unlabeled manifold neighbors when their neighborhoods overlap.",
        )
        self.assertEqual(float(target[0, 0].item()), 0.0)

    def test_neighborhood_consistency_loss_prefers_matching_reciprocal_graph(self):
        from app.core.illumination_module_v2 import NeighborhoodConsistencyLoss

        loss_fn = NeighborhoodConsistencyLoss(
            temperature=0.2,
            topk=2,
            negative_weight=0.5,
            teacher_target="reciprocal",
            use_local=False,
        )
        teacher = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.96, 0.04, 0.0],
                [0.92, 0.08, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.96, 0.04],
            ],
            dtype=torch.float32,
        )
        student_good = teacher.clone().requires_grad_(True)
        student_bad = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.96, 0.04],
                [0.96, 0.04, 0.0],
                [0.92, 0.08, 0.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        labels = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)

        good = loss_fn(student_good, teacher, labels)
        bad = loss_fn(student_bad, teacher, labels)

        self.assertLess(float(good.item()), float(bad.item()))
        self.assertGreater(float(bad.item()), float(good.item()) + 0.05)
        bad.backward()
        self.assertIsNotNone(student_bad.grad)
        self.assertGreater(float(student_bad.grad.abs().sum().item()), 0.0)

    def test_phase2_keeps_coarse_and_fusion_modules_trainable_via_geometry_target(self):
        model = self._build_model()
        model.train()
        model.freeze_backbone(True)
        model.freeze_illumination(False)
        model.freeze_local_extractor(True)
        model.freeze_feature_fusion(False)

        x = torch.rand(2, 3, 128, 128)
        out = model(x, return_illuminated=True)
        loss = out["features"].mean() + out["ipaid_details"]["reflectance"].mean()
        loss.backward()

        coarse_modules = {
            "illumination_estimator": model.illumination.illumination_estimator,
            "sensitivity_estimator": model.illumination.sensitivity_estimator,
            "color_tolerance_net": model.illumination.color_tolerance_net,
            "feature_fusion": model.feature_fusion,
        }

        for name, module in coarse_modules.items():
            grad_sum = 0.0
            grad_count = 0
            for param in module.parameters():
                if param.grad is not None:
                    grad_sum += float(param.grad.abs().sum().item())
                    grad_count += 1
            self.assertGreater(
                grad_count,
                0,
                msg=f"{name} should receive gradients when phase2 fixes f_theta but trains illumination",
            )
            self.assertGreater(
                grad_sum,
                0.0,
                msg=f"{name} gradient magnitude should stay non-zero in phase2",
            )

    def test_forward_illumination_only_skips_reid_outputs(self):
        model = self._build_model()
        model.train()
        model.freeze_backbone(True)
        model.freeze_illumination(False)
        model.freeze_local_extractor(True)

        x = torch.rand(2, 3, 128, 128)
        out = model.forward_illumination_only(x, return_illuminated=True)

        self.assertIsNone(out["features"])
        self.assertIsNone(out["logits"])
        self.assertIn("ipaid_details", out)
        self.assertEqual(out["illuminated"].shape, x.shape)

        loss = out["ipaid_details"]["reflectance"].mean()
        loss.backward()

        grad_sum = 0.0
        grad_count = 0
        for param in model.illumination.parameters():
            if param.grad is not None:
                grad_sum += float(param.grad.abs().sum().item())
                grad_count += 1

        self.assertGreater(grad_count, 0)
        self.assertGreater(grad_sum, 0.0)

    def test_teacher_anchor_loss_is_zero_for_identical_features(self):
        from app.core.illumination_module_v2 import TeacherAnchorLoss

        loss_fn = TeacherAnchorLoss()
        features = torch.eye(4, dtype=torch.float32)

        loss_same = loss_fn(features, features.clone())
        loss_shifted = loss_fn(features, features[[1, 0, 2, 3]])

        self.assertLess(float(loss_same.item()), 1e-6)
        self.assertGreater(float(loss_shifted.item()), 1e-3)

    def test_geometry_preserving_loss_matches_pairwise_structure(self):
        from app.core.illumination_module_v2 import GeometryPreservingLoss

        loss_fn = GeometryPreservingLoss()
        reference = torch.eye(4, dtype=torch.float32)

        loss_same = loss_fn(reference, reference.clone())
        loss_shifted = loss_fn(reference[[1, 0, 2, 3]], reference)

        self.assertLess(float(loss_same.item()), 1e-6)
        self.assertGreater(float(loss_shifted.item()), 1e-3)

    def test_teacher_logit_consistency_loss_is_zero_for_identical_logits(self):
        from app.core.illumination_module_v2 import TeacherLogitConsistencyLoss

        loss_fn = TeacherLogitConsistencyLoss(temperature=2.0)
        logits = torch.tensor(
            [
                [3.0, 1.0, -2.0],
                [1.5, 2.5, -0.5],
                [-1.0, 0.5, 2.0],
            ],
            dtype=torch.float32,
        )

        loss_same = loss_fn(logits, logits.clone())
        loss_shifted = loss_fn(logits.roll(shifts=1, dims=1), logits)

        self.assertLess(float(loss_same.item()), 1e-6)
        self.assertGreater(float(loss_shifted.item()), 1e-4)

    def test_cross_light_prototype_loss_is_zero_for_identical_same_id_features(self):
        from app.core.illumination_module_v2 import CrossLightPrototypeLoss

        loss_fn = CrossLightPrototypeLoss(
            similarity="cosine",
            photometric_scale=10.0,
            photometric_offset=0.1,
            min_gap_weight=0.1,
        )
        features = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        photometric = torch.tensor([0.1, 0.8, 0.2, 0.9], dtype=torch.float32)

        loss = loss_fn(features, labels, photometric)

        self.assertLess(float(loss.item()), 1e-6)

    def test_cross_light_prototype_loss_penalizes_misaligned_same_id_features(self):
        from app.core.illumination_module_v2 import CrossLightPrototypeLoss

        loss_fn = CrossLightPrototypeLoss(
            similarity="cosine",
            photometric_scale=12.0,
            photometric_offset=0.05,
            min_gap_weight=0.1,
        )
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0], dtype=torch.long)
        photometric = torch.tensor([0.0, 1.0], dtype=torch.float32)

        loss = loss_fn(features, labels, photometric)

        self.assertGreater(float(loss.item()), 0.5)

    def test_relative_class_structure_loss_is_translation_invariant_inside_each_identity(self):
        from app.core.illumination_module_v2 import RelativeClassStructureLoss

        loss_fn = RelativeClassStructureLoss(metric="mse", radial_weight=1.0)
        teacher = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 1.0],
                [1.0, 2.0],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        student = torch.tensor(
            [
                [2.0, 1.0],
                [1.0, 2.0],
                [4.0, 2.0],
                [3.0, 3.0],
            ],
            dtype=torch.float32,
        )

        loss = loss_fn(student, teacher, labels)

        self.assertLess(float(loss.item()), 1e-6)

    def test_relative_class_structure_loss_penalizes_intra_class_shape_collapse(self):
        from app.core.illumination_module_v2 import RelativeClassStructureLoss

        loss_fn = RelativeClassStructureLoss(metric="mse", radial_weight=1.0)
        teacher = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 1.0],
                [1.0, 2.0],
            ],
            dtype=torch.float32,
        )
        student = torch.tensor(
            [
                [1.5, 0.5],
                [1.5, 0.5],
                [2.5, 1.5],
                [2.5, 1.5],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        loss = loss_fn(student, teacher, labels)

        self.assertGreater(float(loss.item()), 1e-3)

    def test_nonnegative_gradient_alignment_suppresses_conflicting_auxiliary_gradients(self):
        from tools.train_joint import _compute_nonnegative_gradient_alignment

        anchor = torch.tensor([[1.0, -2.0]], dtype=torch.float32, requires_grad=True)
        loss_reid = (anchor ** 2).sum()
        loss_aux_same = 3.0 * (anchor ** 2).sum()
        loss_aux_conflict = -1.0 * (anchor ** 2).sum()

        aligned = _compute_nonnegative_gradient_alignment(loss_reid, loss_aux_same, anchor)
        conflicting = _compute_nonnegative_gradient_alignment(loss_reid, loss_aux_conflict, anchor)

        self.assertGreater(float(aligned.item()), 0.99)
        self.assertLess(float(conflicting.item()), 1e-6)

    def test_linear_warmup_value_interpolates_from_start_to_end(self):
        from tools.train_joint import _linear_warmup_value

        self.assertAlmostEqual(_linear_warmup_value(0.05, 0.35, epoch=0, warmup_epochs=6), 0.05, places=6)
        self.assertAlmostEqual(_linear_warmup_value(0.05, 0.35, epoch=5, warmup_epochs=6), 0.35, places=6)
        self.assertAlmostEqual(_linear_warmup_value(0.05, 0.35, epoch=2, warmup_epochs=6), 0.17, places=6)
        self.assertAlmostEqual(_linear_warmup_value(0.05, 0.35, epoch=12, warmup_epochs=6), 0.35, places=6)

    def test_joint_trainer_phase3_aux_weight_uses_ramp_when_enabled(self):
        from tools.train_joint import JointTrainer

        trainer = JointTrainer.__new__(JointTrainer)
        trainer.phase3_aux_ramp_enabled = True
        trainer.phase3_aux_ramp_epochs = 6
        trainer.phase3_aux_ramp = {
            "illumination": (0.05, 0.35),
            "iicl": (0.0, 0.10),
            "cross_light": (0.0, 0.12),
        }

        self.assertAlmostEqual(trainer._get_phase3_aux_weight("illumination", phase=3, epoch=0), 0.05, places=6)
        self.assertAlmostEqual(trainer._get_phase3_aux_weight("illumination", phase=3, epoch=5), 0.35, places=6)
        self.assertAlmostEqual(trainer._get_phase3_aux_weight("iicl", phase=3, epoch=0), 0.0, places=6)
        self.assertAlmostEqual(trainer._get_phase3_aux_weight("cross_light", phase=1, epoch=0), 0.12, places=6)

    def test_frozen_modules_keep_batchnorm_statistics_fixed_during_train_mode(self):
        model = self._build_model()
        model.freeze_backbone(True)
        model.freeze_local_extractor(True)
        model.freeze_feature_fusion(True)
        model.train()

        backbone_bn = next(m for m in model.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d))
        extractor_bn = next(m for m in model.local_extractor.modules() if isinstance(m, torch.nn.BatchNorm2d))

        self.assertFalse(backbone_bn.training)
        self.assertFalse(extractor_bn.training)

        backbone_before = backbone_bn.running_mean.clone()
        extractor_before = extractor_bn.running_mean.clone()

        with torch.no_grad():
            model(torch.rand(4, 3, 128, 128))

        self.assertTrue(torch.equal(backbone_bn.running_mean, backbone_before))
        self.assertTrue(torch.equal(extractor_bn.running_mean, extractor_before))

    def test_backbone_input_preparation_clamps_to_physical_rgb_range(self):
        model = self._build_model()
        bad_rgb = torch.tensor(
            [[[[-1.5, 1.5], [3.0, -2.0]], [[0.2, -0.1], [1.2, 0.5]], [[2.4, 0.4], [-3.0, 1.8]]]],
            dtype=torch.float32,
        )

        prepared = model._prepare_backbone_input(bad_rgb)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=prepared.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=prepared.dtype).view(1, 3, 1, 1)
        recovered = prepared * std + mean

        self.assertGreaterEqual(float(recovered.min().item()), 0.0)
        self.assertLessEqual(float(recovered.max().item()), 1.0)

    def test_backbone_random_erasing_runs_in_normalized_space(self):
        from app.core.joint_model import JointReIDModel

        model = JointReIDModel(
            num_classes=4,
            backbone_name="osnet_ain_x1_0",
            pretrained_backbone=False,
            use_backbone_checkpointing=False,
            ipaid_params={
                "_backbone_random_erasing": {
                    "enabled": True,
                    "probability": 1.0,
                    "scale": [1.0, 1.0],
                    "ratio": [1.0, 1.0],
                    "value": "random",
                },
            },
        )
        model.train()

        images = torch.full((1, 3, 16, 16), 0.5, dtype=torch.float32)
        prepared = model._prepare_backbone_input(images)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=prepared.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=prepared.dtype).view(1, 3, 1, 1)
        baseline = (images - mean) / std

        self.assertFalse(torch.allclose(prepared, baseline))
        self.assertTrue(torch.allclose(images, torch.full_like(images, 0.5)))

    def test_phase3_checkpointing_override_enables_only_joint_stage(self):
        from tools.train_joint import JointTrainer

        trainer = JointTrainer.__new__(JointTrainer)
        trainer.config = {
            "hardware": {"use_backbone_checkpointing": False},
            "training": {
                "phases": {
                    "phase3": {
                        "use_backbone_checkpointing": True,
                    }
                },
            }
        }
        trainer.model = type("TinyModel", (), {"use_backbone_checkpointing": False})()

        trainer._set_phase_backbone_checkpointing(phase=1)
        self.assertFalse(trainer.model.use_backbone_checkpointing)

        trainer._set_phase_backbone_checkpointing(phase=3)
        self.assertTrue(trainer.model.use_backbone_checkpointing)

    def test_backbone_checkpointing_runs_even_when_input_tensor_has_no_grad_flag(self):
        from app.core.joint_model import JointReIDModel

        model = JointReIDModel(
            num_classes=4,
            backbone_name="osnet_ain_x1_0",
            pretrained_backbone=False,
            use_backbone_checkpointing=True,
            ipaid_params={
                "base_channels": 8,
                "num_scales": 2,
                "refine_iterations": 1,
                "use_sensitivity": True,
                "use_refinement": True,
                "use_feature_guided": True,
                "use_color_illumination": True,
                "color_illumination_mode": "global_white_balance",
            },
        )
        model.train()

        with mock.patch(
            "app.core.joint_model.checkpoint",
            side_effect=lambda fn, x, use_reentrant=False: fn(x),
        ) as checkpoint_mock:
            _ = model.extract_backbone_features(torch.rand(1, 3, 128, 128))

        self.assertTrue(checkpoint_mock.called)

    def test_raw_feature_fusion_path_keeps_backbone_gradients(self):
        model = self._build_model()
        model.train()
        model.freeze_backbone(False)
        model.freeze_illumination(False)
        model.freeze_local_extractor(False)
        model.freeze_feature_fusion(False)

        raw_feature_map = model._extract_raw_feature_map_for_fusion(torch.rand(2, 3, 128, 128))

        self.assertIsNotNone(raw_feature_map)
        self.assertTrue(raw_feature_map.requires_grad)

        raw_feature_map.mean().backward()
        backbone_grad = sum(
            float(param.grad.abs().sum().item())
            for param in model.backbone.parameters()
            if param.grad is not None
        )
        self.assertGreater(backbone_grad, 0.0)

    def test_reid_dataset_raises_on_unreadable_images(self):
        from app.core.evaluation import ReIDDataset

        dataset = ReIDDataset(samples=[("does_not_exist.jpg", 0)], transform=None)

        with self.assertRaises(FileNotFoundError):
            dataset[0]

    def test_reid_evaluator_restores_model_training_state(self):
        from app.core.evaluation import ReIDEvaluator

        class TinyModel(torch.nn.Module):
            def forward(self, imgs, boxes_list=None, return_illuminated=False):
                return {"features": imgs.mean(dim=(2, 3))}

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            query_img = tmp / "query" / "0" / "q1.jpg"
            gallery_img = tmp / "gallery" / "0" / "g1.jpg"
            query_img.parent.mkdir(parents=True, exist_ok=True)
            gallery_img.parent.mkdir(parents=True, exist_ok=True)
            self._write_image(query_img)
            self._write_image(gallery_img)

            model = TinyModel()
            evaluator = ReIDEvaluator(
                model=model,
                device=torch.device("cpu"),
                img_height=16,
                img_width=16,
                batch_size=1,
                num_workers=0,
                exclude_same_camera=False,
            )

            model.train()
            evaluator.evaluate(str(query_img.parent.parent), str(gallery_img.parent.parent))
            self.assertTrue(model.training)

            model.eval()
            evaluator.evaluate(str(query_img.parent.parent), str(gallery_img.parent.parent))
            self.assertFalse(model.training)

    def test_ipanda50_evaluator_restores_model_training_state(self):
        from app.core.evaluation import ReIDEvaluator

        class TinyModel(torch.nn.Module):
            def forward(self, imgs, boxes_list=None, return_illuminated=False):
                return {"features": imgs.mean(dim=(2, 3))}

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            img1 = tmp / "test" / "0" / "a.jpg"
            img2 = tmp / "test" / "0" / "b.jpg"
            img1.parent.mkdir(parents=True, exist_ok=True)
            self._write_image(img1)
            self._write_image(img2)

            model = TinyModel()
            evaluator = ReIDEvaluator(
                model=model,
                device=torch.device("cpu"),
                img_height=16,
                img_width=16,
                batch_size=1,
                num_workers=0,
                exclude_same_camera=False,
            )

            model.train()
            evaluator.evaluate_ipanda50(str(img1.parent.parent))
            self.assertTrue(model.training)

            model.eval()
            evaluator.evaluate_ipanda50(str(img1.parent.parent))
            self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
