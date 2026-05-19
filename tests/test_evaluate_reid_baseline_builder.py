from pathlib import Path
import sys
import types

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")

from tools.evaluate_reid import _build_model


def test_build_model_baseline_uses_joint_model_when_legacy_baseline_class_is_missing(monkeypatch) -> None:
    captured = {}

    class DummyModel(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.init_kwargs = kwargs
            self.weight = torch.nn.Parameter(torch.zeros(1))

    joint_module = types.ModuleType("app.core.joint_model")
    joint_module.JointReIDModel = DummyModel

    factory_module = types.ModuleType("app.core.model_factory")

    def extract_config_from_checkpoint(checkpoint):
        return checkpoint["config"]

    def resolve_joint_model_init(config, num_classes, backbone_override=None, pretrained_backbone=False):
        captured["config"] = config
        captured["num_classes"] = num_classes
        captured["backbone_override"] = backbone_override
        captured["pretrained_backbone"] = pretrained_backbone
        return {
            "num_classes": num_classes,
            "backbone_name": backbone_override or config["model"]["backbone"],
            "num_stripes": config["model"]["local_extractor"]["num_parts"],
            "pretrained_backbone": pretrained_backbone,
        }

    factory_module.extract_config_from_checkpoint = extract_config_from_checkpoint
    factory_module.resolve_joint_model_init = resolve_joint_model_init

    # Simulate the current codebase state: train_baselines no longer exposes
    # the legacy BaselineReIDModel class, but evaluation must still succeed.
    train_baselines_module = types.ModuleType("tools.train_baselines")

    monkeypatch.setitem(sys.modules, "app.core.joint_model", joint_module)
    monkeypatch.setitem(sys.modules, "app.core.model_factory", factory_module)
    monkeypatch.setitem(sys.modules, "tools.train_baselines", train_baselines_module)

    checkpoint = {
        "num_classes": 7,
        "config": {
            "model": {
                "backbone": "osnet_ain_x1_0",
                "local_extractor": {"num_parts": 6},
            }
        },
        "model_state_dict": {"weight": torch.tensor([3.5])},
    }

    model = _build_model(checkpoint, baseline=True, device=torch.device("cpu"))

    assert isinstance(model, DummyModel)
    assert model.init_kwargs["num_classes"] == 7
    assert model.init_kwargs["backbone_name"] == "osnet_ain_x1_0"
    assert model.init_kwargs["num_stripes"] == 6
    assert model.init_kwargs["pretrained_backbone"] is False
    assert float(model.weight.detach().item()) == 3.5
    assert captured["num_classes"] == 7
    assert captured["backbone_override"] == "osnet_ain_x1_0"
    assert captured["pretrained_backbone"] is False
