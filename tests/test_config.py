from __future__ import annotations

from fracture.utils.config import load_config


def test_base_config_loads_with_group_defaults():
    cfg = load_config()
    assert cfg.model.name == "densenet169"
    assert cfg.data.name == "mura"
    assert cfg.train.loss in {"ce", "focal"}


def test_group_override_switches_model_file():
    cfg = load_config(["model=convnext_tiny"])
    assert cfg.model.name == "convnext_tiny"
    assert cfg.model.drop_path_rate == 0.1


def test_dotlist_override_wins():
    cfg = load_config(["train.lr=1e-5", "data.image_size=384"])
    assert float(cfg.train.lr) == 1e-5
    assert cfg.data.image_size == 384


def test_experiment_layer_applies():
    cfg = load_config(experiment="fast_dev")
    assert cfg.train.max_epochs == 1
    assert cfg.model.pretrained is False


def test_experiment_then_dotlist_precedence():
    cfg = load_config(["train.max_epochs=7"], experiment="fast_dev")
    assert cfg.train.max_epochs == 7


def test_strong_experiment_enables_ema_and_mixup():
    cfg = load_config(experiment="strong")
    assert cfg.train.ema.enabled is True
    assert cfg.train.mixup.enabled is True
