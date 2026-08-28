from __future__ import annotations

import torch

from fracture.losses import FocalLoss, SoftTargetCrossEntropy, build_loss
from fracture.models.backbones import TimmBinaryClassifier
from fracture.models.ema import ModelEMA
from fracture.models.mixup import MixupCutmix


def test_backbone_forward_binary():
    model = TimmBinaryClassifier("resnet50", pretrained=False)
    out = model(torch.randn(2, 3, 96, 96))
    assert out.shape == (2, 2)


def test_densenet_alias_handles_missing_drop_path():
    # densenet timm models reject drop_path_rate; wrapper must fall back
    model = TimmBinaryClassifier("densenet169", pretrained=False, drop_path_rate=0.3)
    assert model(torch.randn(1, 3, 96, 96)).shape == (1, 2)


def test_cam_target_layer_is_conv():
    model = TimmBinaryClassifier("resnet50", pretrained=False)
    assert isinstance(model.cam_target_layer(), torch.nn.Conv2d)


def test_ema_diverges_from_live_weights():
    model = TimmBinaryClassifier("resnet50", pretrained=False)
    ema = ModelEMA(model, decay=0.9, warmup_steps=0)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(4, 3, 96, 96)
    for _ in range(3):
        opt.zero_grad()
        model(x).sum().backward()
        opt.step()
        ema.update(model)
    assert not torch.allclose(next(model.parameters()), next(ema.module.parameters()))


def test_focal_gamma0_equals_ce():
    logits, target = torch.randn(16, 2), torch.randint(0, 2, (16,))
    torch.testing.assert_close(
        FocalLoss(gamma=0.0)(logits, target),
        torch.nn.functional.cross_entropy(logits, target),
        rtol=1e-4,
        atol=1e-4,
    )


def test_focal_downweights_easy():
    easy = torch.tensor([[10.0, -10.0]])
    hard = torch.tensor([[0.05, 0.0]])
    t = torch.tensor([0])
    fl = FocalLoss(gamma=2.0, reduction="none")
    assert fl(easy, t).item() < 0.01 * fl(hard, t).item()


def test_soft_target_ce_matches_hard():
    logits, target = torch.randn(8, 2), torch.randint(0, 2, (8,))
    oh = torch.nn.functional.one_hot(target, 2).float()
    torch.testing.assert_close(
        SoftTargetCrossEntropy()(logits, oh),
        torch.nn.functional.cross_entropy(logits, target),
        rtol=1e-4,
        atol=1e-4,
    )


def test_build_loss_dispatch():
    assert isinstance(build_loss("focal"), FocalLoss)
    assert build_loss("ce").__class__.__name__ == "CrossEntropyLoss"


def test_mixup_soft_targets_sum_to_one():
    mix = MixupCutmix(num_classes=2, prob=1.0, cutmix_alpha=0.0, seed=0)
    _, soft = mix(torch.randn(8, 3, 32, 32), torch.randint(0, 2, (8,)))
    assert soft.shape == (8, 2)
    torch.testing.assert_close(soft.sum(1), torch.ones(8), rtol=1e-4, atol=1e-4)
