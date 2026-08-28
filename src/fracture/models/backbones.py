"""timm backbone factory for binary radiograph classification.

DenseNet-169 is the default because it is the MURA paper's baseline
(Rajpurkar et al., 2018) and remains a strong, memory-frugal choice for
single-channel medical images.
"""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
from torch import nn

BACKBONES: dict[str, str] = {
    "densenet169": "densenet169.tv_in1k",
    "densenet201": "densenet201.tv_in1k",
    "effnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "resnet50": "resnet50.a1_in1k",
}


@dataclass
class BackboneOutput:
    logits: torch.Tensor
    features: torch.Tensor


class TimmBinaryClassifier(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int = 2,
        pretrained: bool = True,
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        timm_name = BACKBONES.get(backbone, backbone)
        self.backbone_name = timm_name
        try:
            self.encoder = timm.create_model(
                timm_name,
                pretrained=pretrained,
                num_classes=0,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
            )
        except TypeError:
            # some backbones (e.g. densenet) do not accept drop_path_rate
            self.encoder = timm.create_model(
                timm_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate
            )
        self.feature_dim: int = self.encoder.num_features  # type: ignore[assignment]
        self.dropout = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.feature_dim, num_classes)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.encoder(x)))

    def forward_with_features(self, x: torch.Tensor) -> BackboneOutput:
        feats = self.encoder(x)
        return BackboneOutput(logits=self.head(self.dropout(feats)), features=feats)

    def cam_target_layer(self) -> nn.Module:
        """Last spatial layer for Grad-CAM: prefer the final Conv2d, fall back to
        the final norm layer (e.g. ConvNeXt's trailing LayerNorm2d)."""
        convs = [m for _, m in self.encoder.named_modules() if isinstance(m, nn.Conv2d)]
        if convs:
            return convs[-1]
        norms = [
            m
            for _, m in self.encoder.named_modules()
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm))
        ]
        if norms:  # pragma: no cover - architecture-dependent
            return norms[-1]
        raise RuntimeError("no CAM-compatible layer found; pass one explicitly")  # pragma: no cover


def create_model(cfg) -> TimmBinaryClassifier:
    return TimmBinaryClassifier(
        backbone=cfg.name,
        num_classes=cfg.get("num_classes", 2),
        pretrained=cfg.get("pretrained", True),
        drop_rate=cfg.get("drop_rate", 0.2),
        drop_path_rate=cfg.get("drop_path_rate", 0.1),
    )
