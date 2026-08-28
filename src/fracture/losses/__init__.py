"""Loss functions for imbalanced binary radiograph classification.

The model emits two logits (normal / abnormal); focal loss with ``gamma=0``
recovers weighted cross-entropy.  Class weights come from the datamodule's
inverse-frequency estimate.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017) with optional class weights.

    ``gamma=0`` reduces to weighted cross-entropy.  ``weight`` is typically the
    effective-number class weight vector produced by the datamodule.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        # ``ce`` is per-sample -(log p_t) including label smoothing + weights.
        weight = self.weight if isinstance(self.weight, Tensor) else None
        ce = F.cross_entropy(
            logits,
            target,
            weight=weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        # p_t recovered from the *unweighted, unsmoothed* CE for the focal term
        # so the modulating factor stays in [0, 1].
        with torch.no_grad():
            pt = torch.exp(-F.cross_entropy(logits, target, reduction="none")).clamp_(1e-6, 1.0)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class SoftTargetCrossEntropy(nn.Module):
    """Cross-entropy against soft targets, for use with MixUp / CutMix."""

    def forward(self, logits: Tensor, soft_target: Tensor) -> Tensor:
        return torch.sum(-soft_target * F.log_softmax(logits, dim=-1), dim=-1).mean()


def build_loss(
    name: str,
    *,
    class_weights: Tensor | None = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.05,
) -> nn.Module:
    name = name.lower()
    if name in {"ce", "cross_entropy"}:
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if name in {"focal", "class_balanced_focal"}:
        return FocalLoss(gamma=gamma, weight=class_weights, label_smoothing=label_smoothing)
    raise ValueError(f"unknown loss '{name}'")


__all__ = ["FocalLoss", "SoftTargetCrossEntropy", "build_loss"]
