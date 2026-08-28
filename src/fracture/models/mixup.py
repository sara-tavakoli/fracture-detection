"""Batch-level MixUp / CutMix producing soft targets."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def _rand_bbox(h: int, w: int, lam: float, rng: np.random.Generator) -> tuple[int, int, int, int]:
    cut_rat = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(h * cut_rat), int(w * cut_rat)
    cy, cx = rng.integers(0, h), rng.integers(0, w)
    y1, y2 = np.clip([cy - cut_h // 2, cy + cut_h // 2], 0, h)
    x1, x2 = np.clip([cx - cut_w // 2, cx + cut_w // 2], 0, w)
    return int(y1), int(y2), int(x1), int(x2)


class MixupCutmix:
    def __init__(
        self,
        num_classes: int,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
        switch_prob: float = 0.5,
        label_smoothing: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.rng = np.random.default_rng(seed)

    def _one_hot(self, target: Tensor) -> Tensor:
        off = self.label_smoothing / self.num_classes
        on = 1.0 - self.label_smoothing + off
        oh = torch.full((target.size(0), self.num_classes), off, device=target.device, dtype=torch.float)
        return oh.scatter_(1, target.unsqueeze(1), on)

    def __call__(self, x: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        soft = self._one_hot(target)
        if self.rng.random() > self.prob:
            return x, soft

        perm = torch.randperm(x.size(0), device=x.device)
        use_cutmix = self.rng.random() < self.switch_prob and self.cutmix_alpha > 0
        if use_cutmix:
            lam = float(self.rng.beta(self.cutmix_alpha, self.cutmix_alpha))
            y1, y2, x1, x2 = _rand_bbox(x.size(2), x.size(3), lam, self.rng)
            x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
            lam = 1.0 - ((y2 - y1) * (x2 - x1) / (x.size(2) * x.size(3)))
        else:
            lam = float(self.rng.beta(self.mixup_alpha, self.mixup_alpha))
            x = lam * x + (1.0 - lam) * x[perm]

        soft = lam * soft + (1.0 - lam) * soft[perm]
        return x, soft
