"""Exponential moving average of model weights (Polyak averaging)."""

from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
from torch import nn


class ModelEMA:
    """Maintains a shadow copy of ``model`` updated as
    ``ema = decay * ema + (1 - decay) * model``.

    A warmup schedule ramps the effective decay so early, noisy weights do not
    dominate the average.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998, warmup_steps: int = 2000) -> None:
        self.decay = decay
        self.warmup_steps = max(warmup_steps, 1)
        self.num_updates = 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def _current_decay(self) -> float:
        ramp = 1.0 - 1.0 / (self.num_updates + 1)
        return min(self.decay, ramp) if self.num_updates < self.warmup_steps else self.decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        d = self._current_decay()
        ema_params = dict(self.module.named_parameters())
        ema_buffers = dict(self.module.named_buffers())
        for name, p in model.named_parameters():
            ema_params[name].mul_(d).add_(p.detach(), alpha=1.0 - d)
        for name, b in model.named_buffers():
            if b.dtype.is_floating_point:
                ema_buffers[name].mul_(d).add_(b.detach(), alpha=1.0 - d)
            else:
                ema_buffers[name].copy_(b)

    @contextmanager
    def average_parameters(self, model: nn.Module):
        """Temporarily swap ``model``'s weights for the EMA weights."""
        backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.module.state_dict(), strict=False)
        try:
            yield
        finally:
            model.load_state_dict(backup)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "num_updates": self.num_updates, "module": self.module.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.num_updates = state["num_updates"]
        self.module.load_state_dict(state["module"])
