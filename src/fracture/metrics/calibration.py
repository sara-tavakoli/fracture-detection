"""Confidence calibration: ECE / MCE, temperature scaling, reliability data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class ReliabilityBins:
    bin_edges: np.ndarray
    bin_acc: np.ndarray
    bin_conf: np.ndarray
    bin_count: np.ndarray

    @property
    def ece(self) -> float:
        w = self.bin_count / max(self.bin_count.sum(), 1)
        return float(np.sum(w * np.abs(self.bin_acc - self.bin_conf)))

    @property
    def mce(self) -> float:
        mask = self.bin_count > 0
        if not mask.any():
            return float("nan")
        return float(np.max(np.abs(self.bin_acc - self.bin_conf)[mask]))


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> ReliabilityBins:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    conf = y_prob.max(1)
    pred = y_prob.argmax(1)
    correct = (pred == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    acc = np.zeros(n_bins)
    avg_conf = np.zeros(n_bins)
    count = np.zeros(n_bins)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        count[b] = m.sum()
        if count[b]:
            acc[b] = correct[m].mean()
            avg_conf[b] = conf[m].mean()
    return ReliabilityBins(edges, acc, avg_conf, count)


class TemperatureScaler(nn.Module):
    """Single-parameter post-hoc calibration (Guo et al., 2017).

    Fit on held-out *validation* logits, then apply to test logits.
    """

    def __init__(self, init_temp: float = 1.0) -> None:
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(float(np.log(init_temp))))

    @property
    def temperature(self) -> float:
        return float(self.log_temp.exp().item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_temp.exp()

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> TemperatureScaler:
        logits = logits.detach()
        labels = labels.detach()
        opt = torch.optim.LBFGS([self.log_temp], lr=0.05, max_iter=max_iter)
        nll = nn.CrossEntropyLoss()

        def _closure():
            opt.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        opt.step(_closure)
        return self


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    return reliability_bins(y_true, y_prob, n_bins).ece
