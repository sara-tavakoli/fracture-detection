"""Predictive uncertainty and selective-prediction (abstention) analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from fracture.utils.numeric import trapezoid


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor, n_samples: int = 30) -> torch.Tensor:
    """Return an ``(n_samples, B, C)`` tensor of softmax probabilities with
    dropout kept active at inference time (Gal & Ghahramani, 2016)."""
    was_training = model.training
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
    try:
        probs = torch.stack([model(x).softmax(-1) for _ in range(n_samples)], dim=0)
    finally:
        model.train(was_training)
    return probs


def predictive_entropy(mean_probs: np.ndarray) -> np.ndarray:
    p = np.clip(mean_probs, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def mutual_information(sampled_probs: np.ndarray) -> np.ndarray:
    """BALD: total (predictive) entropy minus expected (aleatoric) entropy.

    ``sampled_probs`` has shape ``(S, N, C)``.
    """
    mean = sampled_probs.mean(axis=0)
    total = predictive_entropy(mean)
    expected = predictive_entropy(np.clip(sampled_probs, 1e-12, 1.0)).mean(axis=0)
    return total - expected


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Free-energy OOD score (Liu et al., 2020). Lower energy => in-distribution."""
    t = temperature
    return -t * np.log(np.sum(np.exp(logits / t), axis=-1))


@dataclass
class RiskCoverage:
    coverage: np.ndarray
    risk: np.ndarray
    aurc: float  # area under risk-coverage curve (lower is better)
    eaurc: float  # excess AURC vs the optimal ranking


def risk_coverage_curve(correct: np.ndarray, confidence: np.ndarray) -> RiskCoverage:
    """Selective risk as a function of coverage, sorted by descending confidence."""
    correct = np.asarray(correct).astype(float)
    order = np.argsort(-np.asarray(confidence))
    c = correct[order]
    n = c.size
    cum_err = np.cumsum(1.0 - c)
    k = np.arange(1, n + 1)
    risk = cum_err / k
    coverage = k / n
    aurc = trapezoid(risk, coverage)

    # optimal: all correct predictions ranked first
    c_opt = np.sort(correct)[::-1]
    risk_opt = np.cumsum(1.0 - c_opt) / k
    aurc_opt = trapezoid(risk_opt, coverage)
    return RiskCoverage(coverage, risk, aurc, aurc - aurc_opt)
