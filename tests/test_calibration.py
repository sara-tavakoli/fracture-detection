from __future__ import annotations

import numpy as np
import torch

from fracture.metrics.calibration import (
    TemperatureScaler,
    expected_calibration_error,
    reliability_bins,
)
from fracture.uncertainty.selective import (
    mc_dropout_predict,
    predictive_entropy,
    risk_coverage_curve,
)


def test_temperature_scaling_reduces_ece_on_overconfident_logits():
    rng = np.random.default_rng(0)
    n, k = 2000, 7
    y = rng.integers(0, k, size=n)
    logits = np.full((n, k), -2.0)
    logits[np.arange(n), y] = 2.0
    # inflate -> overconfident
    logits *= 3.0
    # add label noise so it is genuinely miscalibrated
    flip = rng.random(n) < 0.25
    logits[flip] = np.roll(logits[flip], 1, axis=1)

    prob_pre = torch.tensor(logits).softmax(1).numpy()
    ece_pre = expected_calibration_error(y, prob_pre)

    scaler = TemperatureScaler().fit(torch.tensor(logits), torch.tensor(y))
    prob_post = (torch.tensor(logits) / scaler.temperature).softmax(1).numpy()
    ece_post = expected_calibration_error(y, prob_post)

    assert scaler.temperature > 1.0
    assert ece_post < ece_pre


def test_reliability_bins_sum_to_n():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, size=500)
    probs = rng.dirichlet(np.ones(5), size=500)
    rb = reliability_bins(y, probs, n_bins=10)
    assert rb.bin_count.sum() == 500
    assert 0.0 <= rb.ece <= 1.0


def test_predictive_entropy_bounds():
    uniform = np.full((1, 4), 0.25)
    peaked = np.array([[0.97, 0.01, 0.01, 0.01]])
    assert predictive_entropy(uniform)[0] > predictive_entropy(peaked)[0]
    np.testing.assert_allclose(predictive_entropy(uniform)[0], np.log(4), rtol=1e-5)


def test_risk_coverage_monotone_ish_and_bounded():
    rng = np.random.default_rng(2)
    correct = (rng.random(300) < 0.8).astype(float)
    conf = correct * 0.3 + rng.random(300) * 0.7
    rc = risk_coverage_curve(correct, conf)
    assert rc.coverage[0] < rc.coverage[-1]
    assert 0.0 <= rc.aurc <= 1.0
    assert rc.eaurc >= -1e-9


def test_mc_dropout_varies_predictions():
    net = torch.nn.Sequential(
        torch.nn.Linear(10, 32), torch.nn.ReLU(), torch.nn.Dropout(0.5), torch.nn.Linear(32, 3)
    )
    x = torch.randn(4, 10)
    samples = mc_dropout_predict(net, x, n_samples=20)
    assert samples.shape == (20, 4, 3)
    assert samples.std(0).sum() > 0
