from __future__ import annotations

import numpy as np
import pandas as pd

from fracture.metrics.classification import (
    aggregate_studies,
    choose_threshold,
    compute_report,
    per_body_part_kappa,
)


def test_perfect_scores_give_unit_auroc_and_kappa():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    s = y.astype(float) * 0.9 + 0.05
    rep = compute_report(y, s, threshold=0.5, n_bootstrap=0)
    assert rep.auroc == 1.0
    assert rep.kappa == 1.0
    assert rep.sensitivity == 1.0 and rep.specificity == 1.0


def test_choose_threshold_meets_target_sensitivity():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(200), np.zeros(200)].astype(int)
    s = np.r_[rng.normal(0.7, 0.15, 200), rng.normal(0.3, 0.15, 200)].clip(0, 1)
    thr = choose_threshold(y, s, target_sensitivity=0.95)
    sens = ((s >= thr)[y == 1]).mean()
    assert sens >= 0.95 - 1e-9


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=400)
    s = (y * 0.4 + rng.random(400) * 0.6).clip(0, 1)
    rep = compute_report(y, s, threshold=0.5, n_bootstrap=400, seed=0)
    lo, hi = rep.ci["auroc"]
    assert lo <= rep.auroc <= hi


def test_study_aggregation_mean_and_shape():
    df = pd.DataFrame(
        {
            "study_id": ["a", "a", "b", "b", "b"],
            "label": [1, 1, 0, 0, 0],
            "body_part": ["wrist"] * 5,
            "score": [0.8, 0.6, 0.2, 0.1, 0.3],
        }
    )
    study = aggregate_studies(df, score_col="score", how="mean")
    assert len(study) == 2
    a = study.set_index("study_id").loc["a"]
    assert abs(a["score"] - 0.7) < 1e-9
    assert a["label"] == 1


def test_per_body_part_kappa_has_overall_row():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "body_part": rng.choice(["wrist", "elbow", "hand"], size=300),
            "label": rng.integers(0, 2, size=300),
            "score": rng.random(300),
        }
    )
    out = per_body_part_kappa(df, threshold=0.5)
    assert "overall" in set(out["body_part"])
    assert out["n"].iloc[-1] == 300
