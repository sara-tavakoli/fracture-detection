"""Binary evaluation metrics for radiograph abnormality detection.

Includes image-level and study-level reports, per-body-part Cohen's kappa (the
MURA benchmark metric), operating-point selection, and bootstrap 95% CIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


@dataclass
class BinaryReport:
    n: int
    prevalence: float
    threshold: float
    accuracy: float
    auroc: float
    ap: float
    kappa: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    f1: float
    confusion: list[list[int]]
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["ci95"] = {k: list(v) for k, v in self.ci.items()}
        d.pop("ci")
        return d


def _point_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else float("nan")
    try:
        auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
        ap = average_precision_score(y_true, y_score) if y_true.any() else float("nan")
    except ValueError:
        auroc = ap = float("nan")
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "auroc": float(auroc),
        "ap": float(ap),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def choose_threshold(
    y_true: np.ndarray, y_score: np.ndarray, *, target_sensitivity: float | None = None
) -> float:
    """If ``target_sensitivity`` is set, return the lowest threshold achieving it;
    otherwise return the Youden-J optimal threshold."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    if target_sensitivity is not None:
        ok = np.where(tpr >= target_sensitivity)[0]
        return float(thr[ok[-1]]) if len(ok) else 0.5
    return float(thr[np.argmax(tpr - fpr)])


def compute_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float | None = None,
    target_sensitivity: float | None = None,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> BinaryReport:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    if threshold is None:
        threshold = choose_threshold(y_true, y_score, target_sensitivity=target_sensitivity)

    m = _point_metrics(y_true, y_score, threshold)
    report = BinaryReport(
        n=int(y_true.size),
        prevalence=float(y_true.mean()),
        threshold=float(threshold),
        confusion=m.pop("confusion"),
        **m,
    )

    if n_bootstrap:
        rng = np.random.default_rng(seed)
        keys = ["accuracy", "auroc", "ap", "kappa", "sensitivity", "specificity"]
        acc: dict[str, list[float]] = {k: [] for k in keys}
        idx_all = np.arange(y_true.size)
        for _ in range(n_bootstrap):
            idx = rng.choice(idx_all, size=idx_all.size, replace=True)
            if np.unique(y_true[idx]).size < 2:
                continue
            bm = _point_metrics(y_true[idx], y_score[idx], threshold)
            for k in keys:
                acc[k].append(bm[k])
        report.ci = {
            k: (
                float(np.nanpercentile(v, 2.5)),
                float(np.nanpercentile(v, 97.5)),
            )
            for k, v in acc.items()
            if v
        }
    return report


def per_body_part_kappa(
    df: pd.DataFrame,
    *,
    score_col: str = "score",
    label_col: str = "label",
    part_col: str = "body_part",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Cohen's kappa per body part at a fixed threshold (MURA-style)."""
    rows = []
    for part, g in df.groupby(part_col):
        y = g[label_col].to_numpy().astype(int)
        pred = (g[score_col].to_numpy() >= threshold).astype(int)
        rows.append(
            {
                "body_part": part,
                "n": len(g),
                "prevalence": float(y.mean()),
                "kappa": float(cohen_kappa_score(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
            }
        )
    out = pd.DataFrame(rows).sort_values("body_part").reset_index(drop=True)
    overall = cohen_kappa_score(df[label_col].astype(int), (df[score_col] >= threshold).astype(int))
    out.loc[len(out)] = ["overall", len(df), float(df[label_col].mean()), float(overall)]
    return out


def aggregate_studies(image_df: pd.DataFrame, *, score_col: str = "score", how: str = "mean") -> pd.DataFrame:
    """Collapse image-level scores to study-level (MURA averages per study).

    ``image_df`` needs ``study_id``, ``label``, ``body_part`` and ``score_col``.
    """
    agg = {score_col: how, "label": "first", "body_part": "first"}
    study = image_df.groupby("study_id").agg(agg).reset_index()
    study = study.rename(columns={score_col: "score"})
    return study
