"""Patient-disjoint, body-part-stratified splitting for MURA-style data.

MURA ships an official patient-level train/valid split.  For model development
and confidence intervals we additionally provide K-fold cross-validation that
groups on ``patient_id`` (never splitting a patient's studies across folds) and
stratifies on ``body_part`` x ``label`` so every fold sees every stratum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class SplitConfig:
    n_folds: int = 5
    test_fold: int = 0
    val_fold: int = 1
    seed: int = 1337


REQUIRED_COLUMNS = {"image_id", "study_id", "patient_id", "body_part", "label"}


def assign_folds(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(f"metadata missing columns: {sorted(missing)}")

    df = df.reset_index(drop=True).copy()
    strata = df["body_part"].astype(str) + "/" + df["label"].astype(str)
    groups = df["patient_id"].astype(str).to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    fold = np.full(len(df), -1, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(df, strata, groups)):
        fold[val_idx] = fold_idx
    if (fold < 0).any():  # pragma: no cover - defensive
        raise RuntimeError("some rows were not assigned to a fold")
    df["fold"] = fold
    _assert_patient_disjoint(df)
    return df


def split_frames(df: pd.DataFrame, cfg: SplitConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "split" in df.columns and df["split"].isin({"train", "valid"}).all():
        # Honour an official split when present: carve val out of official train.
        official_val = df[df["split"] == "valid"]
        dev = df[df["split"] == "train"]
        dev = assign_folds(dev, cfg)
        val = dev[dev["fold"] == cfg.val_fold]
        train = dev[dev["fold"] != cfg.val_fold]
        return (
            train.reset_index(drop=True),
            val.reset_index(drop=True),
            official_val.reset_index(drop=True),
        )

    if "fold" not in df.columns:
        df = assign_folds(df, cfg)
    test = df[df["fold"] == cfg.test_fold]
    val = df[df["fold"] == cfg.val_fold]
    train = df[~df["fold"].isin({cfg.test_fold, cfg.val_fold})]
    for name, part in (("train", train), ("val", val), ("test", test)):
        if part.empty:
            raise RuntimeError(f"{name} split is empty; check SplitConfig")
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def _assert_patient_disjoint(df: pd.DataFrame) -> None:
    per_patient = df.groupby("patient_id")["fold"].nunique()
    leaked = per_patient[per_patient > 1]
    if len(leaked):  # pragma: no cover - defensive
        raise RuntimeError(f"{len(leaked)} patient(s) span multiple folds")
