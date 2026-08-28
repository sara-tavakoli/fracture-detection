#!/usr/bin/env python
"""Assign patient-disjoint folds and write a dataset summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fracture.data.splits import SplitConfig, assign_folds, split_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/mura"))
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--test-fold", type=int, default=0)
    ap.add_argument("--val-fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    df = pd.read_csv(args.data_dir / "metadata.csv")
    cfg = SplitConfig(args.n_folds, args.test_fold, args.val_fold, args.seed)
    df = assign_folds(df, cfg)
    train_df, val_df, test_df = split_frames(df, cfg)

    df[["image_id", "study_id", "patient_id", "body_part", "label", "fold"]].to_csv(
        args.data_dir / "splits.csv", index=False
    )

    def _summ(name, part):
        return {
            "images": len(part),
            "studies": int(part["study_id"].nunique()),
            "patients": int(part["patient_id"].nunique()),
            "prevalence": round(float(part["label"].mean()), 4),
            "by_body_part": part.groupby("body_part")["label"]
            .agg(["count", "mean"])
            .round(3)
            .to_dict("index"),
        }

    t, v, s = set(train_df.patient_id), set(val_df.patient_id), set(test_df.patient_id)
    summary = {
        "config": cfg.__dict__,
        "splits": {n: _summ(n, p) for n, p in (("train", train_df), ("val", val_df), ("test", test_df))},
        "patient_overlap": {"train_val": len(t & v), "train_test": len(t & s), "val_test": len(v & s)},
    }
    (args.data_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                k: {kk: summary["splits"][k][kk] for kk in ("images", "studies", "patients", "prevalence")}
                for k in summary["splits"]
            },
            indent=2,
        )
    )
    print("patient overlap (want 0s):", summary["patient_overlap"])


if __name__ == "__main__":
    main()
