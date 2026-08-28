#!/usr/bin/env python
"""Normalise a downloaded MURA release into ``data/mura/metadata.csv``.

MURA (Stanford ML Group, https://stanfordmlgroup.github.io/competitions/mura/)
requires accepting a research-use agreement; it cannot be auto-downloaded. Once
you have ``MURA-v1.1/`` (with ``train_image_paths.csv``, ``valid_image_paths.csv``,
``train_labeled_studies.csv``, ``valid_labeled_studies.csv``), run:

    python scripts/prepare_mura.py --mura-root /path/to/MURA-v1.1 --out data/mura

This writes a unified ``metadata.csv`` with columns
``image_id, study_id, patient_id, body_part, label, filepath, split`` where
``filepath`` is relative to ``--mura-root``. Labels are study-level (every image
in a study inherits the study label), exactly as MURA defines the task.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# .../XR_WRIST/patient00001/study1_positive/image1.png
_PART_RE = re.compile(r"XR_([A-Z]+)")
_PATIENT_RE = re.compile(r"(patient\d+)")


def _study_key(image_path: str) -> str:
    return str(Path(image_path).parent)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mura-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/mura"))
    args = ap.parse_args()

    frames = []
    for split in ("train", "valid"):
        img_csv = args.mura_root / f"{split}_image_paths.csv"
        study_csv = args.mura_root / f"{split}_labeled_studies.csv"
        if not img_csv.exists():
            raise FileNotFoundError(img_csv)
        images = pd.read_csv(img_csv, header=None, names=["path"])
        studies = pd.read_csv(study_csv, header=None, names=["study_path", "label"])
        studies["key"] = studies["study_path"].str.rstrip("/")
        label_map = dict(zip(studies["key"], studies["label"], strict=False))

        images["path"] = images["path"].str.strip()
        images["key"] = images["path"].apply(_study_key)
        images["label"] = images["key"].map(label_map)
        images = images.dropna(subset=["label"])
        images["label"] = images["label"].astype(int)
        images["body_part"] = images["path"].apply(
            lambda p: _PART_RE.search(p).group(1).lower() if _PART_RE.search(p) else "unknown"
        )
        images["patient_id"] = images["path"].apply(
            lambda p: _PATIENT_RE.search(p).group(1) if _PATIENT_RE.search(p) else "unknown"
        )
        images["study_id"] = images["key"].str.replace("/", "__", regex=False)
        images["image_id"] = images["path"].str.replace("/", "__", regex=False).str.removesuffix(".png")
        images["filepath"] = images["path"]
        images["split"] = split
        frames.append(
            images[["image_id", "study_id", "patient_id", "body_part", "label", "filepath", "split"]]
        )

    df = pd.concat(frames, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "metadata.csv", index=False)

    print(f"wrote {len(df)} rows -> {args.out / 'metadata.csv'}")
    print(df.groupby(["split", "body_part"])["label"].agg(["count", "mean"]).round(3).to_string())
    print("\nNOTE: set data.data_dir to --mura-root (filepaths are relative to it),")
    print("or copy/symlink the XR_* folders under data/mura/.")


if __name__ == "__main__":
    main()
