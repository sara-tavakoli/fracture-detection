"""Torch ``Dataset`` over a MURA-style metadata frame.

Supports 8-bit images (PNG/JPG) and DICOM (via ``pydicom``), applying a simple
percentile window to DICOM pixel data so intensity ranges are comparable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from fracture import BODY_PARTS

_PART_TO_IDX = {p: i for i, p in enumerate(BODY_PARTS)}


def load_image(path: str | Path) -> np.ndarray:
    """Return an HxWx3 uint8 array from a radiograph file (image or DICOM)."""
    path = Path(path)
    if path.suffix.lower() in {".dcm", ".dicom"}:
        import pydicom

        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr  # invert so bone is bright
        lo, hi = np.percentile(arr, [1, 99])
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
        gray = arr.astype(np.uint8)
        return np.stack([gray] * 3, axis=-1)
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


class RadiographDataset(Dataset):
    """Yields ``(image_tensor, label, meta)`` with ``meta`` carrying study /
    body-part ids used for study-level aggregation and per-part metrics."""

    def __init__(
        self, frame: pd.DataFrame, image_root: str | Path, transform, *, return_meta: bool = True
    ) -> None:
        need = {"image_id", "study_id", "body_part", "label"}
        if not need.issubset(frame.columns):
            raise KeyError(f"frame needs columns {sorted(need)}")
        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.return_meta = return_meta
        self.labels = self.frame["label"].astype(int).to_numpy()

    def __len__(self) -> int:
        return len(self.frame)

    def _path(self, row: pd.Series) -> Path:
        if "filepath" in row and isinstance(row["filepath"], str) and row["filepath"]:
            p = Path(row["filepath"])
            return p if p.is_absolute() else self.image_root / p
        return self.image_root / f"{row['image_id']}.png"

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image = load_image(self._path(row))
        image = self.transform(image=image)["image"]
        label = int(self.labels[idx])
        if not self.return_meta:
            return image, label
        meta = {
            "image_id": str(row["image_id"]),
            "study_id": str(row["study_id"]),
            "body_part": str(row["body_part"]),
            "body_part_idx": _PART_TO_IDX.get(str(row["body_part"]), -1),
        }
        return image, label, meta

    # -- helpers --------------------------------------------------------
    def class_counts(self) -> torch.Tensor:
        counts = np.bincount(self.labels, minlength=2)
        return torch.as_tensor(counts, dtype=torch.long)

    def sample_weights(self) -> torch.Tensor:
        """Balance both the label and the body-part so rare parts are not
        swamped by wrist/hand studies."""
        df = self.frame
        key = df["body_part"].astype(str) + "/" + df["label"].astype(str)
        freq = key.map(key.value_counts()).to_numpy().astype(np.float64)
        w = 1.0 / np.clip(freq, 1, None)
        w = w * (len(w) / w.sum())
        return torch.as_tensor(w, dtype=torch.float)
