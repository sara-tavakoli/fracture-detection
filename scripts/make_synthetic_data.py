#!/usr/bin/env python
"""Generate a small synthetic MURA-style radiograph dataset.

Creates grey "bone-like" images grouped into studies (patient x body-part), a
fraction of which contain a bright "fracture line". Label-correlated signal lets
a model beat chance in a smoke test. NOT real data.

    python scripts/make_synthetic_data.py --out data/mura --patients 40
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

BODY_PARTS = ["elbow", "finger", "forearm", "hand", "humerus", "shoulder", "wrist"]


def render_view(size: int, abnormal: bool, rng: np.random.Generator) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size] / size
    # soft tissue gradient + a bright "bone" band
    img = 0.25 + 0.15 * yy + 0.05 * rng.standard_normal((size, size))
    bone_center = 0.5 + 0.15 * rng.standard_normal()
    bone = np.exp(-((xx - bone_center) ** 2) / (2 * 0.06**2))
    img += 0.55 * bone

    if abnormal:
        # a bright, roughly transverse discontinuity across the bone
        y0 = rng.uniform(0.25, 0.75)
        angle = rng.uniform(-0.25, 0.25)
        line = np.exp(-((yy - y0 - angle * (xx - 0.5)) ** 2) / (2 * 0.006**2))
        img += 0.6 * line * bone
        if rng.random() < 0.4:  # sometimes add "hardware" dots
            for _ in range(rng.integers(2, 5)):
                cx, cy = rng.uniform(0.35, 0.65), rng.uniform(0.2, 0.8)
                img += 0.7 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 0.004**2)))

    img += 0.02 * rng.standard_normal((size, size))
    arr = np.clip(img, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/mura"))
    ap.add_argument("--patients", type=int, default=40)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--abnormal-frac", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    img_root = args.out
    img_root.mkdir(parents=True, exist_ok=True)

    for p in range(args.patients):
        patient_id = f"P{p:05d}"
        n_parts = int(rng.integers(1, 4))
        for part in rng.choice(BODY_PARTS, size=n_parts, replace=False):
            n_studies = int(rng.integers(1, 3))
            for s in range(n_studies):
                abnormal = rng.random() < args.abnormal_frac
                study_id = f"{patient_id}_{part}_study{s}_{'positive' if abnormal else 'negative'}"
                rel_dir = Path(part) / patient_id / f"study{s}"
                (img_root / rel_dir).mkdir(parents=True, exist_ok=True)
                for v in range(int(rng.integers(1, 4))):
                    fname = rel_dir / f"image{v}.png"
                    render_view(args.size, abnormal, rng).save(img_root / fname)
                    rows.append(
                        {
                            "image_id": str(fname).replace("/", "__").removesuffix(".png"),
                            "study_id": study_id,
                            "patient_id": patient_id,
                            "body_part": part,
                            "label": int(abnormal),
                            "filepath": str(fname),
                        }
                    )

    with open(args.out / "metadata.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} images across {len({r['study_id'] for r in rows})} studies -> {args.out}")


if __name__ == "__main__":
    main()
