#!/usr/bin/env python
"""Batch Grad-CAM overlays for radiographs (fracture localisation sanity check).

    python scripts/explain.py --checkpoint artifacts/best.ckpt \
        --images data/mura --limit 24 --out artifacts/cams
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fracture.data.dataset import load_image
from fracture.explain.cam import FractureExplainer
from fracture.serve.inference import FracturePredictor

_EXTS = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/cams"))
    ap.add_argument("--method", default="gradcam++", choices=["gradcam", "gradcam++", "xgradcam"])
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    predictor = FracturePredictor(args.checkpoint, device=args.device, use_tta=False)
    explainer = FractureExplainer(predictor.model, method=args.method, device=str(predictor.device))

    if args.images.is_file():
        paths = [args.images]
    else:
        paths = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in _EXTS)[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        arr = load_image(path)
        x = predictor.eval_tf(image=arr)["image"]
        res = explainer.explain(torch.as_tensor(x))
        Image.fromarray(res.overlay).save(args.out / f"{path.stem}_p{res.abnormal_prob:.2f}_cam.png")
        print(f"{path.name}: P(abnormal)={res.abnormal_prob:.3f}")

    print(f"\nwrote {len(paths)} overlays to {args.out}")


if __name__ == "__main__":
    main()
