#!/usr/bin/env python
"""Export a trained checkpoint to TorchScript and (optionally) ONNX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fracture import BODY_PARTS, CLASSES
from fracture.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from fracture.serve.inference import FracturePredictor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/export"))
    ap.add_argument("--onnx", action="store_true")
    ap.add_argument("--image-size", type=int, default=None)
    args = ap.parse_args()

    predictor = FracturePredictor(args.checkpoint, image_size=args.image_size, device="cpu", use_tta=False)
    model = predictor.model.eval()
    size = predictor.image_size
    example = torch.randn(1, 3, size, size)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.jit.trace(model, example).save(str(args.out / "model.torchscript.pt"))
    print("wrote", args.out / "model.torchscript.pt")

    if args.onnx:
        try:
            import onnx  # noqa: F401

            torch.onnx.export(
                model,
                example,
                str(args.out / "model.onnx"),
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
            )
            print("wrote", args.out / "model.onnx")
        except ImportError:
            print("skipping ONNX export: `pip install onnx onnxruntime` to enable it")

    meta = {
        "classes": list(CLASSES),
        "body_parts": list(BODY_PARTS),
        "backbone": predictor.backbone,
        "image_size": size,
        "preprocessing": {
            "longest_max_size": size,
            "pad_to": size,
            "normalize_mean": list(IMAGENET_MEAN),
            "normalize_std": list(IMAGENET_STD),
            "channel_order": "RGB (grayscale replicated)",
        },
        "output": "raw logits [normal, abnormal]; softmax then take index 1; average over a study",
        "disclaimer": "research prototype - not a medical device",
    }
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("wrote", args.out / "metadata.json")


if __name__ == "__main__":
    main()
