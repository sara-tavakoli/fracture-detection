"""Checkpoint loading and study-level inference.

A *study* is a set of radiograph views; the study-level abnormality probability
is the mean of the per-image probabilities (as in the MURA paper), optionally
with horizontal-flip TTA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from fracture import BODY_PARTS
from fracture.data.dataset import load_image
from fracture.data.transforms import eval_transform, tta_transforms
from fracture.models.backbones import TimmBinaryClassifier
from fracture.uncertainty.selective import mc_dropout_predict


@dataclass
class StudyPrediction:
    abnormal_probability: float
    decision: str  # "abnormal" | "normal"
    threshold: float
    per_image: list[float]
    n_images: int
    body_part: str | None = None
    epistemic_uncertainty: float | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _select_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FracturePredictor:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        backbone: str | None = None,
        image_size: int | None = None,
        threshold: float = 0.5,
        device: str = "auto",
        use_tta: bool = True,
    ) -> None:
        self.device = _select_device(device)
        self.threshold = threshold
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        hp = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}
        model_cfg = hp.get("model", {}) if hp else {}
        self.backbone = backbone or model_cfg.get("name", "densenet169")
        self.image_size = image_size or hp.get("data", {}).get("image_size", 320)

        self.model = TimmBinaryClassifier(self.backbone, num_classes=2, pretrained=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(ckpt, dict) and "ema" in ckpt:
            ema_state = {k.replace("module.", "", 1): v for k, v in ckpt["ema"]["module"].items()}
            self.model.load_state_dict(ema_state, strict=False)
        else:
            clean = {k.replace("model.", "", 1): v for k, v in state.items() if k.startswith("model.")}
            self.model.load_state_dict(clean or state, strict=False)
        self.model.eval().to(self.device)

        self.eval_tf = eval_transform(self.image_size)
        self.tta_tfs = tta_transforms(self.image_size) if use_tta else None

    @torch.no_grad()
    def _image_prob(self, arr: np.ndarray) -> float:
        tfs = self.tta_tfs or [self.eval_tf]
        batch = torch.stack([t(image=arr)["image"] for t in tfs]).to(self.device)
        return float(self.model(batch).softmax(1)[:, 1].mean())

    def predict_study(
        self,
        images: list,
        *,
        body_part: str | None = None,
        mc_dropout_samples: int = 0,
    ) -> StudyPrediction:
        if not images:
            raise ValueError("a study needs at least one image")
        arrs = [img if isinstance(img, np.ndarray) else load_image_from_any(img) for img in images]
        per_image = [self._image_prob(a) for a in arrs]
        prob = float(np.mean(per_image))

        epistemic = None
        if mc_dropout_samples > 0:
            xs = torch.stack([self.eval_tf(image=a)["image"] for a in arrs]).to(self.device)
            sampled = mc_dropout_predict(self.model, xs, n_samples=mc_dropout_samples)  # (S, N, 2)
            study_scores = sampled.softmax(-1)[..., 1].mean(dim=1)  # (S,)
            prob = 0.5 * (prob + float(study_scores.mean()))
            epistemic = float(study_scores.var())

        warnings: list[str] = []
        if 0.4 <= prob <= 0.6:
            warnings.append("Borderline study probability; recommend radiologist review.")
        if len(per_image) == 1:
            warnings.append("Single view only; multi-view studies are more reliable.")
        if body_part is not None and body_part not in BODY_PARTS:
            warnings.append(f"Unrecognised body part '{body_part}'; model trained on {list(BODY_PARTS)}.")

        return StudyPrediction(
            abnormal_probability=prob,
            decision="abnormal" if prob >= self.threshold else "normal",
            threshold=self.threshold,
            per_image=per_image,
            n_images=len(per_image),
            body_part=body_part,
            epistemic_uncertainty=epistemic,
            warnings=warnings,
        )


def load_image_from_any(src) -> np.ndarray:
    import io

    from PIL import Image

    if isinstance(src, (bytes, bytearray)):
        # try DICOM first (starts with 'DICM' at offset 128), else PIL
        if len(src) > 132 and src[128:132] == b"DICM":
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as fh:
                fh.write(src)
                return load_image(fh.name)
        return np.asarray(Image.open(io.BytesIO(src)).convert("RGB"))
    return load_image(src)
