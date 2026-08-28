"""Grad-CAM family localisation for radiograph abnormality models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from fracture.data.transforms import IMAGENET_MEAN, IMAGENET_STD

_METHODS = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus, "xgradcam": XGradCAM}


@dataclass
class CamResult:
    heatmap: np.ndarray
    overlay: np.ndarray
    class_idx: int
    abnormal_prob: float


def denormalize(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x.detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


class FractureExplainer:
    def __init__(self, model, method: str = "gradcam++", target_layer=None, device="cpu") -> None:
        if method not in _METHODS:
            raise ValueError(f"method must be one of {sorted(_METHODS)}")
        self.model = model.eval().to(device)
        self.device = device
        inner = getattr(model, "model", model)
        layer = target_layer or inner.cam_target_layer()
        self.cam = _METHODS[method](model=inner, target_layers=[layer])

    @torch.no_grad()
    def _abnormal_prob(self, x: torch.Tensor) -> float:
        inner = getattr(self.model, "model", self.model)
        return float(inner(x.to(self.device)).softmax(1)[0, 1])

    def explain(self, x: torch.Tensor, class_idx: int = 1) -> CamResult:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        prob = self._abnormal_prob(x)
        grayscale = self.cam(
            input_tensor=x.to(self.device),
            targets=[ClassifierOutputTarget(class_idx)],
        )[0]
        rgb = denormalize(x[0])
        overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
        return CamResult(
            heatmap=grayscale.astype(np.float32),
            overlay=overlay.astype(np.uint8),
            class_idx=class_idx,
            abnormal_prob=prob,
        )
