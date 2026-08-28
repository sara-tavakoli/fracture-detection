"""Albumentations pipelines for musculoskeletal radiographs.

X-ray-specific choices
----------------------
* Images are single-channel intensity; we replicate to 3 channels for
  ImageNet-pretrained backbones but never apply hue/saturation jitter.
* CLAHE + random gamma emulate window/level variation across machines and PACS.
* Geometric augmentation is limited to small rotations / shifts / scale and
  horizontal flip: laterality can flip, but a fracture's appearance is
  flip-invariant, so it is safe and standard for MURA.
* No vertical flip (upside-down radiographs are not a real distribution shift
  and confuse the body-part prior).
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(image_size: int, *, strength: str = "medium") -> A.Compose:
    s = {"light": 0.5, "medium": 1.0, "heavy": 1.5}[strength]
    return A.Compose(
        [
            A.LongestMaxSize(max_size=int(image_size * 1.15)),
            A.PadIfNeeded(image_size, image_size, border_mode=0, fill=0),
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(1 - 0.08 * s, 1 + 0.08 * s),
                translate_percent=(0.0, 0.04 * s),
                rotate=(-12 * s, 12 * s),
                p=0.7,
            ),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=2.0, p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.15 * s, contrast_limit=0.15 * s, p=1.0),
                ],
                p=0.7,
            ),
            A.GaussNoise(p=0.15),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                fill=0,
                p=0.25 * min(s, 1.0),
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def eval_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(image_size, image_size, border_mode=0, fill=0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def tta_transforms(image_size: int) -> list[A.Compose]:
    base = [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(image_size, image_size, border_mode=0, fill=0),
    ]
    tail = [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return [A.Compose(base + extra + tail) for extra in ([], [A.HorizontalFlip(p=1.0)])]
