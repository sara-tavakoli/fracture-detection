"""Musculoskeletal radiograph abnormality / fracture detection (MURA-style).

Binary task: ``normal`` (0) vs ``abnormal`` (1), evaluated at both the image and
the *study* level (a study = all views of one body part for one patient), which
is the clinically meaningful unit and the one the MURA benchmark scores.
"""

from __future__ import annotations

__version__ = "0.1.0"

CLASSES: tuple[str, ...] = ("normal", "abnormal")
POSITIVE_CLASS = 1  # "abnormal"

# The seven MURA study types.
BODY_PARTS: tuple[str, ...] = (
    "elbow",
    "finger",
    "forearm",
    "hand",
    "humerus",
    "shoulder",
    "wrist",
)

# Radiologist Cohen's kappa on the MURA test set (Rajpurkar et al., 2018),
# useful as a reference line when plotting per-body-part kappa.
RADIOLOGIST_KAPPA: dict[str, float] = {
    "elbow": 0.710,
    "finger": 0.389,
    "forearm": 0.737,
    "hand": 0.851,
    "humerus": 0.600,
    "shoulder": 0.290,
    "wrist": 0.931,
    "overall": 0.778,
}

__all__ = [
    "BODY_PARTS",
    "CLASSES",
    "POSITIVE_CLASS",
    "RADIOLOGIST_KAPPA",
    "__version__",
]
