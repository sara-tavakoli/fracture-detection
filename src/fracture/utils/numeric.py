"""Small numeric helpers shared across metrics code."""

from __future__ import annotations

import numpy as np

# ``np.trapz`` was renamed to ``np.trapezoid`` in NumPy 2.0; support both.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # noqa: B009


def trapezoid(y, x) -> float:
    """Definite integral of ``y`` w.r.t. ``x`` (NumPy-version agnostic)."""
    return float(_trapz(np.asarray(y, dtype=float), np.asarray(x, dtype=float)))
