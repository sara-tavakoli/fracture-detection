"""Deterministic-as-possible seeding across python / numpy / torch."""

from __future__ import annotations

import contextlib
import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    """Seed every RNG we touch and optionally force deterministic kernels.

    Returns the seed so callers can log the effective value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN / MPS determinism.  ``warn_only`` keeps ops that have no
        # deterministic implementation (e.g. some pooling backward) from
        # hard-crashing a run.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with contextlib.suppress(Exception):  # pragma: no cover - older torch
            torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return seed


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding derived from the base torch seed."""
    base = torch.initial_seed() % 2**32
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)
