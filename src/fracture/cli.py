"""Thin console-script wrappers around the ``scripts/`` entry points."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _run(name: str) -> None:
    sys.argv[0] = str(_SCRIPTS / name)
    runpy.run_path(str(_SCRIPTS / name), run_name="__main__")


def train_entry() -> None:
    _run("train.py")


def evaluate_entry() -> None:
    _run("evaluate.py")
