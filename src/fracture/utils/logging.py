"""Console + file logging helpers with a single Rich handler."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger()
    if not _CONFIGURED:
        root.handlers.clear()
        root.setLevel(logging.DEBUG)
        console = RichHandler(rich_tracebacks=True, show_path=False, markup=True)
        console.setLevel(level.upper())
        console.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        root.addHandler(console)
        _CONFIGURED = True

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
        root.addHandler(fh)

    return logging.getLogger("fracture")
