from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("mura")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_synthetic_data.py"),
            "--out",
            str(out),
            "--patients",
            "24",
            "--size",
            "64",
            "--seed",
            "0",
        ],
        check=True,
    )
    assert (out / "metadata.csv").exists()
    return out


@pytest.fixture
def dummy_metadata() -> pd.DataFrame:
    parts = ["elbow", "finger", "forearm", "hand", "humerus", "shoulder", "wrist"]
    rng = np.random.default_rng(0)
    rows = []
    for p in range(30):
        pid = f"P{p:03d}"
        for part in rng.choice(parts, size=2, replace=False):
            for s in range(2):
                label = int(rng.random() < 0.5)
                sid = f"{pid}_{part}_{s}"
                for v in range(2):
                    rows.append(
                        {
                            "image_id": f"{sid}_{v}",
                            "study_id": sid,
                            "patient_id": pid,
                            "body_part": part,
                            "label": label,
                        }
                    )
    return pd.DataFrame(rows)
