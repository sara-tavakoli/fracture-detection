from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fracture.data.dataset import RadiographDataset
from fracture.data.splits import SplitConfig, assign_folds, split_frames
from fracture.data.transforms import eval_transform, train_transform


def test_folds_are_patient_disjoint(dummy_metadata):
    df = assign_folds(dummy_metadata, SplitConfig(n_folds=5, seed=0))
    assert (df.groupby("patient_id")["fold"].nunique() == 1).all()


def test_split_frames_disjoint_patients(dummy_metadata):
    train, val, test = split_frames(dummy_metadata, SplitConfig(n_folds=5, seed=0))
    a, b, c = (set(p["patient_id"]) for p in (train, val, test))
    assert a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c)


def test_official_split_is_respected():
    rows = []
    for i in range(40):
        split = "train" if i < 30 else "valid"
        rows.append(
            {
                "image_id": f"i{i}",
                "study_id": f"s{i}",
                "patient_id": f"p{i}",
                "body_part": "wrist",
                "label": i % 2,
                "split": split,
            }
        )
    df = pd.DataFrame(rows)
    train, val, test = split_frames(df, SplitConfig(n_folds=5, seed=0))
    assert set(test["split"]) == {"valid"}
    assert set(train["split"]) == {"train"} and set(val["split"]) == {"train"}


def test_missing_columns_raise():
    with pytest.raises(KeyError):
        assign_folds(pd.DataFrame({"image_id": ["a"]}), SplitConfig())


def test_dataset_item_shape_and_meta(synthetic_data):
    df = pd.read_csv(synthetic_data / "metadata.csv")
    ds = RadiographDataset(df, synthetic_data, eval_transform(48))
    img, label, meta = ds[0]
    assert img.shape == (3, 48, 48)
    assert label in (0, 1)
    assert {"image_id", "study_id", "body_part", "body_part_idx"} <= set(meta)


def test_sample_weights_length_and_positivity(synthetic_data):
    df = pd.read_csv(synthetic_data / "metadata.csv")
    ds = RadiographDataset(df, synthetic_data, train_transform(48))
    w = ds.sample_weights()
    assert len(w) == len(ds)
    assert (w > 0).all()


def test_train_transform_is_stochastic(synthetic_data):
    from PIL import Image

    df = pd.read_csv(synthetic_data / "metadata.csv")
    with Image.open(synthetic_data / df.iloc[0]["filepath"]) as im:
        arr = np.asarray(im.convert("RGB"))
    tf = train_transform(48)
    assert not np.allclose(tf(image=arr)["image"].numpy(), tf(image=arr)["image"].numpy())
