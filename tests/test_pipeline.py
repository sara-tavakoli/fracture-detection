"""End-to-end smoke: config -> datamodule -> Lightning fit -> study inference."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_train_one_epoch_and_study_predict(synthetic_data, tmp_path):
    import pytorch_lightning as pl

    from fracture.data.datamodule import MURADataModule
    from fracture.models.classifier import FractureClassifier
    from fracture.utils.config import load_config

    cfg = load_config(
        [
            f"data.data_dir={synthetic_data}",
            "data.image_size=64",
            "data.batch_size=8",
            "data.num_workers=0",
            "data.n_folds=3",
            "data.balanced_sampler=false",
            "model.name=resnet50",
            "model.pretrained=false",
            "model.drop_path_rate=0.0",
            "train.max_epochs=1",
            "train.precision=32-true",
            "train.ema.enabled=false",
            "train.mixup.enabled=false",
        ]
    )
    dm = MURADataModule(
        data_dir=str(synthetic_data),
        image_size=64,
        batch_size=8,
        num_workers=0,
        n_folds=3,
        balanced_sampler=False,
    )
    model = FractureClassifier(cfg)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        limit_train_batches=4,
        limit_val_batches=2,
        limit_test_batches=2,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(model, datamodule=dm)
    metrics = trainer.test(model, datamodule=dm)
    assert "test/auroc" in metrics[0]

    ckpt = tmp_path / "m.ckpt"
    trainer.save_checkpoint(ckpt)

    from fracture.serve.inference import FracturePredictor

    predictor = FracturePredictor(ckpt, backbone="resnet50", image_size=64, device="cpu", use_tta=False)
    study_imgs = sorted((synthetic_data).rglob("*.png"))[:3]
    pred = predictor.predict_study([p.read_bytes() for p in study_imgs], body_part="wrist")
    assert 0.0 <= pred.abnormal_probability <= 1.0
    assert pred.decision in {"normal", "abnormal"}
    assert len(pred.per_image) == 3


@pytest.mark.slow
def test_gradcam_runs(synthetic_data):
    from fracture.explain.cam import FractureExplainer
    from fracture.models.backbones import TimmBinaryClassifier

    model = TimmBinaryClassifier("resnet50", pretrained=False).eval()
    res = FractureExplainer(model, method="gradcam++", device="cpu").explain(torch.randn(3, 64, 64))
    assert res.overlay.shape == (64, 64, 3)
    assert res.heatmap.min() >= 0.0 and res.heatmap.max() <= 1.0
