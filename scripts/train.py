#!/usr/bin/env python
"""Train a musculoskeletal-radiograph abnormality detector.

python scripts/train.py --experiment strong
python scripts/train.py model=convnext_tiny data.image_size=384 train.lr=8e-5
python scripts/train.py --experiment fast_dev            # CI smoke run
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fracture.data.datamodule import MURADataModule
from fracture.models.classifier import FractureClassifier
from fracture.utils.config import load_config, save_config
from fracture.utils.logging import setup_logging
from fracture.utils.seed import seed_everything


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_trainer(cfg, output_dir: Path):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
        RichProgressBar,
    )
    from pytorch_lightning.loggers import CSVLogger

    loggers: list = [CSVLogger(save_dir=str(output_dir), name="logs")]
    try:
        from pytorch_lightning.loggers import MLFlowLogger

        loggers.append(
            MLFlowLogger(
                experiment_name="fracture-detection",
                tracking_uri=cfg.paths.mlflow_uri,
                run_name=f"{cfg.model.name}-{cfg.train.loss}",
            )
        )
    except Exception:
        pass

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="epoch{epoch:02d}-auroc{val/auroc:.4f}",
        monitor=cfg.train.monitor,
        mode=cfg.train.monitor_mode,
        save_top_k=2,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [
        ckpt_cb,
        EarlyStopping(
            monitor=cfg.train.monitor, mode=cfg.train.monitor_mode, patience=cfg.train.early_stop_patience
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]

    precision = cfg.train.precision
    if cfg.trainer.accelerator in ("auto", "cpu") and not torch.cuda.is_available():
        precision = "32-true"

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=precision,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        deterministic="warn" if cfg.get("deterministic", True) else False,
        default_root_dir=str(output_dir),
    )
    return trainer, ckpt_cb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.overrides, experiment=args.experiment)
    output_dir = Path(args.output_dir or cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(log_file=output_dir / "train.log")

    seed_everything(cfg.seed, deterministic=cfg.get("deterministic", True))
    save_config(cfg, output_dir / "config.snapshot.yaml")

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "git_sha": _git_sha(),
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    (output_dir / "env.json").write_text(json.dumps(env, indent=2))
    log.info("environment: %s", env)

    dm = MURADataModule(
        data_dir=cfg.data.data_dir,
        metadata_csv=cfg.data.metadata_csv,
        image_subdir=cfg.data.image_subdir,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        aug_strength=cfg.data.aug_strength,
        balanced_sampler=cfg.data.balanced_sampler,
        n_folds=cfg.data.n_folds,
        test_fold=cfg.data.test_fold,
        val_fold=cfg.data.val_fold,
        seed=cfg.seed,
    )
    model = FractureClassifier(cfg)

    trainer, ckpt_cb = build_trainer(cfg, output_dir)
    trainer.fit(model, datamodule=dm)
    test_metrics = trainer.test(model, datamodule=dm, ckpt_path="best")

    if ckpt_cb.best_model_path:
        (output_dir / "best.ckpt").write_bytes(Path(ckpt_cb.best_model_path).read_bytes())
        log.info("best checkpoint -> %s (%s)", output_dir / "best.ckpt", ckpt_cb.best_model_score)

    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    log.info("test metrics: %s", test_metrics)


if __name__ == "__main__":
    main()
