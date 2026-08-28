"""LightningModule for binary radiograph abnormality detection."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryCohenKappa,
    BinaryF1Score,
    BinarySpecificity,
)

from fracture.losses import SoftTargetCrossEntropy, build_loss
from fracture.models.backbones import create_model
from fracture.models.ema import ModelEMA
from fracture.models.mixup import MixupCutmix


class FractureClassifier(LightningModule):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.num_classes = 2

        self.model = create_model(cfg.model)
        self.criterion = build_loss(
            cfg.train.loss,
            gamma=cfg.train.get("focal_gamma", 2.0),
            label_smoothing=cfg.train.get("label_smoothing", 0.0),
        )
        if getattr(self.criterion, "weight", None) is None:
            self.criterion.register_buffer("weight", torch.ones(self.num_classes))
        self.soft_criterion = SoftTargetCrossEntropy()

        self.mixup = None
        if cfg.train.get("mixup", {}).get("enabled", False):
            m = cfg.train.mixup
            self.mixup = MixupCutmix(
                num_classes=self.num_classes,
                mixup_alpha=m.get("mixup_alpha", 0.2),
                cutmix_alpha=m.get("cutmix_alpha", 0.0),
                prob=m.get("prob", 0.3),
                switch_prob=m.get("switch_prob", 0.0),
                label_smoothing=cfg.train.get("label_smoothing", 0.0),
                seed=cfg.get("seed", 0),
            )

        self._ema: ModelEMA | None = None
        self.class_weights: torch.Tensor | None = None

        metrics = MetricCollection(
            {
                "acc": BinaryAccuracy(),
                "auroc": BinaryAUROC(),
                "ap": BinaryAveragePrecision(),
                "f1": BinaryF1Score(),
                "kappa": BinaryCohenKappa(),
                "spec": BinarySpecificity(),
            }
        )
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    # -- setup ---------------------------------------------------------------
    def setup(self, stage: str | None = None) -> None:
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and getattr(dm, "class_weights", None) is not None:
            self.class_weights = dm.class_weights.to(self.device)
            if hasattr(self.criterion, "weight"):
                self.criterion.weight = self.class_weights
        if self.cfg.train.get("ema", {}).get("enabled", False) and self._ema is None:
            self._ema = ModelEMA(
                self.model,
                decay=self.cfg.train.ema.get("decay", 0.9998),
                warmup_steps=self.cfg.train.ema.get("warmup_steps", 1000),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # -- train -----------------------------------------------------------
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch[0], batch[1]
        if self.mixup is not None:
            x, soft = self.mixup(x, y)
            loss = self.soft_criterion(self.model(x), soft)
        else:
            loss = self.criterion(self.model(x), y)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        opt = self.optimizers()
        if not isinstance(opt, list):
            self.log("train/lr", opt.param_groups[0]["lr"])
        return loss

    def on_before_zero_grad(self, optimizer) -> None:
        if self._ema is not None:
            self._ema.update(self.model)

    # -- eval ----------------------------------------------------------
    def _shared_eval(self, batch, metrics) -> torch.Tensor:
        x, y = batch[0], batch[1]
        model = self._ema.module if self._ema is not None else self.model
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        p_abnormal = logits.softmax(dim=1)[:, 1]
        metrics.update(p_abnormal, y)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        loss = self._shared_eval(batch, self.val_metrics)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx: int) -> None:
        self._shared_eval(batch, self.test_metrics)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        model = self._ema.module if self._ema is not None else self.model
        return model(batch[0]).softmax(dim=1)[:, 1]

    # -- optim -------------------------------------------------------------
    def configure_optimizers(self):
        t = self.cfg.train
        base_lr, wd = t.lr, t.get("weight_decay", 1e-4)
        head_mult = t.get("head_lr_mult", 10.0)
        head_ids = {id(p) for p in self.model.head.parameters()}
        decay, no_decay, head = [], [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in head_ids:
                head.append(p)
            elif p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": wd, "lr": base_lr},
                {"params": no_decay, "weight_decay": 0.0, "lr": base_lr},
                {"params": head, "weight_decay": wd, "lr": base_lr * head_mult},
            ],
            betas=(0.9, 0.999),
        )
        if t.get("scheduler", "cosine") == "none":
            return opt
        total = int(self.trainer.estimated_stepping_batches)
        warmup = int(t.get("warmup_frac", 0.05) * total)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(warmup, 1)
            import math

            progress = (step - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    # -- checkpoint plumbing --------------------------------------------
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self._ema is not None:
            checkpoint["ema"] = self._ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if "ema" in checkpoint:
            if self._ema is None:
                self._ema = ModelEMA(self.model)
            self._ema.load_state_dict(checkpoint["ema"])
