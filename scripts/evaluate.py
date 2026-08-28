#!/usr/bin/env python
"""Image- and study-level evaluation with uncertainty quantification.

Writes to ``<output>/eval/``:
    report_image.json / report_study.json  -- binary metrics + bootstrap 95% CIs
    per_body_part_kappa.csv                 -- MURA metric vs radiologist kappa
    calibration.json                        -- ECE pre/post temperature scaling
    selective.json                          -- AURC / excess-AURC, risk @ coverage
    operating_point.json                    -- threshold @ target sensitivity
    *.png                                   -- ROC, PR, reliability, risk-coverage,
                                               per-body-part kappa bar
    RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fracture import RADIOLOGIST_KAPPA
from fracture.data.datamodule import MURADataModule
from fracture.metrics.calibration import TemperatureScaler, reliability_bins
from fracture.metrics.classification import (
    aggregate_studies,
    choose_threshold,
    compute_report,
    per_body_part_kappa,
)
from fracture.models.classifier import FractureClassifier
from fracture.uncertainty.selective import predictive_entropy, risk_coverage_curve
from fracture.utils.config import load_config
from fracture.utils.seed import seed_everything


@torch.no_grad()
def _collect(model, loader, device):
    model.eval().to(device)
    logits, ys, studies, parts = [], [], [], []
    for batch in loader:
        x, y, meta = batch
        logits.append(model(x.to(device)).cpu())
        ys.append(y)
        studies += list(meta["study_id"])
        parts += list(meta["body_part"])
    return torch.cat(logits).numpy(), torch.cat(ys).numpy(), studies, parts


def _plots(out_dir: Path, y_img, s_img, study_df, temp, bp_kappa):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(y_img, s_img)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label="image-level")
    sfpr, stpr, _ = roc_curve(study_df["label"], study_df["score"])
    ax.plot(sfpr, stpr, label="study-level")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="1 - specificity", ylabel="sensitivity", title="ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roc.png", dpi=150)
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_img, s_img)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec)
    ax.set(xlabel="recall", ylabel="precision", title="Precision-Recall (image-level)")
    fig.tight_layout()
    fig.savefig(out_dir / "pr.png", dpi=150)
    plt.close(fig)

    rb = reliability_bins(y_img, np.stack([1 - s_img, s_img], 1), n_bins=15)
    centres = 0.5 * (rb.bin_edges[:-1] + rb.bin_edges[1:])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.bar(centres, rb.bin_acc, width=1 / 15, alpha=0.7, edgecolor="k", label="accuracy")
    ax.plot(centres, rb.bin_conf, "o-", color="crimson", label="confidence")
    ax.set(title=f"Reliability (ECE={rb.ece:.3f}, T={temp:.2f})", xlabel="confidence", ylabel="accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "reliability.png", dpi=150)
    plt.close(fig)

    correct = ((s_img >= 0.5).astype(int) == y_img).astype(float)
    rc = risk_coverage_curve(correct, np.abs(s_img - 0.5) * 2)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rc.coverage, rc.risk)
    ax.set(xlabel="coverage", ylabel="selective risk", title=f"Risk-coverage (AURC={rc.aurc:.4f})")
    fig.tight_layout()
    fig.savefig(out_dir / "risk_coverage.png", dpi=150)
    plt.close(fig)

    parts = [r for r in bp_kappa["body_part"] if r != "overall"]
    model_k = [bp_kappa.set_index("body_part").loc[p, "kappa"] for p in parts]
    rad_k = [RADIOLOGIST_KAPPA.get(p, np.nan) for p in parts]
    x = np.arange(len(parts))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, model_k, width=0.4, label="model")
    ax.bar(x + 0.2, rad_k, width=0.4, label="radiologist (Rajpurkar 2018)")
    ax.set_xticks(x)
    ax.set_xticklabels(parts, rotation=30)
    ax.set_ylabel("Cohen's kappa")
    ax.set_title("Per-body-part kappa (study-level)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_body_part_kappa.png", dpi=150)
    plt.close(fig)


def _results_md(img_rep, study_rep, calib, selective, bp_kappa, out_path):
    def line(name, r):
        return (
            f"| {name} | {r['auroc']:.4f} | {r['ap']:.4f} | {r['kappa']:.4f} "
            f"| {r['sensitivity']:.3f} | {r['specificity']:.3f} | {r['threshold']:.3f} |"
        )

    lines = [
        "# Results",
        "",
        "| Level | AUROC | AP | Cohen's kappa | Sensitivity | Specificity | Threshold |",
        "|---|---|---|---|---|---|---|",
        line("image", img_rep),
        line("study", study_rep),
        "",
        (
            f"- ECE pre / post temperature: **{calib['ece_pre']:.4f} / "
            f"{calib['ece_post']:.4f}** (T={calib['temperature']:.3f})"
        ),
        f"- AURC / excess-AURC: **{selective['aurc']:.4f} / {selective['eaurc']:.4f}**",
        f"- Risk @ 80% coverage: **{selective['risk_at_coverage_0.8']:.4f}**",
        "",
        "## Per-body-part kappa (study-level) vs radiologist",
        "",
        "| Body part | n | Prevalence | Model kappa | Radiologist kappa |",
        "|---|---|---|---|---|",
    ]
    for _, row in bp_kappa.iterrows():
        rad = RADIOLOGIST_KAPPA.get(row["body_part"], float("nan"))
        lines.append(
            f"| {row['body_part']} | {int(row['n'])} | {row['prevalence']:.3f} "
            f"| {row['kappa']:.3f} | {rad:.3f} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--output-dir", default="artifacts")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--target-sensitivity", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.overrides, experiment=args.experiment)
    seed_everything(cfg.seed, deterministic=False)
    target_sens = args.target_sensitivity or cfg.get("eval", {}).get("target_sensitivity", 0.95)
    device = (
        torch.device("cuda")
        if (args.device in ("auto", "cuda") and torch.cuda.is_available())
        else torch.device("mps")
        if (args.device in ("auto", "mps") and torch.backends.mps.is_available())
        else torch.device("cpu")
    )

    dm = MURADataModule(
        data_dir=cfg.data.data_dir,
        metadata_csv=cfg.data.metadata_csv,
        image_subdir=cfg.data.image_subdir,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        aug_strength="light",
        balanced_sampler=False,
        n_folds=cfg.data.n_folds,
        test_fold=cfg.data.test_fold,
        val_fold=cfg.data.val_fold,
        seed=cfg.seed,
    )
    dm.setup("test")

    model = FractureClassifier.load_from_checkpoint(
        args.checkpoint, cfg=cfg, map_location="cpu", strict=False
    )
    core = model._ema.module if model._ema is not None else model.model

    val_logits, val_y, _, _ = _collect(core, dm.val_dataloader(), device)
    test_logits, test_y, test_studies, test_parts = _collect(core, dm.test_dataloader(), device)

    scaler = TemperatureScaler().fit(torch.tensor(val_logits), torch.tensor(val_y))
    temp = scaler.temperature
    s_img_pre = torch.tensor(test_logits).softmax(1)[:, 1].numpy()
    s_img = (torch.tensor(test_logits) / temp).softmax(1)[:, 1].numpy()

    ece_pre = reliability_bins(test_y, np.stack([1 - s_img_pre, s_img_pre], 1)).ece
    ece_post = reliability_bins(test_y, np.stack([1 - s_img, s_img], 1)).ece

    # choose threshold on the validation set at the target sensitivity
    s_val = (torch.tensor(val_logits) / temp).softmax(1)[:, 1].numpy()
    threshold = choose_threshold(val_y, s_val, target_sensitivity=target_sens)

    img_df = pd.DataFrame(
        {"study_id": test_studies, "body_part": test_parts, "label": test_y, "score": s_img}
    )
    study_df = aggregate_studies(img_df, score_col="score", how="mean")

    img_rep = compute_report(test_y, s_img, threshold=threshold, n_bootstrap=args.n_bootstrap, seed=cfg.seed)
    study_rep = compute_report(
        study_df["label"].to_numpy(),
        study_df["score"].to_numpy(),
        threshold=threshold,
        n_bootstrap=args.n_bootstrap,
        seed=cfg.seed,
    )
    bp_kappa = per_body_part_kappa(study_df, threshold=threshold)

    correct = ((s_img >= threshold).astype(int) == test_y).astype(float)
    conf = np.abs(s_img - 0.5) * 2
    rc = risk_coverage_curve(correct, conf)
    order = np.argsort(-conf)
    cov80 = int(0.8 * len(correct))
    selective = {
        "aurc": rc.aurc,
        "eaurc": rc.eaurc,
        "risk_at_coverage_0.8": float(1 - correct[order[:cov80]].mean()),
        "mean_entropy": float(predictive_entropy(np.stack([1 - s_img, s_img], 1)).mean()),
    }

    out_dir = Path(args.output_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report_image.json").write_text(json.dumps(img_rep.to_dict(), indent=2))
    (out_dir / "report_study.json").write_text(json.dumps(study_rep.to_dict(), indent=2))
    (out_dir / "calibration.json").write_text(
        json.dumps({"temperature": temp, "ece_pre": ece_pre, "ece_post": ece_post}, indent=2)
    )
    (out_dir / "selective.json").write_text(json.dumps(selective, indent=2))
    (out_dir / "operating_point.json").write_text(
        json.dumps({"target_sensitivity": target_sens, "threshold": threshold}, indent=2)
    )
    bp_kappa.to_csv(out_dir / "per_body_part_kappa.csv", index=False)
    _plots(out_dir, test_y, s_img, study_df, temp, bp_kappa)
    _results_md(
        img_rep.to_dict(),
        study_rep.to_dict(),
        {"ece_pre": ece_pre, "ece_post": ece_post, "temperature": temp},
        selective,
        bp_kappa,
        out_dir / "RESULTS.md",
    )

    print((out_dir / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
