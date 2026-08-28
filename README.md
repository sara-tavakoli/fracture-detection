# Fracture Detection — musculoskeletal radiograph abnormality classification

[![CI](https://github.com/OWNER/fracture-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/fracture-detection/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A reproducible research pipeline for **binary abnormality detection on upper-extremity
radiographs**, built around the MURA benchmark (Stanford, Rajpurkar et al. 2018).
The design mirrors how MURA is actually scored — **the study, not the image, is the
unit of analysis** — and pairs it with the trust-oriented machinery a medical-imaging
model needs.

| Concern | What this repo does |
|---|---|
| **Patient leakage** | `StratifiedGroupKFold` on `patient_id`, stratified by `body_part × label`; a patient's studies never split across folds. The official MURA train/valid split is honoured when present. |
| **Study-level evaluation** | Per-image probabilities are mean-pooled per study (as in the MURA paper); metrics reported at **both** image and study level. |
| **The MURA metric** | Per-body-part Cohen's κ, plotted against the published **radiologist κ** for each of the 7 study types. |
| **Class / body-part imbalance** | Inverse-frequency weights over `body_part × label`, class-balanced focal loss, weighted sampler. |
| **Honest metrics** | AUROC, AP, κ, sensitivity, specificity, PPV/NPV — each with **bootstrap 95 % CIs**. |
| **Operating point** | Decision threshold chosen on validation at a target sensitivity (default 95 %) — a miss on a fracture is the costly error. |
| **Calibration** | Temperature scaling fit on validation; ECE + reliability diagram pre/post. |
| **Uncertainty** | MC-Dropout epistemic term, predictive entropy, selective-prediction risk–coverage (AURC / excess-AURC). |
| **Localisation** | Grad-CAM / Grad-CAM++ / XGrad-CAM overlays to check the model attends to the pathology, not the collimator edge or laterality marker. |
| **Inputs** | 8-bit images **and DICOM** (`pydicom`), with MONOCHROME1 inversion and percentile windowing. |
| **Reproducibility** | Seeded, deterministic kernels, composable OmegaConf configs, env + git-SHA capture, MLflow, pinned deps, CI smoke-training. |

> ⚠️ **Not a medical device.** Trained on a public research dataset; not validated
> on a prospective clinical cohort. "Abnormal" in MURA is broader than "fracture"
> (it also covers hardware, degenerative change, lesions, effusions). Do not use
> for diagnosis or triage. See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

---

## Architecture

```
              configs/ (OmegaConf, composable: data/ model/ experiment/)
                     │
     ┌───────────────┼──────────────────────────────┐
     ▼               ▼                              ▼
 data/ (patient    models/ (timm backbone —      metrics/    uncertainty/   explain/
 K-fold, X-ray     DenseNet-169 default — +      (binary CIs, (MC-dropout,   (Grad-CAM
 aug, DICOM,       linear head, EMA, focal),     per-part κ,  risk-coverage)  family)
 balanced sampler) LightningModule               study agg,
     └───────────────┴───────────────┬────── temp scaling)
                                     ▼
                 scripts/  prepare_mura · train · evaluate · explain · export
                                     ▼
                 serve/  FastAPI  (/predict multi-file study, /explain, /model-card)
```

Backbones (`model=<name>`): `densenet169` (default, MURA baseline), `densenet201`,
`effnetv2_s`, `convnext_tiny`, `resnet50` — or any `timm` name.

## Quickstart

```bash
git clone https://github.com/OWNER/fracture-detection && cd fracture-detection
make venv install

# ---- Option A: whole pipeline on synthetic data (no download / no agreement) ----
make smoke                       # synthetic MURA-like data → 1-epoch train → full eval

# ---- Option B: real MURA ----
# 1. Request access + download from https://stanfordmlgroup.github.io/competitions/mura/
python scripts/prepare_mura.py --mura-root /path/to/MURA-v1.1 --out data/mura
python scripts/prepare_splits.py --data-dir data/mura            # patient-disjoint folds
python scripts/train.py --experiment strong                      # or --experiment baseline_ce
python scripts/evaluate.py --checkpoint artifacts/best.ckpt      # image + study metrics, per-part κ
python scripts/explain.py  --checkpoint artifacts/best.ckpt --images data/mura --limit 24
python scripts/export_model.py --checkpoint artifacts/best.ckpt --onnx
```

### Configuration

```bash
python scripts/train.py model=convnext_tiny data.image_size=384 train.lr=8e-5
python scripts/train.py --experiment baseline_ce      # DenseNet-169 + weighted CE (MURA-paper style)
python scripts/train.py --experiment strong           # focal + sampler + heavy aug + EMA + light MixUp
python scripts/train.py --experiment fast_dev         # CI smoke recipe
```

Each run writes `artifacts/{config.snapshot.yaml, env.json, train.log}`, MLflow
metrics, and the best checkpoint.

## Serving

```bash
FRACTURE_CKPT=artifacts/best.ckpt make serve          # uvicorn on :8000
docker compose up --build
```

| Endpoint | Purpose |
|---|---|
| `POST /predict` | multi-file **study** upload → mean study probability, per-image probabilities, decision at the configured threshold, optional MC-dropout epistemic term, warnings |
| `POST /explain` | as `/predict` plus a Grad-CAM++ overlay PNG per view |
| `GET /model-card` | intended use, the 7 body parts, reference radiologist κ |
| `GET /health` | liveness |

Client: [`frontend/index.html`](frontend/index.html) (multi-file study upload).

## Evaluation output (`artifacts/eval/`)

- `report_image.json`, `report_study.json` — metrics + bootstrap CIs + confusion
- `per_body_part_kappa.csv` + `per_body_part_kappa.png` — model vs radiologist κ
- `calibration.json`, `selective.json`, `operating_point.json`
- `roc.png`, `pr.png`, `reliability.png`, `risk_coverage.png`
- `RESULTS.md`

## Repository layout

```
src/fracture/
  data/         patient K-fold splits, DICOM/PNG dataset, X-ray albumentations, DataModule
  models/       timm backbones (DenseNet default), LightningModule, EMA, MixUp
  losses/       class-balanced focal, soft-target CE
  metrics/      binary report + bootstrap CIs, per-body-part κ, study aggregation, temp scaling
  uncertainty/  MC-dropout, entropy, risk-coverage
  explain/      Grad-CAM family
  serve/        study-level inference (TTA, DICOM) + FastAPI
  utils/        seeding, config, logging
scripts/        prepare_mura · make_synthetic_data · prepare_splits · train · evaluate · explain · export
configs/        config.yaml + data/ model/ experiment/
tests/          data, models, losses, metrics, calibration, config, api, end-to-end pipeline
```

## Testing

```bash
make test                 # full suite
pytest -m "not slow"      # skip Lightning end-to-end tests
make lint type            # ruff + mypy (both gate CI)
```

CI runs ruff, `ruff format --check`, mypy, the unit suite, and a full synthetic
**train → evaluate → export** on every push.

## Design notes & limitations

- **"Abnormal" ≠ "fracture".** MURA study labels mark *any* abnormality. This
  model is a fracture-*screening* aid at best, and only a research prototype.
- **Upper extremity only** — elbow, finger, forearm, hand, humerus, shoulder,
  wrist. No spine, pelvis, lower limb, chest, paediatric growth plates.
- **No lesion-level annotations** in MURA, so Grad-CAM is a qualitative check,
  not a validated localiser.
- **Label noise:** MURA labels come from the radiology report, not a re-read;
  the paper's own radiologist κ (shoulder 0.29, finger 0.39) shows how hard some
  parts are. The per-part κ plot is the honest way to read model performance.
- The synthetic generator is for CI only — its "fracture line" is a cartoon.

## Citation

Cite this code via [`CITATION.cff`](CITATION.cff) and the dataset:

> Rajpurkar, P. et al. *MURA: Large Dataset for Abnormality Detection in
> Musculoskeletal Radiographs.* MIDL 2018. arXiv:1712.06957.

## License

MIT — see [`LICENSE`](LICENSE).
