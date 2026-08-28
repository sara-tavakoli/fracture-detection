# Results

> Populate from `artifacts/eval/RESULTS.md` after a real training run. Always
> report the git SHA, config snapshot, and seed (`artifacts/env.json`,
> `artifacts/config.snapshot.yaml`).

## Protocol
- Split: official MURA patient-level train/valid **or** patient-disjoint fold 0 = test.
- Study probability = mean of per-image probabilities.
- Metrics on temperature-scaled probabilities (T fit on val).
- Threshold selected on val at target sensitivity (default 0.95).
- 95 % CIs = 2 000-resample bootstrap.
- Backbone / recipe: _fill in_ (e.g. `densenet169`, `--experiment strong`, 30 epochs).

## Headline (template)

| Level | AUROC | AP | Cohen's κ | Sensitivity | Specificity | Threshold |
|---|---|---|---|---|---|---|
| image | – | – | – | – | – | – |
| study | – | – | – | – | – | – |

- ECE pre / post temperature: – / – (T = –)
- AURC / excess-AURC: – / –
- Risk @ 80 % coverage: –

## Per-body-part κ (study-level) vs radiologist

| Body part | n | Prevalence | Model κ [95 % CI] | Radiologist κ (Rajpurkar 2018) |
|---|---|---|---|---|
| elbow | – | – | – | 0.710 |
| finger | – | – | – | 0.389 |
| forearm | – | – | – | 0.737 |
| hand | – | – | – | 0.851 |
| humerus | – | – | – | 0.600 |
| shoulder | – | – | – | 0.290 |
| wrist | – | – | – | 0.931 |
| **overall** | – | – | – | **0.778** |

## Ablations to report
| Run | Study AUROC | Study κ | Notes |
|---|---|---|---|
| `baseline_ce` | – | – | DenseNet-169 + weighted CE, no MixUp/EMA (MURA-paper style) |
| `strong` | – | – | focal + sampler + heavy aug + EMA + light MixUp |
| `model=convnext_tiny` | – | – | |
| + TTA (h-flip) | – | – | |
| + deep ensemble (×3) | – | – | |
| image-level vs study-level | – | – | quantify the study-aggregation gain |

## Figures
`artifacts/eval/`: `roc.png`, `pr.png`, `reliability.png`, `risk_coverage.png`,
`per_body_part_kappa.png`, plus a Grad-CAM montage from `scripts/explain.py`.

## Comparison points
The MURA paper's DenseNet-169 ensemble reaches overall κ ≈ 0.705 on the test set
(below the best radiologist's 0.778). Single-model numbers are lower; report the
protocol precisely.
