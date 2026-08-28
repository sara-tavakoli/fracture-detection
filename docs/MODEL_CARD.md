# Model Card — Fracture Detection (MURA)

Following Mitchell et al., *Model Cards for Model Reporting* (2019).

## Model details
- **Task:** binary classification of musculoskeletal radiographs as `normal` (0)
  or `abnormal` (1). The clinical unit is the **study** (all views of one body
  part for one patient); study probability = mean of per-image probabilities.
- **Body parts:** elbow, finger, forearm, hand, humerus, shoulder, wrist.
- **Architecture:** ImageNet-pretrained `timm` backbone (default DenseNet-169,
  the MURA-paper baseline) + single linear head with dropout; optional weight EMA.
- **Training:** AdamW, cosine schedule + warmup, class-balanced focal loss
  (`gamma≈1.5`), inverse-frequency `body_part × label` sampler, X-ray-appropriate
  augmentation (CLAHE, random gamma, small affine, horizontal flip), mixed precision.
- **Post-hoc:** temperature scaling on the validation split; decision threshold
  selected on validation at a target sensitivity (default 0.95).

## Intended use
- **Intended:** research on abnormality detection, calibration, uncertainty, and
  explainability for radiographs; teaching; benchmarking against MURA.
- **Users:** ML researchers and students.

## Out-of-scope / prohibited use
- **Any clinical use** — diagnosis, screening, triage, or decision support.
- Body regions or modalities outside training (spine, pelvis, lower limb, chest,
  CT, paediatric growth plates).
- Interpreting "abnormal" as "fracture": MURA labels include hardware,
  degenerative change, effusions, lesions, and post-operative changes.

Not FDA/CE cleared; not prospectively validated.

## Factors
Performance varies strongly **by body part** (radiologist κ ranges 0.29 for
shoulder to 0.93 for wrist) and by view count per study, projection, image
quality, presence of hardware/casts, and acquisition device. Report per-body-part
κ, not just an aggregate.

## Metrics
On a held-out patient-disjoint test fold (or the official MURA validation set):
- Image-level and study-level: AUROC, AP, Cohen's κ, sensitivity, specificity,
  PPV/NPV — each with bootstrap 95 % CIs.
- Per-body-part κ vs the published radiologist κ (Rajpurkar et al. 2018).
- Calibration: ECE and reliability, pre/post temperature scaling.
- Selective prediction: risk–coverage curve, AURC, excess-AURC.
- Operating point: threshold at target sensitivity + achieved specificity.

Populate `docs/RESULTS.md` from `artifacts/eval/RESULTS.md`.

## Training data
MURA v1.1 (Stanford ML Group) — ~40 k images / ~14 k studies / ~12 k patients,
upper-extremity radiographs, labelled `abnormal`/`normal` at the study level from
the original radiology reports. See `docs/DATASET.md`.

## Ethical considerations
- **Automation bias / missed fracture:** a confident false-negative could delay
  care. The threshold is deliberately tuned for sensitivity; the service surfaces
  entropy and warnings and never emits a bare "diagnosis".
- **Label noise:** report-derived labels are imperfect and were not re-read; some
  body parts have low inter-rater agreement even among radiologists.
- **Distribution:** single institution; generalisation to other PACS, detectors,
  and populations is untested.
- **Privacy:** MURA is de-identified for research; do not upload identifiable
  images to any demo.

## Caveats and recommendations
- Re-fit temperature and re-select the threshold on any new data source.
- Always report per-body-part κ with CIs.
- Treat Grad-CAM as a sanity check for shortcut learning (collimator, markers,
  casts), not as localisation ground truth.
