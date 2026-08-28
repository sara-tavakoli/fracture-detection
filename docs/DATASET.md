# Dataset Card — MURA v1.1

## Overview
- **Name:** MURA (musculoskeletal radiographs).
- **Source:** Rajpurkar, P., Irvin, J., Bagul, A., et al. (2018). *MURA: Large
  Dataset for Abnormality Detection in Musculoskeletal Radiographs.* MIDL 2018.
  arXiv:1712.06957. Stanford ML Group.
- **Access:** requires agreeing to the Stanford research-use terms at
  <https://stanfordmlgroup.github.io/competitions/mura/>. **Cannot be
  auto-downloaded.**
- **Size:** ~40,561 images from 14,863 studies of 12,173 patients.
- **Regions:** 7 upper-extremity study types — elbow, finger, forearm, hand,
  humerus, shoulder, wrist.
- **Format:** PNG, variable resolution, grayscale.

## Labels
Binary per **study**: `abnormal` (1) / `normal` (0), assigned from the original
radiology report by the dataset authors. Every image in a study inherits the
study label. There are **no bounding boxes or pixel masks**.

## Directory structure
```
MURA-v1.1/
  train_image_paths.csv          # one image path per line
  valid_image_paths.csv
  train_labeled_studies.csv      # study_path,label
  valid_labeled_studies.csv
  train/XR_WRIST/patient00001/study1_positive/image1.png
  ...
```
`scripts/prepare_mura.py` parses body part from `XR_<PART>`, patient from
`patientNNNNN`, and study from the parent directory, emitting a unified
`data/mura/metadata.csv` with `image_id, study_id, patient_id, body_part, label,
filepath, split`.

## Known biases and hazards
| Issue | Consequence |
|---|---|
| **Report-derived labels, not re-read** | Label noise; the paper reports radiologist κ of only 0.29 (shoulder) / 0.39 (finger). |
| **"Abnormal" is broad** | Fractures, hardware, degenerative change, effusions, lesions, post-op — a model cannot be assumed to be a fracture detector. |
| **Single institution** | Detector/PACS/processing differ elsewhere; unmodelled domain shift. |
| **Per-body-part difficulty varies hugely** | Aggregate metrics hide this; always break down by body part. |
| **Multiple views per study** | Split on `patient_id` (enforced here). Study-level aggregation matters. |
| **Laterality markers, collimation, casts** | Shortcut-learning risk; inspect Grad-CAM. |

## Official vs cross-validation splits
- MURA ships a fixed patient-level train/valid split. `split_frames` honours it
  when a `split` column is present (val is carved from official train; official
  valid becomes the test set).
- Otherwise `StratifiedGroupKFold(n_splits=5)` on `patient_id`, stratified by
  `body_part × label`; `split_summary.json` asserts zero patient overlap.

## Preprocessing
- DICOM (if you supply your own): MONOCHROME1 inversion + 1–99 percentile window.
- Resize longest side to `image_size`, pad to square, ImageNet normalisation
  (grayscale replicated to 3 channels).

## Ethical use
De-identified, research-only under the Stanford agreement. No re-identification.
Do not upload identifiable patient radiographs to demos.
