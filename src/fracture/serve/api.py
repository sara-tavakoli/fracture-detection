"""FastAPI service for study-level radiograph abnormality detection.

    uvicorn fracture.serve.api:app --reload

Environment:
    FRACTURE_CKPT        checkpoint path (default: artifacts/best.ckpt)
    FRACTURE_DEVICE      auto | cpu | cuda | mps
    FRACTURE_THRESHOLD   decision threshold (default 0.5)
    FRACTURE_TTA         1/0 (default 1)
    FRACTURE_CAM_METHOD  gradcam++ | gradcam | xgradcam
"""

from __future__ import annotations

import base64
import os
from functools import lru_cache
from io import BytesIO

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fracture import BODY_PARTS, RADIOLOGIST_KAPPA, __version__

app = FastAPI(
    title="MSK Radiograph Abnormality Detector",
    version=__version__,
    description="Research prototype (MURA-style). NOT a medical device.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MAX_BYTES = 25 * 1024 * 1024
DISCLAIMER = (
    "Research prototype trained on the MURA dataset. Not a medical device; not for "
    "diagnostic use. Abnormality != fracture: MURA labels cover any abnormality "
    "(fracture, hardware, degenerative change, lesion). Always defer to a radiologist."
)


class StudyResponse(BaseModel):
    abnormal_probability: float = Field(..., ge=0.0, le=1.0)
    decision: str
    threshold: float
    per_image: list[float]
    n_images: int
    body_part: str | None = None
    epistemic_uncertainty: float | None = None
    warnings: list[str] = []
    disclaimer: str = DISCLAIMER


class ExplainResponse(StudyResponse):
    cam_method: str
    overlays_png_base64: list[str]


@lru_cache(maxsize=1)
def get_predictor():
    from fracture.serve.inference import FracturePredictor

    ckpt = os.environ.get("FRACTURE_CKPT", "artifacts/best.ckpt")
    if not os.path.exists(ckpt):
        raise RuntimeError(f"checkpoint '{ckpt}' not found; set FRACTURE_CKPT or train a model")
    return FracturePredictor(
        ckpt,
        threshold=float(os.environ.get("FRACTURE_THRESHOLD", "0.5")),
        device=os.environ.get("FRACTURE_DEVICE", "auto"),
        use_tta=os.environ.get("FRACTURE_TTA", "1") == "1",
    )


@lru_cache(maxsize=1)
def get_explainer():
    from fracture.explain.cam import FractureExplainer

    predictor = get_predictor()
    method = os.environ.get("FRACTURE_CAM_METHOD", "gradcam++")
    return FractureExplainer(predictor.model, method=method, device=str(predictor.device)), method


async def _read_all(files: list[UploadFile]) -> list[bytes]:
    out = []
    for f in files:
        raw = await f.read()
        if len(raw) > MAX_BYTES:
            raise HTTPException(413, f"{f.filename} too large (max 25 MB)")
        out.append(raw)
    if not out:
        raise HTTPException(400, "upload at least one image")
    return out


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/model-card")
def model_card() -> dict:
    return {
        "task": "binary abnormality detection on musculoskeletal radiographs",
        "unit_of_analysis": "study (mean of per-image probabilities)",
        "body_parts": list(BODY_PARTS),
        "reference_radiologist_kappa": RADIOLOGIST_KAPPA,
        "training_data": "MURA (Stanford), patient-level split",
        "not_intended_use": "clinical diagnosis, triage, or treatment decisions",
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict", response_model=StudyResponse)
async def predict(
    files: list[UploadFile] = File(...),
    body_part: str | None = None,
    mc_dropout_samples: int = 0,
    predictor=Depends(get_predictor),
) -> StudyResponse:
    raws = await _read_all(files)
    try:
        pred = predictor.predict_study(raws, body_part=body_part, mc_dropout_samples=mc_dropout_samples)
    except Exception as exc:
        raise HTTPException(400, f"inference failed: {exc}") from exc
    return StudyResponse(**pred.as_dict())


@app.post("/explain", response_model=ExplainResponse)
async def explain(
    files: list[UploadFile] = File(...),
    body_part: str | None = None,
    predictor=Depends(get_predictor),
) -> ExplainResponse:
    import torch
    from PIL import Image

    from fracture.serve.inference import load_image_from_any

    raws = await _read_all(files)
    pred = predictor.predict_study(raws, body_part=body_part)
    explainer, method = get_explainer()

    overlays: list[str] = []
    for raw in raws:
        arr = load_image_from_any(raw)
        x = predictor.eval_tf(image=arr)["image"]
        res = explainer.explain(torch.as_tensor(x))
        buf = BytesIO()
        Image.fromarray(res.overlay).save(buf, format="PNG")
        overlays.append(base64.b64encode(buf.getvalue()).decode("ascii"))

    return ExplainResponse(**pred.as_dict(), cam_method=method, overlays_png_base64=overlays)
