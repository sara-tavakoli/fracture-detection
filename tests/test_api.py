from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture
def client():
    from fracture.serve import api
    from fracture.serve.inference import StudyPrediction

    class _FakePred:
        threshold = 0.5

        def predict_study(self, raws, body_part=None, mc_dropout_samples=0):
            return StudyPrediction(
                abnormal_probability=0.42,
                decision="normal",
                threshold=0.5,
                per_image=[0.4, 0.44],
                n_images=len(raws),
                body_part=body_part,
                warnings=[],
            )

    api.get_predictor.cache_clear()
    fake = _FakePred()
    api.app.dependency_overrides[api.get_predictor] = lambda: fake
    try:
        yield fastapi_testclient.TestClient(api.app)
    finally:
        api.app.dependency_overrides.clear()


def _png() -> bytes:
    buf = io.BytesIO()
    Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8")).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_model_card_has_body_parts_and_reference_kappa(client):
    body = client.get("/model-card").json()
    assert len(body["body_parts"]) == 7
    assert "reference_radiologist_kappa" in body
    assert "disclaimer" in body


def test_predict_multi_image_study(client):
    files = [("files", ("a.png", _png(), "image/png")), ("files", ("b.png", _png(), "image/png"))]
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["n_images"] == 2
    assert body["decision"] in {"normal", "abnormal"}
    assert "disclaimer" in body


def test_predict_requires_a_file(client):
    r = client.post("/predict")
    assert r.status_code in (400, 422)
