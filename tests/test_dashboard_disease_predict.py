"""POST /api/disease/predict"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from model_training.disease_model.exporter import DiseaseModelExporter
from model_training.disease_model.trainer import load_disease_config, train_image_pixels_model
from web_dashboard.app import create_app


@pytest.fixture
def app_with_disease_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = Path(__file__).resolve().parent.parent
    cfg = load_disease_config(root / "configs" / "disease_model_config.yaml")

    dm = cfg["disease_model"]
    labels = [str(x) for x in dm["class_names"]]
    resize_side = int(dm["training"]["resize_side"])

    dataset_root = tmp_path / "datasets" / "disease_model" / "v1"
    manifest_path = dataset_root / "manifest.parquet"
    img_dir = dataset_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for label_idx, label in enumerate(labels):
        class_dir = img_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            fname = f"img_{i:02d}.png"
            path = class_dir / fname
            base = 10 + label_idx * 20
            arr = np.full((resize_side, resize_side), base + i, dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(path)
            rows.append(
                {
                    "image_path": str(Path("images") / label / fname),
                    "disease_label": label,
                }
            )

    import pandas as pd

    pd.DataFrame(rows).to_parquet(manifest_path, index=False, engine="pyarrow")

    artifacts = train_image_pixels_model(
        cfg,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
    )
    out = DiseaseModelExporter(cfg).export(
        classifier=artifacts.classifier,
        evaluation_report=artifacts.report,
        output_root=tmp_path,
    )
    bundle_dir = Path(out["bundle_dir"])
    monkeypatch.setenv("AGROEDGE_DISEASE_BUNDLE", str(bundle_dir))
    monkeypatch.setenv("AGROEDGE_DISEASE_LOG", str(tmp_path / "disease_predictions.jsonl"))
    return create_app()


def test_disease_predict_ok(app_with_disease_bundle):
    c = app_with_disease_bundle.test_client()
    dim = 64
    payload = {"features": [0.0] * dim}
    r = c.post("/api/disease/predict", json=payload)
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert "disease_class_name" in data
    assert data["probabilities"] is not None


def test_disease_predict_bad_length(app_with_disease_bundle):
    c = app_with_disease_bundle.test_client()
    r = c.post("/api/disease/predict", json={"features": [1.0, 2.0]})
    assert r.status_code == 400
    assert r.get_json()["error"] == "feature_length_mismatch"


def test_disease_predict_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGROEDGE_DISEASE_BUNDLE", "/nonexistent/bundle/dir")
    c = create_app().test_client()
    r = c.post("/api/disease/predict", json={"features": [0.0] * 64})
    assert r.status_code == 503
    assert r.get_json()["error"] == "disease_bundle_unavailable"


def test_openapi_lists_disease_path():
    c = create_app().test_client()
    spec = c.get("/api/openapi.json").get_json()
    assert "/api/disease/predict" in spec["paths"]
    post = spec["paths"]["/api/disease/predict"]["post"]
    assert post["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "DiseasePredictRequest"
    )


def test_disease_predict_from_image_base64(app_with_disease_bundle):
    c = app_with_disease_bundle.test_client()
    # Create an 8x8 grayscale image -> the placeholder encoder produces 64 features.
    img = Image.new("L", (8, 8), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    r = c.post("/api/disease/predict", json={"image_base64": b64})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert "disease_class_name" in data


def test_disease_predict_image_too_large(app_with_disease_bundle, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGROEDGE_DISEASE_MAX_IMAGE_BYTES", "100")
    c = app_with_disease_bundle.test_client()
    # Build a base64 payload that decodes to 200 bytes.
    import base64 as _base64

    img_bytes = b"a" * 200
    b64 = _base64.b64encode(img_bytes).decode("ascii")
    r = c.post("/api/disease/predict", json={"image_base64": b64})
    assert r.status_code == 413
    data = r.get_json()
    assert data["error"] == "image_too_large"
    assert data["max_image_bytes"] == 100


def test_disease_last_endpoint_is_updated(app_with_disease_bundle):
    c = app_with_disease_bundle.test_client()

    r0 = c.get("/api/disease/last")
    assert r0.status_code == 503
    assert r0.get_json()["error"] == "no_disease_log"

    dim = 64
    post = c.post("/api/disease/predict", json={"features": [0.0] * dim})
    assert post.status_code == 200
    post_data = post.get_json()

    r1 = c.get("/api/disease/last")
    assert r1.status_code == 200
    last = r1.get_json()
    assert last["ok"] is True
    assert last["row"]["disease_class_name"] == post_data["disease_class_name"]


def test_disease_history_endpoint_returns_last_n(app_with_disease_bundle):
    c = app_with_disease_bundle.test_client()

    r0 = c.get("/api/disease/history")
    assert r0.status_code == 503
    assert r0.get_json()["error"] == "no_disease_log"

    dim = 64
    post1 = c.post("/api/disease/predict", json={"features": [0.0] * dim})
    assert post1.status_code == 200
    _ = post1.get_json()

    post2 = c.post("/api/disease/predict", json={"features": [1.0] * dim})
    assert post2.status_code == 200
    last = post2.get_json()

    r1 = c.get("/api/disease/history?limit=2")
    assert r1.status_code == 200
    data = r1.get_json()
    assert data["ok"] is True
    assert isinstance(data["rows"], list)
    assert len(data["rows"]) == 2
    assert data["rows"][0]["disease_class_name"] == last["disease_class_name"]
