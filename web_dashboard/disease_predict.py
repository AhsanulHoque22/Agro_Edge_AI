"""
POST /api/disease/predict — multiclass inference from either:

- A numeric feature vector: JSON {"features": [float, ...]}
- An image: JSON {"image_base64": "..."} or multipart/form-data with file field name "image"

The v0.1 disease model now uses a grayscale *pixel-vector embedding*.
Image preprocessing is a lightweight embedding step:
  - decode image
  - convert to grayscale
  - resize to N x N where N^2 = model feature_dimension (default 64 -> 8x8)
  - flatten and normalize to [0, 1]

When the real CV image pipeline is added, replace this preprocessing while
keeping the API contract stable.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, request

from edge_inference.disease_engine import DiseaseFeatureError, DiseaseInferenceEngine
from web_dashboard.disease_history import append_disease_prediction

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]


EXTENSION_KEY = "agroedge_disease_predict"

MAX_IMAGE_BYTES_DEFAULT = 1_000_000  # 1MB


def _resolve_max_image_bytes() -> int:
    raw = os.getenv("AGROEDGE_DISEASE_MAX_IMAGE_BYTES", "").strip()
    if not raw:
        return MAX_IMAGE_BYTES_DEFAULT
    try:
        n = int(raw)
    except ValueError:
        return MAX_IMAGE_BYTES_DEFAULT
    return max(1, min(n, 10_000_000))  # clamp: 1B..10MB


def _resolve_bundle_dir(project_root: Path) -> Path:
    raw = os.getenv("AGROEDGE_DISEASE_BUNDLE", "").strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = project_root / p
        return p
    return project_root / "model_export" / "disease_model" / "v0.1.0"


def _get_or_load_engine(app: Flask, project_root: Path) -> DiseaseInferenceEngine | None:
    ext: dict[str, Any] = app.extensions.setdefault(EXTENSION_KEY, {})
    if ext.get("loaded"):
        return ext.get("engine")

    ext["loaded"] = True
    bundle = _resolve_bundle_dir(project_root)
    model_path = bundle / "model.joblib"
    if not bundle.is_dir() or not model_path.is_file():
        ext["engine"] = None
        return None

    try:
        ext["engine"] = DiseaseInferenceEngine(bundle_dir=bundle)
    except (OSError, ValueError, FileNotFoundError):
        ext["engine"] = None
    return ext["engine"]


def _decode_image_base64(value: str) -> bytes:
    raw = value.strip()
    # Allow data URI: data:image/png;base64,....
    if raw.startswith("data:") and ";base64," in raw:
        raw = raw.split(",", 1)[1]
    return base64.b64decode(raw, validate=True)


def _image_bytes_to_feature_vector(image_bytes: bytes, feature_dim: int) -> list[float]:
    if Image is None:
        raise RuntimeError("Pillow not available: cannot process image")

    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("L")

    # Default: 64 -> 8x8.
    side = int(round(feature_dim ** 0.5))
    side = max(1, side)

    resized = img.resize((side, side), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32).reshape(-1) / 255.0

    if arr.shape[0] >= feature_dim:
        vec = arr[:feature_dim]
    else:
        vec = np.pad(arr, (0, feature_dim - arr.shape[0]), mode="constant")

    return [float(x) for x in vec.tolist()]


def register_disease_predict_routes(app: Flask, project_root: Path) -> None:
    @app.post("/api/disease/predict")
    def disease_predict() -> tuple[Any, int]:
        engine = _get_or_load_engine(app, project_root)
        max_bytes = _resolve_max_image_bytes()
        if engine is None:
            bundle = _resolve_bundle_dir(project_root)
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "disease_bundle_unavailable",
                        "path": str(bundle),
                        "hint": "Run scripts/train_disease_model.py or set AGROEDGE_DISEASE_BUNDLE.",
                    }
                ),
                503,
            )

        body = request.get_json(silent=True)

        features: list[float] | None = None
        input_kind: str | None = None
        image_sha256: str | None = None

        if isinstance(body, dict) and "features" in body:
            raw_features = body.get("features")
            if not isinstance(raw_features, list):
                return (
                    jsonify(
                        {
                            "ok": False,
                            "error": "missing_features",
                            "expected_feature_dim": engine.feature_dimension,
                        }
                    ),
                    400,
                )
            features = raw_features  # type: ignore[assignment]
            input_kind = "features"
        elif isinstance(body, dict) and "image_base64" in body:
            try:
                image_bytes = _decode_image_base64(str(body.get("image_base64")))
                if len(image_bytes) > max_bytes:
                    return (
                        jsonify(
                            {
                                "ok": False,
                                "error": "image_too_large",
                                "max_image_bytes": max_bytes,
                            }
                        ),
                        413,
                    )
                image_sha256 = hashlib.sha256(image_bytes).hexdigest()
                features = _image_bytes_to_feature_vector(image_bytes, engine.feature_dimension)
                input_kind = "image_base64"
            except base64.binascii.Error:
                return (
                    jsonify(
                        {
                            "ok": False,
                            "error": "invalid_image_base64",
                            "expected_feature_dim": engine.feature_dimension,
                        }
                    ),
                    400,
                )
            except Exception as exc:
                return (
                    jsonify(
                        {
                            "ok": False,
                            "error": "invalid_image",
                            "message": str(exc),
                        }
                    ),
                    400,
                )
        elif "image" in (request.files or {}):
            try:
                file_storage = request.files["image"]
                image_bytes = file_storage.read()
                if len(image_bytes) > max_bytes:
                    return (
                        jsonify(
                            {
                                "ok": False,
                                "error": "image_too_large",
                                "max_image_bytes": max_bytes,
                            }
                        ),
                        413,
                    )
                image_sha256 = hashlib.sha256(image_bytes).hexdigest()
                features = _image_bytes_to_feature_vector(image_bytes, engine.feature_dimension)
                input_kind = "multipart_image"
            except Exception as exc:
                return (
                    jsonify(
                        {
                            "ok": False,
                            "error": "invalid_image",
                            "message": str(exc),
                        }
                    ),
                    400,
                )
        else:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "missing_request_fields",
                        "hint": 'Send JSON {"features": [...]} or {"image_base64": "..."} '
                        'or multipart/form-data with file field name "image".',
                        "expected_feature_dim": engine.feature_dimension,
                    }
                ),
                400,
            )

        try:
            out = engine.predict_vector(features or [])
        except DiseaseFeatureError as exc:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "feature_length_mismatch",
                        "message": str(exc),
                        "expected_feature_dim": engine.feature_dimension,
                    }
                ),
                400,
            )

        # Persist the last prediction for dashboard viewing.
        # (JSONL so it remains robust across crashes / restarts.)
        append_disease_prediction(
            project_root,
            {
                "input_kind": input_kind,
                "image_sha256": image_sha256,
                "features_length": len(features) if isinstance(features, list) else None,
                "disease_class_name": out.disease_class_name,
                "disease_class_index": out.disease_class_index,
                "probabilities": out.probabilities,
                "model_version": out.model_version,
                "inferred_at_utc": out.inferred_at_utc,
            },
        )

        return (
            jsonify(
                {
                    "ok": True,
                    "disease_class_name": out.disease_class_name,
                    "disease_class_index": out.disease_class_index,
                    "probabilities": out.probabilities,
                    "model_version": out.model_version,
                    "inferred_at_utc": out.inferred_at_utc,
                }
            ),
            200,
        )
