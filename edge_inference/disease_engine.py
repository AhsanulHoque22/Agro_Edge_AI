"""
Load exported disease model bundle and run multiclass inference.

The **sanity** bundle expects a fixed-length numeric vector (see
``feature_dimension`` in ``configs/disease_model_config.yaml``). Future image
pipelines should embed CNN outputs into the same vector size or extend this
engine with a preprocessor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np


class DiseaseFeatureError(ValueError):
    """Input vector length or type mismatch."""


@dataclass(frozen=True)
class DiseaseInferenceResult:
    disease_class_index: int
    disease_class_name: str
    probabilities: dict[str, float] | None
    model_version: str
    inferred_at_utc: str


class DiseaseInferenceEngine:
    """Sklearn classifier bundle from ``DiseaseModelExporter``."""

    def __init__(self, bundle_dir: Path) -> None:
        self.bundle_dir = bundle_dir
        model_path = bundle_dir / "model.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Disease model not found: {model_path}")
        payload = joblib.load(model_path)
        for key in ("classifier", "class_names"):
            if key not in payload:
                raise ValueError(f"Invalid disease payload, missing {key}")
        self._clf = payload["classifier"]
        self._class_names = [str(x) for x in payload["class_names"]]
        self._feature_dim = int(payload.get("feature_dimension", 0))
        meta_path = bundle_dir / "metadata.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._version = str(meta.get("version", payload.get("version", "unknown")))
        else:
            self._version = str(payload.get("version", "unknown"))

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    @property
    def feature_dimension(self) -> int:
        return self._feature_dim

    def predict_vector(self, features: np.ndarray | list[float]) -> DiseaseInferenceResult:
        arr = np.asarray(features, dtype=np.float64).reshape(-1)
        if self._feature_dim > 0 and arr.shape[0] != self._feature_dim:
            raise DiseaseFeatureError(
                f"Expected {self._feature_dim} features, got {arr.shape[0]}"
            )
        x = arr.reshape(1, -1)
        idx = int(self._clf.predict(x)[0])
        if idx < 0 or idx >= len(self._class_names):
            raise RuntimeError(f"Classifier returned invalid class index {idx}")
        name = self._class_names[idx]
        probs: dict[str, float] | None = None
        if hasattr(self._clf, "predict_proba"):
            prob_row = self._clf.predict_proba(x)[0]
            probs = {
                self._class_names[i]: float(prob_row[i])
                for i in range(min(len(prob_row), len(self._class_names)))
            }
        return DiseaseInferenceResult(
            disease_class_index=idx,
            disease_class_name=name,
            probabilities=probs,
            model_version=self.model_version,
            inferred_at_utc=datetime.now(UTC).isoformat(),
        )
