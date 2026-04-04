"""
Edge inference engine for AgroEdge irrigation model bundle.

Responsibilities:
  - Load exported model bundle artifacts
  - Validate incoming feature payload against exported feature schema
  - Run classifier + regressor inference
  - Apply safety decision rules before emitting actuation recommendation
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class FeatureValidationError(ValueError):
    """Raised when incoming feature payload violates model schema contract."""


@dataclass
class InferenceResult:
    """Model-only outputs without safety constraints."""

    irrigation_needed: int
    irrigation_probability: float
    irrigation_duration_minutes: float
    model_version: str
    inferred_at_utc: str


@dataclass
class DecisionResult:
    """Safety-aware final decision for edge actuation."""

    should_irrigate: bool
    approved_duration_minutes: float
    blocked_reason: str | None
    model_prediction: InferenceResult
    model_version: str
    decided_at_utc: str


@dataclass
class DecisionRules:
    """
    Safety rule thresholds applied after ML inference.

    - minimum_tank_level_percent: block irrigation below this tank level
    - minimum_irrigation_interval_days: enforce cooldown between events
    - maximum_irrigation_duration_minutes: hard cap to avoid overwatering
    """

    minimum_tank_level_percent: float = 10.0
    minimum_irrigation_interval_days: int = 1
    maximum_irrigation_duration_minutes: float = 90.0


class EdgeInferenceEngine:
    """Loads and serves the exported irrigation model bundle on edge devices."""

    def __init__(self, bundle_dir: Path, decision_rules: DecisionRules | None = None) -> None:
        self.bundle_dir = bundle_dir
        self.decision_rules = decision_rules or DecisionRules()
        self._model_payload = self._load_model_payload()
        self._feature_schema = self._load_json(self.bundle_dir / "feature_schema.json")
        self._metadata = self._load_json(self.bundle_dir / "metadata.json")
        self._feature_specs: list[dict[str, Any]] = self._feature_schema.get("features", [])
        self._feature_order: list[str] = [spec["name"] for spec in self._feature_specs]

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_model_payload(self) -> dict[str, Any]:
        model_path = self.bundle_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        payload = joblib.load(model_path)
        required_keys = {"classifier", "regressor", "feature_names", "version"}
        missing = required_keys - set(payload.keys())
        if missing:
            raise ValueError(f"Invalid model payload missing keys: {sorted(missing)}")
        return payload

    @property
    def model_version(self) -> str:
        return str(self._model_payload["version"])

    def _coerce_and_validate(self, payload: dict[str, Any]) -> list[float]:
        vector: list[float] = []
        for spec in self._feature_specs:
            name = spec["name"]
            ftype = str(spec.get("type", "float"))
            required = bool(spec.get("required", False))
            if name not in payload:
                if required:
                    raise FeatureValidationError(f"Missing required feature: {name}")
                value = 0 if ftype == "int" else 0.0
            else:
                value = payload[name]

            try:
                if ftype == "int":
                    coerced = int(value)
                    vector.append(float(coerced))
                elif ftype == "float":
                    coerced = float(value)
                    vector.append(coerced)
                else:
                    raise FeatureValidationError(f"Unsupported feature type '{ftype}' for {name}")
            except (TypeError, ValueError) as exc:
                raise FeatureValidationError(f"Invalid value for {name}: {value!r}") from exc
        return vector

    def predict(self, feature_payload: dict[str, Any]) -> InferenceResult:
        """Run model inference and return raw predictions."""
        vector = self._coerce_and_validate(feature_payload)
        x = pd.DataFrame([vector], columns=self._feature_order, dtype=float)
        classifier = self._model_payload["classifier"]
        regressor = self._model_payload["regressor"]

        cls_pred = int(classifier.predict(x)[0])
        if hasattr(classifier, "predict_proba"):
            cls_prob = float(classifier.predict_proba(x)[0][1])
        else:
            cls_prob = float(cls_pred)
        reg_pred = float(regressor.predict(x)[0])
        reg_pred = max(0.0, reg_pred)

        return InferenceResult(
            irrigation_needed=cls_pred,
            irrigation_probability=cls_prob,
            irrigation_duration_minutes=reg_pred,
            model_version=self.model_version,
            inferred_at_utc=datetime.now(UTC).isoformat(),
        )

    def decide(self, feature_payload: dict[str, Any]) -> DecisionResult:
        """
        Produce final actuation decision by combining model output with safety rules.
        """
        pred = self.predict(feature_payload)

        tank_level = float(feature_payload.get("water_tank_level_percent", 0.0))
        days_since_last = int(feature_payload.get("days_since_last_irrigation", 0))

        if tank_level < self.decision_rules.minimum_tank_level_percent:
            return DecisionResult(
                should_irrigate=False,
                approved_duration_minutes=0.0,
                blocked_reason="tank_level_below_minimum",
                model_prediction=pred,
                model_version=self.model_version,
                decided_at_utc=datetime.now(UTC).isoformat(),
            )

        if days_since_last < self.decision_rules.minimum_irrigation_interval_days:
            return DecisionResult(
                should_irrigate=False,
                approved_duration_minutes=0.0,
                blocked_reason="minimum_irrigation_interval_not_met",
                model_prediction=pred,
                model_version=self.model_version,
                decided_at_utc=datetime.now(UTC).isoformat(),
            )

        if pred.irrigation_needed == 0:
            return DecisionResult(
                should_irrigate=False,
                approved_duration_minutes=0.0,
                blocked_reason="model_predicted_no_irrigation",
                model_prediction=pred,
                model_version=self.model_version,
                decided_at_utc=datetime.now(UTC).isoformat(),
            )

        approved_duration = min(
            pred.irrigation_duration_minutes,
            self.decision_rules.maximum_irrigation_duration_minutes,
        )
        return DecisionResult(
            should_irrigate=True,
            approved_duration_minutes=float(max(0.0, approved_duration)),
            blocked_reason=None,
            model_prediction=pred,
            model_version=self.model_version,
            decided_at_utc=datetime.now(UTC).isoformat(),
        )
