"""
Feature engineering pipeline for irrigation model training.

Transforms raw TrainingRecord parquet files into a model-ready dataset using the
feature contract defined in configs/feature_schema.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class PreparedDataset:
    """Container for prepared training data splits."""

    features: pd.DataFrame
    target_classification: pd.Series
    target_regression: pd.Series
    metadata: pd.DataFrame


class FeaturePreparer:
    """
    Converts raw dataset records into model-ready features and targets.

    The exact feature order is sourced from configs/feature_schema.yaml and must
    remain stable for compatibility with training, export, and edge inference.
    """

    def __init__(self, feature_schema_path: Path) -> None:
        self.feature_schema_path = feature_schema_path
        self.feature_specs = self._load_feature_specs()
        self.feature_names = [spec["name"] for spec in self.feature_specs]

    def _load_feature_specs(self) -> list[dict[str, Any]]:
        with self.feature_schema_path.open("r", encoding="utf-8") as f:
            schema = yaml.safe_load(f)
        features = schema.get("features", [])
        if not features:
            raise ValueError("No features found in feature schema.")
        return features

    @staticmethod
    def _default_for_type(feature_type: str) -> float | int:
        if feature_type == "int":
            return 0
        return 0.0

    def prepare(self, raw_df: pd.DataFrame) -> PreparedDataset:
        """
        Validate and prepare raw data into feature matrix and target vectors.
        """
        required_feature_names = [
            spec["name"] for spec in self.feature_specs if bool(spec.get("required", False))
        ]
        required_columns = set(required_feature_names + ["irrigation_needed", "irrigation_duration_minutes"])
        missing = sorted(required_columns - set(raw_df.columns))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        df = raw_df.copy()
        for spec in self.feature_specs:
            name = spec["name"]
            feature_type = str(spec.get("type", "float"))
            if name not in df.columns:
                df[name] = self._default_for_type(feature_type)
            if feature_type == "int":
                df[name] = pd.to_numeric(df[name], errors="raise").astype("int64")
            elif feature_type == "float":
                df[name] = pd.to_numeric(df[name], errors="raise").astype("float64")

        # Keep training metadata separately for traceability and experiment logs.
        metadata_cols = [
            "record_id",
            "farm_id",
            "field_id",
            "node_id",
            "collected_at",
            "dataset_version",
            "label_source",
            "schema_version",
        ]
        metadata = df[[c for c in metadata_cols if c in df.columns]].copy()

        features = df[self.feature_names].copy()
        y_cls = pd.to_numeric(df["irrigation_needed"], errors="raise").astype("int64")
        y_reg = pd.to_numeric(df["irrigation_duration_minutes"], errors="raise").astype("float64")

        return PreparedDataset(
            features=features,
            target_classification=y_cls,
            target_regression=y_reg,
            metadata=metadata,
        )

    @staticmethod
    def save(prepared: PreparedDataset, output_dir: Path) -> dict[str, Any]:
        """
        Save prepared dataset artifacts to parquet files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        x_path = output_dir / "X_features.parquet"
        y_cls_path = output_dir / "y_irrigation_needed.parquet"
        y_reg_path = output_dir / "y_irrigation_duration.parquet"
        meta_path = output_dir / "metadata.parquet"

        prepared.features.to_parquet(x_path, index=False, engine="pyarrow")
        prepared.target_classification.to_frame("irrigation_needed").to_parquet(
            y_cls_path, index=False, engine="pyarrow"
        )
        prepared.target_regression.to_frame("irrigation_duration_minutes").to_parquet(
            y_reg_path, index=False, engine="pyarrow"
        )
        prepared.metadata.to_parquet(meta_path, index=False, engine="pyarrow")

        return {
            "X_features": str(x_path),
            "y_irrigation_needed": str(y_cls_path),
            "y_irrigation_duration": str(y_reg_path),
            "metadata": str(meta_path),
            "n_rows": len(prepared.features),
            "n_features": prepared.features.shape[1],
        }
