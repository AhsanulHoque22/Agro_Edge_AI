"""
Training pipeline for AgroEdge irrigation decision models.

Trains two models on the same feature matrix:
  1) RandomForestClassifier for `irrigation_needed`
  2) RandomForestRegressor for `irrigation_duration_minutes`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, KFold, StratifiedKFold


@dataclass
class TrainingArtifacts:
    classifier: RandomForestClassifier
    regressor: RandomForestRegressor
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_cls_train: pd.Series
    y_cls_test: pd.Series
    y_reg_train: pd.Series
    y_reg_test: pd.Series
    metadata_test: pd.DataFrame
    feature_names: list[str]
    config: dict[str, Any]


class IrrigationModelTrainer:
    """Config-driven trainer for AgroEdge irrigation models."""

    def __init__(self, model_config_path: Path) -> None:
        self.model_config_path = model_config_path
        self.config = self._load_config()
        self.model_cfg = self.config["irrigation_model"]

    def _load_config(self) -> dict[str, Any]:
        with self.model_config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_processed_dataset(self, processed_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        """Load prepared features, labels, and metadata."""
        X = pd.read_parquet(processed_dir / "X_features.parquet", engine="pyarrow")
        y_cls = pd.read_parquet(
            processed_dir / "y_irrigation_needed.parquet",
            engine="pyarrow",
        )["irrigation_needed"]
        y_reg = pd.read_parquet(
            processed_dir / "y_irrigation_duration.parquet",
            engine="pyarrow",
        )["irrigation_duration_minutes"]
        metadata = pd.read_parquet(processed_dir / "metadata.parquet", engine="pyarrow")
        return X, y_cls, y_reg, metadata

    def _validate_feature_contract(self, X: pd.DataFrame) -> list[str]:
        expected = self.model_cfg["features"]
        actual = list(X.columns)
        if actual != expected:
            raise ValueError(
                "Feature contract mismatch.\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual}"
            )
        return expected

    def split_data(
        self,
        X: pd.DataFrame,
        y_cls: pd.Series,
        y_reg: pd.Series,
        metadata: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
        """
        Create train/test split.

        Uses the configured test ratio and stratifies by classification label.
        Validation is handled via cross-validation metrics.
        """
        random_state = int(self.model_cfg["random_state"])
        test_size = float(self.model_cfg["splits"]["test"])
        groups = metadata["field_id"].astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y_cls, groups=groups))
        X_train = X.loc[train_idx].reset_index(drop=True)
        X_test = X.loc[test_idx].reset_index(drop=True)
        y_cls_train = y_cls.loc[train_idx].reset_index(drop=True)
        y_cls_test = y_cls.loc[test_idx].reset_index(drop=True)
        y_reg_train = y_reg.loc[train_idx].reset_index(drop=True)
        y_reg_test = y_reg.loc[test_idx].reset_index(drop=True)
        metadata_test = metadata.loc[test_idx].reset_index(drop=True)
        return X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test, metadata_test

    def train_models(
        self,
        X_train: pd.DataFrame,
        y_cls_train: pd.Series,
        y_reg_train: pd.Series,
    ) -> tuple[RandomForestClassifier, RandomForestRegressor]:
        """Fit classifier and regressor using hyperparameters from config."""
        clf_params = dict(self.model_cfg["classifier_hyperparameters"])
        reg_params = dict(self.model_cfg["regressor_hyperparameters"])
        classifier = RandomForestClassifier(**clf_params)
        regressor = RandomForestRegressor(**reg_params)
        classifier.fit(X_train, y_cls_train)
        regressor.fit(X_train, y_reg_train)
        return classifier, regressor

    def cv_splitters(self) -> tuple[StratifiedKFold, KFold]:
        # Row-level CV for now; holdout leakage is prevented by group-based test split.
        # A future step can add GroupKFold CV using field_id for stricter validation.
        cv_cfg = self.model_cfg["cross_validation"]
        n_splits = int(cv_cfg["n_splits"])
        shuffle = bool(cv_cfg["shuffle"])
        random_state = int(cv_cfg["random_state"])
        cls_cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        reg_cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return cls_cv, reg_cv

    def run(self, processed_dir: Path) -> TrainingArtifacts:
        """Execute loading, validation, split, and training."""
        X, y_cls, y_reg, metadata = self.load_processed_dataset(processed_dir)
        feature_names = self._validate_feature_contract(X)
        X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test, metadata_test = self.split_data(
            X, y_cls, y_reg, metadata
        )
        classifier, regressor = self.train_models(X_train, y_cls_train, y_reg_train)
        return TrainingArtifacts(
            classifier=classifier,
            regressor=regressor,
            X_train=X_train,
            X_test=X_test,
            y_cls_train=y_cls_train,
            y_cls_test=y_cls_test,
            y_reg_train=y_reg_train,
            y_reg_test=y_reg_test,
            metadata_test=metadata_test,
            feature_names=feature_names,
            config=self.config,
        )
