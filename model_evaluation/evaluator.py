"""
Evaluation utilities for AgroEdge irrigation models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold


@dataclass
class EvaluationOutput:
    report: dict[str, Any]
    gates_passed: bool
    gate_failures: list[str]


class IrrigationModelEvaluator:
    """Builds classification/regression metrics and enforces performance gates."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.model_cfg = config["irrigation_model"]
        self.gates = self.model_cfg["minimum_performance"]

    @staticmethod
    def _safe_rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    def _cv_metrics_classifier(
        self,
        base_model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: StratifiedKFold,
    ) -> dict[str, float]:
        f1s: list[float] = []
        accs: list[float] = []
        for tr_idx, va_idx in cv.split(X_train, y_train):
            x_tr, x_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            m = clone(base_model)
            m.fit(x_tr, y_tr)
            pred = m.predict(x_va)
            f1s.append(float(f1_score(y_va, pred)))
            accs.append(float(accuracy_score(y_va, pred)))
        return {
            "cv_f1_mean": float(np.mean(f1s)),
            "cv_f1_std": float(np.std(f1s)),
            "cv_accuracy_mean": float(np.mean(accs)),
            "cv_accuracy_std": float(np.std(accs)),
        }

    def _cv_metrics_regressor(
        self,
        base_model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: KFold,
    ) -> dict[str, float]:
        r2s: list[float] = []
        maes: list[float] = []
        rmses: list[float] = []
        for tr_idx, va_idx in cv.split(X_train):
            x_tr, x_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            m = clone(base_model)
            m.fit(x_tr, y_tr)
            pred = m.predict(x_va)
            r2s.append(float(r2_score(y_va, pred)))
            maes.append(float(mean_absolute_error(y_va, pred)))
            rmses.append(self._safe_rmse(y_va, pred))
        return {
            "cv_r2_mean": float(np.mean(r2s)),
            "cv_r2_std": float(np.std(r2s)),
            "cv_mae_mean": float(np.mean(maes)),
            "cv_mae_std": float(np.std(maes)),
            "cv_rmse_mean": float(np.mean(rmses)),
            "cv_rmse_std": float(np.std(rmses)),
        }

    def _per_stage_metrics(
        self,
        y_true_cls: pd.Series,
        y_pred_cls: np.ndarray,
        y_true_reg: pd.Series,
        y_pred_reg: np.ndarray,
        X_test: pd.DataFrame,
    ) -> dict[str, Any]:
        per_stage: dict[str, Any] = {}
        stage_col = "growth_stage_encoded"
        for stage in sorted(X_test[stage_col].unique().tolist()):
            mask = X_test[stage_col] == stage
            cls_t = y_true_cls[mask]
            cls_p = y_pred_cls[mask]
            reg_t = y_true_reg[mask]
            reg_p = y_pred_reg[mask]
            per_stage[str(int(stage))] = {
                "count": int(mask.sum()),
                "f1": float(f1_score(cls_t, cls_p, zero_division=0)),
                "accuracy": float(accuracy_score(cls_t, cls_p)),
                "mae_minutes": float(mean_absolute_error(reg_t, reg_p)),
            }
        return per_stage

    def _check_gates(self, cls_metrics: dict[str, Any], reg_metrics: dict[str, Any]) -> tuple[bool, list[str]]:
        failures: list[str] = []
        if cls_metrics["f1_score"] < float(self.gates["classifier_f1_score"]):
            failures.append(
                f"classifier_f1_score {cls_metrics['f1_score']:.4f} < {self.gates['classifier_f1_score']}"
            )
        if cls_metrics["accuracy"] < float(self.gates["classifier_accuracy"]):
            failures.append(
                f"classifier_accuracy {cls_metrics['accuracy']:.4f} < {self.gates['classifier_accuracy']}"
            )
        if reg_metrics["r2_score"] < float(self.gates["regressor_r2_score"]):
            failures.append(
                f"regressor_r2_score {reg_metrics['r2_score']:.4f} < {self.gates['regressor_r2_score']}"
            )
        mae_key = "mae_minutes_when_irrigating"
        gate_mae = float(self.gates["regressor_mae_minutes"])
        observed_mae = reg_metrics.get(mae_key, reg_metrics["mae_minutes"])
        if observed_mae > gate_mae:
            failures.append(
                f"regressor_mae_minutes_when_irrigating {observed_mae:.4f} > {gate_mae}"
            )
        return len(failures) == 0, failures

    def evaluate_split(
        self,
        classifier,
        regressor,
        X_eval: pd.DataFrame,
        y_cls_eval: pd.Series,
        y_reg_eval: pd.Series,
    ) -> dict[str, Any]:
        """Evaluate one explicit dataset split and return metrics dict."""
        cls_pred = classifier.predict(X_eval)
        cls_proba = classifier.predict_proba(X_eval)[:, 1] if hasattr(classifier, "predict_proba") else None
        reg_pred = regressor.predict(X_eval)
        roc_auc: float | None
        if cls_proba is None or y_cls_eval.nunique() < 2:
            roc_auc = None
        else:
            roc_auc = float(roc_auc_score(y_cls_eval, cls_proba))

        cls_metrics = {
            "accuracy": float(accuracy_score(y_cls_eval, cls_pred)),
            "f1_score": float(f1_score(y_cls_eval, cls_pred)),
            "precision": float(precision_score(y_cls_eval, cls_pred, zero_division=0)),
            "recall": float(recall_score(y_cls_eval, cls_pred, zero_division=0)),
            "roc_auc": roc_auc,
            "confusion_matrix": confusion_matrix(y_cls_eval, cls_pred).tolist(),
        }

        irrig_mask = y_cls_eval == 1
        if int(irrig_mask.sum()) > 1:
            reg_mae_pos = float(mean_absolute_error(y_reg_eval[irrig_mask], reg_pred[irrig_mask]))
            reg_r2_pos = float(r2_score(y_reg_eval[irrig_mask], reg_pred[irrig_mask]))
        else:
            reg_mae_pos = None
            reg_r2_pos = None

        reg_metrics = {
            "r2_score": float(r2_score(y_reg_eval, reg_pred)),
            "mae_minutes": float(mean_absolute_error(y_reg_eval, reg_pred)),
            "rmse_minutes": self._safe_rmse(y_reg_eval, reg_pred),
            "r2_when_irrigating": reg_r2_pos,
            "mae_minutes_when_irrigating": reg_mae_pos,
        }
        return {
            "classifier_metrics": cls_metrics,
            "regressor_metrics": reg_metrics,
            "dataset_info": {
                "size": int(len(X_eval)),
                "n_features": int(X_eval.shape[1]),
                "class_distribution": {
                    "negative_0": int((y_cls_eval == 0).sum()),
                    "positive_1": int((y_cls_eval == 1).sum()),
                },
            },
        }

    def evaluate(
        self,
        classifier,
        regressor,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_cls_train: pd.Series,
        y_cls_test: pd.Series,
        y_reg_train: pd.Series,
        y_reg_test: pd.Series,
        cls_cv: StratifiedKFold,
        reg_cv: KFold,
    ) -> EvaluationOutput:
        """Compute CV and held-out test metrics, then enforce gates."""
        cls_pred = classifier.predict(X_test)
        reg_pred = regressor.predict(X_test)
        split_metrics = self.evaluate_split(
            classifier=classifier,
            regressor=regressor,
            X_eval=X_test,
            y_cls_eval=y_cls_test,
            y_reg_eval=y_reg_test,
        )
        cls_metrics = split_metrics["classifier_metrics"]
        reg_metrics = split_metrics["regressor_metrics"]

        report = {
            "classifier_metrics": cls_metrics,
            "regressor_metrics": reg_metrics,
            "cross_validation": {
                **self._cv_metrics_classifier(classifier, X_train, y_cls_train, cls_cv),
                **self._cv_metrics_regressor(regressor, X_train, y_reg_train, reg_cv),
            },
            "per_stage_metrics": self._per_stage_metrics(
                y_cls_test.reset_index(drop=True),
                cls_pred,
                y_reg_test.reset_index(drop=True),
                reg_pred,
                X_test.reset_index(drop=True),
            ),
            "dataset_info": {
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
                "n_features": int(X_train.shape[1]),
                "class_distribution_test": {
                    "negative_0": int((y_cls_test == 0).sum()),
                    "positive_1": int((y_cls_test == 1).sum()),
                },
            },
        }
        gates_passed, gate_failures = self._check_gates(cls_metrics, reg_metrics)
        report["gates_passed"] = gates_passed
        report["gate_failures"] = gate_failures
        return EvaluationOutput(report=report, gates_passed=gates_passed, gate_failures=gate_failures)
