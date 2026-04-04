from __future__ import annotations

import pytest

from scripts.train_irrigation_model import _build_evaluation_comparison


def test_build_evaluation_comparison_has_expected_deltas() -> None:
    baseline = {
        "classifier_metrics": {"f1_score": 0.80, "accuracy": 0.82},
        "regressor_metrics": {"r2_score": 0.75, "mae_minutes": 10.0},
    }
    candidate = {
        "classifier_metrics": {"f1_score": 0.84, "accuracy": 0.81},
        "regressor_metrics": {"r2_score": 0.79, "mae_minutes": 9.5},
    }
    out = _build_evaluation_comparison(
        baseline=baseline,
        candidate=candidate,
        baseline_path="b.json",
        candidate_path="c.json",
    )
    cmp = out["comparison"]
    assert cmp["classifier_f1_score"]["delta_candidate_minus_baseline"] == pytest.approx(0.04)
    assert cmp["classifier_accuracy"]["delta_candidate_minus_baseline"] == pytest.approx(-0.01)
    assert cmp["regressor_r2_score"]["delta_candidate_minus_baseline"] == pytest.approx(0.04)
    assert cmp["regressor_mae_minutes"]["delta_candidate_minus_baseline"] == pytest.approx(-0.5)
