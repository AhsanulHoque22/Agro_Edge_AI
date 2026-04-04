"""
End-to-end irrigation model training pipeline:
  processed dataset -> train -> evaluate -> gate check -> export
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_evaluation.evaluator import IrrigationModelEvaluator
from model_export.exporter import IrrigationModelExporter
from model_training.irrigation_model.trainer import IrrigationModelTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export AgroEdge irrigation model.")
    parser.add_argument("--dataset-version", type=str, default="v1", help="Processed dataset version (default: v1).")
    parser.add_argument(
        "--model-config-path",
        type=str,
        default="configs/model_config.yaml",
        help="Model config path relative to repo root.",
    )
    parser.add_argument(
        "--feature-schema-path",
        type=str,
        default="configs/feature_schema.yaml",
        help="Feature schema path relative to repo root.",
    )
    parser.add_argument("--allow-failed-gates", action="store_true", help="Export even if performance gates fail.")
    parser.add_argument(
        "--fixed-validation-path",
        type=str,
        default=None,
        help="Optional raw validation parquet for stable version comparison.",
    )
    parser.add_argument(
        "--baseline-evaluation-path",
        type=str,
        default=None,
        help="Optional previous evaluation.json to compare against candidate.",
    )
    parser.add_argument(
        "--comparison-output-path",
        type=str,
        default=None,
        help="Optional output path for baseline-vs-candidate comparison JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    processed_dir = root / "datasets" / "processed" / args.dataset_version
    model_cfg_path = root / args.model_config_path
    feature_schema_path = root / args.feature_schema_path
    export_root = root / "model_export"

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed dataset directory not found: {processed_dir}")

    print("Loading configs...")
    model_cfg = yaml.safe_load(model_cfg_path.read_text(encoding="utf-8"))

    print("Training irrigation models...")
    trainer = IrrigationModelTrainer(model_config_path=model_cfg_path)
    artifacts = trainer.run(processed_dir=processed_dir)
    cls_cv, reg_cv = trainer.cv_splitters()

    print("Evaluating models...")
    evaluator = IrrigationModelEvaluator(config=model_cfg)
    evaluation = evaluator.evaluate(
        classifier=artifacts.classifier,
        regressor=artifacts.regressor,
        X_train=artifacts.X_train,
        X_test=artifacts.X_test,
        y_cls_train=artifacts.y_cls_train,
        y_cls_test=artifacts.y_cls_test,
        y_reg_train=artifacts.y_reg_train,
        y_reg_test=artifacts.y_reg_test,
        cls_cv=cls_cv,
        reg_cv=reg_cv,
    )
    if args.fixed_validation_path:
        fixed_path = root / args.fixed_validation_path
        if not fixed_path.exists():
            raise FileNotFoundError(f"Fixed validation dataset not found: {fixed_path}")
        fixed_df = pd.read_parquet(fixed_path, engine="pyarrow")
        required = set(artifacts.feature_names + ["irrigation_needed", "irrigation_duration_minutes"])
        missing = sorted(required - set(fixed_df.columns))
        if missing:
            raise ValueError(
                f"Fixed validation dataset missing required columns: {', '.join(missing)}"
            )
        fixed_eval = evaluator.evaluate_split(
            classifier=artifacts.classifier,
            regressor=artifacts.regressor,
            X_eval=fixed_df[artifacts.feature_names].copy(),
            y_cls_eval=pd.to_numeric(fixed_df["irrigation_needed"], errors="raise").astype("int64"),
            y_reg_eval=pd.to_numeric(
                fixed_df["irrigation_duration_minutes"], errors="raise"
            ).astype("float64"),
        )
        evaluation.report["fixed_validation_metrics"] = fixed_eval
        evaluation.report["fixed_validation_path"] = str(fixed_path)
        print("\nFixed validation metrics:")
        print(
            f"  classifier_f1_score: {fixed_eval['classifier_metrics']['f1_score']:.4f}"
        )
        print(
            f"  classifier_accuracy: {fixed_eval['classifier_metrics']['accuracy']:.4f}"
        )
        print(
            f"  regressor_r2_score: {fixed_eval['regressor_metrics']['r2_score']:.4f}"
        )
        print(
            f"  regressor_mae_minutes: {fixed_eval['regressor_metrics']['mae_minutes']:.4f}"
        )

    cls = evaluation.report["classifier_metrics"]
    reg = evaluation.report["regressor_metrics"]
    print("\nTest metrics:")
    print(f"  classifier_f1_score: {cls['f1_score']:.4f}")
    print(f"  classifier_accuracy: {cls['accuracy']:.4f}")
    print(f"  regressor_r2_score: {reg['r2_score']:.4f}")
    print(f"  regressor_mae_minutes: {reg['mae_minutes']:.4f}")

    if not evaluation.gates_passed:
        print("\nPerformance gates FAILED:")
        for failure in evaluation.gate_failures:
            print(f"  - {failure}")
        if not args.allow_failed_gates:
            raise RuntimeError("Training completed but export blocked due to failed performance gates.")

    print("\nExporting model bundle...")
    exporter = IrrigationModelExporter(model_config=model_cfg, feature_schema_path=feature_schema_path)
    exported = exporter.export(
        classifier=artifacts.classifier,
        regressor=artifacts.regressor,
        evaluation_report=evaluation.report,
        output_root=export_root,
    )
    print("Export complete:")
    for key, path in exported.items():
        print(f"  {key}: {path}")

    if args.baseline_evaluation_path:
        baseline_path = root / args.baseline_evaluation_path
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline evaluation not found: {baseline_path}")
        candidate_eval_path = Path(exported["evaluation"])
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        candidate = json.loads(candidate_eval_path.read_text(encoding="utf-8"))
        comparison = _build_evaluation_comparison(
            baseline=baseline,
            candidate=candidate,
            baseline_path=str(baseline_path),
            candidate_path=str(candidate_eval_path),
        )
        output = (
            root / args.comparison_output_path
            if args.comparison_output_path
            else candidate_eval_path.with_name("comparison_vs_baseline.json")
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nComparison report written: {output}")


def _build_evaluation_comparison(
    baseline: dict,
    candidate: dict,
    baseline_path: str,
    candidate_path: str,
) -> dict:
    def _get(d: dict, *keys: str):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    def _delta(c_val, b_val):
        if c_val is None or b_val is None:
            return None
        try:
            return float(c_val) - float(b_val)
        except (TypeError, ValueError):
            return None

    pairs = {
        "classifier_f1_score": (
            _get(baseline, "classifier_metrics", "f1_score"),
            _get(candidate, "classifier_metrics", "f1_score"),
        ),
        "classifier_accuracy": (
            _get(baseline, "classifier_metrics", "accuracy"),
            _get(candidate, "classifier_metrics", "accuracy"),
        ),
        "regressor_r2_score": (
            _get(baseline, "regressor_metrics", "r2_score"),
            _get(candidate, "regressor_metrics", "r2_score"),
        ),
        "regressor_mae_minutes": (
            _get(baseline, "regressor_metrics", "mae_minutes"),
            _get(candidate, "regressor_metrics", "mae_minutes"),
        ),
        "regressor_mae_minutes_when_irrigating": (
            _get(baseline, "regressor_metrics", "mae_minutes_when_irrigating"),
            _get(candidate, "regressor_metrics", "mae_minutes_when_irrigating"),
        ),
    }
    out = {"baseline": baseline_path, "candidate": candidate_path, "comparison": {}}
    for name, (b_val, c_val) in pairs.items():
        out["comparison"][name] = {
            "baseline": b_val,
            "candidate": c_val,
            "delta_candidate_minus_baseline": _delta(c_val, b_val),
        }
    return out


if __name__ == "__main__":
    main()
