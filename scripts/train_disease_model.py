"""
Train and export the disease model.

Supported modes (config-driven):
  - ``sanity``: sklearn sanity baseline on synthetic vectors
  - ``image_pixels``: grayscale pixel-vector embedding from ``manifest.parquet`` + images

Usage:
  python scripts/train_disease_model.py --dataset-root datasets/disease_model/v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.disease.manifest_builder import build_disease_manifest
from model_training.disease_model.exporter import DiseaseModelExporter
from model_training.disease_model.trainer import (
    load_disease_config,
    train_image_pixels_model,
    train_sanity_baseline,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to disease_model_config.yaml (default: configs/).",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to datasets/disease_model/v1 (default uses that path).",
    )
    p.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional explicit path to manifest.parquet.",
    )
    p.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Override config training.mode: sanity | image_pixels.",
    )
    p.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Optional cap on number of manifest rows to train on.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    cfg_path = args.config or (root / "configs" / "disease_model_config.yaml")
    config = load_disease_config(cfg_path)

    dm = config["disease_model"]
    mode = args.mode or dm["training"].get("mode", "sanity")

    dataset_root = args.dataset_root or (root / "datasets" / "disease_model" / "v1")
    manifest_path = args.manifest_path

    if mode == "sanity":
        print("Training disease sanity baseline (synthetic features)…")
        artifacts = train_sanity_baseline(config)
    elif mode == "image_pixels":
        print(f"Training disease image-pixels model from {dataset_root}…")
        manifest_path = (
            args.manifest_path
            if args.manifest_path is not None
            else (dataset_root / "manifest.parquet")
        )
        if not manifest_path.is_file():
            print(f"manifest.parquet not found; building from images under {dataset_root}…")
            manifest_path = build_disease_manifest(dataset_root=dataset_root)
        artifacts = train_image_pixels_model(
            config,
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            limit_samples=args.limit_samples,
        )
    else:
        raise ValueError(f"Unknown training mode: {mode!r}")

    print(
        f"  test_accuracy={artifacts.report['test_accuracy']:.4f} "
        f"macro_f1={artifacts.report['test_macro_f1']:.4f}"
    )

    exporter = DiseaseModelExporter(config=config)
    out = exporter.export(
        classifier=artifacts.classifier,
        evaluation_report=artifacts.report,
        output_root=root / "model_export",
    )
    print("Exported:")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
