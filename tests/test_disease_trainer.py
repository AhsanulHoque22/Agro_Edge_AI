"""Disease model export from image-pixel training."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from edge_inference.disease_engine import DiseaseFeatureError, DiseaseInferenceEngine
from model_training.disease_model.exporter import DiseaseModelExporter
from model_training.disease_model.trainer import load_disease_config, train_image_pixels_model


def test_train_image_pixels_and_export(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / "configs" / "disease_model_config.yaml"
    config = load_disease_config(cfg_path)

    dm = config["disease_model"]
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
        for i in range(10):
            fname = f"img_{i:02d}.png"
            path = class_dir / fname
            base = 10 + label_idx * 20
            arr = np.full((resize_side, resize_side), base, dtype=np.uint8)
            arr = np.clip(arr + i, 0, 255).astype(np.uint8)
            Image.fromarray(arr, mode="L").save(path)
            rows.append(
                {
                    "image_path": str(Path("images") / label / fname),
                    "disease_label": label,
                }
            )

    pd.DataFrame(rows).to_parquet(manifest_path, index=False, engine="pyarrow")

    artifacts = train_image_pixels_model(
        config,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
    )
    assert artifacts.report["test_accuracy"] >= 0.0

    out = DiseaseModelExporter(config).export(
        classifier=artifacts.classifier,
        evaluation_report=artifacts.report,
        output_root=tmp_path,
    )
    model_path = Path(out["model"])
    assert model_path.is_file()
    payload = joblib.load(model_path)
    assert "classifier" in payload
    assert len(payload["class_names"]) == 8

    bundle_dir = model_path.parent
    engine = DiseaseInferenceEngine(bundle_dir=bundle_dir)
    vec = [0.0] * int(payload["feature_dimension"])
    result = engine.predict_vector(vec)
    assert result.disease_class_name in engine.class_names
    assert result.probabilities is not None

    with pytest.raises(DiseaseFeatureError):
        engine.predict_vector([1.0, 2.0])
