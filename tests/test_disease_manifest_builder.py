"""Manifest builder for disease CV dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from data_pipeline.disease.manifest_builder import build_disease_manifest


def test_build_disease_manifest_writes_parquet(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets" / "disease_model" / "v1"
    images_dir = dataset_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    labels = ["healthy", "rice_blast"]
    resize_side = 8

    for li, label in enumerate(labels):
        label_dir = images_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            arr = np.full((resize_side, resize_side), 10 + li * 30 + i, dtype=np.uint8)
            img_path = label_dir / f"img_{i:02d}.png"
            Image.fromarray(arr, mode="L").save(img_path)

    manifest_path = build_disease_manifest(
        dataset_root=dataset_root,
        limit_per_class=2,
        data_source="test_manifest_builder",
    )

    assert manifest_path.is_file()
    df = pd.read_parquet(manifest_path, engine="pyarrow")

    for col in [
        "image_path",
        "disease_label",
        "image_width_px",
        "image_height_px",
        "data_source",
        "labeling_method",
    ]:
        assert col in df.columns

    assert set(df["disease_label"].tolist()) == set(labels)
    assert all(str(p).startswith("images/") for p in df["image_path"].tolist())

