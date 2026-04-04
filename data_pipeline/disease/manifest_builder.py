"""
Build `manifest.parquet` for the disease CV dataset.

Expected folder layout:

    datasets/disease_model/<version>/images/<disease_label>/*.jpg|*.png|...

This builder converts the folder structure into a Parquet manifest where each
row corresponds to one image and includes at least:
  - image_path      (relative to dataset_root, e.g. "images/healthy/img_01.png")
  - disease_label   (one of DiseaseLabel)
  - image_width_px, image_height_px

The training pipeline (`train_image_pixels_model`) only requires
`image_path` and `disease_label`, but we store the extra fields to align
with `data_pipeline/schemas/image_record.py`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from data_pipeline.schemas import DiseaseLabel, LabelingMethod


def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def build_disease_manifest(
    *,
    dataset_root: Path,
    images_dirname: str = "images",
    output_manifest: Path | None = None,
    dataset_version: str = "v1",
    schema_version: str = "1.0",
    data_source: str = "folder_dataset",
    limit_per_class: int | None = None,
) -> Path:
    images_dir = dataset_root / images_dirname
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    if output_manifest is None:
        output_manifest = dataset_root / "manifest.parquet"

    rows: list[dict[str, Any]] = []

    # DiseaseLabel enum values are the folder names (e.g. "healthy", "rice_blast", ...).
    valid_labels = {x.value for x in DiseaseLabel}

    for label_dir in sorted(images_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label_name = label_dir.name
        if label_name not in valid_labels:
            continue

        count = 0
        for img_path in sorted(label_dir.iterdir()):
            if not img_path.is_file() or not _is_image_file(img_path):
                continue
            if limit_per_class is not None and count >= int(limit_per_class):
                break

            with Image.open(img_path) as img:
                width, height = int(img.width), int(img.height)

            rel = img_path.relative_to(dataset_root).as_posix()
            # Cheap content hash to allow deduping later (optional).
            h = hashlib.sha256(img_path.read_bytes()).hexdigest()

            rows.append(
                {
                    "image_path": rel,
                    "disease_label": label_name,
                    "image_width_px": width,
                    "image_height_px": height,
                    "data_source": data_source,
                    "label_confidence": None,
                    "labeling_method": LabelingMethod.EXPERT.value,
                    "dataset_version": dataset_version,
                    "schema_version": schema_version,
                    "image_sha256": h,
                }
            )
            count += 1

    if not rows:
        raise RuntimeError(
            f"No images found under {images_dir}. Check labels and file extensions."
        )

    df = pd.DataFrame(rows)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_manifest, index=False, engine="pyarrow", compression="snappy")
    return output_manifest

