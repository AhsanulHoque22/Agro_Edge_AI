"""
ImageRecord — metadata for a rice leaf disease image sample.

Each image file in the disease detection dataset has a corresponding
ImageRecord entry in the dataset manifest (manifest.parquet).

The env_context_record_id field optionally links an image to the closest
SensorRecord in time. This bridge enables future multimodal models
(Model 3 — Advisory System) to train on both visual and environmental data.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from .enums import DiseaseLabel, LabelingMethod


class ImageRecord(BaseModel):
    """
    Metadata record for a single rice leaf disease image.

    Images are stored in a directory hierarchy by disease class:
        datasets/disease_model/v1/images/<disease_label>/

    Each image is referenced by its relative path from the dataset root.
    """

    image_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this image.",
    )
    image_path: str = Field(
        description="Relative path from dataset root e.g. images/rice_blast/img_001.jpg",
    )

    # ── Farm traceability (nullable for externally sourced images) ────────────

    farm_id: Optional[str] = None
    field_id: Optional[str] = None
    node_id: Optional[str] = Field(
        default=None,
        description="Camera device identifier if captured in field.",
    )

    # ── Capture context ───────────────────────────────────────────────────────

    captured_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp of image capture. Null for external datasets.",
    )
    crop_type: str = Field(default="rice")
    crop_variety: Optional[str] = None
    growth_stage: Optional[str] = Field(
        default=None,
        description="Growth stage label at time of capture e.g. 2_tillering.",
    )

    # ── Disease classification ────────────────────────────────────────────────

    disease_label: DiseaseLabel
    label_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Model confidence score. Null for expert-labeled images.",
    )
    labeling_method: LabelingMethod = Field(default=LabelingMethod.EXPERT)

    # ── Image properties ──────────────────────────────────────────────────────

    image_width_px: int = Field(gt=0)
    image_height_px: int = Field(gt=0)

    # ── Data provenance ───────────────────────────────────────────────────────

    data_source: str = Field(
        description="Origin of image e.g. PlantVillage, field_capture, kaggle_rice4.",
    )

    # ── Environmental context link ────────────────────────────────────────────
    # Links this image to the SensorRecord that was closest in time to capture.
    # When present, enables multimodal training (image + sensor features).
    # Null for images from external datasets where no sensor data exists.

    env_context_record_id: Optional[str] = Field(
        default=None,
        description="SensorRecord.record_id closest in time to image capture.",
    )

    dataset_version: str = Field(default="v1")
    schema_version: str = Field(default="1.0")
