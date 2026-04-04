"""
TrainingRecord — labeled record for irrigation model training.

Extends FeatureRecord with both irrigation targets:
  - irrigation_needed           (classification: binary 0/1)
  - irrigation_duration_minutes (regression: continuous float)

Both targets are stored in the same record so a single dataset file
supports dual-output training without loading separate label files.
"""

from __future__ import annotations

from pydantic import Field

from .enums import LabelSource
from .feature_record import FeatureRecord


class TrainingRecord(FeatureRecord):
    """
    Fully labeled record for training the irrigation decision model.

    Inherits all 14 features from FeatureRecord and adds two target labels.
    Provenance fields (label_source, dataset_version) enable filtering
    by data origin during training experiments.
    """

    # ── Labels ────────────────────────────────────────────────────────────────

    irrigation_needed: int = Field(
        ge=0, le=1,
        description="Binary label: 1 = irrigate now, 0 = do not irrigate.",
    )
    irrigation_duration_minutes: float = Field(
        ge=0.0,
        description="Recommended irrigation duration in minutes.",
    )

    # ── Provenance ────────────────────────────────────────────────────────────

    label_source: LabelSource = Field(
        default=LabelSource.SYNTHETIC,
        description="How this label was produced.",
    )
    dataset_version: str = Field(
        default="v1",
        description="Dataset version this record belongs to.",
    )
