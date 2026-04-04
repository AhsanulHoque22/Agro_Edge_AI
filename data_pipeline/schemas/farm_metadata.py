"""
Farm and field metadata schemas.

FarmMetadata  — static configuration for a farm entity.
FieldMetadata — configuration for a single crop plot within a farm.

Growth stage is computed automatically from planting_date by default.
It can also be supplied by a phone image classifier (future Model 2b)
or set manually via the dashboard — the architecture supports all three
without any schema change.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from typing import Optional

from pydantic import BaseModel, Field

from .enums import (
    GROWTH_STAGE_ENCODING,
    GrowthStage,
    SoilType,
    StageDetectionMethod,
)

# ── Growth Stage Calculation ──────────────────────────────────────────────────

# BBCH-based variety-agnostic DAT thresholds.
# Format: (start_day_inclusive, end_day_exclusive, GrowthStage)
# Rice variety may shift these boundaries by ±10 days.
# Source: BBCH monograph for Oryza sativa.
GROWTH_STAGE_DAT_THRESHOLDS: list[tuple[int, int, GrowthStage]] = [
    (0,   7,   GrowthStage.GERMINATION),
    (7,   25,  GrowthStage.SEEDLING),
    (25,  55,  GrowthStage.TILLERING),
    (55,  75,  GrowthStage.STEM_ELONGATION),
    (75,  90,  GrowthStage.BOOTING),
    (90,  110, GrowthStage.HEADING_FLOWERING),
    (110, 130, GrowthStage.GRAIN_FILLING),
    (130, 160, GrowthStage.RIPENING_MATURITY),
]


def calculate_growth_stage(days_after_transplanting: int) -> GrowthStage:
    """
    Return the rice growth stage for a given DAT value.

    Iterates through BBCH thresholds and returns the matching stage.
    If DAT exceeds all thresholds, returns RIPENING_MATURITY.
    """
    for start, end, stage in GROWTH_STAGE_DAT_THRESHOLDS:
        if start <= days_after_transplanting < end:
            return stage
    return GrowthStage.RIPENING_MATURITY


def encode_growth_stage(stage: GrowthStage) -> int:
    """Return the integer encoding for a GrowthStage value."""
    return GROWTH_STAGE_ENCODING[stage]


# ── Schema Models ─────────────────────────────────────────────────────────────

class FarmMetadata(BaseModel):
    """Static configuration for a farm entity."""

    farm_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    farm_name: str
    farm_location_lat: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    farm_location_lon: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    farm_area_hectares: Optional[float] = Field(default=None, gt=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = Field(default="1.0")


class FieldMetadata(BaseModel):
    """
    Configuration for a single crop plot within a farm.

    Growth stage is determined by one of three methods:
      1. DATE_CALCULATED  — derived from planting_date + BBCH thresholds (default).
         Call update_growth_stage() to recompute when the date changes.
      2. IMAGE_CLASSIFIED — supplied by a phone image classifier (future Model 2b).
         Pass method=IMAGE_CLASSIFIED and override_stage=<detected_stage>.
      3. MANUAL           — explicit override by the farmer via the dashboard.
         Pass method=MANUAL and override_stage=<chosen_stage>.

    All three methods use the same update_growth_stage() interface, so adding
    image-based detection later requires no schema changes.
    """

    field_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    farm_id: str
    field_name: str
    field_area_m2: Optional[float] = Field(default=None, gt=0.0)
    soil_type: Optional[SoilType] = None
    crop_type: str = Field(default="rice")
    crop_variety: Optional[str] = Field(
        default=None,
        description="Free-text variety name e.g. BRRI dhan29, IR64, Swarna.",
    )
    planting_date: date
    expected_harvest_date: Optional[date] = None
    node_id: str = Field(description="ESP32 device assigned to this plot.")

    # Growth stage — managed field, not set directly
    growth_stage: Optional[GrowthStage] = Field(
        default=None,
        description="Current growth stage. Set via update_growth_stage().",
    )
    stage_detection_method: StageDetectionMethod = Field(
        default=StageDetectionMethod.DATE_CALCULATED,
    )
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = Field(default="1.0")

    # ── Derived calculations ──────────────────────────────────────────────────

    def days_after_transplanting(self, reference_date: Optional[date] = None) -> int:
        """Return days elapsed since planting_date. Never negative."""
        ref = reference_date or date.today()
        return max(0, (ref - self.planting_date).days)

    def days_to_maturity(self, reference_date: Optional[date] = None) -> int:
        """
        Return days remaining until harvest.
        Uses expected_harvest_date if set, otherwise estimates 150 DAT.
        Returns negative values if harvest date has passed.
        """
        ref = reference_date or date.today()
        if self.expected_harvest_date:
            return (self.expected_harvest_date - ref).days
        dat = self.days_after_transplanting(ref)
        return max(0, 150 - dat)

    # ── Growth stage management ───────────────────────────────────────────────

    def update_growth_stage(
        self,
        reference_date: Optional[date] = None,
        method: StageDetectionMethod = StageDetectionMethod.DATE_CALCULATED,
        override_stage: Optional[GrowthStage] = None,
    ) -> GrowthStage:
        """
        Recompute and store the current growth stage.

        Args:
            reference_date:  Date to use for DAT calculation. Defaults to today.
            method:          How the stage was determined.
            override_stage:  Explicit stage value (used for IMAGE_CLASSIFIED and MANUAL).

        Returns:
            The resolved GrowthStage.
        """
        if override_stage is not None:
            self.growth_stage = override_stage
        else:
            dat = self.days_after_transplanting(reference_date)
            self.growth_stage = calculate_growth_stage(dat)

        self.stage_detection_method = method
        self.updated_at = datetime.now(UTC)
        return self.growth_stage

    def current_growth_stage(self, reference_date: Optional[date] = None) -> GrowthStage:
        """
        Return the current growth stage, computing it if not already set.
        Safe to call repeatedly — only recomputes when growth_stage is None.
        """
        if self.growth_stage is None:
            self.update_growth_stage(reference_date)
        return self.growth_stage  # type: ignore[return-value]
