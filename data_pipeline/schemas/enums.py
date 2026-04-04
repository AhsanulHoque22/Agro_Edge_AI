"""
Shared enumerations and encoding maps for the AgroEdge AI schema layer.

All Enum values and encoding dictionaries used across schema files are
defined here to avoid circular imports and ensure a single source of truth.
"""

from enum import Enum

# ── Growth Stage ──────────────────────────────────────────────────────────────

class GrowthStage(str, Enum):
    """
    Rice growth stages based on the BBCH scale.

    These stages cover all commercially cultivated rice types (indica, japonica,
    hybrid) without tying the system to any specific variety.

    DAT ranges (Days After Transplanting) are defined in feature_schema.yaml.
    """
    GERMINATION       = "0_germination"
    SEEDLING          = "1_seedling"
    TILLERING         = "2_tillering"
    STEM_ELONGATION   = "3_stem_elongation"
    BOOTING           = "4_booting"
    HEADING_FLOWERING = "5_heading_flowering"
    GRAIN_FILLING     = "6_grain_filling"
    RIPENING_MATURITY = "7_ripening_maturity"


class StageDetectionMethod(str, Enum):
    """
    Indicates how the current growth stage was determined.

    DATE_CALCULATED  — computed from planting_date using BBCH DAT thresholds (default)
    IMAGE_CLASSIFIED — detected by a phone image classifier (future Model 2b)
    MANUAL           — explicitly set by the farmer via the dashboard
    """
    DATE_CALCULATED  = "date_calculated"
    IMAGE_CLASSIFIED = "image_classified"
    MANUAL           = "manual"


# ── Farm / Soil ───────────────────────────────────────────────────────────────

class SoilType(str, Enum):
    LOAMY     = "loamy"
    SANDY     = "sandy"
    CLAY      = "clay"
    SILTY     = "silty"
    LOAMY_CLAY = "loamy_clay"


# ── Disease Detection ─────────────────────────────────────────────────────────

class DiseaseLabel(str, Enum):
    """
    Supported rice disease classes for the image classification model.

    Classes are sourced from publicly available datasets:
    PlantVillage, Kaggle RICE4, and IRRI Pathology collections.
    """
    HEALTHY               = "healthy"
    RICE_BLAST            = "rice_blast"
    BROWN_SPOT            = "brown_spot"
    BACTERIAL_LEAF_BLIGHT = "bacterial_leaf_blight"
    SHEATH_BLIGHT         = "sheath_blight"
    TUNGRO                = "tungro"
    LEAF_SCALD            = "leaf_scald"
    UNKNOWN               = "unknown"


class LabelingMethod(str, Enum):
    EXPERT         = "expert"
    COMMUNITY      = "community"
    MODEL_ASSISTED = "model_assisted"


# ── Dataset Provenance ────────────────────────────────────────────────────────

class LabelSource(str, Enum):
    """Records how a training label was produced."""
    SYNTHETIC             = "synthetic"
    MANUAL                = "manual"
    RULE_DERIVED          = "rule_derived"
    THINGSPEAK_HISTORICAL = "thingspeak_historical"


# ── Encoding Maps ─────────────────────────────────────────────────────────────
# These must match the encoding maps in configs/feature_schema.yaml exactly.

GROWTH_STAGE_ENCODING: dict[GrowthStage, int] = {
    GrowthStage.GERMINATION:       0,
    GrowthStage.SEEDLING:          1,
    GrowthStage.TILLERING:         2,
    GrowthStage.STEM_ELONGATION:   3,
    GrowthStage.BOOTING:           4,
    GrowthStage.HEADING_FLOWERING: 5,
    GrowthStage.GRAIN_FILLING:     6,
    GrowthStage.RIPENING_MATURITY: 7,
}

GROWTH_STAGE_DECODING: dict[int, GrowthStage] = {
    v: k for k, v in GROWTH_STAGE_ENCODING.items()
}

CROP_TYPE_ENCODING: dict[str, int] = {
    "rice": 0,
    # Additional crop types are added here when multi-crop support is introduced.
    # crop_type is NOT in the v1 feature vector because the system is rice-only.
}
