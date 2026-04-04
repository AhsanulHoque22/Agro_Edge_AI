"""
data_pipeline.schemas — public API for the AgroEdge AI schema layer.

Import all schema types from here. Do not import directly from submodules
to keep the rest of the codebase decoupled from internal file structure.

Usage:
    from data_pipeline.schemas import SensorRecord, FeatureRecord, TrainingRecord
"""

from .enums import (
    CROP_TYPE_ENCODING,
    GROWTH_STAGE_DECODING,
    GROWTH_STAGE_ENCODING,
    DiseaseLabel,
    GrowthStage,
    LabelingMethod,
    LabelSource,
    SoilType,
    StageDetectionMethod,
)
from .farm_metadata import (
    GROWTH_STAGE_DAT_THRESHOLDS,
    FarmMetadata,
    FieldMetadata,
    calculate_growth_stage,
    encode_growth_stage,
)
from .feature_record import FeatureRecord, compute_temp_humidity_index, compute_vpd
from .image_record import ImageRecord
from .sensor_record import SensorRecord
from .training_record import TrainingRecord

__all__ = [
    # Enums
    "GrowthStage",
    "StageDetectionMethod",
    "SoilType",
    "DiseaseLabel",
    "LabelingMethod",
    "LabelSource",
    # Encoding maps
    "GROWTH_STAGE_ENCODING",
    "GROWTH_STAGE_DECODING",
    "CROP_TYPE_ENCODING",
    # Farm metadata
    "FarmMetadata",
    "FieldMetadata",
    "GROWTH_STAGE_DAT_THRESHOLDS",
    "calculate_growth_stage",
    "encode_growth_stage",
    # Records
    "SensorRecord",
    "FeatureRecord",
    "TrainingRecord",
    "ImageRecord",
    # Utilities
    "compute_vpd",
    "compute_temp_humidity_index",
]
