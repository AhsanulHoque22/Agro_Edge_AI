"""
ThingSpeak ingestion: fetch channel history and build ``TrainingRecord`` datasets.
"""

from .thingspeak_fetch import FetchResult, fetch_feeds_timerange, fetch_latest_feeds
from .thingspeak_training_builder import (
    BuilderStats,
    build_training_records,
    feeds_to_env_dataframe,
    feeds_to_irrigation_dataframe,
)

__all__ = [
    "FetchResult",
    "fetch_feeds_timerange",
    "fetch_latest_feeds",
    "BuilderStats",
    "build_training_records",
    "feeds_to_env_dataframe",
    "feeds_to_irrigation_dataframe",
]
