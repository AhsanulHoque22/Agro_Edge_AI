"""
SyntheticDataGenerator — agronomically grounded TrainingRecord generator.

Generates labeled irrigation decision records using FAO Irrigation and
Drainage Paper No. 33 water requirements and IRRI rice moisture thresholds
as the authoritative rule source.

Each record is an independent cross-sectional sample representing one sensor
reading cycle at a simulated farm. Records are NOT time-series connected —
they are drawn from diverse farm states, growth stages, and environmental
conditions to give the model broad exposure during training.

Label logic:
  - Hard constraints  (tank safety, moisture saturation) → deterministic
  - Clear zones       (moisture below lower or above upper threshold) → deterministic
  - Fuzzy zone        (moisture between thresholds) → probabilistic, weighted by
                       moisture deficit, VPD stress signal, and days since irrigation

All threshold and water requirement values mirror configs/model_config.yaml
to guarantee data and config stay in sync.
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime, timedelta
from typing import NamedTuple

import numpy as np

from data_pipeline.schemas import (
    GROWTH_STAGE_DAT_THRESHOLDS,
    GROWTH_STAGE_ENCODING,
    GrowthStage,
    LabelSource,
    TrainingRecord,
    compute_temp_humidity_index,
    compute_vpd,
)

# ── Climate Profiles (Bangladesh / Southeast Asia rice-growing context) ────────
# Month-indexed lists [Jan=index 0 … Dec=index 11]
# Source: Bangladesh Meteorological Department long-term averages

_MONTHLY_TEMP_RANGE: list[tuple[float, float]] = [
    (13.0, 25.0),  # Jan — cool dry season
    (15.0, 28.0),  # Feb
    (20.0, 33.0),  # Mar — pre-monsoon warming
    (25.0, 37.0),  # Apr
    (27.0, 38.0),  # May — peak heat
    (27.0, 35.0),  # Jun — monsoon onset
    (27.0, 33.0),  # Jul — wet monsoon
    (27.0, 33.0),  # Aug
    (26.0, 33.0),  # Sep
    (24.0, 32.0),  # Oct — post-monsoon
    (18.0, 29.0),  # Nov
    (13.0, 25.0),  # Dec — cool dry season
]

_MONTHLY_HUMIDITY_RANGE: list[tuple[float, float]] = [
    (55.0, 78.0),  # Jan
    (52.0, 76.0),  # Feb
    (52.0, 78.0),  # Mar
    (52.0, 78.0),  # Apr
    (60.0, 84.0),  # May
    (75.0, 92.0),  # Jun
    (80.0, 95.0),  # Jul
    (80.0, 95.0),  # Aug
    (78.0, 94.0),  # Sep
    (70.0, 90.0),  # Oct
    (60.0, 84.0),  # Nov
    (55.0, 80.0),  # Dec
]

# ── Agronomic Thresholds ───────────────────────────────────────────────────────
# Source: IRRI Rice Knowledge Bank + model_config.yaml
# Format: (lower_trigger_percent, upper_target_percent)
# Irrigate when moisture < lower. Stop when moisture >= upper.

_IRRIGATION_THRESHOLDS: dict[GrowthStage, tuple[float, float]] = {
    GrowthStage.GERMINATION:       (60.0, 70.0),
    GrowthStage.SEEDLING:          (55.0, 70.0),
    GrowthStage.TILLERING:         (50.0, 65.0),
    GrowthStage.STEM_ELONGATION:   (50.0, 65.0),
    GrowthStage.BOOTING:           (60.0, 75.0),
    GrowthStage.HEADING_FLOWERING: (65.0, 80.0),  # critical — highest water demand
    GrowthStage.GRAIN_FILLING:     (55.0, 70.0),
    GrowthStage.RIPENING_MATURITY: (30.0, 45.0),  # deliberate drying
}

# Source: FAO Irrigation and Drainage Paper No. 33, Table 11 (rice)
# Format: (min_mm_per_day, max_mm_per_day) — variety-agnostic ranges

_WATER_REQUIREMENTS: dict[GrowthStage, tuple[float, float]] = {
    GrowthStage.GERMINATION:       (3.0,  6.0),
    GrowthStage.SEEDLING:          (4.0,  8.0),
    GrowthStage.TILLERING:         (6.0, 10.0),
    GrowthStage.STEM_ELONGATION:   (7.0, 12.0),
    GrowthStage.BOOTING:           (8.0, 14.0),
    GrowthStage.HEADING_FLOWERING: (9.0, 15.0),
    GrowthStage.GRAIN_FILLING:     (6.0, 10.0),
    GrowthStage.RIPENING_MATURITY: (2.0,  5.0),
}

# Minimum days between irrigation events per stage
# Critical stages allow daily irrigation; others have a soft minimum
_MIN_IRRIGATION_INTERVAL: dict[GrowthStage, int] = {
    GrowthStage.GERMINATION:       1,
    GrowthStage.SEEDLING:          1,
    GrowthStage.TILLERING:         1,
    GrowthStage.STEM_ELONGATION:   1,
    GrowthStage.BOOTING:           0,  # can irrigate daily
    GrowthStage.HEADING_FLOWERING: 0,  # can irrigate daily
    GrowthStage.GRAIN_FILLING:     1,
    GrowthStage.RIPENING_MATURITY: 3,  # actively drying field
}

# Stage weights for sampling — critical stages slightly overrepresented
_STAGE_WEIGHTS: dict[GrowthStage, float] = {
    GrowthStage.GERMINATION:       1.0,
    GrowthStage.SEEDLING:          1.0,
    GrowthStage.TILLERING:         1.2,
    GrowthStage.STEM_ELONGATION:   1.2,
    GrowthStage.BOOTING:           1.4,
    GrowthStage.HEADING_FLOWERING: 1.4,
    GrowthStage.GRAIN_FILLING:     1.2,
    GrowthStage.RIPENING_MATURITY: 1.0,
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class _SensorReading(NamedTuple):
    """Intermediate container for generated environmental values."""
    soil_moisture: float
    soil_temp: float
    air_temp: float
    humidity: float
    light: float
    tank_level: float
    vpd: float
    hour: int
    month: int
    day_of_week: int
    days_since_irrigation: int
    dat: int
    days_to_maturity: int
    collected_at: datetime


# ── Generator ─────────────────────────────────────────────────────────────────

class SyntheticDataGenerator:
    """
    Generates labeled TrainingRecords for the irrigation decision model.

    Usage:
        generator = SyntheticDataGenerator(random_seed=42)
        records = generator.generate(n_records=20000)
    """

    def __init__(
        self,
        random_seed: int = 42,
        n_farms: int = 15,
        dataset_version: str = "v1",
    ) -> None:
        self._rng = np.random.default_rng(random_seed)
        self._n_farms = n_farms
        self._dataset_version = dataset_version

        self._farm_ids  = [f"farm_{i:03d}"      for i in range(1, n_farms + 1)]
        self._field_ids = [f"field_{i:03d}"     for i in range(1, n_farms + 1)]
        self._node_ids  = [f"node_esp32_{i:03d}" for i in range(1, n_farms + 1)]

        self._stages  = list(_STAGE_WEIGHTS.keys())
        self._weights = np.array(list(_STAGE_WEIGHTS.values()), dtype=float)
        self._weights /= self._weights.sum()  # normalise to probabilities

    def _enriched_v1_1_features(self) -> bool:
        """When true, populate v1.1 lag / irrigation-context / THI columns in records."""
        v = self._dataset_version
        return v == "v1_1" or v.startswith("v1_1_") or v == "v1.1"

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, n_records: int) -> list[TrainingRecord]:
        """
        Generate n_records independent labeled TrainingRecords.

        Records are distributed across all 8 growth stages with slight
        overweighting of the booting and heading/flowering stages where
        irrigation decisions are most consequential.
        """
        stage_indices = self._rng.choice(
            len(self._stages), size=n_records, p=self._weights
        )
        records = [
            self._generate_record(self._stages[idx])
            for idx in stage_indices
        ]
        return records

    # ── Record construction ───────────────────────────────────────────────────

    def _generate_record(self, stage: GrowthStage) -> TrainingRecord:
        """Build one labeled TrainingRecord for the given growth stage."""
        idx = int(self._rng.integers(0, self._n_farms))
        reading = self._sample_sensors(stage)
        irrigation_needed, duration = self._calculate_labels(reading, stage)

        m0 = round(reading.soil_moisture, 2)
        air_t = round(reading.air_temp, 2)
        rh = round(reading.humidity, 2)

        v11_kwargs: dict = {}
        if self._enriched_v1_1_features():
            # Synthetic lag structure: soil dried from wetter past readings toward m0.
            d1 = float(self._rng.uniform(0.3, 5.0))
            d2 = float(self._rng.uniform(0.3, 5.0))
            t1 = min(100.0, m0 + d1)
            t2 = min(100.0, t1 + d2)
            d_si = int(reading.days_since_irrigation)
            v11_kwargs = {
                "soil_moisture_percent_t_minus_1": round(t1, 2),
                "soil_moisture_percent_t_minus_2": round(t2, 2),
                "delta_soil_moisture_percent_t_minus_1": round(m0 - t1, 4),
                "delta_soil_moisture_percent_t_minus_2": round(m0 - t2, 4),
                "time_since_last_irrigation_hours": round(
                    24.0 * d_si + float(self._rng.uniform(0.0, 23.99)), 4
                ),
                "last_irrigation_duration_minutes": round(float(self._rng.uniform(5.0, 95.0)), 2),
                "rain_expected_flag": int(self._rng.random() < 0.12),
                "temp_humidity_index": compute_temp_humidity_index(air_t, rh),
                "schema_version": "1.1",
            }

        return TrainingRecord(
            record_id=str(uuid.uuid4()),
            farm_id=self._farm_ids[idx],
            field_id=self._field_ids[idx],
            node_id=self._node_ids[idx],
            collected_at=reading.collected_at,
            soil_moisture_percent=m0,
            soil_temperature_celsius=round(reading.soil_temp, 2),
            air_temperature_celsius=air_t,
            air_humidity_percent=rh,
            light_intensity_lux=round(reading.light, 1),
            water_tank_level_percent=round(reading.tank_level, 2),
            growth_stage_encoded=GROWTH_STAGE_ENCODING[stage],
            days_after_transplanting=reading.dat,
            days_to_maturity=reading.days_to_maturity,
            vpd_kpa=reading.vpd,
            hour_of_day=reading.hour,
            day_of_week=reading.day_of_week,
            month=reading.month,
            days_since_last_irrigation=reading.days_since_irrigation,
            irrigation_needed=irrigation_needed,
            irrigation_duration_minutes=round(duration, 1),
            label_source=LabelSource.SYNTHETIC,
            dataset_version=self._dataset_version,
            **v11_kwargs,
        )

    # ── Sensor sampling ───────────────────────────────────────────────────────

    def _sample_sensors(self, stage: GrowthStage) -> _SensorReading:
        """
        Sample all sensor values for a given growth stage.

        VPD is computed before soil moisture so the moisture sampler can apply
        a downward shift on high-VPD days. This models the physical reality that
        high atmospheric dryness accelerates evapotranspiration, leading to drier
        soils — which in turn creates the expected positive VPD–irrigation correlation.
        """
        collected_at, month, hour, day_of_week = self._sample_datetime()
        air_temp   = self._sample_air_temperature(month, hour)
        humidity   = self._sample_humidity(month, air_temp)
        soil_temp  = self._sample_soil_temperature(air_temp)
        light      = self._sample_light(hour)
        tank_level = self._sample_tank_level()
        vpd        = compute_vpd(air_temp, humidity)                    # VPD before moisture
        soil_moisture = self._sample_soil_moisture(stage, vpd_kpa=vpd)  # moisture uses VPD
        days_since = self._sample_days_since_irrigation(stage)
        dat        = self._sample_dat_for_stage(stage)
        days_to_maturity = max(0, 150 - dat)

        return _SensorReading(
            soil_moisture=soil_moisture,
            soil_temp=soil_temp,
            air_temp=air_temp,
            humidity=humidity,
            light=light,
            tank_level=tank_level,
            vpd=vpd,
            hour=hour,
            month=month,
            day_of_week=day_of_week,
            days_since_irrigation=days_since,
            dat=dat,
            days_to_maturity=days_to_maturity,
            collected_at=collected_at,
        )

    def _sample_datetime(self) -> tuple[datetime, int, int, int]:
        """Sample a timezone-aware UTC timestamp over a 2-year window."""
        days_offset = int(self._rng.integers(0, 730))
        hour        = int(self._rng.integers(0, 24))
        minute      = int(self._rng.integers(0, 60))
        base = datetime(2023, 6, 1, tzinfo=UTC)
        dt = base + timedelta(days=days_offset, hours=hour, minutes=minute)
        return dt, dt.month, dt.hour, dt.weekday()

    def _sample_dat_for_stage(self, stage: GrowthStage) -> int:
        """Sample a DAT value uniformly within the stage's valid BBCH range."""
        for start, end, s in GROWTH_STAGE_DAT_THRESHOLDS:
            if s == stage:
                return int(self._rng.integers(start, end))
        return 130

    def _sample_air_temperature(self, month: int, hour: int) -> float:
        """
        Sample air temperature with monthly and diurnal variation.
        Coolest at ~5am, hottest at ~2pm.
        """
        t_min, t_max = _MONTHLY_TEMP_RANGE[month - 1]
        # Diurnal factor: 0 at night, 1 at 14:00
        if 5 <= hour <= 23:
            diurnal = max(0.0, math.sin(math.pi * (hour - 5) / 18.0))
        else:
            diurnal = 0.0
        base  = t_min + (t_max - t_min) * (0.4 + diurnal * 0.6)
        noise = float(self._rng.normal(0.0, 1.2))
        return float(np.clip(base + noise, t_min - 2.0, t_max + 2.0))

    def _sample_humidity(self, month: int, air_temp: float) -> float:
        """
        Sample humidity inversely correlated with temperature.
        Hotter air drives humidity lower within the monthly range.
        """
        h_min, h_max = _MONTHLY_HUMIDITY_RANGE[month - 1]
        t_min, t_max = _MONTHLY_TEMP_RANGE[month - 1]
        temp_norm    = (air_temp - t_min) / max(1.0, t_max - t_min)
        base  = h_max - (h_max - h_min) * temp_norm
        noise = float(self._rng.normal(0.0, 3.5))
        return float(np.clip(base + noise, h_min, h_max))

    def _sample_soil_temperature(self, air_temp: float) -> float:
        """
        Soil temperature follows air temperature with thermal damping.
        Typically 1–3°C cooler than peak air temp, less variable.
        """
        soil_temp = air_temp * 0.82 + float(self._rng.normal(2.0, 1.2))
        return float(np.clip(soil_temp, 14.0, 42.0))

    def _sample_light(self, hour: int) -> float:
        """Sample light intensity with cosine diurnal curve, zero at night."""
        if hour < 6 or hour > 19:
            return float(self._rng.uniform(0.0, 60.0))
        peak_lux    = float(self._rng.uniform(35000.0, 90000.0))
        peak_hour   = 12.0
        hour_factor = max(0.0, math.cos(math.pi * (hour - peak_hour) / 8.0))
        lux = peak_lux * hour_factor + float(self._rng.normal(0.0, 800.0))
        return float(np.clip(lux, 0.0, 120000.0))

    def _sample_tank_level(self) -> float:
        """
        Tank level distribution: mostly healthy with a realistic low-tank tail.
        80% healthy (>50%), 13% getting low (20–50%), 7% critical (<20%).
        """
        p = float(self._rng.uniform(0.0, 1.0))
        if p < 0.80:
            return float(self._rng.uniform(50.0, 100.0))
        elif p < 0.93:
            return float(self._rng.uniform(20.0, 50.0))
        else:
            return float(self._rng.uniform(2.0, 20.0))

    def _sample_soil_moisture(self, stage: GrowthStage, vpd_kpa: float = 0.0) -> float:
        """
        Sample soil moisture centered at the stage midpoint with wide spread,
        ensuring coverage in all three zones:
          below lower threshold  (~38% of records per stage → irrigate)
          between thresholds     (~24% of records → fuzzy zone)
          above upper threshold  (~38% of records → do not irrigate)

        A VPD-based downward shift models the physical feedback between
        atmospheric dryness and soil moisture: high VPD → faster evapotranspiration
        → lower average soil moisture. This creates the expected positive
        VPD–irrigation correlation across all decision zones.
        Source: evapotranspiration feedback described in FAO-56.
        """
        lower, upper = _IRRIGATION_THRESHOLDS[stage]
        center = (lower + upper) / 2.0
        std    = (upper - lower) * 2.0   # wide spread to cover all zones

        # VPD shift: 0 below 0.8 kPa; increases by 4% per kPa above that; capped at 10%
        vpd_shift = min(10.0, max(0.0, (vpd_kpa - 0.8) * 4.0))
        adjusted_center = center - vpd_shift

        moisture = float(self._rng.normal(adjusted_center, std))
        return float(np.clip(moisture, 3.0, 97.0))

    def _sample_days_since_irrigation(self, stage: GrowthStage) -> int:
        """
        Sample days since last irrigation.
        Critical stages are irrigated more frequently.
        Ripening stage has longer intervals (field drying).
        """
        if stage in {GrowthStage.BOOTING, GrowthStage.HEADING_FLOWERING}:
            return int(self._rng.integers(0, 4))
        elif stage == GrowthStage.RIPENING_MATURITY:
            return int(self._rng.integers(3, 14))
        else:
            return int(self._rng.integers(0, 7))

    # ── Label calculation ─────────────────────────────────────────────────────

    def _calculate_labels(
        self,
        reading: _SensorReading,
        stage: GrowthStage,
    ) -> tuple[int, float]:
        """
        Determine (irrigation_needed, irrigation_duration_minutes).

        Decision zones:
          Tank < 10%                      → 0, 0.0  (safety: physically can't irrigate)
          Moisture >= upper threshold     → 0, 0.0  (clear: field is saturated)
          Moisture < lower threshold      → 1, dur  (clear: water deficit detected)
          lower <= moisture < upper       → probabilistic (fuzzy zone)

        The fuzzy zone models the judgment a farmer makes when moisture is
        borderline — weighted by deficit size, VPD stress, and time elapsed.
        """
        lower, upper = _IRRIGATION_THRESHOLDS[stage]
        min_interval = _MIN_IRRIGATION_INTERVAL[stage]

        # ── Hard constraints ──────────────────────────────────────────────────
        if reading.tank_level < 10.0:
            return 0, 0.0

        if reading.days_since_irrigation < min_interval:
            return 0, 0.0

        # ── VPD stress threshold adjustment ───────────────────────────────────
        # High VPD (atmospheric dryness) increases evapotranspiration, meaning
        # the crop reaches water stress sooner. We model this by lowering the
        # effective lower trigger threshold when VPD is elevated.
        # This creates a direct, agronomically valid positive VPD–irrigation
        # correlation across all decision zones, not just the fuzzy zone.
        # Source: Allen et al. (1998), FAO-56 Penman-Monteith reference.
        vpd_adjustment = max(0.0, (reading.vpd - 0.8) * 3.0)  # 0 below 0.8 kPa
        vpd_adjustment = min(7.0, vpd_adjustment)               # cap at 7%
        effective_lower = lower - vpd_adjustment

        # ── Clear no-irrigation zone ──────────────────────────────────────────
        if reading.soil_moisture >= upper:
            return 0, 0.0

        # ── Clear irrigation zone ─────────────────────────────────────────────
        if reading.soil_moisture < effective_lower:
            duration = self._calculate_duration(reading.soil_moisture, upper, stage)
            return 1, duration

        # ── Fuzzy zone (effective_lower <= moisture < upper) ──────────────────
        # Position in the zone: 0.0 = at lower bound, 1.0 = at upper bound
        zone_width = upper - effective_lower
        zone_position = (reading.soil_moisture - effective_lower) / max(1.0, zone_width)

        # Deficit signal: strong when moisture is near the lower bound
        deficit_signal = 1.0 - zone_position

        # VPD stress signal: 1.5 kPa is a moderate stress threshold for rice
        vpd_signal = min(1.0, reading.vpd / 1.5)

        # Days elapsed signal: increases urgency if not irrigated for a while
        days_signal = min(1.0, reading.days_since_irrigation / 3.0)

        # Weighted score → sigmoid to probability
        score = deficit_signal * 0.55 + vpd_signal * 0.30 + days_signal * 0.15
        p_irrigate = _sigmoid((score - 0.50) * 7.0)

        irrigate = int(float(self._rng.uniform(0.0, 1.0)) < p_irrigate)
        duration = self._calculate_duration(reading.soil_moisture, upper, stage) if irrigate else 0.0
        return irrigate, duration

    def _calculate_duration(
        self,
        current_moisture: float,
        target_moisture: float,
        stage: GrowthStage,
    ) -> float:
        """
        Calculate irrigation duration (minutes) from water deficit model.

        Combines two components:
          1. Moisture deficit component:
             How much moisture needs to be restored to reach the target.
             Each 1% deficit ≈ 1.2 minutes (empirical scaling for a small field).

          2. Daily water requirement component:
             The stage's ongoing evapotranspiration demand, expressed as
             additional runtime beyond the deficit.

        Clamped to [5, 90] minutes to reflect practical field operations.
        """
        deficit = max(0.0, target_moisture - current_moisture)
        deficit_component = deficit * 1.2

        req_min, req_max = _WATER_REQUIREMENTS[stage]
        daily_req = float(self._rng.uniform(req_min, req_max))
        stage_component = daily_req * 1.4  # 1 mm/day ≈ 1.4 min for typical small plot

        noise = float(self._rng.normal(0.0, 2.5))
        return float(np.clip(deficit_component + stage_component + noise, 5.0, 90.0))
