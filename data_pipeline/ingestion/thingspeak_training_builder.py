"""
Build ``TrainingRecord`` rows from ThingSpeak environmental + irrigation channels.

Labels come from the **irrigation log** (duration > 0 → ``irrigation_needed=1``).
Features are taken from the **environmental** channel using the reading at or
just before the irrigation event (backward ``merge_asof``).

Positive-only pulls are valid; add **negative** rows by sampling env readings that
are farther than ``association_hours`` from any irrigation event.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline.schemas import (
    LabelSource,
    TrainingRecord,
    calculate_growth_stage,
    compute_temp_humidity_index,
    compute_vpd,
    encode_growth_stage,
)

# Trigger codes from ``thingspeak_client.client`` (field2 irrigation channel)
_TRIGGER_CODE_TO_NAME = {0: "ai_model", 1: "manual_override", 2: "threshold_fallback"}


@dataclass(frozen=True)
class BuilderStats:
    n_irrigation_rows: int
    n_positive_merged: int
    n_positive_skipped_no_env: int
    n_negative_added: int
    n_positive_skipped_hybrid_rules: int = 0
    n_negative_skipped_hybrid_rules: int = 0


def env_row_to_series(row: dict[str, Any]) -> dict[str, Any] | None:
    """Map one ThingSpeak env feed row to semantic fields (floats)."""
    try:
        return {
            "entry_id": row.get("entry_id"),
            "created_at": _parse_ts(row.get("created_at")),
            "soil_moisture_percent": _to_float(row.get("field1")),
            "soil_temperature_celsius": _to_float(row.get("field2")),
            "air_temperature_celsius": _to_float(row.get("field3")),
            "air_humidity_percent": _to_float(row.get("field4")),
            "light_intensity_lux": _to_float(row.get("field5")),
            "water_tank_level_percent": _to_float(row.get("field6")),
            "water_flow_rate_lph": _to_float_or_none(row.get("field7")),
        }
    except ValueError:
        return None


def irrigation_row_to_series(row: dict[str, Any]) -> dict[str, Any] | None:
    """Map one irrigation feed row; skip invalid or zero-duration events."""
    try:
        ts = _parse_ts(row.get("created_at"))
        duration = float(row.get("field1") or 0)
    except (TypeError, ValueError):
        return None
    if duration <= 0:
        return None
    try:
        trigger_code = int(float(row.get("field2") or 0))
    except (TypeError, ValueError):
        trigger_code = 0
    moisture_before = _to_float_or_none(row.get("field3"))
    moisture_after = _to_float_or_none(row.get("field4"))
    return {
        "entry_id": row.get("entry_id"),
        "created_at": ts,
        "irrigation_duration_minutes": duration,
        "trigger_code": trigger_code,
        "trigger_name": _TRIGGER_CODE_TO_NAME.get(trigger_code, "ai_model"),
        "soil_moisture_before_percent": moisture_before,
        "soil_moisture_after_percent": moisture_after,
    }


def feeds_to_env_dataframe(feeds: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for f in feeds:
        parsed = env_row_to_series(f)
        if parsed is None:
            continue
        rows.append(parsed)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)
    out["created_at"] = pd.to_datetime(out["created_at"], utc=True)
    return out


def feeds_to_irrigation_dataframe(feeds: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for f in feeds:
        parsed = irrigation_row_to_series(f)
        if parsed is None:
            continue
        rows.append(parsed)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)
    out["created_at"] = pd.to_datetime(out["created_at"], utc=True)
    return out


def _days_since_last_irrigation(timestamps: pd.Series, irr_times: pd.Series) -> np.ndarray:
    """For each timestamp, whole days since previous irrigation (0 if same day)."""
    irr_sorted = pd.DatetimeIndex(irr_times, tz="UTC").asi8.astype(np.int64, copy=False)
    irr_sorted = np.sort(irr_sorted)
    out = np.zeros(len(timestamps), dtype=np.int64)
    ts_ns = pd.DatetimeIndex(timestamps, tz="UTC").asi8.astype(np.int64, copy=False)
    for i, t in enumerate(ts_ns):
        pos = int(np.searchsorted(irr_sorted, t, side="right") - 1)
        if pos < 0:
            out[i] = 14  # default: no prior irrigation in window
            continue
        delta_s = (t - irr_sorted[pos]) / 10**9
        out[i] = max(0, int(delta_s // 86400))
    return out


def _lag_soil_moisture(env_df: pd.DataFrame, ts: datetime) -> tuple[float, float]:
    hist = env_df[env_df["created_at"] <= ts].sort_values("created_at")
    if hist.empty:
        return 0.0, 0.0
    current = float(hist.iloc[-1]["soil_moisture_percent"])
    t1 = float(hist.iloc[-2]["soil_moisture_percent"]) if len(hist) >= 2 else current
    t2 = float(hist.iloc[-3]["soil_moisture_percent"]) if len(hist) >= 3 else t1
    return t1, t2


def _last_irrigation_context(irr_df: pd.DataFrame, ts: datetime) -> tuple[float, float]:
    hist = irr_df[irr_df["created_at"] < ts].sort_values("created_at")
    if hist.empty:
        return 24.0 * 14.0, 0.0
    last = hist.iloc[-1]
    delta_hours = max(0.0, (ts - last["created_at"]).total_seconds() / 3600.0)
    return float(delta_hours), float(last["irrigation_duration_minutes"])


def build_training_records(
    env_feeds: list[dict[str, Any]],
    irrigation_feeds: list[dict[str, Any]],
    *,
    farm_id: str,
    field_id: str,
    node_id: str,
    planting_date: date,
    expected_harvest_date: date | None,
    dataset_version: str,
    association_hours: float = 2.0,
    max_negatives: int | None = None,
    negatives_ratio: float | None = None,
    random_seed: int = 42,
    hybrid_labeling: bool = False,
    hybrid_min_negative_soil_moisture: float = 50.0,
    hybrid_max_negative_vpd_kpa: float = 1.2,
) -> tuple[list[TrainingRecord], BuilderStats]:
    """
    Merge env + irrigation history into ``TrainingRecord`` instances.

    Args:
        association_hours: Negative samples must be at least this far (hours)
            from any irrigation ``created_at``.
        max_negatives: Cap negative rows (after ratio).
        negatives_ratio: If set, add up to ``int(n_pos * ratio)`` negatives.
    """
    env_df = feeds_to_env_dataframe(env_feeds)
    irr_df = feeds_to_irrigation_dataframe(irrigation_feeds)

    records: list[TrainingRecord] = []
    if env_df.empty or irr_df.empty:
        stats = BuilderStats(
            n_irrigation_rows=len(irr_df),
            n_positive_merged=0,
            n_positive_skipped_no_env=len(irr_df) if not env_df.empty else 0,
            n_negative_added=0,
        )
        return records, stats

    left = irr_df.sort_values("created_at")
    right = env_df.sort_values("created_at")
    merged = pd.merge_asof(
        left,
        right,
        on="created_at",
        direction="backward",
        suffixes=("_irr", "_env"),
    )

    irr_times_all = irr_df["created_at"]

    n_skipped = 0
    n_positive_skipped_hybrid = 0
    for _, row in merged.iterrows():
        if pd.isna(row.get("soil_moisture_percent")):
            n_skipped += 1
            continue
        if hybrid_labeling and float(row.get("water_tank_level_percent", 0.0)) < 10.0:
            # Logged irrigation with critically low tank is treated as low-confidence.
            n_positive_skipped_hybrid += 1
            continue

        ts: datetime = row["created_at"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        ref_d = ts.astimezone(UTC).date()
        dat = max(0, (ref_d - planting_date).days)
        stage = calculate_growth_stage(dat)
        stage_enc = encode_growth_stage(stage)

        if expected_harvest_date is not None:
            days_to_maturity = (expected_harvest_date - ref_d).days
        else:
            days_to_maturity = max(0, 150 - dat)

        moist = float(row["soil_moisture_percent"])
        air_t = float(row["air_temperature_celsius"])
        rh = float(row["air_humidity_percent"])
        vpd = compute_vpd(air_t, rh)

        dsirr = int(
            _days_since_last_irrigation(
                pd.Series([ts], dtype="datetime64[ns, UTC]"),
                irr_times_all,
            )[0]
        )
        t1, t2 = _lag_soil_moisture(env_df, ts)
        delta_t1 = moist - t1
        delta_t2 = moist - t2
        time_since_last_hours, last_irr_duration = _last_irrigation_context(irr_df, ts)
        thi = compute_temp_humidity_index(air_t, rh)

        mo = ts.month
        hod = ts.hour
        dow_mon0 = ts.weekday()

        duration = float(row["irrigation_duration_minutes"])
        rec = TrainingRecord(
            record_id=str(uuid.uuid4()),
            farm_id=farm_id,
            field_id=field_id,
            node_id=node_id,
            collected_at=ts.astimezone(UTC),
            soil_moisture_percent=round(moist, 2),
            soil_temperature_celsius=round(float(row["soil_temperature_celsius"]), 2),
            air_temperature_celsius=round(air_t, 2),
            air_humidity_percent=round(rh, 2),
            light_intensity_lux=round(float(row["light_intensity_lux"]), 2),
            water_tank_level_percent=round(float(row["water_tank_level_percent"]), 2),
            growth_stage_encoded=stage_enc,
            days_after_transplanting=dat,
            days_to_maturity=days_to_maturity,
            vpd_kpa=vpd,
            hour_of_day=hod,
            day_of_week=dow_mon0,
            month=mo,
            days_since_last_irrigation=dsirr,
            soil_moisture_percent_t_minus_1=round(t1, 2),
            soil_moisture_percent_t_minus_2=round(t2, 2),
            delta_soil_moisture_percent_t_minus_1=round(delta_t1, 2),
            delta_soil_moisture_percent_t_minus_2=round(delta_t2, 2),
            time_since_last_irrigation_hours=round(time_since_last_hours, 2),
            last_irrigation_duration_minutes=round(last_irr_duration, 2),
            rain_expected_flag=0,
            temp_humidity_index=thi,
            irrigation_needed=1,
            irrigation_duration_minutes=round(duration, 2),
            label_source=LabelSource.THINGSPEAK_HISTORICAL,
            dataset_version=dataset_version,
        )
        records.append(rec)

    n_pos = len(records)
    n_neg_target = 0
    if negatives_ratio is not None and n_pos > 0:
        n_neg_target = max(0, int(n_pos * float(negatives_ratio)))
    if max_negatives is not None:
        if n_neg_target > 0:
            n_neg_target = min(n_neg_target, max_negatives)
        else:
            n_neg_target = max_negatives

    n_neg_added = 0
    n_neg_skipped_hybrid = 0
    if n_neg_target and n_neg_target > 0 and len(env_df) > 0:
        rng = np.random.default_rng(random_seed)
        assoc = timedelta(hours=association_hours)
        irr_ts_set = sorted(irr_df["created_at"].tolist())

        def near_irrigation(t: datetime) -> bool:
            for it in irr_ts_set:
                if abs((t - it).total_seconds()) <= assoc.total_seconds():
                    return True
            return False

        candidates = []
        for _, erow in env_df.iterrows():
            ts = erow["created_at"]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if near_irrigation(ts):
                continue
            candidates.append(erow)

        if candidates:
            take = min(n_neg_target, len(candidates))
            idx = rng.choice(len(candidates), size=take, replace=False)
            for i in idx:
                erow = candidates[int(i)]
                ts = erow["created_at"]
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                ts = ts.astimezone(UTC)
                ref_d = ts.date()
                dat = max(0, (ref_d - planting_date).days)
                stage = calculate_growth_stage(dat)
                stage_enc = encode_growth_stage(stage)
                if expected_harvest_date is not None:
                    days_to_maturity = (expected_harvest_date - ref_d).days
                else:
                    days_to_maturity = max(0, 150 - dat)

                air_t = float(erow["air_temperature_celsius"])
                rh = float(erow["air_humidity_percent"])
                vpd = compute_vpd(air_t, rh)
                soil = float(erow["soil_moisture_percent"])
                if hybrid_labeling:
                    # High-confidence negative: moist enough and low atmospheric stress.
                    if soil < float(hybrid_min_negative_soil_moisture) or vpd > float(
                        hybrid_max_negative_vpd_kpa
                    ):
                        n_neg_skipped_hybrid += 1
                        continue
                dsirr = int(
                    _days_since_last_irrigation(
                        pd.Series([ts], dtype="datetime64[ns, UTC]"),
                        irr_times_all,
                    )[0]
                )
                t1, t2 = _lag_soil_moisture(env_df, ts)
                delta_t1 = soil - t1
                delta_t2 = soil - t2
                time_since_last_hours, last_irr_duration = _last_irrigation_context(irr_df, ts)
                thi = compute_temp_humidity_index(air_t, rh)

                rec = TrainingRecord(
                    record_id=str(uuid.uuid4()),
                    farm_id=farm_id,
                    field_id=field_id,
                    node_id=node_id,
                    collected_at=ts,
                    soil_moisture_percent=round(soil, 2),
                    soil_temperature_celsius=round(float(erow["soil_temperature_celsius"]), 2),
                    air_temperature_celsius=round(air_t, 2),
                    air_humidity_percent=round(rh, 2),
                    light_intensity_lux=round(float(erow["light_intensity_lux"]), 2),
                    water_tank_level_percent=round(float(erow["water_tank_level_percent"]), 2),
                    growth_stage_encoded=stage_enc,
                    days_after_transplanting=dat,
                    days_to_maturity=days_to_maturity,
                    vpd_kpa=vpd,
                    hour_of_day=ts.hour,
                    day_of_week=ts.weekday(),
                    month=ts.month,
                    days_since_last_irrigation=dsirr,
                    soil_moisture_percent_t_minus_1=round(t1, 2),
                    soil_moisture_percent_t_minus_2=round(t2, 2),
                    delta_soil_moisture_percent_t_minus_1=round(delta_t1, 2),
                    delta_soil_moisture_percent_t_minus_2=round(delta_t2, 2),
                    time_since_last_irrigation_hours=round(time_since_last_hours, 2),
                    last_irrigation_duration_minutes=round(last_irr_duration, 2),
                    rain_expected_flag=0,
                    temp_humidity_index=thi,
                    irrigation_needed=0,
                    irrigation_duration_minutes=0.0,
                    label_source=(
                        LabelSource.RULE_DERIVED if hybrid_labeling else LabelSource.THINGSPEAK_HISTORICAL
                    ),
                    dataset_version=dataset_version,
                )
                records.append(rec)
                n_neg_added += 1

    stats = BuilderStats(
        n_irrigation_rows=len(irr_df),
        n_positive_merged=n_pos,
        n_positive_skipped_no_env=n_skipped,
        n_negative_added=n_neg_added,
        n_positive_skipped_hybrid_rules=n_positive_skipped_hybrid,
        n_negative_skipped_hybrid_rules=n_neg_skipped_hybrid,
    )
    return records, stats


def _parse_ts(value: Any) -> datetime:
    if value is None:
        raise ValueError("missing created_at")
    s = str(value).replace("Z", "+00:00")
    ts = datetime.fromisoformat(s)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _to_float(value: Any) -> float:
    if value is None or value == "":
        raise ValueError("expected numeric field")
    return float(value)


def _to_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
