# Irrigation Model v1.1 (M1) - Feature Contract and Fallbacks

This document defines the additive v1.1 feature contract introduced for model intelligence upgrades.
It does not replace v1.0. Existing production/demo flows can continue using:

- `configs/feature_schema.yaml` (v1.0)
- `configs/model_config.yaml` (v1.0)

New files for v1.1 planning and implementation:

- `configs/feature_schema_v1_1.yaml`
- `configs/model_config_v1_1.yaml`

## Scope of M1

M1 adds schema/config and data dictionary definitions only. Runtime extraction and training pipeline code changes are M2/M3 work.

## New v1.1 features

| Feature | Type | Source | Default/Fallback (M1 contract) |
|---|---|---|---|
| `soil_moisture_percent_t_minus_1` | float | sensor_history | resolved from lag order rules below |
| `soil_moisture_percent_t_minus_2` | float | sensor_history | resolved from lag order rules below |
| `delta_soil_moisture_percent_t_minus_1` | float | derived | always computed after lag resolution |
| `delta_soil_moisture_percent_t_minus_2` | float | derived | always computed after lag resolution |
| `time_since_last_irrigation_hours` | float | irrigation log | from latest irrigation event; fallback `days_since_last_irrigation * 24.0` |
| `last_irrigation_duration_minutes` | float | irrigation log | from latest irrigation event; fallback `0.0` |
| `rain_expected_flag` | int | weather forecast | `0` when forecast unavailable (temporary M1 policy) |
| `temp_humidity_index` | float | derived | computed from air temperature + humidity; fallback to a deterministic formula result |

## Deterministic derived formulas

- `delta_soil_moisture_percent_t_minus_1 = soil_moisture_percent - soil_moisture_percent_t_minus_1`
- `delta_soil_moisture_percent_t_minus_2 = soil_moisture_percent - soil_moisture_percent_t_minus_2`
- `temp_humidity_index = air_temperature_celsius - (0.55 - 0.0055 * air_humidity_percent) * (air_temperature_celsius - 14.5)`

## Canonical lag fallback order

Use this exact order in both training and runtime:

1. `soil_moisture_percent_t_minus_1`
   - use previous env row moisture if available
   - else fallback to current `soil_moisture_percent`
2. `soil_moisture_percent_t_minus_2`
   - use second previous env row moisture if available
   - else fallback to resolved `soil_moisture_percent_t_minus_1`
3. Compute both delta features only after steps (1) and (2).

With this rule, deltas naturally become `0.0` only when lag values collapse to current.

## Compatibility rules

1. Keep v1.0 schema/config unchanged.
2. Version bump only through new files and exported bundle version (`v1.1.0`).
3. During rollout, v1.1 additional features are optional in schema and can be backfilled by defaults above.
4. Once runtime extraction is stable, optional fields can be revisited and upgraded to required in a later schema revision.

## Planned adoption path

1. M2: training data builder computes new features from ThingSpeak historical rows.
2. M2: hybrid labeling and fixed validation dataset introduced.
3. M3: runtime scheduler populates v1.1 features with cloud-primary and ring-buffer fallback.
4. M4: model comparison and staged rollout decision against fixed baseline.
