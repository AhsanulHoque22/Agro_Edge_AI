# ThingSpeak → training data (retraining)

Pull historical **environmental** and **irrigation log** channels, align rows in time, and write `TrainingRecord` Parquet compatible with `scripts/prepare_features.py`.

## Prerequisites

- Same field mapping as `configs/thingspeak_channels.yaml` (env channel fields 1–7, irrigation log field1 = minutes).
- Read API keys for **both** channels in the environment (see `.env.example`).
- At least one irrigation log row with `field1 > 0` in the selected window (otherwise there is nothing to label as “irrigated”).

## Command

From `agroedge_ai/`:

```bash
python scripts/pull_thingspeak_training_data.py \
  --days 120 \
  --farm-id farm_001 \
  --field-id field_001 \
  --node-id node_esp32_001 \
  --planting-date 2024-06-01 \
  --harvest-date 2024-11-15 \
  --version ts_2024_q3 \
  --negatives-ratio 0.5
```

Outputs `datasets/raw/ts_2024_q3/training_records.parquet` by default.

- **`--audit-profile field`** (default): tank safety + physical bounds + minimum row count. Use **`synthetic`** only if you merged a large, balanced dataset and want the same checks as the generator.
- **Negatives**: env samples farther than `--association-hours` (default 2) from any irrigation timestamp; label `irrigation_needed=0`, `irrigation_duration_minutes=0`.
- **Hybrid labeling (M2, optional):**
  - `--hybrid-labeling`
  - `--hybrid-min-negative-soil-moisture 50`
  - `--hybrid-max-negative-vpd-kpa 1.2`
  - Applies confidence filters and reports skipped low-confidence rows.

`--range-start` / `--range-end` accept ISO datetimes and override `--days`.

## Retrain pipeline

```bash
python scripts/prepare_features.py --version ts_2024_q3
# or explicit file:
python scripts/prepare_features.py --version ts_2024_q3 \
  --raw-parquet datasets/raw/ts_2024_q3/training_records.parquet

python scripts/train_irrigation_model.py   # ensure config points at processed version
```

Ground truth for positives is **what was actually logged** (duration > 0). Quality depends on ESP32/solenoid plumbing and logging discipline; consider merging with synthetic data for cold start.

## Merge with synthetic data

```bash
python scripts/merge_training_parquet.py \
  --input datasets/raw/v1/training_records.parquet \
  --input datasets/raw/ts_live/training_records.parquet \
  --dataset-version merged_v3
```

Then `prepare_features.py --version merged_v3` and retrain.

## Fixed validation set + version comparison (M2)

Build a deterministic fixed validation slice:

```bash
python scripts/build_fixed_validation_set.py \
  --input datasets/raw/merged_v3/training_records.parquet \
  --output datasets/validation/fixed_v1/validation_records.parquet \
  --holdout-ratio 0.2
```

Compare model evaluation reports:

```bash
python scripts/compare_irrigation_evaluations.py \
  --baseline model_export/irrigation_model/v1.0.0/evaluation.json \
  --candidate model_export/irrigation_model/v1.1.0/evaluation.json \
  --output model_export/irrigation_model/v1.1.0/comparison_vs_v1.0.0.json
```

Or run comparison directly from training/export:

```bash
python scripts/train_irrigation_model.py \
  --dataset-version merged_v3 \
  --model-config-path configs/model_config_v1_1.yaml \
  --feature-schema-path configs/feature_schema_v1_1.yaml \
  --fixed-validation-path datasets/validation/fixed_v1/validation_records.parquet \
  --baseline-evaluation-path model_export/irrigation_model/v1.0.0/evaluation.json
```

## v1.1 feature contract (M1)

M1 introduced additive schema/config files for irrigation model intelligence work:

- `configs/feature_schema_v1_1.yaml`
- `configs/model_config_v1_1.yaml`

See [`MODEL_INTELLIGENCE_V1_1.md`](MODEL_INTELLIGENCE_V1_1.md) for feature definitions,
fallback behavior, and rollout compatibility rules.

## Implementation notes

- Fetch uses chunked requests (ThingSpeak max 8000 points/request); see `data_pipeline/ingestion/thingspeak_fetch.py`.
- Features at irrigation time use `merge_asof(..., direction="backward")` on env readings.
- `growth_stage_encoded` / `days_after_transplanting` derive from `--planting-date` and each row’s UTC date (not from ThingSpeak).
