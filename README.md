# AgroEdge AI

AI-first precision-agriculture stack: **data pipeline**, **irrigation model training**, **edge inference**, **ThingSpeak integration**, and a **minimal local dashboard** (Raspberry Pi).

## Quick checks

```bash
cd agroedge_ai
./scripts/verify.sh
```

When this repo is on GitHub, **CI** runs the same script on pushes/PRs to `main` or `master` (see `.github/workflows/ci.yml` at the monorepo root).

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for Raspberry Pi systemd, logs, dashboard, and **HTTPS reverse proxy** examples.

- **Real data retrain:** [docs/INGESTION.md](docs/INGESTION.md) → `scripts/pull_thingspeak_training_data.py`
- **Model intelligence roadmap (v1.1 features):** [docs/MODEL_INTELLIGENCE_V1_1.md](docs/MODEL_INTELLIGENCE_V1_1.md)
- **M2 utilities:** `scripts/build_fixed_validation_set.py`, `scripts/compare_irrigation_evaluations.py`

### Irrigation v1.1 retrain (synthetic parity example)

From `agroedge_ai/`, after `generate_synthetic_data.py --version v1_1` (this version **fills v1.1 lag / THI / irrigation-context columns**):

1. `python scripts/prepare_features.py --version v1_1 --feature-schema-path configs/feature_schema.yaml --processed-version v1_baseline_from_v1_1_raw`
2. `python scripts/prepare_features.py --version v1_1 --feature-schema-path configs/feature_schema_v1_1.yaml --processed-version v1_1`
3. `python scripts/build_fixed_validation_set.py --input datasets/raw/v1_1/training_records.parquet --output datasets/validation/fixed_v1_1/validation_records.parquet --salt agroedge_fixed_validation_v1_1`
4. Train baseline: `python scripts/train_irrigation_model.py --dataset-version v1_baseline_from_v1_1_raw --fixed-validation-path datasets/validation/fixed_v1_1/validation_records.parquet`
5. Copy `model_export/irrigation_model/v1.0.0/evaluation.json` → `datasets/validation/fixed_v1_1/baseline_evaluation.json`
6. Train v1.1 + diff: `python scripts/train_irrigation_model.py --dataset-version v1_1 --model-config-path configs/model_config_v1_1.yaml --feature-schema-path configs/feature_schema_v1_1.yaml --fixed-validation-path datasets/validation/fixed_v1_1/validation_records.parquet --baseline-evaluation-path datasets/validation/fixed_v1_1/baseline_evaluation.json --comparison-output-path model_export/irrigation_model/v1.1.0/comparison_vs_baseline.json`

`prepare_features.py` also accepts `--raw-parquet` if raw data lives outside `datasets/raw/<version>/`.
- **Disease upgrade contract:** [docs/DISEASE_TFLITE_PLAN.md](docs/DISEASE_TFLITE_PLAN.md)

## API documentation (dashboard)

OpenAPI 3.0: [docs/api/openapi.yaml](docs/api/openapi.yaml) — also served at `GET /api/openapi.yaml` and `GET /api/openapi.json` when the dashboard is running. **Disease:** `POST /api/disease/predict` with JSON `{"features": [...]}` (see [docs/API.md](docs/API.md)).

## Layout (high level)

| Path | Role |
|------|------|
| `configs/` | YAML schema + model + ThingSpeak channel definitions |
| `data_pipeline/` | Synthetic data, validation, features, JSONL helpers |
| `model_training/` | Irrigation trainer |
| `model_evaluation/` | Metrics + gates |
| `model_export/` | Versioned `model.joblib` bundles |
| `edge_inference/` | Bundle load, decide, scheduler |
| `cloud_integration/thingspeak_client/` | Read/write ThingSpeak |
| `web_dashboard/` | Flask monitoring app |
| `scripts/` | CLI entry points + `verify.sh` + `run_dashboard_gunicorn.sh` |
| `deploy/` | systemd, logrotate, env templates, `reverse-proxy/` (Caddy, nginx) |
| `requirements-prod.txt` | Optional: **Gunicorn** for dashboard on the Pi |
| `firmware/esp32/` | ThingSpeak telemetry **Arduino template** |
| `configs/disease_model_config.yaml` | Disease model scaffold (sanity train, not images yet) |

**More tooling:** `scripts/merge_training_parquet.py` (blend datasets), `scripts/train_disease_model.py` (baseline bundle), `scripts/disease_inference_smoke_test.py`. **Edge:** `edge_inference.DiseaseInferenceEngine` for the exported disease pack. Optional dashboard login: `AGROEDGE_DASHBOARD_BASIC_*` in `.env.example`. Real image data layout: `datasets/disease_model/v1/README.md`.

## License

Proprietary — all rights reserved unless otherwise stated by the project owner.
