# AgroEdge — HTTP API (local dashboard)

The Flask monitoring app serves a small read-only API. The machine-readable contract is **OpenAPI 3.0**:

- **YAML:** `GET /api/openapi.yaml`
- **JSON:** `GET /api/openapi.json`

Source file in the repo: [`docs/api/openapi.yaml`](api/openapi.yaml).

Also documented interactively from the dashboard home page (links at the top).

When `AGROEDGE_DASHBOARD_BASIC_USER` and `AGROEDGE_DASHBOARD_BASIC_PASSWORD` are set, all routes **except** `GET /health` require **HTTP Basic** credentials (`Authorization: Basic ...`). Use only with HTTPS or a trusted network.

## Endpoints (summary)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| GET | `/api/last-cycle` | Last JSONL runtime row |
| GET | `/api/thingspeak-links` | ThingSpeak web channel URLs from env |
| GET | `/api/openapi.yaml` | OpenAPI spec (YAML) |
| GET | `/api/openapi.json` | OpenAPI spec (JSON) |
| POST | `/api/disease/predict` | Disease inference: JSON `{"features": [float, ...]}` OR `{"image_base64": "..."}` (length = bundle dimension; image payload uses grayscale pixel-vector embedding). |
| GET | `/api/disease/last` | Last disease prediction (from `AGROEDGE_DISEASE_LOG`) |

ThingSpeak **REST** (`api.thingspeak.com`) is used by the runtime ingestion client, not by these routes.

**Disease bundle:** `AGROEDGE_DISEASE_BUNDLE` or default `model_export/disease_model/v0.1.0`. Export with `scripts/train_disease_model.py`.
