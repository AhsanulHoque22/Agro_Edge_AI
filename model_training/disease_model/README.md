# Disease / CV model

**Rice disease image classification** (and optional TFLite) per the AgroEdge roadmap.

## What exists today

- `configs/disease_model_config.yaml` — class list + export knobs.
- `trainer.py` — real-image baseline: convert each image to a fixed-length grayscale pixel vector
  (resize to `resize_side` x `resize_side`, default 8x8 -> 64 dims) and train an sklearn `MLPClassifier`.
- `data_pipeline/disease/manifest_builder.py` — build `datasets/disease_model/<version>/manifest.parquet` from `images/<label>/*`.
- `exporter.py` — writes `model_export/disease_model/v0.1.0/` (`model.joblib`, `metadata.json`, `evaluation.json`).
- CLI: `python scripts/train_disease_model.py`

This establishes a working end-to-end pipeline for **image -> vector -> classifier -> exported bundle**.
You can later replace the embedding step with CNN/MobileNet features without changing the dashboard/API contract.

## Planned (real imagery)

1. Image ingestion — `data_pipeline/schemas/image_record.py`.
2. Dataset layout under `datasets/disease_model/v1/images/<label>/` (already done here).
3. Optional: evaluation gates per-class + better metrics (confusion matrix, class-wise F1).
4. Optional: replace pixel-vector embedding with CNN/MobileNet embeddings.
