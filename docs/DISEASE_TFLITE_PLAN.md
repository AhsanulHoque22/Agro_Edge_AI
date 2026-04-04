# Disease Model Upgrade Plan (MobileNetV2 -> TFLite)

This document defines the deployment contract for the disease model upgrade path.
It does not replace the current image-pixels MLP baseline.

## Current baseline

- Training: grayscale 8x8 pixel vectors -> sklearn MLP
- Export: `model_export/disease_model/v0.1.0/model.joblib`
- Inference API: `POST /api/disease/predict` (features or image payload)

## Target upgrade (M4+)

- Architecture: MobileNetV2 transfer learning (cloud training)
- Edge format: TensorFlow Lite bundle for Raspberry Pi 3
- API contract: unchanged (`/api/disease/predict` response schema remains stable)

## Proposed TFLite bundle contract

Under `model_export/disease_model/v<version>/`:

- `model.tflite` - quantized or float model
- `labels.json` - ordered class names
- `metadata.json` - model version, input size, preprocessing params
- `evaluation.json` - validation metrics

## Pi 3 constraints

- Memory budget: keep model and inference pipeline lightweight (target <= 1 GB total system RAM with runtime services)
- Throughput: single-image synchronous inference acceptable; avoid large batch assumptions
- Preprocessing: deterministic resize/normalize identical to training

## Migration safety rules

1. Keep `model.joblib` path supported until TFLite path is validated.
2. Add TFLite engine as additive code path behind config/env selection.
3. Maintain backward compatibility for dashboard routes and OpenAPI schema.
4. Keep smoke tests for both baseline and new bundle format during transition.
