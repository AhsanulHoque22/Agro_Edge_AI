# Disease image dataset layout (v1)

Place **rice leaf** imagery and a manifest here when training the real CV model.

## Directory layout

```text
datasets/disease_model/v1/
  README.md          # this file
  images/
    healthy/
    rice_blast/
    brown_spot/
    bacterial_leaf_blight/
    sheath_blight/
    tungro/
    leaf_scald/
    unknown/
  manifest.parquet   # one row per image — built from ImageRecord (see schema)
```

Class folder names must match `DiseaseLabel` in `data_pipeline/schemas/enums.py`.

## Manifest

Build `manifest.parquet` automatically from the folder structure:

```bash
python scripts/build_disease_manifest.py --dataset-root datasets/disease_model/v1
```

Alternatively, you can create it yourself (the trainer requires at least):

- `image_id` (optional but recommended),
- `image_path`,
- `disease_label`,
- `image_width_px`,
- `image_height_px`,
- `data_source`
- Optional: `env_context_record_id` to join environmental telemetry for multimodal training.

The current disease trainer consumes `manifest.parquet` + images and trains an sklearn `MLPClassifier` on fixed-length grayscale pixel vectors (default 8x8 -> 64 dims).
