"""
Disease model scaffolding: sklearn sanity baseline (no real images yet).

Produces an exportable bundle so CI and edge-adapters can rely on a stable
directory layout. Replace with CNN / MobileNet fine-tuning when
``datasets/images`` is populated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DiseaseSanityArtifacts:
    classifier: RandomForestClassifier
    X_test: Any
    y_test: Any
    report: dict[str, Any]


@dataclass(frozen=True)
class DiseaseImagePixelsArtifacts:
    classifier: Any
    X_test: Any
    y_test: Any
    report: dict[str, Any]


def load_disease_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "disease_model" not in raw:
        raise ValueError("disease_model_config.yaml missing disease_model key")
    return raw


def train_sanity_baseline(config: dict[str, Any]) -> DiseaseSanityArtifacts:
    dm = config["disease_model"]
    n_classes = int(dm["n_classes"])
    n_features = int(dm["feature_dimension"])
    n_samples = int(dm["training"]["sanity_samples"])
    rs = int(dm["training"]["sanity_random_state"])
    test_size = float(dm["training"]["test_size"])

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(32, n_features),
        n_redundant=max(0, n_features // 4),
        n_classes=n_classes,
        n_clusters_per_class=2,
        random_state=rs,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rs, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=40,
        max_depth=12,
        random_state=rs,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = {
        "mode": "sanity_baseline",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro")),
    }
    return DiseaseSanityArtifacts(
        classifier=clf,
        X_test=X_test,
        y_test=y_test,
        report=report,
    )


def train_image_pixels_model(
    config: dict[str, Any],
    *,
    dataset_root: Path,
    manifest_path: Path | None = None,
    limit_samples: int | None = None,
) -> DiseaseImagePixelsArtifacts:
    """
    Real CV baseline: convert each disease image into a fixed-length grayscale
    pixel vector, then train a multiclass sklearn MLP classifier.

    Expects:
      - ``{dataset_root}/manifest.parquet`` by default
      - manifest includes ``image_path`` and ``disease_label`` columns.
    """
    if Image is None:
        raise RuntimeError("Pillow is required for image pixel training.")

    dm = config["disease_model"]
    resize_side = int(dm["training"]["resize_side"])
    feature_dim = int(dm["feature_dimension"])
    expected_dim = resize_side * resize_side
    if feature_dim != expected_dim:
        raise ValueError(
            f"feature_dimension={feature_dim} must equal resize_side^2={expected_dim}"
        )

    class_names = [str(x) for x in dm["class_names"]]
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    if manifest_path is None:
        manifest_path = dataset_root / "manifest.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.parquet not found: {manifest_path}")

    df = pd.read_parquet(manifest_path, engine="pyarrow")
    if "image_path" not in df.columns or "disease_label" not in df.columns:
        raise ValueError("manifest.parquet missing required columns: image_path, disease_label")

    if limit_samples is not None:
        df = df.head(int(limit_samples))

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for _, row in df.iterrows():
        label_raw = row.get("disease_label")
        label = str(label_raw)
        if label not in label_to_idx:
            continue

        img_rel = str(row.get("image_path"))
        img_path = dataset_root / img_rel
        if not img_path.is_file():
            continue

        img = Image.open(img_path)
        img = img.convert("L").resize((resize_side, resize_side), Image.BILINEAR)
        vec = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
        if vec.shape[0] != feature_dim:
            continue

        X_list.append(vec)
        y_list.append(label_to_idx[label])

    if not X_list:
        raise RuntimeError("No usable training images found. Check manifest and image paths.")

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)

    rs = int(dm["training"]["random_state"])
    test_size = float(dm["training"]["test_size"])

    hidden_layer_sizes = dm["training"]["hidden_layer_sizes"]
    max_iter = int(dm["training"]["max_iter"])
    early_stopping = bool(dm["training"].get("early_stopping", True))

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=rs, stratify=y
        )
    except ValueError:
        # Very small datasets may not allow stratified splitting.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=rs
        )

    clf = MLPClassifier(
        hidden_layer_sizes=tuple(int(v) for v in hidden_layer_sizes),
        max_iter=max_iter,
        random_state=rs,
        early_stopping=early_stopping,
        n_iter_no_change=8,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = {
        "mode": "image_pixels_mlp",
        "n_samples": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(feature_dim),
        "n_classes": int(len(class_names)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro")),
    }

    return DiseaseImagePixelsArtifacts(
        classifier=clf,
        X_test=X_test,
        y_test=y_test,
        report=report,
    )
