"""Export disease model bundle (joblib + metadata)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib


class DiseaseModelExporter:
    """Write ``model_export/disease_model/v<version>/`` artifacts."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.dm = config["disease_model"]

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def export(
        self,
        classifier: Any,
        evaluation_report: dict[str, Any],
        output_root: Path,
    ) -> dict[str, str]:
        version = str(self.dm["version"])
        bundle_dir = output_root / "disease_model" / f"v{version}"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        compression = int(self.dm["export"]["compression"])

        model_payload = {
            "classifier": classifier,
            "model_type": self.dm["model_type"],
            "version": version,
            "feature_dimension": self.dm["feature_dimension"],
            "class_names": list(self.dm["class_names"]),
        }
        model_path = bundle_dir / "model.joblib"
        joblib.dump(model_payload, model_path, compress=compression)

        eval_path = bundle_dir / "evaluation.json"
        self._write_json(eval_path, evaluation_report)

        metadata = {
            "model_name": "disease_model",
            "model_type": self.dm["model_type"],
            "version": version,
            "exported_at_utc": datetime.now(UTC).isoformat(),
            "schema_version": self.config.get("schema_version", "1.0"),
            "n_classes": len(self.dm["class_names"]),
            "note": (
                "Baseline trained from image pixel vectors + manifest.parquet. "
                "Replace with CNN/mobile embeddings when you move beyond this v0 pipeline."
            ),
        }
        meta_path = bundle_dir / "metadata.json"
        self._write_json(meta_path, metadata)

        return {
            "bundle_dir": str(bundle_dir),
            "model": str(model_path),
            "evaluation": str(eval_path),
            "metadata": str(meta_path),
        }
