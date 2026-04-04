"""
Model export packager for AgroEdge irrigation model artifacts.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import yaml


class IrrigationModelExporter:
    """Writes versioned export bundle for edge inference usage."""

    def __init__(self, model_config: dict[str, Any], feature_schema_path: Path) -> None:
        self.model_config = model_config
        self.feature_schema_path = feature_schema_path
        self.ir_cfg = model_config["irrigation_model"]

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def export(
        self,
        classifier,
        regressor,
        evaluation_report: dict[str, Any],
        output_root: Path,
    ) -> dict[str, str]:
        version = str(self.ir_cfg["version"])
        bundle_dir = output_root / "irrigation_model" / f"v{version}"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        compression = int(self.ir_cfg["export"]["compression"])
        model_payload = {
            "classifier": classifier,
            "regressor": regressor,
            "feature_names": self.ir_cfg["features"],
            "model_type": self.ir_cfg["model_type"],
            "version": version,
        }
        model_path = bundle_dir / "model.joblib"
        joblib.dump(model_payload, model_path, compress=compression)

        feature_schema = yaml.safe_load(self.feature_schema_path.read_text(encoding="utf-8"))
        feature_schema_path = bundle_dir / "feature_schema.json"
        self._write_json(feature_schema_path, feature_schema)

        encoder_map = {
            "growth_stage_encoding": feature_schema.get("growth_stage_encoding", {}),
            "crop_type_encoding": feature_schema.get("crop_type_encoding", {}),
            "disease_labels": feature_schema.get("disease_labels", []),
        }
        encoder_map_path = bundle_dir / "encoder_map.json"
        self._write_json(encoder_map_path, encoder_map)

        evaluation_path = bundle_dir / "evaluation.json"
        self._write_json(evaluation_path, evaluation_report)

        metadata = {
            "model_name": "irrigation_model",
            "model_type": self.ir_cfg["model_type"],
            "version": version,
            "exported_at_utc": datetime.now(UTC).isoformat(),
            "schema_version": self.model_config.get("schema_version", "1.0"),
            "feature_count": len(self.ir_cfg["features"]),
        }
        metadata_path = bundle_dir / "metadata.json"
        self._write_json(metadata_path, metadata)

        return {
            "bundle_dir": str(bundle_dir),
            "model": str(model_path),
            "feature_schema": str(feature_schema_path),
            "encoder_map": str(encoder_map_path),
            "evaluation": str(evaluation_path),
            "metadata": str(metadata_path),
        }
