"""
ThingSpeak read client for AgroEdge runtime ingestion.

Read-only client that fetches latest environmental telemetry row from the
configured ThingSpeak channel.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class ThingSpeakConfig:
    base_url: str
    env_channel_id: str
    env_read_api_key: str
    irrigation_channel_id: str | None = None
    irrigation_read_api_key: str | None = None
    irrigation_write_api_key: str | None = None
    timeout_seconds: float = 10.0


class ThingSpeakReadClient:
    """Minimal read client for ThingSpeak channel feeds API."""

    def __init__(self, config: ThingSpeakConfig) -> None:
        self.config = config

    def fetch_latest_environment_row(self) -> dict[str, Any]:
        """
        Fetch latest feed entry and map ThingSpeak fields to semantic names.
        """
        url = f"{self.config.base_url.rstrip('/')}/channels/{self.config.env_channel_id}/feeds.json"
        params = {"api_key": self.config.env_read_api_key, "results": 1}
        response = requests.get(url, params=params, timeout=self.config.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        feeds = payload.get("feeds", [])
        if not feeds:
            raise ValueError("ThingSpeak returned no feed rows.")
        row = feeds[-1]
        return _map_environment_feed_row(row)

    def fetch_latest_environment_rows(self, results: int = 3) -> list[dict[str, Any]]:
        """Fetch the latest N environmental rows (oldest->newest)."""
        n = max(1, min(int(results), 50))
        url = f"{self.config.base_url.rstrip('/')}/channels/{self.config.env_channel_id}/feeds.json"
        params = {"api_key": self.config.env_read_api_key, "results": n}
        response = requests.get(url, params=params, timeout=self.config.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        feeds = payload.get("feeds", [])
        out: list[dict[str, Any]] = []
        for row in feeds:
            out.append(_map_environment_feed_row(row))
        if not out:
            raise ValueError("ThingSpeak returned no feed rows.")
        return out

    def fetch_latest_irrigation_event_row(self) -> dict[str, Any]:
        """
        Fetch latest irrigation log row.

        Requires irrigation channel id and irrigation read API key.
        """
        if not self.config.irrigation_channel_id:
            raise ValueError("irrigation_channel_id is required for irrigation reads.")
        if not self.config.irrigation_read_api_key:
            raise ValueError("irrigation_read_api_key is required for irrigation reads.")

        url = f"{self.config.base_url.rstrip('/')}/channels/{self.config.irrigation_channel_id}/feeds.json"
        params = {"api_key": self.config.irrigation_read_api_key, "results": 1}
        response = requests.get(url, params=params, timeout=self.config.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        feeds = payload.get("feeds", [])
        if not feeds:
            raise ValueError("ThingSpeak returned no irrigation rows.")
        row = feeds[-1]
        return {
            "created_at": row.get("created_at"),
            "irrigation_duration_minutes": _to_float_or_none(row.get("field1")) or 0.0,
            "trigger_code": int(float(row.get("field2") or 0)),
            "soil_moisture_before_percent": _to_float_or_none(row.get("field3")),
            "soil_moisture_after_percent": _to_float_or_none(row.get("field4")),
            "entry_id": row.get("entry_id"),
        }


class ThingSpeakWriteClient:
    """Write client for irrigation decision logs to ThingSpeak channel 2."""

    def __init__(self, config: ThingSpeakConfig) -> None:
        self.config = config
        if not self.config.irrigation_channel_id:
            raise ValueError("irrigation_channel_id is required for write client.")
        if not self.config.irrigation_write_api_key:
            raise ValueError("irrigation_write_api_key is required for write client.")

    def publish_irrigation_log(self, action_payload: dict[str, Any]) -> int:
        """
        Publish one irrigation decision event.

        ThingSpeak field mapping (irrigation_log channel):
          field1: irrigation_duration_minutes
          field2: trigger_source (0=ai_model,1=manual_override,2=threshold_fallback)
          field3: soil_moisture_before_percent
          field4: soil_moisture_after_percent (optional, blank for immediate decision logs)
          field5: node_id_hash (CRC32 numeric hash)
        """
        node_id = str(action_payload.get("node_id", "unknown"))
        trigger_source = str(action_payload.get("trigger_source", "ai_model"))
        trigger_code = _trigger_source_code(trigger_source)
        duration = float(action_payload.get("approved_duration_minutes", 0.0))
        moisture_before = float(action_payload.get("soil_moisture_before_percent", 0.0))
        moisture_after = action_payload.get("soil_moisture_after_percent")
        node_hash = int(zlib.crc32(node_id.encode("utf-8")) & 0xFFFFFFFF)

        url = f"{self.config.base_url.rstrip('/')}/update.json"
        params = {
            "api_key": self.config.irrigation_write_api_key,
            "field1": round(duration, 2),
            "field2": trigger_code,
            "field3": round(moisture_before, 2),
            "field5": node_hash,
        }
        if moisture_after is not None:
            params["field4"] = round(float(moisture_after), 2)

        response = requests.post(url, data=params, timeout=self.config.timeout_seconds)
        response.raise_for_status()
        entry_id = int(response.text.strip())
        if entry_id <= 0:
            raise ValueError(f"ThingSpeak update rejected (response={response.text!r}).")
        return entry_id


def _to_float(value: Any) -> float:
    if value is None or value == "":
        raise ValueError("Expected numeric ThingSpeak field but got null/empty.")
    return float(value)


def _to_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _trigger_source_code(trigger_source: str) -> int:
    mapping = {
        "ai_model": 0,
        "manual_override": 1,
        "threshold_fallback": 2,
    }
    return mapping.get(trigger_source, 0)


def _map_environment_feed_row(row: dict[str, Any]) -> dict[str, Any]:
    # ThingSpeak fields mapped using configs/thingspeak_channels.yaml
    return {
        "collected_at": row.get("created_at"),
        "soil_moisture_percent": _to_float(row.get("field1")),
        "soil_temperature_celsius": _to_float(row.get("field2")),
        "air_temperature_celsius": _to_float(row.get("field3")),
        "air_humidity_percent": _to_float(row.get("field4")),
        "light_intensity_lux": _to_float(row.get("field5")),
        "water_tank_level_percent": _to_float(row.get("field6")),
        "water_flow_rate_lph": _to_float_or_none(row.get("field7")),
        "reserved_future": row.get("field8"),
        "entry_id": row.get("entry_id"),
    }
