"""
ThingSpeak channel feed fetching with time-window chunking.

ThingSpeak limits a single feed request to **8000** points. For longer ranges,
this module walks backward in time in chunks and deduplicates by ``entry_id``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import requests


@dataclass(frozen=True)
class FetchResult:
    """Raw feed rows plus diagnostics."""

    feeds: list[dict[str, Any]]
    requests_made: int


def _format_ts(dt: datetime) -> str:
    """ThingSpeak expects ``YYYY-MM-DD HH:MM:SS`` (UTC recommended)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt_utc = dt.astimezone(UTC)
    return dt_utc.strftime("%Y-%m-%d %H:%M:%S")


def fetch_feeds_chunk(
    *,
    base_url: str,
    channel_id: str,
    api_key: str,
    start: datetime | None = None,
    end: datetime | None = None,
    results: int = 8000,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """
    One GET feeds.json request.

    If ``start`` and ``end`` are both set, ThingSpeak returns rows in that range.
    If omitted, returns up to ``results`` most recent rows (newest last in list).
    """
    url = f"{base_url.rstrip('/')}/channels/{channel_id}/feeds.json"
    params: dict[str, str | int] = {
        "api_key": api_key,
        "results": min(results, 8000),
    }
    if start is not None:
        params["start"] = _format_ts(start)
    if end is not None:
        params["end"] = _format_ts(end)

    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    feeds = payload.get("feeds") or []
    return feeds


def fetch_feeds_timerange(
    *,
    base_url: str,
    channel_id: str,
    api_key: str,
    range_start: datetime,
    range_end: datetime,
    chunk_days: float = 21.0,
    results_per_request: int = 8000,
    timeout: float = 30.0,
    pause_seconds: float = 1.0,
    max_feeds: int | None = None,
) -> FetchResult:
    """
    Walk backward from ``range_end`` to ``range_start``, merging chunks.

    Default chunk is 21 days (~2016 samples at a 15-minute cadence), safely
    under the 8000 cap.

    Args:
        pause_seconds: Sleep between HTTP calls to respect ThingSpeak rate limits.
        max_feeds: Optional cap on total rows (most recent by ``entry_id``).
    """
    if range_end <= range_start:
        raise ValueError("range_end must be after range_start")

    seen: dict[Any, dict[str, Any]] = {}
    requests_made = 0
    cursor_end = range_end

    while cursor_end > range_start:
        chunk_start_dt = max(range_start, cursor_end - timedelta(days=chunk_days))
        feeds = fetch_feeds_chunk(
            base_url=base_url,
            channel_id=channel_id,
            api_key=api_key,
            start=chunk_start_dt,
            end=cursor_end,
            results=results_per_request,
            timeout=timeout,
        )
        requests_made += 1

        for row in feeds:
            eid = row.get("entry_id")
            if eid is not None:
                seen[eid] = row

        if pause_seconds > 0:
            time.sleep(pause_seconds)

        if not feeds:
            break

        oldest_ts = None
        for row in feeds:
            cat = row.get("created_at")
            if not cat:
                continue
            ts = _parse_created_at(str(cat))
            if oldest_ts is None or ts < oldest_ts:
                oldest_ts = ts

        if oldest_ts is None or oldest_ts <= chunk_start_dt:
            cursor_end = chunk_start_dt - timedelta(seconds=1)
        else:
            cursor_end = oldest_ts - timedelta(seconds=1)

        if max_feeds is not None and len(seen) >= max_feeds:
            break

    merged = sorted(seen.values(), key=lambda r: _parse_created_at(str(r["created_at"])))
    if max_feeds is not None and len(merged) > max_feeds:
        merged = merged[-max_feeds:]

    return FetchResult(feeds=merged, requests_made=requests_made)


def fetch_latest_feeds(
    *,
    base_url: str,
    channel_id: str,
    api_key: str,
    results: int = 8000,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Convenience: latest ``results`` rows (no date filter)."""
    return fetch_feeds_chunk(
        base_url=base_url,
        channel_id=channel_id,
        api_key=api_key,
        start=None,
        end=None,
        results=results,
        timeout=timeout,
    )


def _parse_created_at(value: str) -> datetime:
    """Parse ThingSpeak ``created_at`` into UTC-aware datetime."""
    ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)
