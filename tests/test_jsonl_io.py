"""Tests for data_pipeline.jsonl_io."""

from __future__ import annotations

import json

import pytest

from data_pipeline.jsonl_io import read_last_jsonl_row


def test_read_last_jsonl_row_empty(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert read_last_jsonl_row(p) is None


def test_read_last_jsonl_row_missing(tmp_path):
    assert read_last_jsonl_row(tmp_path / "nope.jsonl") is None


def test_read_last_jsonl_row_returns_last_object(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(
        '{"a": 1}\n{"b": 2}\n\n{"c": 3}\n',
        encoding="utf-8",
    )
    row = read_last_jsonl_row(p)
    assert row == {"c": 3}


def test_read_last_jsonl_row_invalid_json_raises(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("not json\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_last_jsonl_row(p)
