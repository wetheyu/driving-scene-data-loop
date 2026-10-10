"""The one byte format every JSON and JSONL artifact is written in."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from driving_scene_data_loop.json_files import read_json, read_jsonl, write_jsonl, write_report


def test_report_and_rows_keep_their_fixed_bytes(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    rows = tmp_path / "rows.jsonl"

    write_report(report, {"b": 1, "a": [0.5, None]})
    write_jsonl(rows, [{"b": 1, "a": True}, {"c": "x"}])

    assert report.read_text(encoding="utf-8") == (
        '{\n  "a": [\n    0.5,\n    null\n  ],\n  "b": 1\n}\n'
    )
    assert rows.read_text(encoding="utf-8") == '{"b":1,"a":true}\n{"c":"x"}\n'
    assert read_json(report) == {"a": [0.5, None], "b": 1}
    assert read_jsonl(rows) == [{"b": 1, "a": True}, {"c": "x"}]


def test_a_nan_never_reaches_an_artifact(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_report(tmp_path / "report.json", {"ap": math.nan})
