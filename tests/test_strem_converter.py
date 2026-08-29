"""Tests for the pinned external nuScenes-to-Strem converter boundary."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from driving_scene_data_loop import strem_converter
from driving_scene_data_loop.strem_converter import (
    StremConverterError,
    load_numeric_id_map,
    run_nuscenes_converter,
)

_FAKE_CONVERTER = r'''#!/usr/bin/env python3
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--version-dir")
parser.add_argument("--out-dir")
parser.add_argument("--classes")
parser.add_argument("--coordinate-frame")
parser.add_argument("--include-ego", action="store_true")
parser.add_argument("--version-field")
args = parser.parse_args()

from pathlib import Path
output = Path(args.out_dir)
output.mkdir(parents=True)
(output / "nuscenes_scene-0001.json").write_text(
    json.dumps({"version": args.version_field, "frames": []})
)
(output / "nuscenes_numeric_id_map.json").write_text(json.dumps({
    "schema_version": "1.0",
    "reserved_ids": {"0": "ego"},
    "numeric_id_to_instance_token": {"17": "instance-token-17"},
}))
'''


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_converter_produces_stream_and_traceable_id_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    converter = tmp_path / "converter.py"
    converter.write_text(_FAKE_CONVERTER, encoding="utf-8")
    monkeypatch.setattr(strem_converter, "CONVERTER_SHA256", _sha256(converter))

    result = run_nuscenes_converter(
        metadata_root,
        tmp_path / "output",
        converter,
    )

    assert result.stream_count == 1
    assert load_numeric_id_map(result.numeric_id_map_path) == {
        17: "instance-token-17"
    }


def test_converter_rejects_an_existing_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_root = tmp_path / "metadata"
    output_dir = tmp_path / "output"
    metadata_root.mkdir()
    output_dir.mkdir()
    converter = tmp_path / "converter.py"
    converter.write_text(_FAKE_CONVERTER, encoding="utf-8")
    monkeypatch.setattr(strem_converter, "CONVERTER_SHA256", _sha256(converter))

    with pytest.raises(StremConverterError, match="must not already exist"):
        run_nuscenes_converter(metadata_root, output_dir, converter)


def test_numeric_id_map_must_be_one_to_one(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text(
        '{"schema_version":"1.0","reserved_ids":{"0":"ego"},'
        '"numeric_id_to_instance_token":{"17":"same","18":"same"}}',
        encoding="utf-8",
    )

    with pytest.raises(StremConverterError, match="one-to-one"):
        load_numeric_id_map(path)
