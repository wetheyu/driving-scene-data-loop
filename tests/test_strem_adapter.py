"""Tests for the small external Strem boundary."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import pytest

from driving_scene_data_loop import strem_adapter
from driving_scene_data_loop.strem_adapter import StremAdapter, StremAdapterError

_FAKE_STREM = r"""#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

if sys.argv[1:] == ["--version"]:
    print(__VERSION__)
    raise SystemExit(0)

if "--first-frame-start" not in sys.argv:
    raise SystemExit(2)
first_frame_start_index = sys.argv.index("--first-frame-start")
if sys.argv[first_frame_start_index + 1] != "exact":
    raise SystemExit(2)

spec_path = Path(sys.argv[4])
mode = json.loads(spec_path.read_text())["mode"]
if mode == "timeout":
    time.sleep(1)
if mode == "error":
    print(json.dumps({"diagnostic": {"message": "fixture failure"}}), file=sys.stderr)
    raise SystemExit(2)
if mode == "malformed":
    print("not json")
    raise SystemExit(0)

status = "match" if mode in ("match", "open_match") else "no_match"
intervals = []
if status == "match":
    intervals = [{
        "start_frame_index": 0,
        "end_frame_index": 4,
        "start_time_semantics": "exact" if mode == "match" else "interval",
        "start_lower_timestamp": 0.0,
        "start_upper_timestamp": 0.0 if mode == "match" else 0.5,
        "start_lower_inclusive": True,
        "start_upper_inclusive": mode == "match",
        "end_lower_timestamp": 2.0,
        "end_upper_timestamp": None if mode == "open_match" else 2.5,
        "constraints": ["1.000s < t' - t"],
        "bindings": {"ego": 0, "pedestrian": 17},
    }]
print(json.dumps({
    "schema_version": "2.0",
    "status": status,
    "specification": {"name": f"fixture_{mode}"},
    "stream": {"first_frame_start": "exact"},
    "intervals": intervals,
}))
raise SystemExit(0 if status == "match" else 1)
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_binary(path: Path, version: str = "strem 0.3.0") -> None:
    path.write_text(_FAKE_STREM.replace("__VERSION__", repr(version)), encoding="utf-8")
    path.chmod(0o700)


@pytest.fixture
def strem_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, Path]]:
    binary = tmp_path / "strem"
    stream = tmp_path / "scene.json"
    _write_binary(binary)
    stream.write_text("{}", encoding="utf-8")
    specs = {}
    for mode in ("match", "open_match", "no_match", "error", "timeout", "malformed"):
        spec = tmp_path / f"{mode}.json"
        spec.write_text(json.dumps({"mode": mode}), encoding="utf-8")
        specs[mode] = spec
    monkeypatch.setitem(
        strem_adapter.STREM_SHA256_BY_PLATFORM,
        (platform.system(), platform.machine().lower()),
        _sha256(binary),
    )
    return binary, stream, specs


def test_match_keeps_complete_symbolic_interval(
    strem_files: tuple[Path, Path, dict[str, Path]],
) -> None:
    binary, stream, specs = strem_files

    result = StremAdapter(binary).run_scene(stream, specs["match"])

    assert result.status == "match"
    interval = result.intervals[0]
    assert interval.start_frame_index == 0
    assert interval.end_frame_index == 4
    assert interval.start_time_semantics == "exact"
    assert interval.start_lower_timestamp == 0.0
    assert interval.start_upper_timestamp == 0.0
    assert interval.start_lower_inclusive
    assert interval.start_upper_inclusive
    assert interval.end_lower_timestamp == 2.0
    assert interval.end_upper_timestamp == 2.5
    assert interval.constraints == ("1.000s < t' - t",)
    assert interval.bindings == (("ego", 0), ("pedestrian", 17))


def test_open_ended_interval_keeps_none(
    strem_files: tuple[Path, Path, dict[str, Path]],
) -> None:
    binary, stream, specs = strem_files

    result = StremAdapter(binary).run_scene(stream, specs["open_match"])

    interval = result.intervals[0]
    assert interval.start_time_semantics == "interval"
    assert interval.start_lower_timestamp == 0.0
    assert interval.start_upper_timestamp == 0.5
    assert interval.start_lower_inclusive
    assert not interval.start_upper_inclusive
    assert interval.end_upper_timestamp is None


def test_no_match_is_not_an_error(
    strem_files: tuple[Path, Path, dict[str, Path]],
) -> None:
    binary, stream, specs = strem_files

    result = StremAdapter(binary).run_scene(stream, specs["no_match"])

    assert result.status == "no_match"
    assert result.intervals == ()


def test_exit_two_is_not_treated_as_no_match(
    strem_files: tuple[Path, Path, dict[str, Path]],
) -> None:
    binary, stream, specs = strem_files

    with pytest.raises(StremAdapterError, match="fixture failure"):
        StremAdapter(binary).run_scene(stream, specs["error"])


def test_timeout_is_not_treated_as_no_match(
    strem_files: tuple[Path, Path, dict[str, Path]],
) -> None:
    binary, stream, specs = strem_files

    with pytest.raises(StremAdapterError, match="timeout"):
        StremAdapter(binary, timeout_s=0.01).run_scene(stream, specs["timeout"])


def test_malformed_output_is_rejected(
    strem_files: tuple[Path, Path, dict[str, Path]],
) -> None:
    binary, stream, specs = strem_files

    with pytest.raises(StremAdapterError, match="malformed JSON"):
        StremAdapter(binary).run_scene(stream, specs["malformed"])


def test_wrong_release_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "strem"
    _write_binary(binary, "strem 0.2.1")
    monkeypatch.setitem(
        strem_adapter.STREM_SHA256_BY_PLATFORM,
        (platform.system(), platform.machine().lower()),
        _sha256(binary),
    )

    with pytest.raises(StremAdapterError, match="must report"):
        StremAdapter(binary)
