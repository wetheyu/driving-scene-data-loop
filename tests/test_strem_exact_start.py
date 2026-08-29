"""Real-release check for Strem's exact first-frame start semantics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from driving_scene_data_loop.strem_adapter import StremAdapter


@pytest.mark.parametrize("first_timestamp", [0.0, 12.5])
def test_first_frame_uses_its_actual_timestamp(
    tmp_path: Path,
    first_timestamp: float,
) -> None:
    binary_value = os.environ.get("STREM_BIN")
    if not binary_value:
        pytest.skip("set STREM_BIN to run the real Strem integration test")

    stream = tmp_path / "scene.json"
    specification = tmp_path / "first-frame-pedestrian.json"
    stream.write_text(
        json.dumps(_first_frame_stream(first_timestamp)),
        encoding="utf-8",
    )
    specification.write_text(json.dumps(_first_frame_specification()), encoding="utf-8")

    result = StremAdapter(Path(binary_value)).run_scene(stream, specification)

    assert result.status == "match"
    interval = result.intervals[0]
    assert (interval.start_frame_index, interval.end_frame_index) == (0, 0)
    assert interval.bindings == (("p", 17),)
    assert interval.start_time_semantics == "exact"
    assert interval.start_lower_timestamp == pytest.approx(first_timestamp)
    assert interval.start_upper_timestamp == pytest.approx(first_timestamp)
    assert interval.start_lower_inclusive
    assert interval.start_upper_inclusive


def _first_frame_stream(first_timestamp: float) -> dict[str, object]:
    frames = []
    for index, timestamp in enumerate((first_timestamp, first_timestamp + 0.5)):
        annotations = []
        if index == 0:
            annotations.append(
                {
                    "class": "pedestrian",
                    "id": 17,
                    "score": 1.0,
                    "bbox": {
                        "type": "@stremf/bbox/aabb",
                        "region": {
                            "center": {"x": 0.0, "y": 0.0},
                            "dimensions": {"w": 1.0, "h": 1.0},
                        },
                    },
                }
            )
        frames.append(
            {
                "index": index,
                "timestamp": timestamp,
                "samples": [
                    {
                        "type": "@stremf/sample/detection",
                        "channel": "test",
                        "image": {
                            "path": f"frame-{index}.json",
                            "dimensions": {"width": 10, "height": 10},
                        },
                        "annotations": annotations,
                    }
                ],
            }
        )
    return {"version": "0.2.1", "frames": frames}


def _first_frame_specification() -> dict[str, object]:
    return {
        "name": "first_frame_pedestrian",
        "states": ["q0", "q1", "qf"],
        "initial_states": ["q0"],
        "accepting_states": ["qf"],
        "clocks": [],
        "persistent_bindings": ["p"],
        "transitions": [
            {
                "from": "q0",
                "to": "q1",
                "spatial_guard": "[E(p := [:pedestrian:])(@id(p) == 17)]",
            },
            {"from": "q1", "to": "qf", "terminal": True},
        ],
    }
