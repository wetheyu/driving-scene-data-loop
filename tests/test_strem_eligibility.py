"""Tests for CAM_FRONT filtering before scenario mining."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray

from driving_scene_data_loop.strem_eligibility import (
    SUMMARY_FILENAME,
    build_cam_front_eligible_streams,
)


class FakeNuScenes:
    """Only the devkit interface used by the eligibility builder."""

    def __init__(self) -> None:
        categories = {
            "ped": "human.pedestrian.adult",
            "car": "vehicle.car",
            "truck": "vehicle.truck",
            "bus": "vehicle.bus.rigid",
            "construction": "vehicle.construction",
        }
        self.instance = [
            {"token": f"{name}-instance", "category_token": name}
            for name in categories
        ]
        self.sample = [{"token": "sample", "timestamp": 1_000_000}]
        self.scene = [
            {
                "name": "scene-0001",
                "first_sample_token": "sample",
            }
        ]
        visibility = {"ped": "2", "car": "4", "truck": "1", "bus": "3", "construction": "2"}
        self.sample_annotation = [
            {
                "token": f"{name}-annotation",
                "sample_token": "sample",
                "instance_token": f"{name}-instance",
                "category_name": category,
                "visibility_token": visibility[name],
            }
            for name, category in categories.items()
        ]
        self._categories = categories

    def get(self, table: str, token: str) -> dict[str, object]:
        if table == "category":
            return {"name": self._categories[token]}
        if table == "sample":
            return {
                "token": "sample",
                "timestamp": 1_000_000,
                "next": "",
                "data": {"CAM_FRONT": "camera"},
            }
        if table == "sample_data":
            return {
                "channel": "CAM_FRONT",
                "is_key_frame": True,
                "width": 640,
                "height": 480,
            }
        raise KeyError((table, token))

    def get_sample_data(
        self, *_: object, **__: object
    ) -> tuple[str, list[object], NDArray[np.float64]]:
        corners = np.asarray(
            [[-1, 1] * 4, [-1, -1, 1, 1] * 2, [9] * 4 + [11] * 4],
            dtype=np.float64,
        )
        box = SimpleNamespace(center=np.asarray([0, 0, 10]), corners=lambda: corners)
        intrinsic = np.asarray(
            [[100, 0, 320], [0, 100, 240], [0, 0, 1]], dtype=np.float64
        )
        return "frame.jpg", [box], intrinsic


def test_filter_keeps_eligible_objects_and_normalizes_motor_classes(tmp_path: Path) -> None:
    raw_streams = tmp_path / "raw-streams"
    output = tmp_path / "eligible-streams"
    raw_streams.mkdir()
    _write_raw_streams(raw_streams)

    result = build_cam_front_eligible_streams(FakeNuScenes(), raw_streams, output)

    stream = json.loads(
        (output / "nuscenes_scene-0001.json").read_text(encoding="utf-8")
    )
    annotations = stream["frames"][0]["samples"][0]["annotations"]
    assert [(item["class"], item["id"]) for item in annotations] == [
        ("ego", 0),
        ("pedestrian", 17),
        ("motor_vehicle", 18),
        ("motor_vehicle", 20),
        ("motor_vehicle", 21),
    ]
    assert result.target_annotation_count == 5
    assert result.eligible_annotation_count == 4
    assert (output / SUMMARY_FILENAME).is_file()


def _write_raw_streams(root: Path) -> None:
    annotations = [
        {"class": "ego", "id": 0},
        {"class": "pedestrian", "id": 17},
        {"class": "car", "id": 18},
        {"class": "truck", "id": 19},
        {"class": "bus", "id": 20},
        {"class": "construction_vehicle", "id": 21},
    ]
    _write(
        root / "nuscenes_scene-0001.json",
        {
            "frames": [
                {
                    "index": 0,
                    "samples": [{"annotations": annotations}],
                    "timestamp": 0.0,
                }
            ]
        },
    )
    _write(
        root / "nuscenes_numeric_id_map.json",
        {
            "numeric_id_to_instance_token": {
                "17": "ped-instance",
                "18": "car-instance",
                "19": "truck-instance",
                "20": "bus-instance",
                "21": "construction-instance",
            },
            "reserved_ids": {"0": "ego"},
            "schema_version": "1.0",
        },
    )


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")
