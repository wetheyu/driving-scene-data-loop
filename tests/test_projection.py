"""Tests for the CAM_FRONT projection rule."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray

from driving_scene_data_loop.projection import (
    ProjectionObservation,
    is_visually_eligible,
    project_annotation,
    project_camera_corners,
)


def _corners(*, x_center: float = 0.0, z: float = 10.0) -> NDArray[np.float64]:
    return np.asarray(
        [
            [x_center - 1, x_center + 1] * 4,
            [-1, -1, 1, 1] * 2,
            [z - 1] * 4 + [z + 1] * 4,
        ],
        dtype=np.float64,
    )


def _intrinsic() -> NDArray[np.float64]:
    return np.asarray([[100, 0, 320], [0, 100, 240], [0, 0, 1]], dtype=np.float64)


def test_project_camera_corners_handles_visible_and_behind_boxes() -> None:
    visible = project_camera_corners(
        _corners(), _intrinsic(), image_width=640, image_height=480, center_depth_m=10
    )
    behind = project_camera_corners(
        _corners(z=-10),
        _intrinsic(),
        image_width=640,
        image_height=480,
        center_depth_m=-10,
    )

    assert visible.projectable
    assert visible.clipped_area_fraction == 1.0
    assert visible.letterbox_height_px >= 16
    assert not behind.projectable


def test_project_annotation_uses_devkit_camera_space_box() -> None:
    box = SimpleNamespace(center=np.asarray([0, 0, 10]), corners=lambda: _corners())

    class FakeNuScenes:
        def get(self, table: str, token: str) -> dict[str, object]:
            assert (table, token) == ("sample_data", "camera")
            return {
                "channel": "CAM_FRONT",
                "is_key_frame": True,
                "width": 640,
                "height": 480,
            }

        def get_sample_data(
            self, *_: object, **__: object
        ) -> tuple[str, list[object], NDArray[np.float64]]:
            return "frame.jpg", [box], _intrinsic()

    observation = project_annotation(FakeNuScenes(), "annotation", "camera")

    assert observation.projectable
    assert observation.center_depth_m == 10


def test_visual_filter_requires_size_retention_and_visibility() -> None:
    observation = ProjectionObservation(
        projectable=True,
        height_eligible=True,
        visible_corner_count=4,
        center_depth_m=10.0,
        unclipped_bbox_xyxy=(0.0, 0.0, 20.0, 20.0),
        clipped_bbox_xyxy=(0.0, 0.0, 20.0, 20.0),
        clipped_area_fraction=0.5,
        letterbox_height_px=16.0,
    )

    assert is_visually_eligible(observation, "2")
    assert not is_visually_eligible(observation, "1")
    assert not is_visually_eligible(
        replace(observation, clipped_area_fraction=0.49), "4"
    )
