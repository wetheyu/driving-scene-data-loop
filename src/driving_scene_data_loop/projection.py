"""Project one nuScenes annotation into its CAM_FRONT image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from nuscenes.utils.geometry_utils import BoxVisibility  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from nuscenes.nuscenes import NuScenes  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class ProjectionObservation:
    """The image-space evidence used by the frozen visual filter."""

    projectable: bool
    height_eligible: bool
    visible_corner_count: int
    center_depth_m: float
    unclipped_bbox_xyxy: tuple[float, float, float, float] | None
    clipped_bbox_xyxy: tuple[float, float, float, float] | None
    clipped_area_fraction: float
    letterbox_height_px: float


MINIMUM_LETTERBOX_HEIGHT_PX = 16.0
MINIMUM_CLIPPED_AREA_FRACTION = 0.5
ELIGIBLE_VISIBILITY_TOKENS = frozenset({"2", "3", "4"})


def project_annotation(
    nusc: NuScenes,
    annotation_token: str,
    camera_token: str,
) -> ProjectionObservation:
    """Use the devkit transform, then apply the project filter."""

    camera = nusc.get("sample_data", camera_token)
    if camera.get("channel") != "CAM_FRONT" or camera.get("is_key_frame") is not True:
        raise ValueError("camera_token must reference a CAM_FRONT keyframe")

    _, boxes, intrinsic = nusc.get_sample_data(
        camera_token,
        box_vis_level=BoxVisibility.NONE,
        selected_anntokens=[annotation_token],
    )
    if len(boxes) != 1 or intrinsic is None:
        raise ValueError("devkit did not return one camera-space annotation box")
    return project_camera_corners(
        boxes[0].corners(),
        np.asarray(intrinsic, dtype=np.float64),
        image_width=camera["width"],
        image_height=camera["height"],
        center_depth_m=float(boxes[0].center[2]),
    )


def project_camera_corners(
    corners_camera: NDArray[np.float64],
    camera_intrinsic: NDArray[np.float64],
    *,
    image_width: int,
    image_height: int,
    center_depth_m: float,
    letterbox_size: int = 518,
    minimum_letterbox_height_px: float = MINIMUM_LETTERBOX_HEIGHT_PX,
) -> ProjectionObservation:
    """Project eight camera-space box corners and measure the clipped box."""

    corners = np.asarray(corners_camera, dtype=np.float64)
    intrinsic = np.asarray(camera_intrinsic, dtype=np.float64)
    if corners.shape != (3, 8) or intrinsic.shape != (3, 3):
        raise ValueError("corners must be 3x8 and camera_intrinsic must be 3x3")
    if image_width <= 0 or image_height <= 0 or letterbox_size <= 0:
        raise ValueError("image and letterbox dimensions must be positive")

    depths = corners[2]
    if not np.all(depths > 0.1):
        return _not_projectable(center_depth_m)

    projected = intrinsic @ corners
    projected[:2] /= projected[2]
    x_values, y_values = projected[0], projected[1]
    visible = (
        (x_values > 0)
        & (x_values < image_width)
        & (y_values > 0)
        & (y_values < image_height)
    )
    visible_corner_count = int(np.count_nonzero(visible))
    if visible_corner_count == 0:
        return _not_projectable(center_depth_m)

    raw_x_min, raw_x_max = float(x_values.min()), float(x_values.max())
    raw_y_min, raw_y_max = float(y_values.min()), float(y_values.max())
    x_min, x_max = max(0.0, raw_x_min), min(float(image_width), raw_x_max)
    y_min, y_max = max(0.0, raw_y_min), min(float(image_height), raw_y_max)
    raw_area = (raw_x_max - raw_x_min) * (raw_y_max - raw_y_min)
    clipped_area = (x_max - x_min) * (y_max - y_min)
    if raw_area <= 0 or clipped_area <= 0:
        return _not_projectable(center_depth_m)

    letterbox_scale = min(letterbox_size / image_width, letterbox_size / image_height)
    letterbox_height = (y_max - y_min) * letterbox_scale
    return ProjectionObservation(
        projectable=True,
        height_eligible=letterbox_height >= minimum_letterbox_height_px,
        visible_corner_count=visible_corner_count,
        center_depth_m=center_depth_m,
        unclipped_bbox_xyxy=(raw_x_min, raw_y_min, raw_x_max, raw_y_max),
        clipped_bbox_xyxy=(x_min, y_min, x_max, y_max),
        clipped_area_fraction=clipped_area / raw_area,
        letterbox_height_px=letterbox_height,
    )


def is_visually_eligible(
    observation: ProjectionObservation,
    visibility_token: str,
) -> bool:
    """Apply the frozen CAM_FRONT visual-cleaning rule."""

    return (
        observation.projectable
        and observation.height_eligible
        and observation.clipped_area_fraction >= MINIMUM_CLIPPED_AREA_FRACTION
        and visibility_token in ELIGIBLE_VISIBILITY_TOKENS
    )


def _not_projectable(center_depth_m: float) -> ProjectionObservation:
    return ProjectionObservation(
        projectable=False,
        height_eligible=False,
        visible_corner_count=0,
        center_depth_m=center_depth_m,
        unclipped_bbox_xyxy=None,
        clipped_bbox_xyxy=None,
        clipped_area_fraction=0.0,
        letterbox_height_px=0.0,
    )
