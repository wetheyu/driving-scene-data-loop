"""Build five-frame model windows from scene-level Strem evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal

from driving_scene_data_loop.scenario_events import ScenarioEvent
from driving_scene_data_loop.splits import SplitManifest
from driving_scene_data_loop.strem_adapter import StremRunResult

if TYPE_CHECKING:
    from nuscenes.nuscenes import NuScenes  # type: ignore[import-untyped]

JsonObject = dict[str, object]
LabelState = Literal["positive", "negative", "ignore", "invalid"]

WINDOW_SIZE = 5
CAMERA_CHANNEL = "CAM_FRONT"
PUBLIC_U_FIELDS = (
    "window_id",
    "scene_token",
    "scene_name",
    "log_token",
    "partition",
    "start_frame_index",
    "end_frame_index",
    "sample_tokens",
    "timestamps_s",
    "frame_refs",
)


class WindowDatasetError(ValueError):
    """Raised when model windows would contain incorrect evidence or metadata."""


@dataclass(frozen=True, slots=True)
class FrameReference:
    """One annotated CAM_FRONT frame joined to its nuScenes sample."""

    sample_token: str
    media_ref: str


@dataclass(frozen=True, slots=True)
class WindowTarget:
    """One scenario label and its loss mask for one five-frame window."""

    state: LabelState
    label: int | None
    loss_mask: bool
    event_group_ids: tuple[str, ...] = ()


def five_frame_ranges(frame_count: int) -> tuple[tuple[int, int], ...]:
    """Return inclusive ranges for stride-one five-frame windows."""

    if frame_count < 0:
        raise WindowDatasetError("frame_count must not be negative")
    return tuple(
        (start, start + WINDOW_SIZE - 1)
        for start in range(max(0, frame_count - WINDOW_SIZE + 1))
    )


def slice_scene_stream(
    stream: JsonObject,
    start_frame_index: int,
    end_frame_index: int,
) -> JsonObject:
    """Copy one five-frame substream without rebasing its indices or timestamps."""

    frames = stream.get("frames")
    if not isinstance(frames, list):
        raise WindowDatasetError("scene stream frames must be a list")
    if end_frame_index - start_frame_index + 1 != WINDOW_SIZE:
        raise WindowDatasetError("a model window must contain exactly five frames")
    if start_frame_index < 0 or end_frame_index >= len(frames):
        raise WindowDatasetError("window is outside the scene stream")

    selected = frames[start_frame_index : end_frame_index + 1]
    for expected_index, value in zip(
        range(start_frame_index, end_frame_index + 1),
        selected,
        strict=True,
    ):
        if not isinstance(value, dict) or value.get("index") != expected_index:
            raise WindowDatasetError("scene stream frame indices are not contiguous")
    result = dict(stream)
    result["frames"] = selected
    return result


def overlapping_events(
    events: tuple[ScenarioEvent, ...],
    start_frame_index: int,
    end_frame_index: int,
) -> tuple[ScenarioEvent, ...]:
    """Return complete-scene events that touch one model window."""

    return tuple(
        event
        for event in events
        if event.start_frame_index <= end_frame_index
        and event.end_frame_index >= start_frame_index
    )


def label_window(
    events: tuple[ScenarioEvent, ...],
    bounded_result: StremRunResult | None,
) -> WindowTarget:
    """Map full-scene overlap plus a bounded Strem result to one target."""

    if not events:
        if bounded_result is not None:
            raise WindowDatasetError("a non-overlapping window must not run bounded Strem")
        return WindowTarget(state="negative", label=0, loss_mask=True)
    if bounded_result is None:
        raise WindowDatasetError("an event-overlapping window needs bounded Strem evidence")
    if bounded_result.status == "no_match":
        return WindowTarget(state="ignore", label=None, loss_mask=False)

    matched_bindings = {interval.bindings for interval in bounded_result.intervals}
    event_group_ids = tuple(
        event.event_group_id
        for event in events
        if event.bindings in matched_bindings
    )
    if not event_group_ids:
        raise WindowDatasetError(
            "bounded Strem bindings do not match the overlapping full-scene event"
        )
    return WindowTarget(
        state="positive",
        label=1,
        loss_mask=True,
        event_group_ids=event_group_ids,
    )


def invalid_target() -> WindowTarget:
    """Represent missing or failed evidence without turning it into a negative."""

    return WindowTarget(state="invalid", label=None, loss_mask=False)


def public_u_record(private_window: JsonObject) -> JsonObject:
    """Keep only selector-safe identity, time, and CAM_FRONT references."""

    if private_window.get("partition") != "u":
        raise WindowDatasetError("only U windows may enter the public pool")
    return {field: private_window[field] for field in PUBLIC_U_FIELDS}


def load_cam_front_frames(
    nusc: NuScenes,
    split_manifest: SplitManifest,
) -> dict[str, tuple[FrameReference, ...]]:
    """Walk each scene and collect its annotated CAM_FRONT keyframes."""

    result: dict[str, tuple[FrameReference, ...]] = {}
    for scene_token, scene_name, log_token, _ in split_manifest.scene_assignments:
        scene = nusc.get("scene", scene_token)
        if scene["name"] != scene_name or scene["log_token"] != log_token:
            raise WindowDatasetError(f"split metadata differs for scene {scene_token}")

        frames: list[FrameReference] = []
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            camera_token = sample["data"].get(CAMERA_CHANNEL)
            if not camera_token:
                raise WindowDatasetError(
                    f"sample has no CAM_FRONT keyframe: {sample_token}"
                )
            camera = nusc.get("sample_data", camera_token)
            media_ref = camera["filename"]
            _validate_media_ref(media_ref)
            frames.append(
                FrameReference(
                    sample_token=sample_token,
                    media_ref=media_ref,
                )
            )
            sample_token = sample["next"]
        if not frames:
            raise WindowDatasetError(f"scene has no samples: {scene_token}")
        result[scene_token] = tuple(frames)
    return result


def _validate_media_ref(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.parts[:2] != ("samples", CAMERA_CHANNEL):
        raise WindowDatasetError(f"invalid CAM_FRONT media reference: {value}")
