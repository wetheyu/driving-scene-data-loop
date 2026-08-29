"""Tests for five-frame labels and the public/private data boundary."""

from __future__ import annotations

import pytest

from driving_scene_data_loop.scenario_events import ScenarioEvent
from driving_scene_data_loop.splits import SplitManifest
from driving_scene_data_loop.strem_adapter import StremInterval, StremRunResult
from driving_scene_data_loop.window_dataset import (
    WindowDatasetError,
    five_frame_ranges,
    invalid_target,
    label_window,
    load_cam_front_frames,
    overlapping_events,
    public_u_record,
    slice_scene_stream,
)

JsonObject = dict[str, object]


def test_cam_front_frames_follow_the_devkit_scene_chain() -> None:
    records: dict[tuple[str, str], JsonObject] = {
        ("scene", "scene-token"): {
            "name": "scene-0001",
            "log_token": "log-token",
            "first_sample_token": "sample-1",
        },
        ("sample", "sample-1"): {
            "data": {"CAM_FRONT": "camera-1"},
            "next": "sample-2",
        },
        ("sample", "sample-2"): {
            "data": {"CAM_FRONT": "camera-2"},
            "next": "",
        },
        ("sample_data", "camera-1"): {
            "filename": "samples/CAM_FRONT/frame-1.jpg"
        },
        ("sample_data", "camera-2"): {
            "filename": "samples/CAM_FRONT/frame-2.jpg"
        },
    }

    class FakeNuScenes:
        def get(self, table: str, token: str) -> JsonObject:
            return records[(table, token)]

    split = SplitManifest(
        split_version="split-v1",
        dataset_release="v1.0-trainval",
        official_split_source="test",
        scene_assignments=(
            ("scene-token", "scene-0001", "log-token", "task_design"),
        ),
    )

    result = load_cam_front_frames(FakeNuScenes(), split)

    assert [frame.sample_token for frame in result["scene-token"]] == [
        "sample-1",
        "sample-2",
    ]


def test_stride_one_ranges_stay_inside_one_scene() -> None:
    assert five_frame_ranges(4) == ()
    assert five_frame_ranges(5) == ((0, 4),)
    assert five_frame_ranges(7) == ((0, 4), (1, 5), (2, 6))


def test_substream_preserves_original_indices_and_timestamps() -> None:
    stream: JsonObject = {
        "version": "0.2.1",
        "frames": [
            {"index": index, "timestamp": index * 0.4, "samples": []}
            for index in range(7)
        ],
    }

    result = slice_scene_stream(stream, 1, 5)

    frames = result["frames"]
    assert isinstance(frames, list)
    assert [frame["index"] for frame in frames] == [1, 2, 3, 4, 5]
    assert [frame["timestamp"] for frame in frames] == pytest.approx(
        [0.4, 0.8, 1.2, 1.6, 2.0]
    )


def test_full_event_overlap_only_decides_whether_bounded_strem_is_needed() -> None:
    event = _event(3, 7, (("pedestrian", 17),))

    assert overlapping_events((event,), 0, 2) == ()
    assert overlapping_events((event,), 0, 4) == (event,)
    assert overlapping_events((event,), 7, 11) == (event,)


def test_window_states_keep_partial_and_failed_evidence_out_of_loss() -> None:
    event = _event(1, 5, (("pedestrian", 17),))

    negative = label_window((), None)
    partial = label_window((event,), StremRunResult("no_match", "scenario", ()))
    failed = invalid_target()

    assert (negative.state, negative.label, negative.loss_mask) == ("negative", 0, True)
    assert (partial.state, partial.label, partial.loss_mask) == ("ignore", None, False)
    assert (failed.state, failed.label, failed.loss_mask) == ("invalid", None, False)


def test_bounded_match_must_keep_the_same_object_binding() -> None:
    event = _event(1, 4, (("pedestrian", 17),))
    matching = StremRunResult(
        "match",
        "scenario",
        (_interval(1, 4, (("pedestrian", 17),)),),
    )
    replacing_object = StremRunResult(
        "match",
        "scenario",
        (_interval(1, 4, (("pedestrian", 18),)),),
    )

    target = label_window((event,), matching)

    assert target.state == "positive"
    assert target.label == 1
    assert target.loss_mask
    assert target.event_group_ids == (event.event_group_id,)
    with pytest.raises(WindowDatasetError, match="bindings"):
        label_window((event,), replacing_object)


def test_public_u_record_contains_no_oracle_evidence() -> None:
    private = {
        "window_id": "scene:0001",
        "scene_token": "scene",
        "scene_name": "scene-0001",
        "log_token": "log",
        "partition": "u",
        "start_frame_index": 1,
        "end_frame_index": 5,
        "sample_tokens": ["s1", "s2", "s3", "s4", "s5"],
        "timestamps_s": [0.5, 1.0, 1.5, 2.0, 2.5],
        "frame_refs": ["frame-1", "frame-2", "frame-3", "frame-4", "frame-5"],
        "scenario_ids": ["scenario"],
        "labels": [1],
        "loss_mask": [True],
        "label_states": ["positive"],
        "event_group_ids": [["event-1"]],
        "bindings": {"pedestrian": 17},
    }

    public = public_u_record(private)

    assert set(public) == {
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
    }
    assert "labels" not in public
    assert "bindings" not in public


def _event(
    start: int,
    end: int,
    bindings: tuple[tuple[str, int], ...],
) -> ScenarioEvent:
    interval = _interval(start, end, bindings)
    return ScenarioEvent(
        event_group_id="scene:scenario:event-0001",
        scene_id="scene",
        scenario_id="scenario",
        start_frame_index=start,
        end_frame_index=end,
        bindings=bindings,
        source_intervals=(interval,),
    )


def _interval(
    start: int,
    end: int,
    bindings: tuple[tuple[str, int], ...],
) -> StremInterval:
    return StremInterval(
        start_frame_index=start,
        end_frame_index=end,
        start_time_semantics="exact",
        start_lower_timestamp=float(start),
        start_upper_timestamp=float(start),
        start_lower_inclusive=True,
        start_upper_inclusive=True,
        end_lower_timestamp=float(end),
        end_upper_timestamp=float(end + 1),
        constraints=(),
        bindings=bindings,
    )
