"""Build private five-frame labels and a label-free public U manifest."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO, cast

from nuscenes.nuscenes import NuScenes  # type: ignore[import-untyped]

from driving_scene_data_loop.scenario_events import ScenarioEvent, build_scenario_events
from driving_scene_data_loop.splits import OFFICIAL_PARTITIONS, load_split_manifest
from driving_scene_data_loop.strem_adapter import (
    StremAdapter,
    StremAdapterError,
    StremRunResult,
)
from driving_scene_data_loop.window_dataset import (
    JsonObject,
    WindowDatasetError,
    WindowTarget,
    five_frame_ranges,
    invalid_target,
    label_window,
    load_cam_front_frames,
    overlapping_events,
    public_u_record,
    slice_scene_stream,
)


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """One stable project label and its executable Strem specification."""

    scenario_id: str
    path: Path
    strem_name: str


def main() -> None:
    """Build Stage-A artifacts from frozen metadata, streams, split, and specs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--stream-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--spec-dir", type=Path, required=True)
    parser.add_argument("--strem-bin", type=Path, required=True)
    parser.add_argument("--eligibility-summary", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--public-output-dir", type=Path, required=True)
    args = parser.parse_args()

    private_output = args.private_output_dir.resolve()
    public_output = args.public_output_dir.resolve()
    if private_output.exists() or public_output.exists():
        raise WindowDatasetError("output directories must not already exist")

    split = load_split_manifest(args.split_manifest)
    specs = _load_specs(args.spec_dir)
    nusc = NuScenes(
        version="v1.0-trainval",
        dataroot=str(args.dataroot),
        verbose=False,
    )
    frames_by_scene = load_cam_front_frames(nusc, split)
    adapter = StremAdapter(args.strem_bin)
    eligibility_summary = _load_object(args.eligibility_summary)

    private_output.mkdir(parents=True)
    public_output.mkdir(parents=True)
    profile = _empty_profile(specs, eligibility_summary)

    with (
        (private_output / "events.jsonl").open("w", encoding="utf-8") as event_file,
        (private_output / "windows.jsonl").open("w", encoding="utf-8") as window_file,
        (public_output / "u_windows.jsonl").open("w", encoding="utf-8") as public_file,
        tempfile.TemporaryDirectory(prefix="window-labels-") as temporary_dir,
    ):
        bounded_stream_path = Path(temporary_dir) / "window.json"
        for scene_token, scene_name, log_token, partition in split.scene_assignments:
            stream_path = args.stream_dir / f"nuscenes_{scene_name}.json"
            stream = _load_object(stream_path)
            frames = _validated_stream_frames(
                stream,
                frames_by_scene[scene_token],
                scene_name,
            )
            events_by_scenario: dict[str, tuple[ScenarioEvent, ...]] = {}
            full_run_failed: set[str] = set()

            for spec in specs:
                try:
                    run = adapter.run_scene(stream_path, spec.path)
                    _check_spec_name(run, spec)
                except StremAdapterError:
                    full_run_failed.add(spec.scenario_id)
                    _increment(profile, "strem_failures", partition, spec.scenario_id)
                    events_by_scenario[spec.scenario_id] = ()
                    continue
                events = build_scenario_events(scene_token, run)
                events_by_scenario[spec.scenario_id] = events
                _write_events(
                    event_file,
                    events,
                    spec,
                    scene_name,
                    log_token,
                    partition,
                )
                _increment_by(
                    profile,
                    len(events),
                    "event_groups",
                    partition,
                    spec.scenario_id,
                )

            for start_index, end_index in five_frame_ranges(len(frames)):
                events_for_window = {
                    spec.scenario_id: overlapping_events(
                        events_by_scenario[spec.scenario_id],
                        start_index,
                        end_index,
                    )
                    for spec in specs
                }
                needs_bounded_run = any(events_for_window.values())
                if needs_bounded_run:
                    bounded_stream = slice_scene_stream(stream, start_index, end_index)
                    bounded_stream_path.write_text(
                        json.dumps(
                            bounded_stream,
                            allow_nan=False,
                            separators=(",", ":"),
                        ),
                        encoding="utf-8",
                    )

                targets: list[WindowTarget] = []
                for spec in specs:
                    if spec.scenario_id in full_run_failed:
                        target = invalid_target()
                    else:
                        window_events = events_for_window[spec.scenario_id]
                        if not window_events:
                            target = label_window((), None)
                        else:
                            try:
                                bounded_run = adapter.run_scene(
                                    bounded_stream_path,
                                    spec.path,
                                )
                                _check_spec_name(bounded_run, spec)
                                target = label_window(window_events, bounded_run)
                            except StremAdapterError:
                                target = invalid_target()
                                _increment(
                                    profile,
                                    "strem_failures",
                                    partition,
                                    spec.scenario_id,
                                )
                    targets.append(target)
                    _increment(
                        profile,
                        "label_states",
                        partition,
                        spec.scenario_id,
                        target.state,
                    )

                window_frames = frames_by_scene[scene_token][
                    start_index : end_index + 1
                ]
                timestamps = tuple(_frame_timestamp(frame, scene_name) for frame in frames[
                    start_index : end_index + 1
                ])
                window = _window_record(
                    scene_token=scene_token,
                    scene_name=scene_name,
                    log_token=log_token,
                    partition=partition,
                    start_index=start_index,
                    end_index=end_index,
                    frame_refs=tuple(frame.media_ref for frame in window_frames),
                    sample_tokens=tuple(frame.sample_token for frame in window_frames),
                    timestamps=timestamps,
                    specs=specs,
                    targets=tuple(targets),
                )
                _write_json_line(window_file, window)
                _increment(profile, "window_counts", partition)
                if partition == "u":
                    _write_json_line(public_file, public_u_record(window))

            _increment(profile, "scene_counts", partition)
            profile["frames"] = cast(int, profile["frames"]) + len(frames)

    profile["logs_by_partition"] = split.log_counts()
    profile["public_u_windows"] = cast(
        dict[str, int], profile["window_counts"]
    )["u"]
    (private_output / "data_profile.json").write_text(
        json.dumps(profile, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(profile, allow_nan=False, sort_keys=True))


def _load_specs(spec_dir: Path) -> tuple[ScenarioSpec, ...]:
    specs: list[ScenarioSpec] = []
    for path in sorted(spec_dir.glob("*.json")):
        document = _load_object(path)
        name = document.get("name")
        if not isinstance(name, str) or not name:
            raise WindowDatasetError(f"specification has no name: {path}")
        specs.append(ScenarioSpec(path.stem, path.resolve(), name))
    if not specs:
        raise WindowDatasetError("spec directory contains no JSON specifications")
    return tuple(specs)


def _validated_stream_frames(
    stream: JsonObject,
    frame_references: tuple[object, ...],
    scene_name: str,
) -> list[JsonObject]:
    raw_frames = stream.get("frames")
    if not isinstance(raw_frames, list) or any(
        not isinstance(frame, dict) for frame in raw_frames
    ):
        raise WindowDatasetError(f"{scene_name} frames must be objects")
    frames = [dict(frame) for frame in raw_frames]
    if len(frames) != len(frame_references):
        raise WindowDatasetError(f"{scene_name} stream and CAM_FRONT frame counts differ")
    previous_timestamp = -math.inf
    for index, frame in enumerate(frames):
        if frame.get("index") != index:
            raise WindowDatasetError(f"{scene_name} frame indices are not contiguous")
        timestamp = _frame_timestamp(frame, scene_name)
        if timestamp <= previous_timestamp:
            raise WindowDatasetError(f"{scene_name} timestamps are not increasing")
        previous_timestamp = timestamp
    return frames


def _frame_timestamp(frame: JsonObject, scene_name: str) -> float:
    value = frame.get("timestamp")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise WindowDatasetError(f"{scene_name} frame timestamp must be numeric")
    timestamp = float(value)
    if not math.isfinite(timestamp):
        raise WindowDatasetError(f"{scene_name} frame timestamp must be finite")
    return timestamp


def _check_spec_name(result: StremRunResult, spec: ScenarioSpec) -> None:
    if result.specification_name != spec.strem_name:
        raise WindowDatasetError(
            f"Strem returned {result.specification_name!r} for {spec.scenario_id!r}"
        )


def _window_record(
    *,
    scene_token: str,
    scene_name: str,
    log_token: str,
    partition: str,
    start_index: int,
    end_index: int,
    frame_refs: tuple[str, ...],
    sample_tokens: tuple[str, ...],
    timestamps: tuple[float, ...],
    specs: tuple[ScenarioSpec, ...],
    targets: tuple[WindowTarget, ...],
) -> JsonObject:
    return {
        "window_id": f"{scene_token}:{start_index:04d}",
        "scene_token": scene_token,
        "scene_name": scene_name,
        "log_token": log_token,
        "partition": partition,
        "start_frame_index": start_index,
        "end_frame_index": end_index,
        "sample_tokens": list(sample_tokens),
        "timestamps_s": list(timestamps),
        "frame_refs": list(frame_refs),
        "scenario_ids": [spec.scenario_id for spec in specs],
        "labels": [target.label for target in targets],
        "loss_mask": [target.loss_mask for target in targets],
        "label_states": [target.state for target in targets],
        "event_group_ids": [list(target.event_group_ids) for target in targets],
    }


def _write_events(
    destination: TextIO,
    events: tuple[ScenarioEvent, ...],
    spec: ScenarioSpec,
    scene_name: str,
    log_token: str,
    partition: str,
) -> None:
    for event in events:
        row: JsonObject = {
            "event_group_id": event.event_group_id,
            "scenario_id": spec.scenario_id,
            "strem_specification_name": spec.strem_name,
            "scene_token": event.scene_id,
            "scene_name": scene_name,
            "log_token": log_token,
            "partition": partition,
            "start_frame_index": event.start_frame_index,
            "end_frame_index": event.end_frame_index,
            "bindings": dict(event.bindings),
            "source_intervals": [asdict(interval) for interval in event.source_intervals],
        }
        _write_json_line(destination, row)


def _empty_profile(
    specs: tuple[ScenarioSpec, ...],
    eligibility_summary: JsonObject,
) -> JsonObject:
    scenario_ids = tuple(spec.scenario_id for spec in specs)
    partition_scenario_counts = {
        partition: {scenario_id: 0 for scenario_id in scenario_ids}
        for partition in OFFICIAL_PARTITIONS
    }
    label_states = {
        partition: {
            scenario_id: {
                "positive": 0,
                "negative": 0,
                "ignore": 0,
                "invalid": 0,
            }
            for scenario_id in scenario_ids
        }
        for partition in OFFICIAL_PARTITIONS
    }
    target = eligibility_summary.get("target_annotations")
    eligible = eligibility_summary.get("eligible_annotations")
    filtered = (
        target - eligible
        if isinstance(target, int) and isinstance(eligible, int)
        else None
    )
    return {
        "schema_version": "1.0",
        "window_size": 5,
        "window_stride": 1,
        "scenario_ids": list(scenario_ids),
        "scene_counts": {partition: 0 for partition in OFFICIAL_PARTITIONS},
        "window_counts": {partition: 0 for partition in OFFICIAL_PARTITIONS},
        "event_groups": partition_scenario_counts,
        "label_states": label_states,
        "strem_failures": {
            partition: {scenario_id: 0 for scenario_id in scenario_ids}
            for partition in OFFICIAL_PARTITIONS
        },
        "frames": 0,
        "visual_cleaning": {
            "summary": eligibility_summary,
            "filtered_annotations": filtered,
            "filter_reason_counts": None,
            "filter_reason_note": (
                "The existing full-trainval v1 eligibility summary records total "
                "retention, not mutually exclusive rejection-reason counts."
            ),
        },
        "public_u_allowed_fields": [
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
        ],
    }


def _increment(profile: JsonObject, *path: str) -> None:
    _increment_by(profile, 1, *path)


def _increment_by(profile: JsonObject, amount: int, *path: str) -> None:
    current: object = profile
    for key in path[:-1]:
        if not isinstance(current, dict):
            raise WindowDatasetError("profile counter path is invalid")
        current = current[key]
    if not isinstance(current, dict):
        raise WindowDatasetError("profile counter parent is invalid")
    leaf = current[path[-1]]
    if not isinstance(leaf, int):
        raise WindowDatasetError("profile counter is not an integer")
    current[path[-1]] = leaf + amount


def _load_object(path: Path) -> JsonObject:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise WindowDatasetError(f"missing JSON file: {path}") from error
    except json.JSONDecodeError as error:
        raise WindowDatasetError(f"invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise WindowDatasetError(f"JSON document must be an object: {path}")
    return dict(value)


def _write_json_line(destination: TextIO, row: JsonObject) -> None:
    destination.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
