"""Filter Strem streams to objects that are learnable from CAM_FRONT."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from driving_scene_data_loop.projection import (
    ELIGIBLE_VISIBILITY_TOKENS,
    MINIMUM_CLIPPED_AREA_FRACTION,
    MINIMUM_LETTERBOX_HEIGHT_PX,
    is_visually_eligible,
    project_annotation,
)
from driving_scene_data_loop.strem_converter import ID_MAP_FILENAME, load_numeric_id_map

if TYPE_CHECKING:
    from nuscenes.nuscenes import NuScenes  # type: ignore[import-untyped]

JsonObject = dict[str, object]
CAMERA_CHANNEL = "CAM_FRONT"
MOTOR_VEHICLE_CATEGORIES = frozenset(
    {"car", "truck", "bus", "trailer", "construction_vehicle"}
)
TARGET_CATEGORIES = MOTOR_VEHICLE_CATEGORIES | {"pedestrian"}
FORMAL_MOTOR_CLASS = "motor_vehicle"
SUMMARY_FILENAME = "cam_front_eligibility_summary.json"

_NUSCENES_MOTOR_CATEGORY_MAP = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.trailer": "trailer",
    "vehicle.construction": "construction_vehicle",
}


class StremEligibilityError(ValueError):
    """Raised when nuScenes records and Strem streams disagree."""


@dataclass(frozen=True, slots=True)
class EligibilityBuildResult:
    """Summary of one CAM_FRONT filtering run."""

    output_dir: Path
    stream_count: int
    frame_count: int
    target_annotation_count: int
    eligible_annotation_count: int
    eligible_by_category: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "class_normalization": {
                category: FORMAL_MOTOR_CLASS
                for category in sorted(MOTOR_VEHICLE_CATEGORIES)
            },
            "eligible_annotations": self.eligible_annotation_count,
            "eligible_by_category": dict(self.eligible_by_category),
            "filter": {
                "camera_channel": CAMERA_CHANNEL,
                "minimum_clipped_area_fraction": MINIMUM_CLIPPED_AREA_FRACTION,
                "minimum_letterbox_height_px": MINIMUM_LETTERBOX_HEIGHT_PX,
                "visibility_tokens": sorted(ELIGIBLE_VISIBILITY_TOKENS),
            },
            "frames": self.frame_count,
            "schema_version": "1.0",
            "streams": self.stream_count,
            "target_annotations": self.target_annotation_count,
        }


def build_cam_front_eligible_streams(
    nusc: NuScenes,
    raw_stream_dir: Path,
    output_dir: Path,
) -> EligibilityBuildResult:
    """Project target annotations and remove visually unusable objects."""

    raw_stream_dir = raw_stream_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise StremEligibilityError("eligibility output directory must not already exist")

    numeric_to_instance = load_numeric_id_map(raw_stream_dir / ID_MAP_FILENAME)
    instance_to_numeric = {token: number for number, token in numeric_to_instance.items()}
    category_by_instance = {
        row["token"]: _normalize_category(
            nusc.get("category", row["category_token"])["name"]
        )
        for row in nusc.instance
    }
    category_by_instance = {
        token: category
        for token, category in category_by_instance.items()
        if category is not None
    }
    if category_by_instance.keys() - instance_to_numeric.keys():
        raise StremEligibilityError("target instances are missing from the numeric ID map")

    eligible_ids_by_sample: dict[str, set[int]] = {
        sample["token"]: set() for sample in nusc.sample
    }
    eligible_by_category = {category: 0 for category in sorted(TARGET_CATEGORIES)}
    target_annotation_count = 0
    for annotation in nusc.sample_annotation:
        category = _normalize_category(annotation["category_name"])
        if category is None:
            continue
        target_annotation_count += 1
        sample = nusc.get("sample", annotation["sample_token"])
        camera_token = sample["data"][CAMERA_CHANNEL]
        observation = project_annotation(nusc, annotation["token"], camera_token)
        if is_visually_eligible(observation, annotation["visibility_token"]):
            numeric_id = instance_to_numeric[annotation["instance_token"]]
            eligible_ids_by_sample[sample["token"]].add(numeric_id)
            eligible_by_category[category] += 1

    output_dir.mkdir(parents=True)
    shutil.copy2(raw_stream_dir / ID_MAP_FILENAME, output_dir / ID_MAP_FILENAME)
    frame_count = 0
    retained_dynamic_count = 0
    for scene in sorted(nusc.scene, key=lambda row: row["name"]):
        scene_name = scene["name"]
        source_path = raw_stream_dir / f"nuscenes_{scene_name}.json"
        document = _load_object(source_path)
        frames = _list(document, "frames", source_path.name)
        samples = _scene_samples(nusc, scene)
        if len(frames) != len(samples):
            raise StremEligibilityError(f"stream frame count differs for {scene_name}")

        first_timestamp = _integer(samples[0], "timestamp", scene_name)
        for index, (frame_value, sample) in enumerate(zip(frames, samples, strict=True)):
            frame = _object(frame_value, f"{scene_name}.frames[{index}]")
            sample_timestamp = _integer(sample, "timestamp", scene_name)
            expected_timestamp = (sample_timestamp - first_timestamp) / 1_000_000
            observed_timestamp = _number(frame, "timestamp", scene_name)
            if not math.isclose(observed_timestamp, expected_timestamp, abs_tol=1e-9):
                raise StremEligibilityError(f"stream timestamp differs for {scene_name}")

            sample_record = _single_object(frame, "samples", scene_name)
            annotations = _list(sample_record, "annotations", scene_name)
            sample_token = cast(str, sample["token"])
            expected_ids = eligible_ids_by_sample[sample_token]
            seen_ids: set[int] = set()
            retained: list[object] = []
            for value in annotations:
                annotation = _object(value, f"{scene_name}.annotations")
                numeric_id = _integer(annotation, "id", scene_name)
                if numeric_id == 0:
                    retained.append(annotation)
                elif numeric_id in expected_ids:
                    normalized = dict(annotation)
                    category = category_by_instance[numeric_to_instance[numeric_id]]
                    if category in MOTOR_VEHICLE_CATEGORIES:
                        normalized["class"] = FORMAL_MOTOR_CLASS
                    retained.append(normalized)
                    seen_ids.add(numeric_id)
            if seen_ids != expected_ids:
                raise StremEligibilityError(
                    f"eligible objects are missing from {scene_name} frame {index}"
                )
            sample_record["annotations"] = retained
            frame_count += 1
            retained_dynamic_count += len(seen_ids)

        (output_dir / source_path.name).write_text(
            json.dumps(document, allow_nan=False, separators=(",", ":")),
            encoding="utf-8",
        )

    eligible_annotation_count = sum(eligible_by_category.values())
    if retained_dynamic_count != eligible_annotation_count:
        raise StremEligibilityError("stream and projection counts differ")
    result = EligibilityBuildResult(
        output_dir=output_dir,
        stream_count=len(nusc.scene),
        frame_count=frame_count,
        target_annotation_count=target_annotation_count,
        eligible_annotation_count=eligible_annotation_count,
        eligible_by_category=tuple(sorted(eligible_by_category.items())),
    )
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(result.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _scene_samples(nusc: NuScenes, scene: JsonObject) -> list[JsonObject]:
    samples: list[JsonObject] = []
    sample_token = cast(str, scene["first_sample_token"])
    while sample_token:
        sample = cast(JsonObject, nusc.get("sample", sample_token))
        samples.append(sample)
        sample_token = cast(str, sample["next"])
    return samples


def _normalize_category(category_name: str) -> str | None:
    if category_name.startswith("human.pedestrian"):
        return "pedestrian"
    if category_name.startswith("vehicle.bus."):
        return "bus"
    return _NUSCENES_MOTOR_CATEGORY_MAP.get(category_name)


def _load_object(path: Path) -> JsonObject:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StremEligibilityError(f"cannot load JSON: {path}") from error
    return _object(value, path.name)


def _object(value: object, context: str) -> JsonObject:
    if not isinstance(value, dict):
        raise StremEligibilityError(f"{context} must be an object")
    return cast(JsonObject, value)


def _list(row: JsonObject, field: str, context: str) -> list[object]:
    value = row.get(field)
    if not isinstance(value, list):
        raise StremEligibilityError(f"{context}.{field} must be a list")
    return value


def _single_object(row: JsonObject, field: str, context: str) -> JsonObject:
    values = _list(row, field, context)
    if len(values) != 1:
        raise StremEligibilityError(f"{context}.{field} must contain one item")
    return _object(values[0], f"{context}.{field}[0]")


def _integer(row: JsonObject, field: str, context: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise StremEligibilityError(f"{context}.{field} must be an integer")
    return value


def _number(row: JsonObject, field: str, context: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise StremEligibilityError(f"{context}.{field} must be numeric")
    return float(value)
