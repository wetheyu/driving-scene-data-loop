"""Deterministic whole-log partitioning for nuScenes trainval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nuscenes.nuscenes import NuScenes  # type: ignore[import-untyped]

SPLIT_VERSION = "split-v1"
DATASET_RELEASE = "v1.0-trainval"
OFFICIAL_PARTITIONS = ("task_design", "l0", "development", "u", "frozen_test")
_TRAIN_PARTITIONS = OFFICIAL_PARTITIONS[:-1]
_CUMULATIVE_PERCENTAGES = (10, 35, 50, 100)

SceneAssignment = tuple[str, str, str, str]


class SplitBuildError(ValueError):
    """Raised when metadata cannot produce a leak-free split manifest."""


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Persisted scene assignments for one deterministic whole-log split."""

    split_version: str
    dataset_release: str
    official_split_source: str
    scene_assignments: tuple[SceneAssignment, ...]

    def __post_init__(self) -> None:
        if self.split_version != SPLIT_VERSION:
            raise SplitBuildError(f"split_version must be {SPLIT_VERSION!r}")
        if self.dataset_release != DATASET_RELEASE:
            raise SplitBuildError(f"dataset_release must be {DATASET_RELEASE!r}")
        if not self.official_split_source:
            raise SplitBuildError("official_split_source must not be empty")
        if not self.scene_assignments:
            raise SplitBuildError("scene_assignments must not be empty")
        if self.scene_assignments != tuple(sorted(self.scene_assignments)):
            raise SplitBuildError("scene_assignments must be sorted")

        scene_tokens: set[str] = set()
        scene_names: set[str] = set()
        partition_by_log: dict[str, str] = {}
        for scene_token, scene_name, log_token, partition in self.scene_assignments:
            if not scene_token or not scene_name or not log_token:
                raise SplitBuildError("scene assignment references must not be empty")
            if partition not in OFFICIAL_PARTITIONS:
                raise SplitBuildError(f"unknown partition: {partition}")
            if scene_token in scene_tokens:
                raise SplitBuildError(f"duplicate scene token: {scene_token}")
            if scene_name in scene_names:
                raise SplitBuildError(f"duplicate scene name: {scene_name}")
            scene_tokens.add(scene_token)
            scene_names.add(scene_name)

            previous = partition_by_log.setdefault(log_token, partition)
            if previous != partition:
                raise SplitBuildError(f"scenes from log {log_token} must share one partition")

    @property
    def log_assignments(self) -> tuple[tuple[str, str], ...]:
        """Derive the unique log assignments recorded by the scene rows."""

        return tuple(
            sorted(
                {
                    log_token: partition for _, _, log_token, partition in self.scene_assignments
                }.items()
            )
        )

    @property
    def manifest_sha256(self) -> str:
        """Hash the logical split content, excluding derived fields."""

        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def log_counts(self) -> dict[str, int]:
        """Count assigned logs by partition."""

        return {
            partition: sum(value == partition for _, value in self.log_assignments)
            for partition in OFFICIAL_PARTITIONS
        }

    def scene_counts(self) -> dict[str, int]:
        """Count assigned scenes by partition."""

        return {
            partition: sum(value == partition for _, _, _, value in self.scene_assignments)
            for partition in OFFICIAL_PARTITIONS
        }

    def to_dict(self) -> dict[str, object]:
        """Return the minimal persisted split document."""

        return {
            "dataset_release": self.dataset_release,
            "official_split_source": self.official_split_source,
            "scene_assignments": [
                {
                    "log_token": log_token,
                    "partition": partition,
                    "scene_name": scene_name,
                    "scene_token": scene_token,
                }
                for scene_token, scene_name, log_token, partition in self.scene_assignments
            ],
            "split_version": self.split_version,
        }

    def to_json(self) -> str:
        """Serialize the logical split as deterministic compact JSON."""

        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


def _required_string(row: dict[str, object], field: str, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise SplitBuildError(f"{context}.{field} must be a non-empty string")
    return value


def load_split_manifest(path: Path) -> SplitManifest:
    """Load a split manifest, ignoring fields from the older verbose format."""

    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SplitBuildError(f"missing split manifest: {path}") from error
    except json.JSONDecodeError as error:
        raise SplitBuildError(f"invalid JSON in split manifest: {error.msg}") from error
    if not isinstance(raw, dict):
        raise SplitBuildError("split manifest must be a JSON object")

    rows = raw.get("scene_assignments")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise SplitBuildError("split manifest scene_assignments must be a list of objects")
    assignments = tuple(
        sorted(
            (
                _required_string(row, "scene_token", "scene_assignment"),
                _required_string(row, "scene_name", "scene_assignment"),
                _required_string(row, "log_token", "scene_assignment"),
                _required_string(row, "partition", "scene_assignment"),
            )
            for row in rows
        )
    )
    return SplitManifest(
        split_version=_required_string(raw, "split_version", "split_manifest"),
        dataset_release=_required_string(raw, "dataset_release", "split_manifest"),
        official_split_source=_required_string(raw, "official_split_source", "split_manifest"),
        scene_assignments=assignments,
    )


def _hash_log_token(log_token: str) -> str:
    payload = f"{SPLIT_VERSION}\0{DATASET_RELEASE}\0{log_token}".encode()
    return hashlib.sha256(payload).hexdigest()


def build_split_manifest(
    nusc: NuScenes,
    *,
    official_train_scene_names: tuple[str, ...],
    official_val_scene_names: tuple[str, ...],
    official_split_source: str,
) -> SplitManifest:
    """Build split-v1 while keeping every complete log in one partition."""

    log_tokens = [_required_string(row, "token", "log") for row in nusc.log]
    if len(log_tokens) != len(set(log_tokens)):
        raise SplitBuildError("log.json contains duplicate tokens")
    known_logs = set(log_tokens)

    scenes: list[tuple[str, str, str]] = []
    for row in nusc.scene:
        scene_token = _required_string(row, "token", "scene")
        scene_name = _required_string(row, "name", f"scene:{scene_token}")
        log_token = _required_string(row, "log_token", f"scene:{scene_token}")
        if log_token not in known_logs:
            raise SplitBuildError(f"scene {scene_token} references unknown log {log_token}")
        scenes.append((scene_token, scene_name, log_token))

    scene_tokens = [token for token, _, _ in scenes]
    scene_names = [name for _, name, _ in scenes]
    if len(scene_tokens) != len(set(scene_tokens)):
        raise SplitBuildError("scene.json contains duplicate scene tokens")
    if len(scene_names) != len(set(scene_names)):
        raise SplitBuildError("scene.json contains duplicate scene names")

    train_names = set(official_train_scene_names)
    val_names = set(official_val_scene_names)
    if train_names & val_names:
        raise SplitBuildError("official train and val scene names overlap")
    if set(scene_names) != train_names | val_names:
        raise SplitBuildError("official split must exactly cover dataset scenes")

    train_logs = {log for _, name, log in scenes if name in train_names}
    val_logs = {log for _, name, log in scenes if name in val_names}
    if train_logs & val_logs:
        raise SplitBuildError("official train and val share a complete log")
    if train_logs | val_logs != known_logs:
        raise SplitBuildError("log.json contains logs not represented by scene.json")

    if not train_logs:
        raise SplitBuildError("official train must contain at least one log")
    boundaries = tuple(
        (len(train_logs) * percentage + 50) // 100 for percentage in _CUMULATIVE_PERCENTAGES
    )
    if len(set(boundaries)) != len(boundaries) or boundaries[0] <= 0:
        raise SplitBuildError("too few train logs for all frozen partitions")
    ranked_logs = sorted(train_logs, key=lambda token: (_hash_log_token(token), token))
    partition_by_log: dict[str, str] = {}
    for index, log_token in enumerate(ranked_logs):
        partition_by_log[log_token] = next(
            partition
            for partition, exclusive_end in zip(_TRAIN_PARTITIONS, boundaries, strict=True)
            if index < exclusive_end
        )
    partition_by_log.update({log_token: "frozen_test" for log_token in val_logs})

    return SplitManifest(
        split_version=SPLIT_VERSION,
        dataset_release=DATASET_RELEASE,
        official_split_source=official_split_source,
        scene_assignments=tuple(
            sorted((token, name, log, partition_by_log[log]) for token, name, log in scenes)
        ),
    )


def build_nuscenes_trainval_split(nusc: NuScenes) -> SplitManifest:
    """Build split-v1 from nuScenes-devkit's official train and val lists."""

    from nuscenes.utils.splits import create_splits_scenes  # type: ignore[import-untyped]

    if nusc.version != DATASET_RELEASE:
        raise SplitBuildError(
            f"expected nuScenes {DATASET_RELEASE}, got {nusc.version}"
        )

    official_splits = create_splits_scenes()
    return build_split_manifest(
        nusc,
        official_train_scene_names=tuple(official_splits["train"]),
        official_val_scene_names=tuple(official_splits["val"]),
        official_split_source=(
            f"nuscenes-devkit=={version('nuscenes-devkit')}:"
            "nuscenes.utils.splits.create_splits_scenes"
        ),
    )
