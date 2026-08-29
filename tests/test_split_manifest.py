"""Tests for the leak-free whole-log nuScenes split."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from driving_scene_data_loop.splits import (
    SplitBuildError,
    SplitManifest,
    _hash_log_token,
    build_split_manifest,
    load_split_manifest,
)

SPLIT_SOURCE = "test-fixture:official-splits"


def split_fixture(
    *,
    reverse_rows: bool = False,
    shared_train_val_log: bool = False,
    unknown_log_reference: bool = False,
) -> tuple[SimpleNamespace, tuple[str, ...], tuple[str, ...]]:
    """Create 20 train logs, 4 val logs, and one repeated-log scene."""

    logs = [{"token": f"train-log-{index:02d}"} for index in range(20)]
    logs += [{"token": f"val-log-{index:02d}"} for index in range(4)]
    scenes: list[dict[str, str]] = []
    train_names = [f"train-scene-{index:02d}" for index in range(20)]
    val_names = [f"val-scene-{index:02d}" for index in range(4)]

    for index, name in enumerate(train_names):
        scenes.append(
            {
                "token": f"train-scene-token-{index:02d}",
                "name": name,
                "log_token": f"train-log-{index:02d}",
            }
        )
    train_names.append("train-scene-same-log")
    scenes.append(
        {
            "token": "train-scene-token-same-log",
            "name": train_names[-1],
            "log_token": "train-log-00",
        }
    )
    for index, name in enumerate(val_names):
        log_token = f"val-log-{index:02d}"
        if shared_train_val_log and index == 0:
            log_token = "train-log-00"
        if unknown_log_reference and index == 0:
            log_token = "missing-log"
        scenes.append(
            {
                "token": f"val-scene-token-{index:02d}",
                "name": name,
                "log_token": log_token,
            }
        )

    if reverse_rows:
        logs.reverse()
        scenes.reverse()
    return SimpleNamespace(log=logs, scene=scenes), tuple(train_names), tuple(val_names)


def build_fixture_manifest() -> SplitManifest:
    nusc, train_names, val_names = split_fixture()
    return build_split_manifest(
        nusc,
        official_train_scene_names=train_names,
        official_val_scene_names=val_names,
        official_split_source=SPLIT_SOURCE,
    )


def test_split_groups_complete_logs_and_freezes_official_val() -> None:
    manifest = build_fixture_manifest()

    assert manifest.log_counts() == {
        "task_design": 2,
        "l0": 5,
        "development": 3,
        "u": 10,
        "frozen_test": 4,
    }
    partition_by_log = dict(manifest.log_assignments)
    assert all(
        partition_by_log[log_token] == partition
        for _, _, log_token, partition in manifest.scene_assignments
    )
    assert {
        partition
        for _, name, _, partition in manifest.scene_assignments
        if name.startswith("val-scene")
    } == {"frozen_test"}


def test_split_identity_is_independent_of_metadata_row_order() -> None:
    first_nusc, train_names, val_names = split_fixture()
    second_nusc, _, _ = split_fixture(reverse_rows=True)

    first = build_split_manifest(
        first_nusc,
        official_train_scene_names=train_names,
        official_val_scene_names=val_names,
        official_split_source=SPLIT_SOURCE,
    )
    second = build_split_manifest(
        second_nusc,
        official_train_scene_names=train_names,
        official_val_scene_names=val_names,
        official_split_source=SPLIT_SOURCE,
    )

    assert first.to_json() == second.to_json()
    assert first.manifest_sha256 == second.manifest_sha256


def test_log_hash_keeps_the_frozen_split_v1_bytes() -> None:
    expected = hashlib.sha256(b"split-v1\0v1.0-trainval\0train-log-00").hexdigest()

    assert _hash_log_token("train-log-00") == expected


@pytest.mark.parametrize(
    ("fixture_options", "message"),
    [
        ({"shared_train_val_log": True}, "train and val share"),
        ({"unknown_log_reference": True}, "unknown log"),
    ],
)
def test_split_rejects_leaking_or_broken_metadata(
    fixture_options: dict[str, bool],
    message: str,
) -> None:
    nusc, train_names, val_names = split_fixture(**fixture_options)

    with pytest.raises(SplitBuildError, match=message):
        build_split_manifest(
            nusc,
            official_train_scene_names=train_names,
            official_val_scene_names=val_names,
            official_split_source=SPLIT_SOURCE,
        )


def test_split_requires_the_official_lists_to_cover_every_scene() -> None:
    nusc, train_names, val_names = split_fixture()

    with pytest.raises(SplitBuildError, match="exactly cover"):
        build_split_manifest(
            nusc,
            official_train_scene_names=train_names[:-1],
            official_val_scene_names=val_names,
            official_split_source=SPLIT_SOURCE,
        )


def test_loader_accepts_old_manifest_fields_and_rejects_log_leakage(
    tmp_path: Path,
) -> None:
    manifest = build_fixture_manifest()
    raw = manifest.to_dict()
    raw.update(
        {
            "created_at": "2026-08-22T08:00:00Z",
            "hash_algorithm": "sha256",
            "manifest_sha256": "old-format-hash",
            "schema_version": "split-manifest-v1",
        }
    )
    path = tmp_path / "split.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    assert load_split_manifest(path) == manifest

    assignments = raw["scene_assignments"]
    assert isinstance(assignments, list)
    repeated_log_rows = [row for row in assignments if row["log_token"] == "train-log-00"]
    repeated_log_rows[0]["partition"] = "frozen_test"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SplitBuildError, match="must share one partition"):
        load_split_manifest(path)
