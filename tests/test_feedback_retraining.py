"""Tests for joining one frozen Oracle-feedback training arm."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from driving_scene_data_loop.feedback_retraining import (
    FeedbackRetrainingError,
    OracleArm,
    get_oracle_arm,
    load_feedback_windows,
)
from driving_scene_data_loop.window_dataset import PUBLIC_U_FIELDS

SCENARIOS = ("near", "hold", "corridor")


def test_feedback_join_uses_public_frames_and_revealed_targets(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    data = load_feedback_windows(
        private_windows_path=paths["private"],
        public_pool_windows_path=paths["public"],
        revealed_labels_path=paths["revealed"],
        reveal_profile_path=paths["profile"],
        frame_index_path=paths["frame_index"],
        arm=OracleArm("fixture", "mining", 2),
    )

    assert data.scenario_ids == SCENARIOS
    assert data.partitions == ("l0", "development", "feedback", "feedback")
    assert data.window_ids[-2:] == ("pool-0", "pool-1")
    assert data.frame_feature_indices[-2:].tolist() == [
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
    ]
    assert data.labels[-2:].tolist() == [[1, -1, 0], [0, 1, 0]]
    assert data.loss_mask[-2:].tolist() == [
        [True, False, True],
        [True, True, True],
    ]


def test_feedback_join_rejects_a_different_scenario_order(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    paths["profile"].write_text(
        json.dumps({"scenario_ids": ["corridor", "near", "hold"]}),
        encoding="utf-8",
    )

    with pytest.raises(FeedbackRetrainingError, match="scenario order differ"):
        load_feedback_windows(
            private_windows_path=paths["private"],
            public_pool_windows_path=paths["public"],
            revealed_labels_path=paths["revealed"],
            reveal_profile_path=paths["profile"],
            frame_index_path=paths["frame_index"],
            arm=OracleArm("fixture", "mining", 2),
        )


def test_formal_arm_list_is_closed() -> None:
    assert get_oracle_arm("mining-300-oracle").budget == 300
    with pytest.raises(FeedbackRetrainingError, match="unknown Oracle arm"):
        get_oracle_arm("new-arm-after-reveal")


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    frame_refs = [f"frame-{index}" for index in range(20)]
    frame_index = tmp_path / "frame_index.jsonl"
    _write_jsonl(
        frame_index,
        [
            {"feature_index": index, "media_ref": frame_ref}
            for index, frame_ref in enumerate(frame_refs)
        ],
    )

    private = tmp_path / "windows.jsonl"
    _write_jsonl(
        private,
        [
            _private_window("l0-0", "l0", frame_refs[0:5], [1, 0, 1]),
            _private_window("dev-0", "development", frame_refs[5:10], [0, 1, 0]),
            # These private U labels deliberately disagree with the reveal fixture.
            # The feedback join must not use them.
            _private_window("pool-0", "u", frame_refs[10:15], [0, 0, 0]),
            _private_window("pool-1", "u", frame_refs[15:20], [1, 0, 1]),
        ],
    )

    public = tmp_path / "u_windows.jsonl"
    public_rows = [
        _public_window("pool-0", frame_refs[10:15]),
        _public_window("pool-1", frame_refs[15:20]),
    ]
    assert all(set(row) == set(PUBLIC_U_FIELDS) for row in public_rows)
    _write_jsonl(public, public_rows)

    revealed = tmp_path / "revealed_labels.jsonl"
    _write_jsonl(
        revealed,
        [
            {
                "method": "mining",
                "rank": 1,
                "window_id": "pool-0",
                "labels": [1, None, 0],
                "loss_mask": [True, False, True],
            },
            {
                "method": "mining",
                "rank": 2,
                "window_id": "pool-1",
                "labels": [0, 1, 0],
                "loss_mask": [True, True, True],
            },
        ],
    )
    profile = tmp_path / "label_profile.json"
    profile.write_text(json.dumps({"scenario_ids": list(SCENARIOS)}), encoding="utf-8")
    return {
        "private": private,
        "public": public,
        "revealed": revealed,
        "profile": profile,
        "frame_index": frame_index,
    }


def _private_window(
    window_id: str,
    partition: str,
    frame_refs: list[str],
    labels: list[int],
) -> dict[str, object]:
    return {
        "window_id": window_id,
        "partition": partition,
        "scenario_ids": list(SCENARIOS),
        "frame_refs": frame_refs,
        "labels": labels,
        "loss_mask": [True, True, True],
    }


def _public_window(window_id: str, frame_refs: list[str]) -> dict[str, object]:
    return {
        "window_id": window_id,
        "scene_token": "scene-u",
        "scene_name": "scene-name-u",
        "log_token": "log-u",
        "partition": "u",
        "start_frame_index": 0,
        "end_frame_index": 4,
        "sample_tokens": [f"sample-{window_id}-{index}" for index in range(5)],
        "timestamps_s": [index * 0.5 for index in range(5)],
        "frame_refs": frame_refs,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
