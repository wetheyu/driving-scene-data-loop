"""Tests for revealing Oracle labels on frozen selected IDs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from driving_scene_data_loop.oracle_reveal import (
    OracleRevealError,
    reveal_selected_labels,
)

SCENARIOS = ("near", "hold", "corridor")
BUDGETS = (2, 4)


def test_reveal_keeps_the_budget_and_nests_prefixes(tmp_path: Path) -> None:
    ranking, windows = _write_fixture(tmp_path)

    report = reveal_selected_labels(
        ranking_paths=(ranking,),
        windows_path=windows,
        output_dir=tmp_path / "reveal",
        budgets=BUDGETS,
    )

    rows = _read_jsonl(tmp_path / "reveal" / "revealed_labels.jsonl")
    assert [row["window_id"] for row in rows] == [f"scene-a:{i:04d}" for i in range(4)]
    assert report["scenario_ids"] == list(SCENARIOS)

    profiles = report["methods"]["random"]["budgets"]
    # ignore and invalid stay in the batch, so windows always equal the budget.
    assert [profiles[str(b)]["windows"] for b in BUDGETS] == list(BUDGETS)
    assert profiles["4"]["usable_labels"] < profiles["4"]["maximum_possible_labels"]

    near = profiles["4"]["per_class"]["near"]
    assert near["label_states"] == {
        "positive": 1,
        "negative": 1,
        "ignore": 1,
        "invalid": 1,
    }
    assert near["usable_labels"] == 2
    assert near["positive_yield"] == pytest.approx(0.25)
    assert near["distinct_positive_event_groups"] == 1
    # A prefix profile only ever describes a prefix of the same ranked list.
    assert profiles["2"]["per_class"]["near"]["label_states"]["ignore"] == 0


def test_reveal_rejects_a_window_outside_the_pool(tmp_path: Path) -> None:
    ranking, windows = _write_fixture(tmp_path, leak_partition="development")

    with pytest.raises(OracleRevealError, match="outside the Pool"):
        reveal_selected_labels(
            ranking_paths=(ranking,),
            windows_path=windows,
            output_dir=tmp_path / "reveal",
            budgets=BUDGETS,
        )


def test_reveal_rejects_a_row_whose_class_order_differs(tmp_path: Path) -> None:
    # One row ordering its classes differently would silently misalign every
    # label against the reported scenario_ids, so the reveal must refuse it.
    ranking, windows = _write_fixture(tmp_path, shuffle_class_order_on="scene-a:0001")

    with pytest.raises(OracleRevealError, match="different class order"):
        reveal_selected_labels(
            ranking_paths=(ranking,),
            windows_path=windows,
            output_dir=tmp_path / "reveal",
            budgets=BUDGETS,
        )


def test_reveal_does_not_modify_the_frozen_inputs(tmp_path: Path) -> None:
    ranking, windows = _write_fixture(tmp_path)
    before = {path: _sha256(path) for path in (ranking, windows)}

    reveal_selected_labels(
        ranking_paths=(ranking,),
        windows_path=windows,
        output_dir=tmp_path / "reveal",
        budgets=BUDGETS,
    )

    assert {path: _sha256(path) for path in (ranking, windows)} == before


def _write_fixture(
    tmp_path: Path,
    *,
    leak_partition: str | None = None,
    shuffle_class_order_on: str | None = None,
) -> tuple[Path, Path]:
    states = ["positive", "negative", "ignore", "invalid"]
    windows: list[dict[str, object]] = []
    for index, state in enumerate(states):
        window_id = f"scene-a:{index:04d}"
        near = state
        row_scenarios = list(SCENARIOS)
        labels = [_label(near), 0, 0]
        masks = [_mask(near), True, True]
        row_states = [near, "negative", "negative"]
        events = [["event-near"] if near == "positive" else [], [], []]
        if window_id == shuffle_class_order_on:
            order = [2, 0, 1]
            row_scenarios = [SCENARIOS[i] for i in order]
            labels = [labels[i] for i in order]
            masks = [masks[i] for i in order]
            row_states = [row_states[i] for i in order]
            events = [events[i] for i in order]
        windows.append(
            {
                "window_id": window_id,
                "scene_token": "scene-a",
                "scene_name": "scene-name-a",
                "log_token": "log-0",
                "partition": (
                    leak_partition if leak_partition and index == 0 else "u"
                ),
                "start_frame_index": index * 5,
                "end_frame_index": index * 5 + 4,
                "scenario_ids": row_scenarios,
                "labels": labels,
                "loss_mask": masks,
                "label_states": row_states,
                "event_group_ids": events,
            }
        )
    # A Development window the selector never chose must stay hidden.
    windows.append(
        {
            "window_id": "scene-b:0000",
            "scene_token": "scene-b",
            "scene_name": "scene-name-b",
            "log_token": "log-1",
            "partition": "development",
            "start_frame_index": 0,
            "end_frame_index": 4,
            "scenario_ids": list(SCENARIOS),
            "labels": [1, 1, 1],
            "loss_mask": [True, True, True],
            "label_states": ["positive", "positive", "positive"],
            "event_group_ids": [["secret"], ["secret"], ["secret"]],
        }
    )
    windows_path = tmp_path / "windows.jsonl"
    _write_jsonl(windows_path, windows)

    ranking_path = tmp_path / "random_ranked.jsonl"
    _write_jsonl(
        ranking_path,
        [
            {
                "rank": index + 1,
                "method": "random",
                "window_id": f"scene-a:{index:04d}",
                "scene_token": "scene-a",
                "log_token": "log-0",
                "start_frame_index": index * 5,
                "end_frame_index": index * 5 + 4,
            }
            for index in range(4)
        ],
    )
    return ranking_path, windows_path


def _label(state: str) -> int | None:
    return {"positive": 1, "negative": 0}.get(state)


def _mask(state: str) -> bool:
    return state in {"positive", "negative"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
