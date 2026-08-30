"""Tests for the pre-registered v0.10 rankings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from driving_scene_data_loop.feedback_retraining import get_oracle_arm
from driving_scene_data_loop.selection import SelectionError, freeze_v010_rankings

SCENARIOS = ("near", "hold", "corridor")


def test_v010_rankings_sort_by_their_declared_signals(tmp_path: Path) -> None:
    pool_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "v010"

    report = freeze_v010_rankings(
        pool_rows_path=pool_dir / "pool2_windows.jsonl",
        pool_report_path=pool_dir / "pool2_report.json",
        output_dir=output_dir,
        budgets=(3, 6),
        random_seeds=(201, 202),
    )

    forbidden = {"labels", "loss_mask", "event_group_ids", "bindings", "similarity"}
    names = ["disagreement_v010", "prob_ranked_v010", "random_seed201", "random_seed202"]
    for method in names:
        rows = _read_jsonl(output_dir / f"{method}_ranked.jsonl")
        assert len(rows) == 6
        assert len({r["window_id"] for r in rows}) == 6
        assert not set().union(*(set(r) for r in rows)) & forbidden

    # Within each class queue the declared signal must be non-increasing.
    dis = _read_jsonl(output_dir / "disagreement_v010_ranked.jsonl")
    prob = _read_jsonl(output_dir / "prob_ranked_v010_ranked.jsonl")
    for scenario in SCENARIOS:
        spreads = [cast(float, r["disagreement"]) for r in dis
                   if r["query_scenario_id"] == scenario]
        assert spreads == sorted(spreads, reverse=True)
        means = [cast(float, r["base_probability"]) for r in prob
                 if r["query_scenario_id"] == scenario]
        assert means == sorted(means, reverse=True)

    # Random rows carry no selector diagnostics at all.
    rand = _read_jsonl(output_dir / "random_seed201_ranked.jsonl")
    assert not set().union(*(set(r) for r in rand)) & {"disagreement", "base_probability"}
    prefixes = cast(dict[str, dict[str, dict[str, object]]], report["prefixes"])
    assert prefixes["disagreement_v010"]["3"]["count"] == 3


def test_v010_rejects_a_negative_spread(tmp_path: Path) -> None:
    pool_dir = _write_fixture(tmp_path)
    rows = _read_jsonl(pool_dir / "pool2_windows.jsonl")
    rows[0]["probability_std"] = [-0.1, 0.1, 0.1]
    _write_jsonl(pool_dir / "pool2_windows.jsonl", rows)

    with pytest.raises(SelectionError, match="probability_std"):
        freeze_v010_rankings(
            pool_rows_path=pool_dir / "pool2_windows.jsonl",
            pool_report_path=pool_dir / "pool2_report.json",
            output_dir=tmp_path / "v010",
            budgets=(3, 6),
            random_seeds=(201, 202),
        )


def test_v010_arms_are_registered() -> None:
    arm = get_oracle_arm("v010-disagreement-900")
    assert arm.method == "disagreement_v010"
    assert arm.budget == 900
    assert get_oracle_arm("v010-random203-300").method == "random_seed203"


def _write_fixture(tmp_path: Path) -> Path:
    pool_dir = tmp_path / "pool2"
    pool_dir.mkdir()
    rows: list[dict[str, object]] = []
    for index in range(18):
        rows.append(
            {
                "window_id": f"window-{index:02d}",
                "scene_token": f"scene-{index // 2:02d}",
                "scene_name": f"scene-name-{index // 2:02d}",
                "log_token": f"log-{index // 6}",
                "partition": "u",
                "start_frame_index": index % 2,
                "end_frame_index": index % 2 + 4,
                "sample_tokens": [f"sample-{index}-{f}" for f in range(5)],
                "timestamps_s": [f * 0.5 for f in range(5)],
                "frame_refs": [f"frame-{index}-{f}" for f in range(5)],
                "pool_index": index,
                "base_probabilities": [
                    0.10 + (index % 7) * 0.09,
                    0.90 - (index % 5) * 0.08,
                    0.50 + ((index * 3) % 11) * 0.03,
                ],
                "probability_std": [
                    0.01 + ((index * 5) % 9) * 0.02,
                    0.02 + (index % 4) * 0.03,
                    0.01 + ((index * 7) % 6) * 0.025,
                ],
            }
        )
    _write_jsonl(pool_dir / "pool2_windows.jsonl", rows)
    (pool_dir / "pool2_report.json").write_text(
        json.dumps(
            {
                "scenario_ids": list(SCENARIOS),
                "base": {"thresholds": {s: 0.5 for s in SCENARIOS}},
            }
        ),
        encoding="utf-8",
    )
    return pool_dir


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
