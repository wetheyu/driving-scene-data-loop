"""Tests for the Development-informed pure-probability selector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from driving_scene_data_loop.selection import (
    SelectionError,
    freeze_score_ranked_rankings,
)

SCENARIOS = ("near", "hold", "corridor")
BUDGETS = (3, 6)


def test_score_ranking_is_pure_probability_order_with_no_similarity_field(
    tmp_path: Path,
) -> None:
    pool_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "score"

    report = freeze_score_ranked_rankings(
        pool_rows_path=pool_dir / "pool_windows.jsonl",
        pool_report_path=pool_dir / "pool_report.json",
        output_dir=output_dir,
        budgets=BUDGETS,
    )

    assert report["development_informed"] is True
    rows = _read_jsonl(output_dir / "mining_v2_score_ranked_ranked.jsonl")
    assert len(rows) == 6
    assert len({row["window_id"] for row in rows}) == 6
    forbidden = {"similarity", "boundary_margin", "cluster_id", "labels", "loss_mask"}
    assert not set().union(*(set(row) for row in rows)) & forbidden

    # Within each class's queue, probability must be strictly non-increasing.
    for scenario in SCENARIOS:
        probs = [
            float(cast(float, row["base_probability"]))
            for row in rows
            if row["query_scenario_id"] == scenario
        ]
        assert probs == sorted(probs, reverse=True)


def test_score_ranking_rejects_an_uneven_budget(tmp_path: Path) -> None:
    pool_dir = _write_fixture(tmp_path)

    with pytest.raises(SelectionError, match="split evenly"):
        freeze_score_ranked_rankings(
            pool_rows_path=pool_dir / "pool_windows.jsonl",
            pool_report_path=pool_dir / "pool_report.json",
            output_dir=tmp_path / "score",
            budgets=(4,),
        )


def _write_fixture(tmp_path: Path) -> Path:
    pool_dir = tmp_path / "pool"
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
                "sample_tokens": [f"sample-{index}-{frame}" for frame in range(5)],
                "timestamps_s": [frame * 0.5 for frame in range(5)],
                "frame_refs": [f"frame-{index}-{frame}" for frame in range(5)],
                "pool_index": index,
                "base_probabilities": [
                    0.10 + (index % 7) * 0.09,
                    0.90 - (index % 5) * 0.08,
                    0.50 + ((index * 3) % 11) * 0.03,
                ],
            }
        )
    _write_jsonl(pool_dir / "pool_windows.jsonl", rows)
    (pool_dir / "pool_report.json").write_text(
        json.dumps(
            {
                "scenario_ids": list(SCENARIOS),
                "base": {"thresholds": {scenario: 0.5 for scenario in SCENARIOS}},
            }
        ),
        encoding="utf-8",
    )
    return pool_dir


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
