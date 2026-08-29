"""Tests for frozen label-free selection rankings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from driving_scene_data_loop.selection import (
    RANDOM_SEED,
    SelectionError,
    boundary_margin,
    freeze_random_variance_rankings,
    freeze_selection_rankings,
)

SCENARIOS = ("near", "hold", "corridor")


def test_boundary_margin_uses_the_frozen_threshold() -> None:
    assert boundary_margin(0.8, 0.8) == pytest.approx(0.0)
    assert boundary_margin(0.75, 0.8) < boundary_margin(0.5, 0.8)


def test_rankings_are_nested_balanced_unique_and_temporally_separated(
    tmp_path: Path,
) -> None:
    pool_dir, fn_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "selection"

    report = freeze_selection_rankings(
        pool_rows_path=pool_dir / "pool_windows.jsonl",
        pool_embeddings_path=pool_dir / "pool_embeddings.npy",
        pool_report_path=pool_dir / "pool_report.json",
        fn_rows_path=fn_dir / "fn_bank.jsonl",
        fn_embeddings_path=fn_dir / "fn_embeddings.npy",
        output_dir=output_dir,
        budgets=(3, 6),
        shortlist_size=18,
        cluster_count=2,
    )

    forbidden = {"labels", "loss_mask", "event_group_ids", "bindings"}
    for method in ("random", "mining"):
        rows = _read_jsonl(output_dir / f"{method}_ranked.jsonl")
        assert len(rows) == 6
        assert len({row["window_id"] for row in rows}) == 6
        assert not set().union(*(set(row) for row in rows)) & forbidden
        _assert_temporal_separation(rows)
        assert [row["rank"] for row in rows] == list(range(1, 7))

    mining = _read_jsonl(output_dir / "mining_ranked.jsonl")
    assert [row["query_scenario_id"] for row in mining[:3]] == list(SCENARIOS)
    assert {scenario: 2 for scenario in SCENARIOS} == {
        scenario: sum(row["query_scenario_id"] == scenario for row in mining)
        for scenario in SCENARIOS
    }
    assert all("cluster_id" in row for row in mining)
    assert report["prefixes"]["mining"]["3"]["query_class_counts"] == {
        scenario: 1 for scenario in SCENARIOS
    }
    assert report["prefixes"]["mining"]["6"]["cluster_profile"]


def test_selector_rejects_an_oracle_field(tmp_path: Path) -> None:
    pool_dir, fn_dir = _write_fixture(tmp_path)
    rows = _read_jsonl(pool_dir / "pool_windows.jsonl")
    rows[0]["labels"] = [1, 0, 0]
    _write_jsonl(pool_dir / "pool_windows.jsonl", rows)

    with pytest.raises(SelectionError, match="invalid public Pool row"):
        freeze_selection_rankings(
            pool_rows_path=pool_dir / "pool_windows.jsonl",
            pool_embeddings_path=pool_dir / "pool_embeddings.npy",
            pool_report_path=pool_dir / "pool_report.json",
            fn_rows_path=fn_dir / "fn_bank.jsonl",
            fn_embeddings_path=fn_dir / "fn_embeddings.npy",
            output_dir=tmp_path / "selection",
            budgets=(3, 6),
            shortlist_size=18,
            cluster_count=2,
        )


def test_variance_batches_are_valid_and_distinct(tmp_path: Path) -> None:
    pool_dir, fn_dir = _write_fixture(tmp_path)
    frozen_dir = tmp_path / "selection"
    freeze_selection_rankings(
        pool_rows_path=pool_dir / "pool_windows.jsonl",
        pool_embeddings_path=pool_dir / "pool_embeddings.npy",
        pool_report_path=pool_dir / "pool_report.json",
        fn_rows_path=fn_dir / "fn_bank.jsonl",
        fn_embeddings_path=fn_dir / "fn_embeddings.npy",
        output_dir=frozen_dir,
        budgets=(3, 6),
        shortlist_size=18,
        cluster_count=2,
    )
    variance_dir = tmp_path / "variance"
    report = freeze_random_variance_rankings(
        pool_rows_path=pool_dir / "pool_windows.jsonl",
        pool_report_path=pool_dir / "pool_report.json",
        output_dir=variance_dir,
        seeds=(102, 103),
        budgets=(3, 6),
    )

    forbidden = {"labels", "loss_mask", "event_group_ids", "bindings"}
    batches = {
        seed: _read_jsonl(variance_dir / f"random_seed{seed}_ranked.jsonl")
        for seed in (102, 103)
    }
    for seed, rows in batches.items():
        assert len(rows) == 6
        assert len({row["window_id"] for row in rows}) == 6
        assert not set().union(*(set(row) for row in rows)) & forbidden
        assert {row["method"] for row in rows} == {f"random_seed{seed}"}
        assert [row["rank"] for row in rows] == list(range(1, 7))
        _assert_temporal_separation(rows)
        assert report["prefixes"][str(seed)]["3"]["count"] == 3

    # A variance estimate is meaningless if the extra draws repeat the frozen one.
    frozen_ids = [row["window_id"] for row in _read_jsonl(frozen_dir / "random_ranked.jsonl")]
    ordered = [frozen_ids] + [[row["window_id"] for row in rows] for rows in batches.values()]
    assert len({tuple(ids) for ids in ordered}) == 3


def test_variance_seeds_must_exclude_the_frozen_random_seed(tmp_path: Path) -> None:
    pool_dir, _ = _write_fixture(tmp_path)

    with pytest.raises(SelectionError, match="exclude the frozen Random seed"):
        freeze_random_variance_rankings(
            pool_rows_path=pool_dir / "pool_windows.jsonl",
            pool_report_path=pool_dir / "pool_report.json",
            output_dir=tmp_path / "variance",
            seeds=(RANDOM_SEED, 102),
            budgets=(3, 6),
        )


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    pool_dir = tmp_path / "pool"
    fn_dir = tmp_path / "fn"
    pool_dir.mkdir()
    fn_dir.mkdir()
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
                    0.30 + index * 0.02,
                    0.35 + index * 0.015,
                    0.40 + index * 0.01,
                ],
            }
        )
    _write_jsonl(pool_dir / "pool_windows.jsonl", rows)
    (pool_dir / "pool_report.json").write_text(
        json.dumps(
            {
                "scenario_ids": list(SCENARIOS),
                "base": {
                    "thresholds": {scenario: 0.5 for scenario in SCENARIOS}
                },
            }
        ),
        encoding="utf-8",
    )

    rng = np.random.default_rng(7)
    pool_embeddings = rng.normal(size=(18, 768)).astype(np.float32)
    pool_embeddings /= np.linalg.norm(pool_embeddings, axis=1, keepdims=True)
    np.save(pool_dir / "pool_embeddings.npy", pool_embeddings)

    fn_rows = [
        {"bank_index": index, "scenario_id": scenario}
        for index, scenario in enumerate(SCENARIOS)
    ]
    _write_jsonl(fn_dir / "fn_bank.jsonl", fn_rows)
    np.save(fn_dir / "fn_embeddings.npy", pool_embeddings[:3])
    return pool_dir, fn_dir


def _assert_temporal_separation(rows: list[dict[str, object]]) -> None:
    starts_by_scene: dict[str, list[int]] = {}
    for row in rows:
        scene = str(row["scene_token"])
        start_value = row["start_frame_index"]
        assert isinstance(start_value, int)
        start = start_value
        assert all(abs(start - other) >= 5 for other in starts_by_scene.get(scene, []))
        starts_by_scene.setdefault(scene, []).append(start)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
