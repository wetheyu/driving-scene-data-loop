"""Tests for scoring frozen predictions, including a hand-checkable AP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from driving_scene_data_loop.prediction_scoring import (
    PredictionScoringError,
    load_partition_targets,
    score_prediction_file,
    score_runs,
    seed_paired_contrast,
)

SCENARIOS = ("near", "hold", "corridor")


def test_ap_matches_a_hand_computation(tmp_path: Path) -> None:
    windows, predictions = _write_fixture(tmp_path)
    scenario_ids, targets = load_partition_targets(windows, "development")

    result = score_prediction_file(predictions, scenario_ids, targets)

    # Class "near", by hand. Valid rows and scores (w2 is masked out):
    #   w0 label=1 p=0.9, w1 label=0 p=0.8, w3 label=1 p=0.6, w4 label=0 p=0.2
    # Descending: w0(1) -> precision 1/1; w1(0); w3(1) -> precision 2/3.
    # AP = mean over positives = (1 + 2/3) / 2 = 5/6.
    classes = cast(dict[str, Any], result["classes"])
    assert classes["near"]["average_precision"] == pytest.approx(5 / 6)
    assert classes["near"]["valid"] == 4
    assert classes["near"]["positives"] == 2


def test_masked_windows_do_not_enter_the_score(tmp_path: Path) -> None:
    windows, predictions = _write_fixture(tmp_path)
    scenario_ids, targets = load_partition_targets(windows, "development")

    baseline = score_prediction_file(predictions, scenario_ids, targets)
    # Flip the masked window's near label; a leak through the mask would move AP.
    rows = [json.loads(line) for line in windows.read_text().splitlines()]
    rows[2]["labels"][0] = 0
    windows.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _, targets = load_partition_targets(windows, "development")

    flipped = score_prediction_file(predictions, scenario_ids, targets)
    assert cast(dict[str, Any], flipped["classes"])["near"] == cast(
        dict[str, Any], baseline["classes"]
    )["near"]


def test_missing_and_foreign_predictions_are_rejected(tmp_path: Path) -> None:
    windows, predictions = _write_fixture(tmp_path)
    scenario_ids, targets = load_partition_targets(windows, "development")

    lines = predictions.read_text().splitlines()
    predictions.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(PredictionScoringError, match="no prediction"):
        score_prediction_file(predictions, scenario_ids, targets)

    foreign = dict(json.loads(lines[0]))
    foreign["window_id"] = "scene-x:9999"
    predictions.write_text("\n".join(lines + [json.dumps(foreign)]) + "\n")
    with pytest.raises(PredictionScoringError, match="outside the partition"):
        score_prediction_file(predictions, scenario_ids, targets)


def test_score_runs_pairs_by_shared_seed(tmp_path: Path) -> None:
    windows, _ = _write_fixture(tmp_path)
    for name, bump in (("a", 0.0), ("b", 0.05)):
        run_dir = tmp_path / name
        run_dir.mkdir()
        for seed in ("17", "29", "43"):
            _write_predictions(run_dir / f"predictions_seed_{seed}.jsonl", bump)
    report = score_runs(
        windows_path=windows,
        partition="development",
        runs={"a": tmp_path / "a", "b": tmp_path / "b"},
        output_path=tmp_path / "scores.json",
    )
    contrast = cast(dict[str, Any], report["seed_paired_contrasts"])["a - b"]["macro"]
    assert contrast["seeds"] == 3
    # Identical prediction files per seed make every per-seed difference equal,
    # so the paired mean is exact and the spread collapses.
    assert contrast["stderr"] == pytest.approx(0.0)


def test_seed_paired_contrast_needs_shared_seeds() -> None:
    with pytest.raises(PredictionScoringError, match="three shared seeds"):
        seed_paired_contrast({"17": 0.1, "29": 0.2}, {"17": 0.1, "29": 0.2})


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    windows = tmp_path / "windows.jsonl"
    rows = []
    labels = [(1, 1, 0), (0, 0, 1), (1, 1, 1), (1, 0, 0), (0, 1, 1)]
    masks = [(True,) * 3, (True,) * 3, (False, True, True), (True,) * 3, (True,) * 3]
    for i, (lab, mask) in enumerate(zip(labels, masks, strict=True)):
        rows.append(
            {
                "window_id": f"scene-a:{i:04d}",
                "partition": "development",
                "scenario_ids": list(SCENARIOS),
                "labels": [v if m else None for v, m in zip(lab, mask, strict=True)],
                "loss_mask": list(mask),
            }
        )
    # A frozen_test row that development scoring must never touch.
    rows.append(
        {
            "window_id": "scene-t:0000",
            "partition": "frozen_test",
            "scenario_ids": list(SCENARIOS),
            "labels": [1, 1, 1],
            "loss_mask": [True, True, True],
        }
    )
    windows.write_text("".join(json.dumps(r) + "\n" for r in rows))
    predictions = tmp_path / "predictions_seed_17.jsonl"
    _write_predictions(predictions, 0.0)
    return windows, predictions


def _write_predictions(path: Path, bump: float) -> None:
    probs = [0.9, 0.8, 0.5, 0.6, 0.2]
    with path.open("w", encoding="utf-8") as f:
        for i, p in enumerate(probs):
            f.write(
                json.dumps(
                    {
                        "window_id": f"scene-a:{i:04d}",
                        "partition": "development",
                        "probabilities": [min(p + bump, 0.99), 0.5, 1.0 - p],
                    }
                )
                + "\n"
            )
