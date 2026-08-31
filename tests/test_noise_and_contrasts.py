"""Tests for the D2 noise injection and the Development contrast helper."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from driving_scene_data_loop.feedback_retraining import get_oracle_arm
from driving_scene_data_loop.prediction_scoring import (
    PredictionScoringError,
    development_paired_contrasts,
)

SCENARIOS = ["near", "hold", "corridor"]


def _write_reveal(tmp_path: Path, count: int = 40) -> Path:
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    rows = []
    for rank in range(1, count + 1):
        rows.append(
            {
                "method": "disagreement_v010",
                "rank": rank,
                "window_id": f"w{rank}",
                "scene_token": "s",
                "log_token": "l",
                "start_frame_index": rank,
                "labels": [1, None, 0],
                "loss_mask": [True, False, True],
                "label_states": ["positive", "ignore", "negative"],
                "event_group_ids": [[], [], []],
            }
        )
    with (oracle / "revealed_labels.jsonl").open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    (oracle / "label_profile.json").write_text(
        json.dumps(
            {
                "artifact": "oracle_revealed_labels",
                "scenario_ids": SCENARIOS,
                "label_order": "order",
            }
        ),
        encoding="utf-8",
    )
    return oracle


def _run_injection(oracle: Path, out: Path, rate: int) -> dict[str, int]:
    subprocess.run(
        [
            sys.executable,
            "scripts/build_noise_injected_labels.py",
            "--oracle-dir", str(oracle),
            "--budget", "40",
            "--rate-percent", str(rate),
            "--output-dir", str(out),
        ],
        check=True,
        capture_output=True,
    )
    profile = json.loads((out / "label_profile.json").read_text(encoding="utf-8"))
    assert isinstance(profile, dict)
    return {k: v for k, v in profile.items() if isinstance(v, int)} | {
        "artifact_is_noise": int(profile["artifact"] == "oracle_noise_injected")
    }


def test_noise_injection_is_deterministic_and_leaves_masks_alone(tmp_path: Path) -> None:
    oracle = _write_reveal(tmp_path)
    profile_a = _run_injection(oracle, tmp_path / "a", 30)
    _run_injection(oracle, tmp_path / "b", 30)

    rows_a = (tmp_path / "a" / "revealed_labels.jsonl").read_text()
    rows_b = (tmp_path / "b" / "revealed_labels.jsonl").read_text()
    assert rows_a == rows_b
    assert profile_a["artifact_is_noise"] == 1
    assert profile_a["decided_labels"] == 80
    assert 0 < profile_a["flipped_labels"] < 80

    for line in rows_a.splitlines():
        row = json.loads(line)
        assert row["loss_mask"] == [True, False, True]
        assert row["labels"][1] is None and row["label_states"][1] == "ignore"
        assert row["label_states"][0] in ("positive", "negative")
        assert row["labels"][0] == (1 if row["label_states"][0] == "positive" else 0)


def test_noise_arms_declare_the_injected_label_source() -> None:
    arm = get_oracle_arm("v010-disagreement-900-noise20")
    assert arm.label_source == "oracle_noise"
    assert (arm.method, arm.budget) == ("disagreement_v010", 900)


def _report(tmp_path: Path, name: str, values: dict[str, list[float]]) -> Path:
    runs = {}
    for index, seed in enumerate(("17", "29", "43", "61")):
        runs[seed] = {
            "normal_order": {
                "classes": {
                    scenario: {"average_precision": values[scenario][index]}
                    for scenario in SCENARIOS
                }
            }
        }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"scenario_ids": SCENARIOS, "runs": runs}), encoding="utf-8")
    return path


def test_paired_contrasts_match_hand_computation(tmp_path: Path) -> None:
    from typing import Any, cast

    base = {s: [0.10, 0.12, 0.11, 0.13] for s in SCENARIOS}
    deltas = [0.02, 0.03, 0.01, 0.04]
    lift = {s: [v + d for v, d in zip(base[s], deltas, strict=True)] for s in SCENARIOS}
    document = development_paired_contrasts(
        {"arm": _report(tmp_path, "arm", lift), "base": _report(tmp_path, "base", base)},
        (("arm", "base"),),
    )

    rows = {
        cast(dict[str, Any], row)["metric"]: cast(dict[str, Any], row)
        for row in cast(list[object], document["rows"])
    }
    assert document["seed_count"] == 4
    for metric in (*SCENARIOS, "macro"):
        # mean 0.025; sd of diffs 0.012910; stderr = sd/2 = 0.006455
        assert rows[metric]["effect"] == pytest.approx(0.025)
        assert rows[metric]["stderr"] == pytest.approx(0.006455, rel=1e-3)
        assert rows[metric]["sigma"] == pytest.approx(3.873, rel=1e-3)


def test_paired_contrasts_reject_a_scenario_order_mismatch(tmp_path: Path) -> None:
    good = _report(tmp_path, "good", {s: [0.1, 0.1, 0.1, 0.1] for s in SCENARIOS})
    swapped = tmp_path / "swapped.json"
    payload = json.loads(good.read_text())
    payload["scenario_ids"] = list(reversed(SCENARIOS))
    swapped.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PredictionScoringError, match="scenario order"):
        development_paired_contrasts(
            {"a": good, "b": swapped}, (("a", "b"),)
        )
