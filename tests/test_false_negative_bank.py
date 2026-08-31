"""Tests for the event-deduplicated Development false-negative bank."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from driving_scene_data_loop.false_negative_bank import (
    build_false_negative_bank,
    build_selection_embedding,
)

SOURCE_SCENARIOS = ("near", "hold", "corridor")
BASE_SCENARIOS = SOURCE_SCENARIOS


def test_selection_embedding_contains_mean_and_change() -> None:
    features = np.zeros((5, 384), dtype=np.float32)
    features[:, 0] = np.arange(1, 6, dtype=np.float32)

    result = build_selection_embedding(features, (0, 1, 2, 3, 4))

    expected = np.zeros(768, dtype=np.float32)
    expected[0] = 3.0
    expected[384] = 4.0
    expected /= np.linalg.norm(expected)
    assert result.shape == (768,)
    assert result.dtype == np.float32
    assert np.allclose(result, expected)
    assert np.linalg.norm(result) == pytest.approx(1.0)


def test_bank_maps_class_names_deduplicates_events_and_ignores_u(tmp_path: Path) -> None:
    windows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []

    # One near event has two missed views and one correctly detected view.
    _append_example(windows, predictions, "near-a-high", "near", "near-event-0", 0.2)
    _append_example(windows, predictions, "near-a-low", "near", "near-event-0", 0.1)
    _append_example(windows, predictions, "near-a-aaa", "near", "near-event-0", 0.1)
    _append_example(windows, predictions, "near-a-correct", "near", "near-event-0", 0.8)
    for index in range(1, 5):
        _append_example(
            windows,
            predictions,
            f"near-{index}",
            "near",
            f"near-event-{index}",
            0.2,
        )
    for index in range(5):
        _append_example(
            windows,
            predictions,
            f"hold-{index}",
            "hold",
            f"hold-event-{index}",
            0.2,
        )
    for index in range(5):
        _append_example(
            windows,
            predictions,
            f"corridor-{index}",
            "corridor",
            f"corridor-event-{index}",
            0.2,
        )
    # Equality is classified positive by `p >= threshold`, so this is not an FN.
    _append_example(windows, predictions, "near-equal", "near", "near-equal", 0.5)
    windows.append(_window("u-private-label", "u", "near", "forbidden-u-event"))

    windows_path = tmp_path / "windows.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "gru_report.json"
    windows_path.write_text(
        "".join(json.dumps(row) + "\n" for row in windows), encoding="utf-8"
    )
    predictions_path.write_text(
        "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
    )
    report_path.write_text(
        json.dumps(
            {
                "base_seed": 17,
                "evaluation_partition": "development",
                "scenario_ids": list(BASE_SCENARIOS),
                # This preserves the old Gate-B diagnostic while proving that
                # the weak third class still enters the new FN bank.
                "retained_classes": ["near", "corridor"],
                "base_thresholds": {
                    scenario_id: {"threshold": 0.5, "f1": 0.4}
                    for scenario_id in BASE_SCENARIOS
                },
                "runs": {"17": {"checkpoint": "gru_seed_17.pt"}},
            }
        ),
        encoding="utf-8",
    )
    features = np.zeros((5, 384), dtype=np.float32)
    features[:, 0] = np.arange(1, 6, dtype=np.float32)
    output_dir = tmp_path / "fn-bank"

    result = build_false_negative_bank(
        windows_path=windows_path,
        predictions_path=predictions_path,
        base_report_path=report_path,
        frame_features=features,
        feature_index={f"frame-{index}": index for index in range(5)},
        feature_model_id="test-dino",
        feature_model_revision="revision-1",
        output_dir=output_dir,
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "fn_bank.jsonl").read_text().splitlines()
    ]
    embeddings = np.load(output_dir / "fn_embeddings.npy")
    near_zero = next(row for row in rows if row["event_group_id"] == "near-event-0")
    assert result["source_window_count"] == len(predictions)
    assert result["bank_event_count"] == 15
    assert result["bank_unique_window_count"] == 15
    assert near_zero["window_id"] == "near-a-aaa"
    assert near_zero["base_probability"] == pytest.approx(0.1)
    assert result["classes"]["near"] == {
        "threshold": 0.5,
        "raw_false_negative_windows": 7,
        "positive_event_groups": 6,
        "event_groups_with_any_false_negative": 5,
        "event_groups_with_all_positive_windows_false_negative": 4,
        "unique_representative_windows": 5,
        "minimum_event_gate": 5,
        "passed_minimum_event_gate": True,
    }
    assert result["classes"]["hold"]["event_groups_with_any_false_negative"] == 5
    assert embeddings.shape == (15, 768)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)
    assert all("labels" not in row and "bindings" not in row for row in rows)
    assert all(row["window_id"] != "u-private-label" for row in rows)


def _append_example(
    windows: list[dict[str, object]],
    predictions: list[dict[str, object]],
    window_id: str,
    scenario_id: str,
    event_group_id: str,
    probability: float,
) -> None:
    windows.append(_window(window_id, "development", scenario_id, event_group_id))
    scenario_index = BASE_SCENARIOS.index(scenario_id)
    label = [0, 0, 0]
    label[scenario_index] = 1
    probabilities = [0.1, 0.1, 0.1]
    probabilities[scenario_index] = probability
    predictions.append(
        {
            "window_id": window_id,
            "partition": "development",
            "probabilities": probabilities,
            "reversed_probabilities": [0.99, 0.99, 0.99],
            "labels": label,
            "loss_mask": [True, True, True],
        }
    )


def _window(
    window_id: str,
    partition: str,
    scenario_id: str,
    event_group_id: str,
) -> dict[str, object]:
    scenario_index = SOURCE_SCENARIOS.index(scenario_id)
    labels = [0, 0, 0]
    labels[scenario_index] = 1
    event_groups: list[list[str]] = [[], [], []]
    event_groups[scenario_index] = [event_group_id]
    return {
        "window_id": window_id,
        "partition": partition,
        "scenario_ids": list(SOURCE_SCENARIOS),
        "labels": labels,
        "loss_mask": [True, True, True],
        "event_group_ids": event_groups,
        "frame_refs": [f"frame-{index}" for index in range(5)],
        "scene_token": "scene-token",
        "scene_name": "scene-name",
        "log_token": "log-token",
        "start_frame_index": 0,
        "end_frame_index": 4,
    }
