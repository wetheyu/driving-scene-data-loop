"""Tests for leak-free LastFrame-LR and Mean5-LR baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from driving_scene_data_loop.lr_baselines import (
    BaselineWindowData,
    LrBaselineError,
    build_window_representations,
    load_baseline_windows,
    train_lr_baselines,
    validate_formal_feature_manifest,
)


def test_load_baseline_windows_uses_only_l0_and_development(tmp_path: Path) -> None:
    frame_index = tmp_path / "frame_index.jsonl"
    frame_index.write_text(
        "".join(
            json.dumps(
                {
                    "feature_index": index,
                    "media_ref": f"samples/CAM_FRONT/frame-{index}.jpg",
                }
            )
            + "\n"
            for index in range(7)
        ),
        encoding="utf-8",
    )
    windows = tmp_path / "windows.jsonl"
    rows = [
        _window("l0", "l0-window", 0, [1, 0], [True, True]),
        _window("development", "dev-window", 1, [None, 1], [False, True]),
        _window("u", "u-window", 2, [1, 1], [True, True]),
        _window("frozen_test", "test-window", 2, [1, 1], [True, True]),
    ]
    windows.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = load_baseline_windows(windows, frame_index)

    assert result.window_ids == ("l0-window", "dev-window")
    assert result.partitions == ("l0", "development")
    assert result.labels.tolist() == [[1, 0], [-1, 1]]
    assert result.loss_mask.tolist() == [[True, True], [False, True]]


def test_representations_distinguish_last_frame_from_order_free_mean() -> None:
    features = np.zeros((7, 384), dtype=np.float32)
    features[:, 0] = np.arange(7, dtype=np.float32)
    indices = np.asarray([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]], dtype=np.int64)

    result = build_window_representations(features, indices)

    assert result["last_frame_lr"][:, 0].tolist() == [4.0, 6.0]
    assert result["mean5_lr"][:, 0].tolist() == [2.0, 4.0]


def test_formal_manifest_rejects_smoke_cache(tmp_path: Path) -> None:
    manifest = tmp_path / "feature_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model_id": "facebook/dinov2-small",
                "model_revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
                "feature_dimension": 384,
                "dtype": "float32",
                "frame_count": 58,
                "limited_smoke": True,
            }
        ),
        encoding="utf-8",
    )

    try:
        validate_formal_feature_manifest(manifest)
    except LrBaselineError as error:
        assert "frame_count" in str(error)
    else:
        raise AssertionError("smoke cache must not pass the formal training gate")


def test_training_fits_scaler_on_l0_and_writes_development_predictions(
    tmp_path: Path,
) -> None:
    features = np.zeros((16, 384), dtype=np.float32)
    features[:, 0] = np.arange(16, dtype=np.float32)
    features[:, 1] = np.asarray([0, 1] * 8, dtype=np.float32)
    frame_indices = np.asarray(
        [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
            [4, 5, 6, 7, 8],
            [5, 6, 7, 8, 9],
            [6, 7, 8, 9, 10],
            [7, 8, 9, 10, 11],
            [10, 11, 12, 13, 14],
            [11, 12, 13, 14, 15],
        ],
        dtype=np.int64,
    )
    data = BaselineWindowData(
        scenario_ids=("scenario_a", "scenario_b"),
        window_ids=tuple(f"window-{index}" for index in range(8)),
        partitions=(
            "l0",
            "l0",
            "l0",
            "l0",
            "development",
            "development",
            "development",
            "development",
        ),
        frame_feature_indices=frame_indices,
        labels=np.asarray(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ],
            dtype=np.int8,
        ),
        loss_mask=np.ones((8, 2), dtype=np.bool_),
    )
    output = tmp_path / "lr-run"

    report = train_lr_baselines(
        window_data=data,
        frame_features=features,
        output_dir=output,
    )

    assert report["l0_window_count"] == 4
    assert report["development_window_count"] == 4
    parameters = np.load(output / "last_frame_lr_parameters.npz")
    expected_l0_last_mean = features[frame_indices[:4, -1]].mean(axis=0)
    assert np.allclose(parameters["scaler_mean"], expected_l0_last_mean)
    assert len((output / "last_frame_lr_predictions.jsonl").read_text().splitlines()) == 4
    assert (output / "mean5_lr_predictions.jsonl").is_file()


def _window(
    partition: str,
    window_id: str,
    start: int,
    labels: list[int | None],
    loss_mask: list[bool],
) -> dict[str, object]:
    return {
        "partition": partition,
        "window_id": window_id,
        "scenario_ids": ["scenario_a", "scenario_b"],
        "frame_refs": [f"samples/CAM_FRONT/frame-{index}.jpg" for index in range(start, start + 5)],
        "labels": labels,
        "loss_mask": loss_mask,
    }
