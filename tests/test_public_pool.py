"""Tests for label-free public-Pool inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from driving_scene_data_loop.gru_baseline import GlobalGRU  # noqa: E402
from driving_scene_data_loop.public_pool import (  # noqa: E402
    PublicPoolError,
    prepare_public_pool,
)

SCENARIOS = ("near", "hold", "corridor")


def test_public_pool_writes_only_safe_metadata_predictions_and_vectors(
    tmp_path: Path,
) -> None:
    features = np.zeros((6, 384), dtype=np.float32)
    features[:, 0] = np.arange(1, 7, dtype=np.float32)
    windows_path = tmp_path / "u_windows.jsonl"
    windows_path.write_text(
        "".join(
            json.dumps(_public_window(f"window-{index}", index)) + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )

    base_dir = tmp_path / "base"
    base_dir.mkdir()
    model = GlobalGRU(class_count=3)
    torch.save(model.state_dict(), base_dir / "gru_seed_17.pt")
    report_path = base_dir / "gru_report.json"
    report_path.write_text(
        json.dumps(
            {
                "base_seed": 17,
                "scenario_ids": list(SCENARIOS),
                "base_thresholds": {
                    scenario_id: {"threshold": 0.5} for scenario_id in SCENARIOS
                },
                "runs": {"17": {"checkpoint": "gru_seed_17.pt"}},
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "public-inference"
    result = prepare_public_pool(
        public_windows_path=windows_path,
        base_report_path=report_path,
        frame_features=features,
        feature_index={f"frame-{index}": index for index in range(6)},
        feature_model_id="test-dino",
        feature_model_revision="revision-1",
        output_dir=output_dir,
        batch_size=1,
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "pool_windows.jsonl").read_text().splitlines()
    ]
    embeddings = np.load(output_dir / "pool_embeddings.npy")
    assert result["window_count"] == 2
    assert rows[0]["pool_index"] == 0
    assert len(rows[0]["base_probabilities"]) == 3
    assert not {"labels", "loss_mask", "event_group_ids", "bindings"} & set(rows[0])
    assert embeddings.shape == (2, 768)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)


def test_public_pool_rejects_an_oracle_field(tmp_path: Path) -> None:
    row = _public_window("window-0", 0)
    row["labels"] = [1, 0, 0]
    windows_path = tmp_path / "u_windows.jsonl"
    windows_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(PublicPoolError, match="non-public fields"):
        prepare_public_pool(
            public_windows_path=windows_path,
            base_report_path=tmp_path / "unused.json",
            frame_features=np.zeros((5, 384), dtype=np.float32),
            feature_index={},
            feature_model_id="test-dino",
            feature_model_revision="revision-1",
            output_dir=tmp_path / "output",
        )


def _public_window(window_id: str, start: int) -> dict[str, object]:
    return {
        "window_id": window_id,
        "scene_token": "scene-token",
        "scene_name": "scene-name",
        "log_token": "log-token",
        "partition": "u",
        "start_frame_index": start,
        "end_frame_index": start + 4,
        "sample_tokens": [f"sample-{index}" for index in range(start, start + 5)],
        "timestamps_s": [index * 0.5 for index in range(start, start + 5)],
        "frame_refs": [f"frame-{index}" for index in range(start, start + 5)],
    }
