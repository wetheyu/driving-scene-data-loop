"""Focused tests for the minimal PyTorch GRU protocol."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

torch = pytest.importorskip("torch")

from driving_scene_data_loop.gru_baseline import (  # noqa: E402
    GlobalGRU,
    GruBaselineError,
    SeedRun,
    best_f1_threshold,
    build_sequence_features,
    calculate_ap_metrics,
    calculate_pos_weight,
    masked_bce_with_logits,
    train_gru_baselines,
)
from driving_scene_data_loop.lr_baselines import BaselineWindowData  # noqa: E402


def test_forward_masked_loss_gradient_and_checkpoint(tmp_path: Path) -> None:
    model = GlobalGRU(class_count=3)
    inputs = torch.randn(4, 5, 384)
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    mask = torch.tensor(
        [[True, True, False], [True, True, True], [True, False, True], [True, True, True]]
    )
    logits = model(inputs)
    assert tuple(logits.shape) == (4, 3)

    loss = masked_bce_with_logits(logits, targets, mask, torch.ones(3))
    loss.backward()  # type: ignore[no-untyped-call,unused-ignore]
    assert any(
        parameter.grad is not None and bool((parameter.grad != 0).any())
        for parameter in model.parameters()
    )

    model.eval()
    with torch.inference_mode():
        expected = model(inputs)
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded = GlobalGRU(class_count=3)
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    reloaded.eval()
    with torch.inference_mode():
        assert torch.equal(expected, reloaded(inputs))


def test_masked_bce_ignores_masked_target() -> None:
    logits = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[True, False]])
    first = masked_bce_with_logits(
        logits,
        torch.tensor([[1.0, -1.0]]),
        mask,
        torch.ones(2),
    )
    second = masked_bce_with_logits(
        logits,
        torch.tensor([[1.0, 999.0]]),
        mask,
        torch.ones(2),
    )
    assert first.item() == pytest.approx(np.log(2.0))
    assert first.item() == second.item()


def test_sequence_weight_ap_threshold_and_reverse_contract() -> None:
    frame_features = np.zeros((6, 384), dtype=np.float32)
    frame_features[:, 0] = np.arange(6, dtype=np.float32)
    indices = np.asarray([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=np.int64)
    sequences = build_sequence_features(frame_features, indices)
    assert sequences[:, :, 0].tolist() == [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]
    assert np.flip(sequences, axis=1)[:, :, 0].tolist() == [
        [4, 3, 2, 1, 0],
        [5, 4, 3, 2, 1],
    ]

    labels = np.asarray([[1], [0], [1], [0]], dtype=np.int8)
    mask = np.asarray([[True], [True], [True], [False]], dtype=np.bool_)
    assert calculate_pos_weight(labels, mask).tolist() == [0.5]

    ap_labels = np.asarray([[1], [0], [1], [0]], dtype=np.int8)
    ap_mask = np.ones((4, 1), dtype=np.bool_)
    scores = np.asarray([[0.9], [0.8], [0.7], [0.1]], dtype=np.float64)
    metrics = calculate_ap_metrics(ap_labels, ap_mask, scores)
    assert metrics.average_precision[0] == pytest.approx(5.0 / 6.0)

    threshold, f1 = best_f1_threshold(
        np.asarray([1, 0, 1, 0], dtype=np.int8),
        np.asarray([0.9, 0.8, 0.7, 0.1], dtype=np.float64),
    )
    assert threshold == pytest.approx(0.7)
    assert f1 == pytest.approx(0.8)


def test_feedback_training_keeps_l0_pos_weight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = np.asarray(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.int8,
    )
    data = BaselineWindowData(
        scenario_ids=("near", "hold", "corridor"),
        window_ids=("l0-a", "l0-b", "dev-a", "dev-b", "selected-u"),
        partitions=("l0", "l0", "development", "development", "feedback"),
        frame_feature_indices=np.tile(np.arange(5, dtype=np.int64), (5, 1)),
        labels=labels,
        loss_mask=np.ones_like(labels, dtype=np.bool_),
    )
    observed_weights: list[list[float]] = []

    def fake_train_one_seed(**kwargs: object) -> SeedRun:
        pos_weight = kwargs["pos_weight"]
        train_sequences = kwargs["train_sequences"]
        assert isinstance(pos_weight, np.ndarray)
        assert isinstance(train_sequences, np.ndarray)
        observed_weights.append(pos_weight.tolist())
        assert len(train_sequences) == 3
        probabilities = np.asarray(
            [[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]],
            dtype=np.float64,
        )
        return SeedRun(
            report={"checkpoint": "fixture.pt"},
            probabilities=probabilities,
            reversed_probabilities=probabilities,
        )

    monkeypatch.setattr(
        "driving_scene_data_loop.gru_baseline._train_one_seed",
        fake_train_one_seed,
    )
    report = train_gru_baselines(
        window_data=data,
        frame_features=np.zeros((5, 384), dtype=np.float32),
        output_dir=tmp_path / "run",
        run_name="fixture-feedback",
    )

    assert observed_weights == [[1.0, 1.0, 1.0]] * 3
    assert report["feedback_window_count"] == 1
    assert report["training_window_count"] == 3
    assert report["pos_weight_source"] == "l0_only"


def test_warm_start_resumes_each_seed_and_lowers_the_learning_rate(tmp_path: Path) -> None:
    window_data, features = _tiny_window_data()
    base_dir = tmp_path / "base"
    base_report = train_gru_baselines(
        window_data=window_data,
        frame_features=features,
        output_dir=base_dir,
        num_threads=1,
    )
    base_config = cast(dict[str, object], base_report["config"])
    assert base_config["initialization"] == "random"
    assert base_config["learning_rate"] == pytest.approx(1e-3)

    warm_report = train_gru_baselines(
        window_data=window_data,
        frame_features=features,
        output_dir=tmp_path / "warm",
        num_threads=1,
        run_name="warm",
        warm_start_dir=base_dir,
    )
    warm_config = cast(dict[str, object], warm_report["config"])
    assert warm_config["initialization"] == "warm_start:base"
    assert warm_config["learning_rate"] == pytest.approx(1e-4)


def test_warm_start_requires_every_seed_checkpoint(tmp_path: Path) -> None:
    window_data, features = _tiny_window_data()
    base_dir = tmp_path / "base"
    train_gru_baselines(
        window_data=window_data,
        frame_features=features,
        output_dir=base_dir,
        num_threads=1,
    )
    (base_dir / "gru_seed_29.pt").unlink()

    with pytest.raises(GruBaselineError, match="warm start checkpoint is missing"):
        train_gru_baselines(
            window_data=window_data,
            frame_features=features,
            output_dir=tmp_path / "warm",
            num_threads=1,
            warm_start_dir=base_dir,
        )


def _tiny_window_data() -> tuple[BaselineWindowData, NDArray[np.float32]]:
    rng = np.random.default_rng(3)
    features = rng.normal(size=(40, 384)).astype(np.float32)
    partitions = ["l0"] * 16 + ["development"] * 8
    labels, masks, indices = [], [], []
    for row in range(24):
        indices.append([(row * 5 + step) % 40 for step in range(5)])
        labels.append([row % 2, (row + 1) % 2, row % 2])
        masks.append([True, True, True])
    return (
        BaselineWindowData(
            scenario_ids=("a", "b", "c"),
            window_ids=tuple(f"w-{row:02d}" for row in range(24)),
            partitions=tuple(partitions),
            frame_feature_indices=np.asarray(indices, dtype=np.int64),
            labels=np.asarray(labels, dtype=np.int8),
            loss_mask=np.asarray(masks, dtype=np.bool_),
        ),
        features,
    )
