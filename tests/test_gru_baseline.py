"""Focused tests for the minimal PyTorch GRU protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from driving_scene_data_loop.gru_baseline import (  # noqa: E402
    GlobalGRU,
    best_f1_threshold,
    build_sequence_features,
    calculate_ap_metrics,
    calculate_pos_weight,
    masked_bce_with_logits,
)


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
