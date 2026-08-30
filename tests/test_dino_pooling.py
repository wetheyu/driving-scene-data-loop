"""Tests for the patch-token pooling used by the representation probe."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from driving_scene_data_loop.dino_features import (  # noqa: E402
    DinoFeatureError,
    max_pool_patch_tokens,
)


def test_max_pool_drops_cls_and_takes_channelwise_max() -> None:
    # The CLS row is deliberately the largest value in the tensor, so a bug that
    # failed to drop it would change the result rather than pass unnoticed.
    hidden_state = torch.tensor(
        [
            [
                [99.0, 99.0],  # CLS token, must be excluded
                [1.0, 5.0],
                [3.0, 2.0],
                [0.0, 4.0],
                [2.0, 1.0],
            ]
        ]
    )
    pooled = max_pool_patch_tokens(hidden_state, patch_size=259)  # 518/259 -> 2x2 = 4

    assert pooled.shape == (1, 2)
    assert torch.equal(pooled, torch.tensor([[3.0, 5.0]]))


def test_max_pool_rejects_the_wrong_patch_count() -> None:
    # 518/14 -> 37x37 = 1369 patches, so 5 tokens is a resolution or config drift.
    with pytest.raises(DinoFeatureError, match="expected"):
        max_pool_patch_tokens(torch.zeros((1, 5, 2)), patch_size=14)
