"""Tests for the patch-max-pool spatial feature probe."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from driving_scene_data_loop.dino_features import DinoFeatureError  # noqa: E402
from driving_scene_data_loop.dino_spatial_features import max_pool_patch_tokens  # noqa: E402


def test_max_pool_drops_cls_and_takes_channelwise_max() -> None:
    # 1 CLS token + 4 patch tokens, hidden size 2. The CLS row is deliberately
    # the largest value in the tensor, so a bug that fails to drop it would be
    # caught by the assertion below rather than silently passing.
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
    pooled = max_pool_patch_tokens(hidden_state, patch_size=259)  # 518/259=2 -> 2x2=4

    assert pooled.shape == (1, 2)
    assert torch.equal(pooled, torch.tensor([[3.0, 5.0]]))


def test_max_pool_rejects_the_wrong_patch_count() -> None:
    hidden_state = torch.zeros((1, 5, 2))  # 4 patches expected, only 4 present is fine
    with pytest.raises(DinoFeatureError, match="expected"):
        max_pool_patch_tokens(hidden_state, patch_size=14)  # 518/14=37 -> needs 1370 tokens
