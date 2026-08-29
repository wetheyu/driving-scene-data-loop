"""Tests for DINO frame selection and frozen letterbox preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from driving_scene_data_loop.dino_features import (
    INPUT_SIZE,
    LETTERBOX_FILL,
    DinoFeatureError,
    collect_unique_frame_refs,
    letterbox_image,
)


def test_collect_unique_frame_refs_deduplicates_overlapping_windows(
    tmp_path: Path,
) -> None:
    windows = tmp_path / "windows.jsonl"
    first = [f"samples/CAM_FRONT/frame-{index}.jpg" for index in range(5)]
    second = [f"samples/CAM_FRONT/frame-{index}.jpg" for index in range(1, 6)]
    windows.write_text(
        "\n".join(
            json.dumps({"frame_refs": refs})
            for refs in (first, second)
        )
        + "\n",
        encoding="utf-8",
    )

    result = collect_unique_frame_refs(windows)

    assert result == tuple(sorted(set(first + second)))


def test_collect_unique_frame_refs_rejects_non_front_camera(tmp_path: Path) -> None:
    windows = tmp_path / "windows.jsonl"
    windows.write_text(
        json.dumps(
            {
                "frame_refs": [
                    "samples/CAM_FRONT/a.jpg",
                    "samples/CAM_FRONT/b.jpg",
                    "samples/CAM_FRONT/c.jpg",
                    "samples/CAM_FRONT/d.jpg",
                    "samples/CAM_BACK/e.jpg",
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DinoFeatureError, match="CAM_FRONT"):
        collect_unique_frame_refs(windows)


def test_letterbox_preserves_aspect_ratio_and_uses_fixed_padding() -> None:
    source = Image.new("RGB", (1600, 900), (255, 0, 0))

    result = letterbox_image(source)

    assert result.size == (INPUT_SIZE, INPUT_SIZE)
    assert result.getpixel((10, 0)) == LETTERBOX_FILL
    assert result.getpixel((10, 112)) == LETTERBOX_FILL
    assert result.getpixel((10, 113)) == (255, 0, 0)
    assert result.getpixel((10, 403)) == (255, 0, 0)
    assert result.getpixel((10, 404)) == LETTERBOX_FILL
