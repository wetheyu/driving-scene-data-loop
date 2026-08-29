"""Extract a patch-max-pooled DINOv2 feature: a representation-ceiling probe.

Section 8/9 of `docs/FINDINGS.md` found that a Development-informed selector
bought 2 to 5 times more positive labels than Random at N=300 without a
corresponding Development Macro-AP gain. That result weakened the "budget is
too small" hypothesis and strengthened "the representation is the ceiling":
`extract_feature_cache` in `dino_features.py` pools DINOv2's CLS token, which
is a global average over the whole frame, and a pedestrian at 15-20 m spans
only 1-2 of the 37x37 patch tokens the encoder actually computes.

This module answers one narrow question first, before trying an actual spatial
model: does simply keeping the encoder's own patch tokens and taking a
per-channel maximum over them, instead of discarding them for the CLS token,
recover signal a global average pooling washes out? Max pooling is the
cheapest way to test that a channel fired strongly *somewhere* in the frame,
without introducing spatial coordinates, a new model architecture, or a wider
GRU input. Everything downstream of the frame-feature cache (`lr_baselines.py`,
`gru_baseline.py`, `false_negative_bank.py`, `public_pool.py`, `selection.py`)
consumes `[frame_count, 384]` float32 rows through `frame_index.jsonl`, so
this cache is a drop-in replacement and needs no change to that code.

The model identity, letterboxing, and frame-reference collection are shared
with `dino_features.py`; only the pooled model output differs.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from driving_scene_data_loop.dino_features import (
    FEATURE_DIMENSION,
    INPUT_SIZE,
    LETTERBOX_FILL,
    MODEL_ID,
    MODEL_REVISION,
    DinoFeatureError,
    collect_unique_frame_refs,
    letterbox_image,
)


def max_pool_patch_tokens(
    last_hidden_state: Any,
    patch_size: int,
) -> Any:
    """Drop the CLS token and take a per-channel max over the remaining patches.

    `last_hidden_state` is `[batch, 1 + num_patches, hidden]`; this rejects any
    other patch-token count, so a config or resolution drift is caught rather
    than silently pooling the wrong number of tokens. Returns `[batch, hidden]`.
    """

    import torch  # type: ignore[import-not-found,unused-ignore]

    hidden_state = torch.as_tensor(last_hidden_state)
    if hidden_state.ndim != 3:
        raise DinoFeatureError("last_hidden_state must have shape [batch,tokens,hidden]")
    if INPUT_SIZE % patch_size != 0:
        raise DinoFeatureError("INPUT_SIZE must be an exact multiple of patch_size")
    expected_patches = (INPUT_SIZE // patch_size) ** 2
    if hidden_state.shape[1] != expected_patches + 1:
        raise DinoFeatureError(
            f"expected {expected_patches + 1} tokens (1 CLS + {expected_patches} "
            f"patches), got {hidden_state.shape[1]}"
        )
    patch_tokens = hidden_state[:, 1:, :]
    return patch_tokens.max(dim=1).values


def extract_patch_max_feature_cache(
    *,
    windows_path: Path,
    media_root: Path,
    output_dir: Path,
    cache_dir: Path,
    batch_size: int,
    num_threads: int,
    limit: int | None = None,
    local_files_only: bool = False,
) -> dict[str, object]:
    """Run the pinned model once per unique frame; pool patch tokens, not CLS."""

    if output_dir.exists():
        raise DinoFeatureError("output directory must not already exist")
    if batch_size <= 0 or num_threads <= 0:
        raise DinoFeatureError("batch_size and num_threads must be positive")
    if limit is not None and limit <= 0:
        raise DinoFeatureError("limit must be positive")

    frame_refs = collect_unique_frame_refs(windows_path)
    if limit is not None:
        frame_refs = frame_refs[:limit]
    missing = [value for value in frame_refs if not (media_root / value).is_file()]
    if missing:
        raise DinoFeatureError(
            f"missing {len(missing)} images below media_root; first={missing[0]}"
        )

    try:
        import torch  # type: ignore[import-not-found,unused-ignore]
        from transformers import (  # type: ignore[import-not-found,unused-ignore]
            AutoImageProcessor,
            AutoModel,
        )
    except ImportError as error:
        raise DinoFeatureError(
            "install the project ml extra before extracting DINO features"
        ) from error

    torch.set_num_threads(num_threads)
    processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call,unused-ignore]
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        use_fast=False,
    )
    model = AutoModel.from_pretrained(  # type: ignore[no-untyped-call,unused-ignore]
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        use_safetensors=True,
    )
    model.eval()
    patch_size = int(model.config.patch_size)

    output_dir.mkdir(parents=True)
    features = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
        output_dir / "frame_features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(frame_refs), FEATURE_DIMENSION),
    )
    started = time.perf_counter()
    for start, batch_refs in _batches(frame_refs, batch_size):
        images: list[Image.Image] = []
        for media_ref in batch_refs:
            try:
                with Image.open(media_root / media_ref) as source:
                    images.append(letterbox_image(source))
            except OSError as error:
                raise DinoFeatureError(f"cannot read image: {media_ref}") from error

        inputs = processor(
            images=images,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        pixel_values = inputs["pixel_values"]
        if tuple(pixel_values.shape[1:]) != (3, INPUT_SIZE, INPUT_SIZE):
            raise DinoFeatureError("processor returned an unexpected pixel shape")
        with torch.inference_mode():
            output = model(pixel_values=pixel_values)
        pooled = max_pool_patch_tokens(output.last_hidden_state, patch_size)
        batch_features = np.asarray(pooled.detach().cpu(), dtype=np.float32)
        expected_shape = (len(batch_refs), FEATURE_DIMENSION)
        if batch_features.shape != expected_shape or not np.isfinite(batch_features).all():
            raise DinoFeatureError("DINO returned invalid patch-pooled features")
        features[start : start + len(batch_refs)] = batch_features
    features.flush()
    elapsed_seconds = time.perf_counter() - started

    with (output_dir / "frame_index.jsonl").open("w", encoding="utf-8") as destination:
        for index, media_ref in enumerate(frame_refs):
            destination.write(
                json.dumps(
                    {"feature_index": index, "media_ref": media_ref},
                    separators=(",", ":"),
                )
                + "\n"
            )

    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_output": "patch_token_max_pool",
        "patch_size": patch_size,
        "patches_per_side": INPUT_SIZE // patch_size,
        "input": {
            "camera": "CAM_FRONT",
            "letterbox_size": INPUT_SIZE,
            "letterbox_fill_rgb": list(LETTERBOX_FILL),
            "processor_resize": False,
            "processor_center_crop": False,
            "processor_normalization": True,
            "processor_use_fast": False,
        },
        "frame_count": len(frame_refs),
        "feature_dimension": FEATURE_DIMENSION,
        "dtype": "float32",
        "batch_size": batch_size,
        "num_threads": num_threads,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "frames_per_second": round(len(frame_refs) / elapsed_seconds, 6),
        "limited_smoke": limit is not None,
    }
    (output_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _batches(values: tuple[str, ...], batch_size: int) -> Iterator[tuple[int, tuple[str, ...]]]:
    for start in range(0, len(values), batch_size):
        yield start, values[start : start + batch_size]
