"""Extract one frozen DINOv2 feature per unique CAM_FRONT frame.

Two poolings of the same frozen encoder are supported, and a cache records
which one produced it:

- `cls` pools the `pooler_output` CLS token, a global summary of the frame.
  This produced the formal cache every result before the spatial probe used.
- `patch_max` drops the CLS token and takes a per-channel maximum over the
  encoder's own 37x37 patch tokens. It probes the representation-ceiling
  hypothesis in `docs/FINDINGS.md`: a pedestrian at 15-20 m spans only 1-2
  patches, so a global average may wash it out where a maximum would not.

Both write `[frame_count, 384]` float32, so every downstream consumer accepts
either without change. The manifest's `model_output` is what tells them apart,
and it is recorded in the reports of runs that consume a cache.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
from PIL import Image

MODEL_ID = "facebook/dinov2-small"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
INPUT_SIZE = 518
FEATURE_DIMENSION = 384
LETTERBOX_FILL = (124, 116, 104)

FeaturePooling = Literal["cls", "patch_max"]
MODEL_OUTPUT_BY_POOLING: dict[str, str] = {
    "cls": "pooler_output_cls",
    "patch_max": "patch_token_max_pool",
}


class DinoFeatureError(ValueError):
    """Raised when images cannot produce the frozen feature cache."""


def collect_unique_frame_refs(windows_path: Path) -> tuple[str, ...]:
    """Read the private window JSONL and return sorted unique image references."""

    frame_refs: set[str] = set()
    try:
        with windows_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    row: object = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DinoFeatureError(
                        f"invalid window JSON on line {line_number}"
                    ) from error
                if not isinstance(row, dict):
                    raise DinoFeatureError(f"window line {line_number} must be an object")
                raw_refs = row.get("frame_refs")
                if not isinstance(raw_refs, list) or len(raw_refs) != 5:
                    raise DinoFeatureError(
                        f"window line {line_number} must contain five frame_refs"
                    )
                for value in raw_refs:
                    if not isinstance(value, str) or not _is_cam_front_ref(value):
                        raise DinoFeatureError(
                            f"window line {line_number} has an invalid CAM_FRONT reference"
                        )
                    frame_refs.add(value)
    except FileNotFoundError as error:
        raise DinoFeatureError(f"missing window file: {windows_path}") from error
    if not frame_refs:
        raise DinoFeatureError("window file contains no frame references")
    return tuple(sorted(frame_refs))


def letterbox_image(image: Image.Image) -> Image.Image:
    """Resize with preserved aspect ratio and pad to the frozen 518x518 input."""

    image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise DinoFeatureError("image dimensions must be positive")
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = image.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BICUBIC,
    )
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), LETTERBOX_FILL)
    offset = (
        (INPUT_SIZE - resized_width) // 2,
        (INPUT_SIZE - resized_height) // 2,
    )
    canvas.paste(resized, offset)
    return canvas


def max_pool_patch_tokens(last_hidden_state: Any, patch_size: int) -> Any:
    """Drop the CLS token and take a per-channel max over the remaining patches.

    `last_hidden_state` is `[batch, 1 + num_patches, hidden]`. Any other patch
    count is rejected, so a config or resolution drift is caught rather than
    silently pooling the wrong number of tokens.
    """

    import torch  # type: ignore[import-not-found,unused-ignore]

    hidden_state = torch.as_tensor(last_hidden_state)
    if hidden_state.ndim != 3:
        raise DinoFeatureError("last_hidden_state must have shape [batch,tokens,hidden]")
    if patch_size <= 0 or INPUT_SIZE % patch_size != 0:
        raise DinoFeatureError("INPUT_SIZE must be an exact multiple of patch_size")
    expected_patches = (INPUT_SIZE // patch_size) ** 2
    if hidden_state.shape[1] != expected_patches + 1:
        raise DinoFeatureError(
            f"expected {expected_patches + 1} tokens (1 CLS + {expected_patches} "
            f"patches), got {hidden_state.shape[1]}"
        )
    return hidden_state[:, 1:, :].max(dim=1).values


def extract_feature_cache(
    *,
    windows_path: Path,
    media_root: Path,
    output_dir: Path,
    cache_dir: Path,
    batch_size: int,
    num_threads: int,
    pooling: FeaturePooling = "cls",
    limit: int | None = None,
    local_files_only: bool = False,
) -> dict[str, object]:
    """Run the pinned model once per unique frame and write an indexed NumPy cache."""

    if output_dir.exists():
        raise DinoFeatureError("output directory must not already exist")
    if batch_size <= 0 or num_threads <= 0:
        raise DinoFeatureError("batch_size and num_threads must be positive")
    if pooling not in MODEL_OUTPUT_BY_POOLING:
        raise DinoFeatureError(f"unknown pooling: {pooling!r}")
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
        pooled = (
            output.pooler_output
            if pooling == "cls"
            else max_pool_patch_tokens(output.last_hidden_state, patch_size)
        )
        batch_features = np.asarray(pooled.detach().cpu(), dtype=np.float32)
        expected_shape = (len(batch_refs), FEATURE_DIMENSION)
        if batch_features.shape != expected_shape or not np.isfinite(batch_features).all():
            raise DinoFeatureError(f"DINO returned invalid {pooling} features")
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
        "model_output": MODEL_OUTPUT_BY_POOLING[pooling],
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


def _is_cam_front_ref(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and path.parts[:2] == ("samples", "CAM_FRONT")
        and path.suffix.lower() in {".jpg", ".jpeg"}
    )
