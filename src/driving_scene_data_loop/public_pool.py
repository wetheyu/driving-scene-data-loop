"""Run the frozen Base model on the label-free selection pool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch  # type: ignore[import-not-found,unused-ignore]
from numpy.typing import NDArray

from driving_scene_data_loop.false_negative_bank import build_selection_embedding
from driving_scene_data_loop.gru_baseline import GlobalGRU
from driving_scene_data_loop.window_dataset import PUBLIC_U_FIELDS

JsonObject = dict[str, Any]


class PublicPoolError(ValueError):
    """Raised when public-Pool inference would violate the experiment."""


def prepare_public_pool(
    *,
    public_windows_path: Path,
    base_report_path: Path,
    frame_features: NDArray[np.float32],
    feature_index: dict[str, int],
    feature_model_id: str,
    feature_model_revision: str,
    output_dir: Path,
    batch_size: int = 256,
) -> JsonObject:
    """Save Base probabilities and selection embeddings without Oracle fields."""

    if output_dir.exists():
        raise PublicPoolError("output directory must not already exist")
    if batch_size <= 0:
        raise PublicPoolError("batch_size must be positive")

    windows = _read_public_windows(public_windows_path)
    report = _read_json(base_report_path)
    if report.get("base_seed") != 17:
        raise PublicPoolError("public-Pool inference requires Base seed 17")
    scenario_ids = cast(tuple[str, ...], tuple(report["scenario_ids"]))
    if len(scenario_ids) != 3:
        raise PublicPoolError("the closed loop requires the three-class Base")
    thresholds = {
        scenario_id: float(report["base_thresholds"][scenario_id]["threshold"])
        for scenario_id in scenario_ids
    }
    checkpoint_name = cast(str, report["runs"]["17"]["checkpoint"])
    checkpoint_path = base_report_path.parent / checkpoint_name

    sequences = np.empty((len(windows), 5, 384), dtype=np.float32)
    embeddings = np.empty((len(windows), 768), dtype=np.float32)
    for row_index, window in enumerate(windows):
        frame_refs = cast(tuple[str, ...], tuple(window["frame_refs"]))
        try:
            frame_indices = cast(
                tuple[int, int, int, int, int],
                tuple(feature_index[frame_ref] for frame_ref in frame_refs),
            )
        except KeyError as error:
            raise PublicPoolError(
                f"public window references an uncached frame: {window['window_id']}"
            ) from error
        if len(frame_indices) != 5:
            raise PublicPoolError("every public window must contain five frames")
        sequences[row_index] = frame_features[list(frame_indices)]
        embeddings[row_index] = build_selection_embedding(
            frame_features,
            frame_indices,
        )

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Derive the head's width from the checkpoint rather than assuming the
    # default, so a Base trained under a different config still loads correctly.
    hidden_size = int(state["classifier.weight"].shape[1])
    model = GlobalGRU(len(scenario_ids), hidden_size)
    model.load_state_dict(state)
    probabilities = _predict(model, sequences, batch_size)

    output_dir.mkdir(parents=True)
    rows_path = output_dir / "pool_windows.jsonl"
    with rows_path.open("w", encoding="utf-8") as destination:
        for pool_index, (window, probability) in enumerate(
            zip(windows, probabilities, strict=True)
        ):
            row = dict(window)
            row["pool_index"] = pool_index
            row["base_probabilities"] = probability.tolist()
            destination.write(
                json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"
            )
    np.save(output_dir / "pool_embeddings.npy", embeddings)

    output_report: JsonObject = {
        "schema_version": "1.0",
        "artifact": "public_pool_base_inference",
        "partition": "u",
        "window_count": len(windows),
        "scenario_ids": list(scenario_ids),
        "base": {
            "run": base_report_path.parent.name,
            "seed": 17,
            "checkpoint": checkpoint_name,
            "thresholds": thresholds,
        },
        "embedding": {
            "model_id": feature_model_id,
            "model_revision": feature_model_revision,
            "formula": "L2(concat(mean(f0..f4), f4-f0))",
            "dimension": 768,
            "dtype": "float32",
        },
        "files": {
            "rows": rows_path.name,
            "embeddings": "pool_embeddings.npy",
        },
    }
    (output_dir / "pool_report.json").write_text(
        json.dumps(output_report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_report


def _predict(
    model: GlobalGRU,
    sequences: NDArray[np.float32],
    batch_size: int,
) -> NDArray[np.float64]:
    model.eval()
    batches: list[NDArray[np.float64]] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            inputs = torch.from_numpy(sequences[start : start + batch_size])
            batches.append(
                np.asarray(torch.sigmoid(model(inputs)).numpy(), dtype=np.float64)
            )
    return np.concatenate(batches, axis=0)


def _read_public_windows(path: Path) -> list[JsonObject]:
    windows: list[JsonObject] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            row = cast(JsonObject, json.loads(line))
            if set(row) != set(PUBLIC_U_FIELDS) or row.get("partition") != "u":
                raise PublicPoolError(
                    f"public window line {line_number} has non-public fields"
                )
            window_id = row.get("window_id")
            if not isinstance(window_id, str) or window_id in seen_ids:
                raise PublicPoolError(f"invalid public window ID on line {line_number}")
            seen_ids.add(window_id)
            windows.append(row)
    if not windows:
        raise PublicPoolError("public Pool is empty")
    return windows


def _read_json(path: Path) -> JsonObject:
    return cast(JsonObject, json.loads(path.read_text(encoding="utf-8")))
