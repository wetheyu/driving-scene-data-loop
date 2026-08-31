"""Build an event-deduplicated Development false-negative bank."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

JsonObject = dict[str, Any]


class FalseNegativeBankError(ValueError):
    """Raised when an error could corrupt the FN bank experiment."""


@dataclass(frozen=True, slots=True)
class _Candidate:
    scenario_id: str
    event_group_id: str
    window_id: str
    scene_token: str
    scene_name: str
    log_token: str
    start_frame_index: int
    end_frame_index: int
    frame_refs: tuple[str, ...]
    probability: float


def load_feature_index(path: Path) -> dict[str, int]:
    """Map a cached frame reference to its DINO feature row."""

    rows = _read_jsonl(path)
    result = {row["media_ref"]: row["feature_index"] for row in rows}
    if len(result) != len(rows) or sorted(result.values()) != list(range(len(rows))):
        raise FalseNegativeBankError("frame feature index is not unique and contiguous")
    return cast(dict[str, int], result)


def build_selection_embedding(
    frame_features: NDArray[np.float32],
    frame_indices: tuple[int, int, int, int, int],
) -> NDArray[np.float32]:
    """Return L2-normalized `concat(mean(f0..f4), f4-f0)`."""

    if frame_features.ndim != 2 or frame_features.shape[1] != 384:
        raise FalseNegativeBankError("frame features must have shape [N,384]")
    sequence = np.asarray(frame_features[list(frame_indices)], dtype=np.float32)
    embedding = np.concatenate(
        (sequence.mean(axis=0), sequence[-1] - sequence[0])
    ).astype(np.float32)
    norm = float(np.linalg.norm(embedding))
    if not math.isfinite(norm) or norm == 0.0:
        raise FalseNegativeBankError("selection embedding has zero or invalid norm")
    return cast(NDArray[np.float32], embedding / norm)


def build_false_negative_bank(
    *,
    windows_path: Path,
    predictions_path: Path,
    base_report_path: Path,
    frame_features: NDArray[np.float32],
    feature_index: dict[str, int],
    feature_model_id: str,
    feature_model_revision: str,
    output_dir: Path,
    minimum_events_per_class: int = 5,
    partition: str = "development",
) -> JsonObject:
    """Find window FNs and retain the lowest-score view of each real event."""

    if output_dir.exists():
        raise FalseNegativeBankError("output directory must not already exist")

    scenario_ids, thresholds, checkpoint = _load_base_protocol(base_report_path)
    predictions = _load_predictions(predictions_path, len(scenario_ids), partition)
    scenario_order = {scenario_id: index for index, scenario_id in enumerate(scenario_ids)}
    seen_development_ids: set[str] = set()
    raw_fn_windows = {scenario_id: 0 for scenario_id in scenario_ids}
    event_window_misses: dict[tuple[str, str], list[bool]] = {}
    fn_window_counts: dict[tuple[str, str], int] = {}
    representatives: dict[tuple[str, str], _Candidate] = {}

    for window in _read_jsonl(windows_path):
        if window["partition"] != partition:
            continue
        window_id = cast(str, window["window_id"])
        if window_id in seen_development_ids or window_id not in predictions:
            raise FalseNegativeBankError(f"invalid {partition} prediction join: {window_id}")
        seen_development_ids.add(window_id)
        prediction = predictions[window_id]

        source_indices = {
            scenario_id: window["scenario_ids"].index(scenario_id)
            for scenario_id in scenario_ids
        }
        frame_refs = cast(tuple[str, ...], tuple(window["frame_refs"]))
        if len(frame_refs) != 5:
            raise FalseNegativeBankError(f"window must contain five frames: {window_id}")

        for class_index, scenario_id in enumerate(scenario_ids):
            source_index = source_indices[scenario_id]
            label = window["labels"][source_index]
            mask = window["loss_mask"][source_index]
            # Training-stage prediction files carry targets and are cross-checked;
            # frozen prediction files are label-free by design, and truth comes
            # from the private windows file either way.
            if "labels" in prediction and (
                prediction["labels"][class_index] != label
                or prediction["loss_mask"][class_index] != mask
            ):
                raise FalseNegativeBankError(
                    f"Base targets disagree with window labels: {window_id}"
                )
            if mask is not True or label != 1:
                continue

            event_group_ids = cast(list[str], window["event_group_ids"][source_index])
            if not event_group_ids:
                raise FalseNegativeBankError(
                    f"positive window has no event group: {window_id}"
                )
            probability = cast(float, prediction["probabilities"][class_index])
            is_false_negative = probability < thresholds[scenario_id]
            if is_false_negative:
                raw_fn_windows[scenario_id] += 1

            for event_group_id in event_group_ids:
                key = (scenario_id, event_group_id)
                event_window_misses.setdefault(key, []).append(is_false_negative)
                if not is_false_negative:
                    continue
                fn_window_counts[key] = fn_window_counts.get(key, 0) + 1
                candidate = _Candidate(
                    scenario_id=scenario_id,
                    event_group_id=event_group_id,
                    window_id=window_id,
                    scene_token=window["scene_token"],
                    scene_name=window["scene_name"],
                    log_token=window["log_token"],
                    start_frame_index=window["start_frame_index"],
                    end_frame_index=window["end_frame_index"],
                    frame_refs=frame_refs,
                    probability=probability,
                )
                current = representatives.get(key)
                if current is None or (probability, window_id) < (
                    current.probability,
                    current.window_id,
                ):
                    representatives[key] = candidate

    if seen_development_ids != set(predictions):
        raise FalseNegativeBankError(f"predictions and {partition} windows are not one-to-one")

    ordered = sorted(
        representatives.values(),
        key=lambda item: (scenario_order[item.scenario_id], item.event_group_id),
    )
    for scenario_id in scenario_ids:
        count = sum(item.scenario_id == scenario_id for item in ordered)
        if count < minimum_events_per_class:
            raise FalseNegativeBankError(
                f"{scenario_id} has only {count} false-negative events"
            )

    embeddings = np.empty((len(ordered), 768), dtype=np.float32)
    rows: list[JsonObject] = []
    for bank_index, candidate in enumerate(ordered):
        indices = cast(
            tuple[int, int, int, int, int],
            tuple(feature_index[ref] for ref in candidate.frame_refs),
        )
        embeddings[bank_index] = build_selection_embedding(frame_features, indices)
        rows.append(
            {
                "bank_index": bank_index,
                "scenario_id": candidate.scenario_id,
                "event_group_id": candidate.event_group_id,
                "window_id": candidate.window_id,
                "scene_token": candidate.scene_token,
                "scene_name": candidate.scene_name,
                "log_token": candidate.log_token,
                "start_frame_index": candidate.start_frame_index,
                "end_frame_index": candidate.end_frame_index,
                "base_probability": candidate.probability,
                "source_false_negative_window_count": fn_window_counts[
                    (candidate.scenario_id, candidate.event_group_id)
                ],
            }
        )

    classes: dict[str, object] = {}
    for scenario_id in scenario_ids:
        class_events = [
            misses
            for (event_scenario, _), misses in event_window_misses.items()
            if event_scenario == scenario_id
        ]
        class_rows = [row for row in rows if row["scenario_id"] == scenario_id]
        classes[scenario_id] = {
            "threshold": thresholds[scenario_id],
            "raw_false_negative_windows": raw_fn_windows[scenario_id],
            "positive_event_groups": len(class_events),
            "event_groups_with_any_false_negative": len(class_rows),
            "event_groups_with_all_positive_windows_false_negative": sum(
                all(misses) for misses in class_events
            ),
            "unique_representative_windows": len(
                {row["window_id"] for row in class_rows}
            ),
            "minimum_event_gate": minimum_events_per_class,
            "passed_minimum_event_gate": True,
        }

    report: JsonObject = {
        "schema_version": "1.0",
        "artifact": f"{partition}_false_negative_bank",
        "partition": partition,
        "definition": (
            "Ground-Truth-positive windows below the frozen Base threshold, "
            "deduplicated by scenario_id and event_group_id"
        ),
        "representative_rule": (
            "lowest Base probability; ties use lexicographically smallest window_id"
        ),
        "base": {
            "run": base_report_path.parent.name,
            "seed": 17,
            "checkpoint": checkpoint,
            "scenario_ids": list(scenario_ids),
        },
        "embedding": {
            "model_id": feature_model_id,
            "model_revision": feature_model_revision,
            "formula": "L2(concat(mean(f0..f4), f4-f0))",
            "dimension": 768,
            "dtype": "float32",
        },
        "source_window_count": len(seen_development_ids),
        "bank_event_count": len(rows),
        "bank_unique_window_count": len({row["window_id"] for row in rows}),
        "classes": classes,
        "files": {"rows": "fn_bank.jsonl", "embeddings": "fn_embeddings.npy"},
    }

    output_dir.mkdir(parents=True)
    with (output_dir / "fn_bank.jsonl").open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(
                json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"
            )
    np.save(output_dir / "fn_embeddings.npy", embeddings)
    (output_dir / "fn_bank_report.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _load_base_protocol(path: Path) -> tuple[tuple[str, ...], dict[str, float], str]:
    report = _read_json(path)
    if report["base_seed"] != 17 or report["evaluation_partition"] != "development":
        raise FalseNegativeBankError("FN bank requires Development Base seed 17")
    scenario_ids = cast(tuple[str, ...], tuple(report["scenario_ids"]))
    # Gate B is diagnostic. Every Gate-A class with enough independent
    # Development false negatives remains eligible for the data loop.
    thresholds = {
        scenario_id: float(report["base_thresholds"][scenario_id]["threshold"])
        for scenario_id in scenario_ids
    }
    if any(not 0.0 <= value <= 1.0 for value in thresholds.values()):
        raise FalseNegativeBankError("Base threshold is outside [0,1]")
    checkpoint = cast(str, report["runs"]["17"]["checkpoint"])
    return scenario_ids, thresholds, checkpoint


def _load_predictions(
    path: Path, class_count: int, partition: str = "development"
) -> dict[str, JsonObject]:
    result: dict[str, JsonObject] = {}
    for row in _read_jsonl(path):
        window_id = cast(str, row["window_id"])
        probabilities = np.asarray(row["probabilities"], dtype=np.float64)
        if (
            row["partition"] != partition
            or probabilities.shape != (class_count,)
            or not np.isfinite(probabilities).all()
            or bool((probabilities < 0.0).any())
            or bool((probabilities > 1.0).any())
            or window_id in result
        ):
            raise FalseNegativeBankError(f"invalid Base prediction: {window_id}")
        row["probabilities"] = probabilities.tolist()
        result[window_id] = row
    return result


def _read_json(path: Path) -> JsonObject:
    return cast(JsonObject, json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[JsonObject]:
    with path.open(encoding="utf-8") as source:
        return [cast(JsonObject, json.loads(line)) for line in source]
