"""Train the two simple visual baselines on frozen frame features."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import average_precision_score  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from driving_scene_data_loop.dino_features import (
    FEATURE_DIMENSION,
    MODEL_ID,
    MODEL_REVISION,
)

JsonObject = dict[str, object]


class LrBaselineError(ValueError):
    """Raised when features or labels cannot support a fair LR baseline."""


FORMAL_FRAME_COUNT = 34_149


@dataclass(frozen=True, slots=True)
class BaselineWindowData:
    """L0 and Development windows joined to frozen frame-feature rows."""

    scenario_ids: tuple[str, ...]
    window_ids: tuple[str, ...]
    partitions: tuple[str, ...]
    frame_feature_indices: NDArray[np.int64]
    labels: NDArray[np.int8]
    loss_mask: NDArray[np.bool_]


def load_baseline_windows(
    windows_path: Path,
    frame_index_path: Path,
) -> BaselineWindowData:
    """Load only L0 and Development; TaskDesign/U/Test never enter this baseline."""

    feature_index = _load_feature_index(frame_index_path)
    scenario_ids: tuple[str, ...] | None = None
    window_ids: list[str] = []
    partitions: list[str] = []
    frame_indices: list[list[int]] = []
    labels: list[list[int]] = []
    masks: list[list[bool]] = []

    for line_number, row in _json_lines(windows_path):
        partition = row.get("partition")
        if partition not in ("l0", "development"):
            continue
        current_scenarios = _string_list(row, "scenario_ids", line_number)
        if scenario_ids is None:
            scenario_ids = tuple(current_scenarios)
        elif tuple(current_scenarios) != scenario_ids:
            raise LrBaselineError("scenario order changes between windows")

        refs = _string_list(row, "frame_refs", line_number)
        if len(refs) != 5:
            raise LrBaselineError(f"window line {line_number} must contain five frames")
        try:
            frame_indices.append([feature_index[ref] for ref in refs])
        except KeyError as error:
            raise LrBaselineError(
                f"window line {line_number} references an uncached frame"
            ) from error

        raw_labels = row.get("labels")
        raw_mask = row.get("loss_mask")
        if not isinstance(raw_labels, list) or not isinstance(raw_mask, list):
            raise LrBaselineError(f"window line {line_number} has invalid targets")
        if len(raw_labels) != len(current_scenarios) or len(raw_mask) != len(current_scenarios):
            raise LrBaselineError(f"window line {line_number} target sizes differ")
        parsed_labels: list[int] = []
        parsed_masks: list[bool] = []
        for label, mask in zip(raw_labels, raw_mask, strict=True):
            if not isinstance(mask, bool):
                raise LrBaselineError(f"window line {line_number} mask must be boolean")
            if mask and (isinstance(label, bool) or label not in (0, 1)):
                raise LrBaselineError(f"window line {line_number} has an invalid label")
            if not mask and label is not None:
                raise LrBaselineError(f"window line {line_number} masked label must be null")
            parsed_labels.append(int(label) if mask else -1)
            parsed_masks.append(mask)

        window_id = row.get("window_id")
        if not isinstance(window_id, str) or not window_id:
            raise LrBaselineError(f"window line {line_number} has no window_id")
        window_ids.append(window_id)
        partitions.append(partition)
        labels.append(parsed_labels)
        masks.append(parsed_masks)

    if scenario_ids is None or not window_ids:
        raise LrBaselineError("no L0 or Development windows were loaded")
    if len(window_ids) != len(set(window_ids)):
        raise LrBaselineError("window IDs are not unique")
    return BaselineWindowData(
        scenario_ids=scenario_ids,
        window_ids=tuple(window_ids),
        partitions=tuple(partitions),
        frame_feature_indices=np.asarray(frame_indices, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int8),
        loss_mask=np.asarray(masks, dtype=np.bool_),
    )


def validate_formal_feature_manifest(path: Path) -> dict[str, object]:
    """Reject a smoke cache or a different encoder, and return the cache identity.

    The identity travels into each run's report. Two caches can share every
    validated field and still be different representations -- `pooler_output_cls`
    against `patch_token_max_pool` -- so `model_output` has to be recorded or a
    representation comparison cannot say which run used which features.
    """

    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise LrBaselineError(f"cannot read feature manifest: {path}") from error
    if not isinstance(value, dict):
        raise LrBaselineError("feature manifest must be a JSON object")
    expected = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "feature_dimension": FEATURE_DIMENSION,
        "dtype": "float32",
        "frame_count": FORMAL_FRAME_COUNT,
        "limited_smoke": False,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise LrBaselineError(f"feature manifest {field} must be {expected_value!r}")
    model_output = value.get("model_output")
    if not isinstance(model_output, str) or not model_output:
        raise LrBaselineError("feature manifest must name its model_output")
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_output": model_output,
        "feature_dimension": FEATURE_DIMENSION,
        "frame_count": FORMAL_FRAME_COUNT,
    }


def load_frame_features(
    feature_path: Path,
    expected_frame_count: int,
) -> NDArray[np.float32]:
    """Memory-map the formal float32 matrix and check its public contract."""

    try:
        value = np.load(feature_path, mmap_mode="r")
    except (OSError, ValueError) as error:
        raise LrBaselineError(f"cannot load feature matrix: {feature_path}") from error
    if value.shape != (expected_frame_count, 384) or value.dtype != np.float32:
        raise LrBaselineError("feature matrix must have shape [frame_count,384] float32")
    if not np.isfinite(value).all():
        raise LrBaselineError("feature matrix contains non-finite values")
    return cast(NDArray[np.float32], value)


def build_window_representations(
    frame_features: NDArray[np.float32],
    frame_indices: NDArray[np.int64],
) -> dict[str, NDArray[np.float32]]:
    """Create last-frame and order-free five-frame mean representations."""

    if frame_indices.ndim != 2 or frame_indices.shape[1] != 5:
        raise LrBaselineError("frame indices must have shape [window_count,5]")
    if frame_indices.size == 0 or frame_indices.min() < 0:
        raise LrBaselineError("frame indices must not be empty or negative")
    if frame_indices.max() >= len(frame_features):
        raise LrBaselineError("frame index is outside the feature matrix")

    last_frame = np.asarray(frame_features[frame_indices[:, -1]], dtype=np.float32)
    mean_five = np.zeros_like(last_frame)
    for position in range(5):
        mean_five += frame_features[frame_indices[:, position]]
    mean_five /= 5.0
    return {"last_frame_lr": last_frame, "mean5_lr": mean_five}


def train_lr_baselines(
    *,
    window_data: BaselineWindowData,
    frame_features: NDArray[np.float32],
    output_dir: Path,
    seed: int = 17,
    feature_cache: JsonObject | None = None,
) -> JsonObject:
    """Fit both frozen protocols on L0 and score only Development AP."""

    if output_dir.exists():
        raise LrBaselineError("output directory must not already exist")
    output_dir.mkdir(parents=True)
    representations = build_window_representations(
        frame_features,
        window_data.frame_feature_indices,
    )
    train_rows = np.asarray(
        [partition == "l0" for partition in window_data.partitions],
        dtype=np.bool_,
    )
    development_rows = np.asarray(
        [partition == "development" for partition in window_data.partitions],
        dtype=np.bool_,
    )
    if not train_rows.any() or not development_rows.any():
        raise LrBaselineError("both L0 and Development windows are required")

    reports: dict[str, object] = {}
    for baseline_name, representation in representations.items():
        reports[baseline_name] = _train_one_baseline(
            baseline_name=baseline_name,
            representation=representation,
            window_data=window_data,
            train_rows=train_rows,
            development_rows=development_rows,
            output_dir=output_dir,
            seed=seed,
        )

    report: JsonObject = {
        "schema_version": "1.0",
        "scenario_ids": list(window_data.scenario_ids),
        "feature_cache": feature_cache,
        "feature_dimension": int(frame_features.shape[1]),
        "train_partition": "l0",
        "evaluation_partition": "development",
        "l0_window_count": int(train_rows.sum()),
        "development_window_count": int(development_rows.sum()),
        "seed": seed,
        "baselines": reports,
    }
    (output_dir / "lr_baseline_report.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _train_one_baseline(
    *,
    baseline_name: str,
    representation: NDArray[np.float32],
    window_data: BaselineWindowData,
    train_rows: NDArray[np.bool_],
    development_rows: NDArray[np.bool_],
    output_dir: Path,
    seed: int,
) -> JsonObject:
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(representation[train_rows])
    scaled_development = scaler.transform(representation[development_rows])
    train_labels = window_data.labels[train_rows]
    train_mask = window_data.loss_mask[train_rows]
    development_labels = window_data.labels[development_rows]
    development_mask = window_data.loss_mask[development_rows]

    probabilities = np.zeros(
        (int(development_rows.sum()), len(window_data.scenario_ids)),
        dtype=np.float64,
    )
    coefficients = np.zeros(
        (len(window_data.scenario_ids), representation.shape[1]),
        dtype=np.float64,
    )
    intercepts = np.zeros(len(window_data.scenario_ids), dtype=np.float64)
    class_reports: dict[str, object] = {}
    average_precisions: list[float] = []

    for class_index, scenario_id in enumerate(window_data.scenario_ids):
        valid_train = train_mask[:, class_index]
        valid_development = development_mask[:, class_index]
        y_train = train_labels[valid_train, class_index]
        y_development = development_labels[valid_development, class_index]
        train_positive = int(y_train.sum())
        train_negative = len(y_train) - train_positive
        development_positive = int(y_development.sum())
        development_negative = len(y_development) - development_positive
        if (
            min(
                train_positive,
                train_negative,
                development_positive,
                development_negative,
            )
            <= 0
        ):
            raise LrBaselineError(f"{scenario_id} needs both classes in L0 and Development")

        positive_weight = train_negative / train_positive
        model = LogisticRegression(
            C=1.0,
            class_weight={0: 1.0, 1: positive_weight},
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
        )
        model.fit(scaled_train[valid_train], y_train)
        iterations = int(model.n_iter_[0])
        if iterations >= model.max_iter:
            raise LrBaselineError(f"{scenario_id} LR did not converge")
        probabilities[:, class_index] = model.predict_proba(scaled_development)[:, 1]
        coefficients[class_index] = model.coef_[0]
        intercepts[class_index] = model.intercept_[0]
        score = float(
            average_precision_score(
                y_development,
                probabilities[valid_development, class_index],
            )
        )
        if not math.isfinite(score):
            raise LrBaselineError(f"{scenario_id} produced a non-finite AP")
        average_precisions.append(score)
        class_reports[scenario_id] = {
            "average_precision": score,
            "development_prevalence": development_positive / len(y_development),
            "l0_positive": train_positive,
            "l0_negative": train_negative,
            "development_positive": development_positive,
            "development_negative": development_negative,
            "positive_weight": positive_weight,
            "iterations": iterations,
        }

    development_ids = tuple(
        window_id
        for window_id, is_development in zip(
            window_data.window_ids,
            development_rows,
            strict=True,
        )
        if is_development
    )
    with (output_dir / f"{baseline_name}_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as destination:
        for row_index, window_id in enumerate(development_ids):
            destination.write(
                json.dumps(
                    {
                        "window_id": window_id,
                        "partition": "development",
                        "probabilities": probabilities[row_index].tolist(),
                        "labels": [
                            int(label) if mask else None
                            for label, mask in zip(
                                development_labels[row_index],
                                development_mask[row_index],
                                strict=True,
                            )
                        ],
                        "loss_mask": development_mask[row_index].tolist(),
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    np.savez_compressed(
        output_dir / f"{baseline_name}_parameters.npz",
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        coefficients=coefficients,
        intercepts=intercepts,
    )
    return {
        "macro_average_precision": float(np.mean(average_precisions)),
        "standard_scaler_fit_partition": "l0",
        "logistic_regression": {
            "C": 1.0,
            "regularization": "L2 (scikit-learn default)",
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight_ratio_source": "l0_only",
        },
        "classes": class_reports,
    }


def _load_feature_index(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for line_number, row in _json_lines(path):
        media_ref = row.get("media_ref")
        feature_index = row.get("feature_index")
        if (
            not isinstance(media_ref, str)
            or not media_ref
            or isinstance(feature_index, bool)
            or not isinstance(feature_index, int)
            or feature_index != len(result)
            or media_ref in result
        ):
            raise LrBaselineError(f"invalid feature index on line {line_number}")
        result[media_ref] = feature_index
    if not result:
        raise LrBaselineError("feature index is empty")
    return result


def _json_lines(path: Path) -> list[tuple[int, JsonObject]]:
    rows: list[tuple[int, JsonObject]] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    value: object = json.loads(line)
                except json.JSONDecodeError as error:
                    raise LrBaselineError(
                        f"invalid JSON on line {line_number} of {path.name}"
                    ) from error
                if not isinstance(value, dict):
                    raise LrBaselineError(f"line {line_number} of {path.name} must be an object")
                rows.append((line_number, dict(value)))
    except FileNotFoundError as error:
        raise LrBaselineError(f"missing file: {path}") from error
    return rows


def _string_list(row: JsonObject, field: str, line_number: int) -> list[str]:
    value = row.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise LrBaselineError(f"window line {line_number} has invalid {field}")
    return cast(list[str], value)
