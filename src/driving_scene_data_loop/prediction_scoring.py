"""Score frozen prediction files against private window labels.

The final evaluation is two stages by contract: inference writes predictions,
then scoring reads those frozen files plus the labels. This module is the
scoring half. It is deliberately torch-free so the arithmetic is unit-testable
anywhere, and it is validated by reproducing the Development AP recorded in
each run's own training report before it is ever pointed at the Held-out Test.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import cast

import numpy as np
from sklearn.metrics import average_precision_score  # type: ignore[import-untyped]

JsonObject = dict[str, object]


class PredictionScoringError(ValueError):
    """Raised when scoring would silently misjoin predictions and labels."""


def load_partition_targets(
    windows_path: Path,
    partition: str,
) -> tuple[tuple[str, ...], dict[str, tuple[list[int], list[bool]]]]:
    """Read one partition's labels and loss masks, keyed by window ID."""

    scenario_ids: tuple[str, ...] | None = None
    targets: dict[str, tuple[list[int], list[bool]]] = {}
    with windows_path.open(encoding="utf-8") as source:
        for line in source:
            row = cast(JsonObject, json.loads(line))
            if row["partition"] != partition:
                continue
            row_sids = tuple(cast(list[str], row["scenario_ids"]))
            if scenario_ids is None:
                scenario_ids = row_sids
            elif row_sids != scenario_ids:
                raise PredictionScoringError("scenario order changes between windows")
            window_id = cast(str, row["window_id"])
            if window_id in targets:
                raise PredictionScoringError(f"duplicate window: {window_id}")
            labels = [
                int(cast(int, v)) if v is not None else -1
                for v in cast(list[object], row["labels"])
            ]
            masks = [bool(v) for v in cast(list[object], row["loss_mask"])]
            targets[window_id] = (labels, masks)
    if scenario_ids is None or not targets:
        raise PredictionScoringError(f"no windows found for partition {partition!r}")
    return scenario_ids, targets


def score_prediction_file(
    predictions_path: Path,
    scenario_ids: tuple[str, ...],
    targets: dict[str, tuple[list[int], list[bool]]],
) -> JsonObject:
    """Compute masked per-class AP and Macro-AP for one seed's predictions."""

    labels = np.full((len(targets), len(scenario_ids)), -1, dtype=np.int8)
    masks = np.zeros((len(targets), len(scenario_ids)), dtype=np.bool_)
    probs = np.full((len(targets), len(scenario_ids)), np.nan, dtype=np.float64)
    order = {window_id: index for index, window_id in enumerate(sorted(targets))}
    for window_id, (row_labels, row_masks) in targets.items():
        index = order[window_id]
        labels[index] = row_labels
        masks[index] = row_masks

    seen = 0
    with predictions_path.open(encoding="utf-8") as source:
        for line in source:
            row = cast(JsonObject, json.loads(line))
            window_id = cast(str, row["window_id"])
            if window_id not in order:
                raise PredictionScoringError(
                    f"prediction for a window outside the partition: {window_id}"
                )
            values = np.asarray(cast(list[float], row["probabilities"]), dtype=np.float64)
            if values.shape != (len(scenario_ids),) or not np.isfinite(values).all():
                raise PredictionScoringError(f"invalid probabilities: {window_id}")
            index = order[window_id]
            if not np.isnan(probs[index]).all():
                raise PredictionScoringError(f"duplicate prediction: {window_id}")
            probs[index] = values
            seen += 1
    if seen != len(targets):
        raise PredictionScoringError(
            f"{len(targets) - seen} partition windows have no prediction"
        )

    result: JsonObject = {"classes": {}}
    average_precisions: list[float] = []
    for class_index, scenario_id in enumerate(scenario_ids):
        valid = masks[:, class_index]
        y_true = labels[valid, class_index]
        y_score = probs[valid, class_index]
        if len(y_true) == 0 or min(int(y_true.sum()), len(y_true) - int(y_true.sum())) <= 0:
            raise PredictionScoringError(f"AP needs both label values: {scenario_id}")
        ap = float(average_precision_score(y_true, y_score))
        cast(dict[str, object], result["classes"])[scenario_id] = {
            "average_precision": ap,
            "positives": int(y_true.sum()),
            "valid": int(valid.sum()),
        }
        average_precisions.append(ap)
    result["macro_average_precision"] = float(np.mean(average_precisions))
    return result


def seed_paired_contrast(
    a_by_seed: dict[str, float],
    b_by_seed: dict[str, float],
) -> JsonObject:
    """Mean, stderr, and sigma of per-seed differences a - b over shared seeds."""

    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    if len(seeds) < 3:
        raise PredictionScoringError("a paired contrast needs at least three shared seeds")
    diffs = [a_by_seed[s] - b_by_seed[s] for s in seeds]
    mean = statistics.mean(diffs)
    stderr = statistics.stdev(diffs) / math.sqrt(len(diffs))
    return {
        "seeds": len(seeds),
        "mean": mean,
        "stderr": stderr,
        # None, not infinity: a zero spread means the contrast is deterministic,
        # and the report must stay allow_nan=False serialisable.
        "sigma": abs(mean) / stderr if stderr > 0 else None,
    }


def _score_matrix(
    predictions_dir: Path,
    scenario_ids: tuple[str, ...],
    targets: dict[str, tuple[list[int], list[bool]]],
) -> dict[str, JsonObject]:
    """Score every predictions_seed_*.jsonl in one run's prediction directory."""

    files = sorted(predictions_dir.glob("predictions_seed_*.jsonl"))
    if not files:
        raise PredictionScoringError(f"no prediction files in {predictions_dir}")
    return {
        path.stem.removeprefix("predictions_seed_"): score_prediction_file(
            path, scenario_ids, targets
        )
        for path in files
    }


def score_runs(
    *,
    windows_path: Path,
    partition: str,
    runs: dict[str, Path],
    output_path: Path,
    window_list: set[str] | None = None,
) -> JsonObject:
    """Score named prediction directories and every pairwise seed-paired contrast.

    `window_list` restricts scoring to a declared evaluation subset (Test2 is a
    subset of the `u` partition); every listed window must exist in the
    partition, so a typo cannot silently shrink the evaluation.
    """

    if output_path.exists():
        raise PredictionScoringError("output path must not already exist")
    scenario_ids, targets = load_partition_targets(windows_path, partition)
    if window_list is not None:
        missing = window_list - set(targets)
        if missing:
            raise PredictionScoringError(
                f"{len(missing)} listed windows are not in partition {partition!r}"
            )
        targets = {w: t for w, t in targets.items() if w in window_list}

    per_run: JsonObject = {}
    macro_by_seed: dict[str, dict[str, float]] = {}
    class_by_seed: dict[str, dict[str, dict[str, float]]] = {}
    for name, directory in sorted(runs.items()):
        scored = _score_matrix(directory, scenario_ids, targets)
        macro_by_seed[name] = {
            seed: cast(float, result["macro_average_precision"])
            for seed, result in scored.items()
        }
        class_by_seed[name] = {
            sid: {
                seed: cast(
                    float,
                    cast(JsonObject, cast(JsonObject, result["classes"])[sid])[
                        "average_precision"
                    ],
                )
                for seed, result in scored.items()
            }
            for sid in scenario_ids
        }
        macros = list(macro_by_seed[name].values())
        per_run[name] = {
            "seeds": sorted(scored),
            "macro_average_precision_mean": statistics.mean(macros),
            "macro_average_precision_std": statistics.stdev(macros) if len(macros) > 1 else 0.0,
            "classes": {
                sid: {
                    "mean": statistics.mean(class_by_seed[name][sid].values()),
                    "std": (
                        statistics.stdev(class_by_seed[name][sid].values())
                        if len(class_by_seed[name][sid]) > 1
                        else 0.0
                    ),
                }
                for sid in scenario_ids
            },
            "per_seed": scored,
        }

    contrasts: JsonObject = {}
    names = sorted(runs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            entry: JsonObject = {"macro": seed_paired_contrast(macro_by_seed[a], macro_by_seed[b])}
            for sid in scenario_ids:
                entry[sid] = seed_paired_contrast(class_by_seed[a][sid], class_by_seed[b][sid])
            contrasts[f"{a} - {b}"] = entry

    report: JsonObject = {
        "schema_version": "1.0",
        "artifact": "prediction_scores",
        "partition": partition,
        "window_list_size": len(targets) if window_list is not None else None,
        "scenario_ids": list(scenario_ids),
        "runs": per_run,
        "seed_paired_contrasts": contrasts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report

def development_paired_contrasts(
    run_reports: dict[str, Path],
    contrasts: tuple[tuple[str, str], ...],
) -> JsonObject:
    """Seed-paired Development contrasts read straight from training reports.

    For diagnostics only: Development carries no headline claim, and this reads
    the per-seed normal-order APs the training stage already computed rather
    than re-deriving anything.
    """

    per_run: dict[str, dict[str, dict[str, float]]] = {}
    scenario_ids: tuple[str, ...] | None = None
    for name, path in run_reports.items():
        report = cast(JsonObject, json.loads(path.read_text(encoding="utf-8")))
        run_scenarios = tuple(cast(list[str], report["scenario_ids"]))
        if scenario_ids is None:
            scenario_ids = run_scenarios
        elif run_scenarios != scenario_ids:
            raise PredictionScoringError(f"scenario order differs in run {name}")
        seeds: dict[str, dict[str, float]] = {}
        for seed, record in cast(JsonObject, report["runs"]).items():
            normal = cast(JsonObject, cast(JsonObject, record)["normal_order"])
            classes = cast(JsonObject, normal["classes"])
            values = {
                scenario_id: float(
                    cast(
                        "float", cast(JsonObject, classes[scenario_id])["average_precision"]
                    )
                )
                for scenario_id in run_scenarios
            }
            values["macro"] = sum(values.values()) / len(run_scenarios)
            seeds[seed] = values
        per_run[name] = seeds

    assert scenario_ids is not None
    shared = set.intersection(*(set(seeds) for seeds in per_run.values()))
    if len(shared) < 2:
        raise PredictionScoringError("paired contrasts need at least two shared seeds")
    ordered = sorted(shared, key=int)

    rows: list[JsonObject] = []
    for left, right in contrasts:
        if left not in per_run or right not in per_run:
            raise PredictionScoringError(f"unknown run in contrast {left} - {right}")
        for key in (*scenario_ids, "macro"):
            diffs = [per_run[left][seed][key] - per_run[right][seed][key] for seed in ordered]
            mean = sum(diffs) / len(diffs)
            variance = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
            stderr = (variance / len(diffs)) ** 0.5
            rows.append(
                {
                    "contrast": f"{left} - {right}",
                    "metric": key,
                    "effect": round(mean, 6),
                    "stderr": round(stderr, 6),
                    "sigma": round(mean / stderr, 3) if stderr else None,
                }
            )
    return {
        "schema_version": "1.0",
        "artifact": "development_paired_contrasts",
        "evidence_level": "Development diagnostic; carries no held-out claim",
        "seed_count": len(ordered),
        "seeds": ordered,
        "scenario_ids": list(scenario_ids),
        "rows": rows,
    }

