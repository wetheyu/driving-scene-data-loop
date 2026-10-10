"""Score frozen VLM labels against the already-revealed Oracle labels.

The Oracle stays the private metric benchmark: a disagreement is reported as a
disagreement, never resolved in the VLM's favour. Windows the Oracle calls
`ignore` or `invalid` are the definitionally ambiguous region and are profiled
separately instead of being counted as VLM errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from driving_scene_data_loop.json_files import read_json, read_jsonl, write_report
from driving_scene_data_loop.vlm_labeling import VERDICTS

JsonObject = dict[str, Any]

ORACLE_ARTIFACT = "oracle_revealed_labels"
REPORT_FILENAME = "vlm_label_report.json"


class VlmEvaluationError(ValueError):
    """Raised when label sources cannot be compared as declared."""


def evaluate_vlm_labels(
    *,
    vlm_labels_path: Path,
    oracle_dir: Path,
    method: str,
    budget: int,
    output_dir: Path,
    run_manifest_path: Path | None = None,
    protocol: str = "v0.11",
) -> JsonObject:
    """Join one budget prefix of a method to the Oracle and score the labels."""

    if output_dir.exists():
        raise VlmEvaluationError("output directory must not already exist")

    scenario_ids, oracle_rows = load_oracle_prefix(oracle_dir, method, budget)
    vlm_rows = load_vlm_labels(vlm_labels_path, oracle_rows, scenario_ids)

    per_class = {
        scenario_id: _score_class(
            [
                (oracle_rows[window_id], vlm_rows[window_id])
                for window_id in sorted(oracle_rows)
            ],
            class_index,
        )
        for class_index, scenario_id in enumerate(scenario_ids)
    }
    f1_values = [
        cast(float, per_class[scenario_id]["f1"])
        for scenario_id in scenario_ids
        if per_class[scenario_id]["f1"] is not None
    ]
    schema_valid = sum(
        bool(vlm_rows[window_id]["schema_valid"]) for window_id in oracle_rows
    )
    agreed_positives = sum(
        cast(int, per_class[scenario_id]["true_positives"]) for scenario_id in scenario_ids
    )

    report: JsonObject = {
        "schema_version": "1.0",
        "artifact": "vlm_label_quality",
        "protocol": protocol,
        "method": method,
        "budget": budget,
        "scenario_ids": list(scenario_ids),
        "benchmark": "private Strem Oracle labels for the same frozen window IDs",
        "macro_f1": round(sum(f1_values) / len(f1_values), 6) if f1_values else None,
        "macro_f1_rule": "mean over classes with at least one scored window",
        "schema_valid_rate": round(schema_valid / budget, 6),
        "agreed_positive_pairs": agreed_positives,
        "downstream_gate": {
            "rule": "the downstream arm runs only at 15 or more agreed positive pairs",
            "passed": agreed_positives >= 15,
        },
        "per_class": per_class,
    }
    if run_manifest_path is not None:
        report["run_cost"] = _cost_summary(read_json(run_manifest_path), agreed_positives)

    output_dir.mkdir(parents=True)
    write_report(output_dir / REPORT_FILENAME, report)
    return report


def load_oracle_prefix(
    oracle_dir: Path, method: str, budget: int
) -> tuple[tuple[str, ...], dict[str, JsonObject]]:
    """Read one method's revealed budget prefix, keyed by window ID."""

    oracle_profile = read_json(oracle_dir / "label_profile.json")
    if oracle_profile.get("artifact") != ORACLE_ARTIFACT:
        raise VlmEvaluationError("oracle directory is not an Oracle reveal")
    scenario_ids = tuple(cast(list[str], oracle_profile["scenario_ids"]))
    oracle_rows = {
        cast(str, row["window_id"]): row
        for row in read_jsonl(oracle_dir / "revealed_labels.jsonl")
        if row.get("method") == method and cast(int, row["rank"]) <= budget
    }
    if len(oracle_rows) != budget:
        raise VlmEvaluationError(f"Oracle prefix is incomplete for {method} at {budget}")
    return scenario_ids, oracle_rows


def load_vlm_labels(
    path: Path, oracle_rows: dict[str, JsonObject], scenario_ids: tuple[str, ...]
) -> dict[str, JsonObject]:
    """Read VLM label rows and require one for every Oracle window, in class order."""

    vlm_rows = {cast(str, row["window_id"]): row for row in read_jsonl(path)}
    missing = set(oracle_rows) - set(vlm_rows)
    if missing:
        raise VlmEvaluationError(f"{len(missing)} ranked windows have no VLM label")
    for row in vlm_rows.values():
        if tuple(cast(list[str], row["scenario_ids"])) != scenario_ids:
            raise VlmEvaluationError("VLM and Oracle scenario order differ")
    return vlm_rows


def paired_macro_f1_bootstrap(
    *,
    oracle_rows: dict[str, JsonObject],
    left: dict[str, JsonObject],
    right: dict[str, JsonObject],
    scenario_ids: tuple[str, ...],
    resamples: int = 10_000,
    seed: int = 17,
) -> JsonObject:
    """Bootstrap Macro-F1(left) - Macro-F1(right) by resampling windows jointly.

    Both labelers are scored on the same resampled windows, so the interval is
    paired. F1 counts only Oracle-decided windows with a decided verdict, exactly
    as `_score_class` does; coverage reports how many Oracle-decided windows each
    labeler decided at all, because F1 alone rewards abstaining.
    """

    window_ids = sorted(oracle_rows)
    counts = {
        side: _confusion_arrays(oracle_rows, labels, window_ids, scenario_ids)
        for side, labels in (("left", left), ("right", right))
    }
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(window_ids), size=(resamples, len(window_ids)))

    def macro(side: str, index: np.ndarray[Any, Any] | None) -> np.ndarray[Any, Any]:
        f1s = []
        for tp, fp, fn in counts[side]:
            if index is None:
                totals = [np.asarray([array.sum()]) for array in (tp, fp, fn)]
            else:
                totals = [array[index].sum(axis=1) for array in (tp, fp, fn)]
            denominator = 2 * totals[0] + totals[1] + totals[2]
            f1s.append(np.where(denominator > 0, 2 * totals[0] / np.maximum(denominator, 1), 0.0))
        return cast(np.ndarray[Any, Any], np.mean(f1s, axis=0))

    point = float(macro("left", None)[0] - macro("right", None)[0])
    differences = macro("left", draws) - macro("right", draws)
    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "difference": round(point, 6),
        "ci95": [round(float(low), 6), round(float(high), 6)],
        "resamples": resamples,
        "seed": seed,
        "left_macro_f1": round(float(macro("left", None)[0]), 6),
        "right_macro_f1": round(float(macro("right", None)[0]), 6),
        "left_coverage": _coverage(oracle_rows, left, window_ids, scenario_ids),
        "right_coverage": _coverage(oracle_rows, right, window_ids, scenario_ids),
    }


def _confusion_arrays(
    oracle_rows: dict[str, JsonObject],
    labels: dict[str, JsonObject],
    window_ids: list[str],
    scenario_ids: tuple[str, ...],
) -> list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
    result = []
    for class_index in range(len(scenario_ids)):
        oracle = [cast(list[str], oracle_rows[w]["label_states"])[class_index] for w in window_ids]
        verdict = [cast(list[str], labels[w]["verdicts"])[class_index] for w in window_ids]
        tp = [o == "positive" and v == "positive" for o, v in zip(oracle, verdict, strict=True)]
        fp = [o == "negative" and v == "positive" for o, v in zip(oracle, verdict, strict=True)]
        fn = [o == "positive" and v == "negative" for o, v in zip(oracle, verdict, strict=True)]
        result.append(tuple(np.asarray(values, dtype=np.int64) for values in (tp, fp, fn)))
    return cast(
        list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]], result
    )


def _coverage(
    oracle_rows: dict[str, JsonObject],
    labels: dict[str, JsonObject],
    window_ids: list[str],
    scenario_ids: tuple[str, ...],
) -> float:
    """Mean over classes of decided verdicts on Oracle-decided windows."""

    rates = []
    for class_index in range(len(scenario_ids)):
        decided = [
            w for w in window_ids
            if cast(list[str], oracle_rows[w]["label_states"])[class_index]
            in ("positive", "negative")
        ]
        answered = sum(
            cast(list[str], labels[w]["verdicts"])[class_index] in ("positive", "negative")
            for w in decided
        )
        rates.append(answered / len(decided))
    return round(sum(rates) / len(rates), 6)


def _score_class(
    pairs: list[tuple[JsonObject, JsonObject]],
    class_index: int,
) -> JsonObject:
    """Score one class: decided windows only, with the rest profiled beside them."""

    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    vlm_states = dict.fromkeys((*VERDICTS, "invalid"), 0)
    oracle_states: dict[str, int] = {}
    ambiguous: dict[str, dict[str, int]] = {}
    evidence_out_of_range = 0
    oracle_decided = 0
    undecidable_on_decided = 0

    for oracle_row, vlm_row in pairs:
        oracle_state = cast(list[str], oracle_row["label_states"])[class_index]
        vlm_state = cast(list[str], vlm_row["verdicts"])[class_index]
        if vlm_state not in vlm_states:
            vlm_state = "invalid"
        vlm_states[vlm_state] += 1
        oracle_states[oracle_state] = oracle_states.get(oracle_state, 0) + 1
        if cast(list[bool], vlm_row["evidence_out_of_range"])[class_index]:
            evidence_out_of_range += 1

        if oracle_state not in ("positive", "negative"):
            bucket = ambiguous.setdefault(oracle_state, dict.fromkeys(vlm_states, 0))
            bucket[vlm_state] += 1
            continue

        oracle_decided += 1
        if vlm_state not in ("positive", "negative"):
            undecidable_on_decided += 1
            continue
        if vlm_state == "positive":
            counts["tp" if oracle_state == "positive" else "fp"] += 1
        else:
            counts["fn" if oracle_state == "positive" else "tn"] += 1

    scored = sum(counts.values())
    precision = _ratio(counts["tp"], counts["tp"] + counts["fp"])
    recall = _ratio(counts["tp"], counts["tp"] + counts["fn"])
    # Decided-only recall reads like positive coverage but is not: a positive the
    # VLM abstained on is invisible to it. Both readings are emitted so no
    # downstream quotation can accidentally upgrade one into the other.
    recall_all = _ratio(counts["tp"], oracle_states.get("positive", 0))
    if not scored:
        f1: float | None = None
    elif precision and recall:
        f1 = round(2 * precision * recall / (precision + recall), 6)
    else:
        f1 = 0.0
    return {
        "windows": len(pairs),
        "oracle_decided_windows": oracle_decided,
        "scored_windows": scored,
        "true_positives": counts["tp"],
        "false_positives": counts["fp"],
        "false_negatives": counts["fn"],
        "true_negatives": counts["tn"],
        "precision": precision,
        "recall": recall,
        "recall_rule": "tp over VLM-decided oracle positives only",
        "recall_including_abstentions": recall_all,
        "f1": f1,
        "undecidable_rate_on_decided": _ratio(undecidable_on_decided, oracle_decided),
        "vlm_verdicts": vlm_states,
        "oracle_label_states": oracle_states,
        "ambiguous_oracle_windows": ambiguous,
        "evidence_out_of_range_windows": evidence_out_of_range,
    }


def _cost_summary(manifest: JsonObject, agreed_positives: int) -> JsonObject:
    """Report what the labels cost, including the unit an operator actually plans on."""

    usage = cast(JsonObject, manifest.get("usage", {}))
    # The measured figure prices the tokens the provider reported; the estimate
    # only sized the pre-run ceiling. Reported cost must be the measured one.
    measured = usage.get("measured_cost_usd")
    if measured is None:
        measured = usage.get("estimated_cost_usd", 0.0)
    spend = float(cast(float, measured))
    return {
        "model": manifest.get("identity", {}).get("model"),
        "transport": manifest.get("transport"),
        "requests": usage.get("requests"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "measured_cost_usd": round(spend, 4),
        "estimated_cost_usd": usage.get("estimated_cost_usd"),
        "cost_per_window_usd": _ratio_float(spend, cast(int, usage.get("requests", 0))),
        "cost_per_agreed_positive_usd": _ratio_float(spend, agreed_positives),
        "elapsed_seconds": usage.get("elapsed_seconds"),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _ratio_float(numerator: float, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None
