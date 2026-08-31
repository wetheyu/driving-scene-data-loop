"""Score frozen VLM labels against the already-revealed Oracle labels.

The Oracle stays the private metric benchmark: a disagreement is reported as a
disagreement, never resolved in the VLM's favour. Windows the Oracle calls
`ignore` or `invalid` are the definitionally ambiguous region and are profiled
separately instead of being counted as VLM errors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

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
) -> JsonObject:
    """Join one budget prefix of a method to the Oracle and score the labels."""

    if output_dir.exists():
        raise VlmEvaluationError("output directory must not already exist")

    oracle_profile = _read_json(oracle_dir / "label_profile.json")
    if oracle_profile.get("artifact") != ORACLE_ARTIFACT:
        raise VlmEvaluationError("oracle directory is not an Oracle reveal")
    scenario_ids = tuple(cast(list[str], oracle_profile["scenario_ids"]))

    oracle_rows = {
        cast(str, row["window_id"]): row
        for row in _read_jsonl(oracle_dir / "revealed_labels.jsonl")
        if row.get("method") == method and cast(int, row["rank"]) <= budget
    }
    if len(oracle_rows) != budget:
        raise VlmEvaluationError(f"Oracle prefix is incomplete for {method} at {budget}")

    vlm_rows = {cast(str, row["window_id"]): row for row in _read_jsonl(vlm_labels_path)}
    missing = set(oracle_rows) - set(vlm_rows)
    if missing:
        raise VlmEvaluationError(f"{len(missing)} ranked windows have no VLM label")
    for row in vlm_rows.values():
        if tuple(cast(list[str], row["scenario_ids"])) != scenario_ids:
            raise VlmEvaluationError("VLM and Oracle scenario order differ")

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
        "protocol": "v0.11",
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
        report["run_cost"] = _cost_summary(_read_json(run_manifest_path), agreed_positives)

    output_dir.mkdir(parents=True)
    (output_dir / REPORT_FILENAME).write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


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


def _read_json(path: Path) -> JsonObject:
    return cast(JsonObject, json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[JsonObject]:
    with path.open(encoding="utf-8") as source:
        return [cast(JsonObject, json.loads(line)) for line in source]
