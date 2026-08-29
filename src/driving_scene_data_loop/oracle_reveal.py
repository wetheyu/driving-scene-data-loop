"""Join frozen selection IDs to private Oracle labels and profile the batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from driving_scene_data_loop.selection import BUDGETS

JsonObject = dict[str, Any]

LABEL_STATES = ("positive", "negative", "ignore", "invalid")
POOL_PARTITION = "u"
REVEALED_FILENAME = "revealed_labels.jsonl"


class OracleRevealError(ValueError):
    """Raised when a reveal would leave the frozen selection or lose budget."""


def reveal_selected_labels(
    *,
    ranking_paths: tuple[Path, ...],
    windows_path: Path,
    output_dir: Path,
    budgets: tuple[int, ...] = BUDGETS,
) -> JsonObject:
    """Reveal Oracle labels for frozen selected IDs and report what they bought.

    Only window IDs present in the frozen rankings are read out of the private
    window file, and every one of them must belong to the Simulated Unlabeled
    Pool. `ignore` and `invalid` stay in the batch and keep consuming budget, so
    a method's window count always equals its budget exactly.
    """

    if output_dir.exists():
        raise OracleRevealError("output directory must not already exist")
    if not budgets or tuple(sorted(set(budgets))) != budgets or budgets[0] <= 0:
        raise OracleRevealError("budgets must be positive, unique, and increasing")
    if not ranking_paths:
        raise OracleRevealError("at least one frozen ranking is required")

    rankings = _read_rankings(ranking_paths, budgets[-1])
    selected_ids = {
        cast(str, row["window_id"]) for rows in rankings.values() for row in rows
    }
    scenario_ids, oracle = _read_selected_windows(windows_path, selected_ids)

    revealed: list[JsonObject] = []
    for method in sorted(rankings):
        for row in rankings[method]:
            window = oracle[cast(str, row["window_id"])]
            revealed.append(
                {
                    "method": method,
                    "rank": row["rank"],
                    "window_id": row["window_id"],
                    "scene_token": window["scene_token"],
                    "log_token": window["log_token"],
                    "start_frame_index": window["start_frame_index"],
                    "labels": list(window["labels"]),
                    "loss_mask": list(window["loss_mask"]),
                    "label_states": list(window["label_states"]),
                    "event_group_ids": [
                        list(group) for group in window["event_group_ids"]
                    ],
                }
            )

    report: JsonObject = {
        "schema_version": "1.0",
        "artifact": "oracle_revealed_labels",
        "private_windows_run": windows_path.parent.name,
        "scenario_ids": list(scenario_ids),
        "label_order": "labels, loss_mask, label_states, and event_group_ids follow scenario_ids",
        "budgets": list(budgets),
        "primary_budget": 300 if 300 in budgets else budgets[0],
        "budget_rule": "ignore and invalid consume budget and are never replaced",
        "revealed_window_count": len(selected_ids),
        "methods": {
            method: {
                "ranked_count": len(rows),
                "budgets": {
                    str(budget): _budget_profile(
                        [
                            item
                            for item in revealed
                            if item["method"] == method
                            and cast(int, item["rank"]) <= budget
                        ],
                        scenario_ids,
                    )
                    for budget in budgets
                },
            }
            for method, rows in sorted(rankings.items())
        },
        "files": {"rows": REVEALED_FILENAME},
    }

    output_dir.mkdir(parents=True)
    with (output_dir / REVEALED_FILENAME).open("w", encoding="utf-8") as destination:
        for row in revealed:
            destination.write(
                json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"
            )
    (output_dir / "label_profile.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _budget_profile(
    rows: list[JsonObject],
    scenario_ids: tuple[str, ...],
) -> JsonObject:
    """Summarize what one budget prefix bought, per class and overall."""

    per_class: JsonObject = {}
    for class_index, scenario_id in enumerate(scenario_ids):
        states = {
            state: sum(row["label_states"][class_index] == state for row in rows)
            for state in LABEL_STATES
        }
        events: set[str] = set()
        for row in rows:
            events.update(row["event_group_ids"][class_index])
        usable = states["positive"] + states["negative"]
        per_class[scenario_id] = {
            "label_states": states,
            "usable_labels": usable,
            "positive_yield": states["positive"] / len(rows) if rows else 0.0,
            "distinct_positive_event_groups": len(events),
        }

    usable_total = sum(
        int(mask) for row in rows for mask in cast(list[bool], row["loss_mask"])
    )
    return {
        "windows": len(rows),
        "unique_windows": len({row["window_id"] for row in rows}),
        "log_coverage": len({row["log_token"] for row in rows}),
        "scene_coverage": len({row["scene_token"] for row in rows}),
        "usable_labels": usable_total,
        "maximum_possible_labels": len(rows) * len(scenario_ids),
        "per_class": per_class,
    }


def _read_rankings(
    ranking_paths: tuple[Path, ...],
    maximum_budget: int,
) -> dict[str, list[JsonObject]]:
    """Load each frozen ranking and check it is intact before any label is read."""

    rankings: dict[str, list[JsonObject]] = {}
    for path in ranking_paths:
        rows = [
            cast(JsonObject, json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        methods = {cast(str, row["method"]) for row in rows}
        if len(methods) != 1:
            raise OracleRevealError(f"ranking mixes selection methods: {path}")
        method = methods.pop()
        if method in rankings:
            raise OracleRevealError(f"duplicate ranking for method {method}")
        if [row["rank"] for row in rows] != list(range(1, len(rows) + 1)):
            raise OracleRevealError(f"ranking is not contiguously ranked: {path}")
        if len({row["window_id"] for row in rows}) != len(rows):
            raise OracleRevealError(f"ranking repeats a window: {path}")
        if len(rows) < maximum_budget:
            raise OracleRevealError(f"ranking is shorter than the largest budget: {path}")
        rankings[method] = rows
    return rankings


def _read_selected_windows(
    windows_path: Path,
    selected_ids: set[str],
) -> tuple[tuple[str, ...], dict[str, JsonObject]]:
    """Read only the selected private windows, leaving the rest of the Pool hidden."""

    scenario_ids: tuple[str, ...] | None = None
    oracle: dict[str, JsonObject] = {}
    with windows_path.open(encoding="utf-8") as source:
        for line in source:
            window = cast(JsonObject, json.loads(line))
            window_id = cast(str, window["window_id"])
            if window_id not in selected_ids:
                continue
            if window_id in oracle:
                raise OracleRevealError(f"private windows repeat {window_id}")
            if window["partition"] != POOL_PARTITION:
                raise OracleRevealError(
                    f"selected window is outside the Pool: {window_id}"
                )
            row_scenarios = tuple(cast(list[str], window["scenario_ids"]))
            if scenario_ids is None:
                scenario_ids = row_scenarios
            elif row_scenarios != scenario_ids:
                raise OracleRevealError(
                    f"window declares a different class order: {window_id}"
                )
            oracle[window_id] = window

    missing = selected_ids - set(oracle)
    if missing:
        raise OracleRevealError(f"{len(missing)} selected IDs are not in the Pool")
    if scenario_ids is None:
        raise OracleRevealError("no selected window was found")
    return scenario_ids, oracle
