"""Build the Development-only smoke batch for VLM prompt iteration.

Protocol v0.11 allows prompt editing on TaskDesign and Development only. This
step reads the private Development windows once and splits them into a public
request view and a private Oracle view, so the labeling script itself never
opens a file that carries labels.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, cast

from driving_scene_data_loop.window_dataset import PUBLIC_U_FIELDS

JsonObject = dict[str, Any]
SMOKE_METHOD = "smoke_development"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--ambiguous", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit("output directory must not already exist")

    windows = [
        row
        for row in _read_jsonl(args.windows)
        if row.get("partition") == "development"
    ]
    if not windows:
        raise SystemExit("no Development windows found")
    scenario_ids = tuple(cast(list[str], windows[0]["scenario_ids"]))
    windows.sort(key=lambda row: cast(str, row["window_id"]))
    rng = random.Random(args.seed)

    chosen: dict[str, JsonObject] = {}
    for class_index, _ in enumerate(scenario_ids):
        for state in ("positive", "negative"):
            pool = [
                row
                for row in windows
                if cast(list[str], row["label_states"])[class_index] == state
                and cast(str, row["window_id"]) not in chosen
            ]
            for row in rng.sample(pool, min(args.per_class, len(pool))):
                chosen[cast(str, row["window_id"])] = row
    ambiguous_pool = [
        row
        for row in windows
        if "ignore" in cast(list[str], row["label_states"])
        and cast(str, row["window_id"]) not in chosen
    ]
    for row in rng.sample(ambiguous_pool, min(args.ambiguous, len(ambiguous_pool))):
        chosen[cast(str, row["window_id"])] = row

    selected = [chosen[window_id] for window_id in sorted(chosen)]
    oracle_dir = args.output_dir / "oracle"
    oracle_dir.mkdir(parents=True)

    _write_jsonl(
        args.output_dir / "smoke_windows.jsonl",
        [{field: row[field] for field in PUBLIC_U_FIELDS} for row in selected],
    )
    _write_jsonl(
        args.output_dir / "smoke_ranking.jsonl",
        [
            {
                "method": SMOKE_METHOD,
                "rank": rank,
                "window_id": row["window_id"],
                "scene_token": row["scene_token"],
                "log_token": row["log_token"],
                "start_frame_index": row["start_frame_index"],
            }
            for rank, row in enumerate(selected, start=1)
        ],
    )
    _write_jsonl(
        oracle_dir / "revealed_labels.jsonl",
        [
            {
                "method": SMOKE_METHOD,
                "rank": rank,
                "window_id": row["window_id"],
                "scene_token": row["scene_token"],
                "log_token": row["log_token"],
                "start_frame_index": row["start_frame_index"],
                "labels": list(row["labels"]),
                "loss_mask": list(row["loss_mask"]),
                "label_states": list(row["label_states"]),
                "event_group_ids": [list(group) for group in row["event_group_ids"]],
            }
            for rank, row in enumerate(selected, start=1)
        ],
    )
    profile = {
        "schema_version": "1.0",
        "artifact": "oracle_revealed_labels",
        "protocol": "v0.11-smoke",
        "partition": "development",
        "scenario_ids": list(scenario_ids),
        "window_count": len(selected),
        "seed": args.seed,
        "purpose": "prompt iteration only; never a reported label-quality result",
        "files": {"rows": "revealed_labels.jsonl"},
    }
    (oracle_dir / "label_profile.json").write_text(
        json.dumps(profile, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"windows": len(selected), "method": SMOKE_METHOD}, sort_keys=True))


def _read_jsonl(path: Path) -> list[JsonObject]:
    with path.open(encoding="utf-8") as source:
        return [cast(JsonObject, json.loads(line)) for line in source]


def _write_jsonl(path: Path, rows: list[JsonObject]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
