"""Flip decided Oracle labels at a declared rate for the D2 noise diagnostic.

Reads one method's budget prefix from an existing Oracle reveal and flips each
decided label (positive <-> negative) independently with the given probability,
deterministically per (window_id, class). Masked entries are never touched: the
diagnostic measures label noise, not budget or masking changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

JsonObject = dict[str, Any]


def flip_decides(rate_percent: int, seed: int, window_id: str, class_index: int) -> bool:
    """Deterministic per-label coin: stable across runs, machines, and orderings."""

    digest = hashlib.sha256(
        f"noise-{seed}-{rate_percent}\0{window_id}\0{class_index}".encode()
    ).digest()
    draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return draw < rate_percent / 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--method", default="disagreement_v010")
    parser.add_argument("--budget", type=int, default=900)
    parser.add_argument("--rate-percent", type=int, required=True, choices=(10, 20, 30))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit("output directory must not already exist")
    profile = cast(
        JsonObject,
        json.loads((args.oracle_dir / "label_profile.json").read_text(encoding="utf-8")),
    )
    if profile.get("artifact") != "oracle_revealed_labels":
        raise SystemExit("oracle directory is not an Oracle reveal")
    scenario_ids = cast(list[str], profile["scenario_ids"])

    rows: list[JsonObject] = []
    flipped = 0
    decided = 0
    with (args.oracle_dir / "revealed_labels.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = cast(JsonObject, json.loads(line))
            if row.get("method") != args.method or cast(int, row["rank"]) > args.budget:
                continue
            labels = list(cast(list[Any], row["labels"]))
            states = list(cast(list[str], row["label_states"]))
            for class_index, masked in enumerate(cast(list[bool], row["loss_mask"])):
                if not masked:
                    continue
                decided += 1
                if flip_decides(
                    args.rate_percent, args.seed, cast(str, row["window_id"]), class_index
                ):
                    flipped += 1
                    labels[class_index] = 1 - cast(int, labels[class_index])
                    states[class_index] = (
                        "positive" if states[class_index] == "negative" else "negative"
                    )
            rows.append({**row, "labels": labels, "label_states": states})

    if [cast(int, row["rank"]) for row in rows] != list(range(1, args.budget + 1)):
        raise SystemExit("reveal prefix is incomplete for the requested method and budget")

    report: JsonObject = {
        "schema_version": "1.0",
        "artifact": "oracle_noise_injected",
        "protocol": "v0.12-D2",
        "source_artifact": "oracle_revealed_labels",
        "method": args.method,
        "budget": args.budget,
        "scenario_ids": scenario_ids,
        "label_order": profile["label_order"],
        "flip_rule": "each decided label flips independently; masked entries untouched",
        "rate_percent": args.rate_percent,
        "flip_seed": args.seed,
        "decided_labels": decided,
        "flipped_labels": flipped,
        "observed_flip_rate": round(flipped / decided, 6) if decided else None,
        "files": {"rows": "revealed_labels.jsonl"},
    }
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "revealed_labels.jsonl").open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    (args.output_dir / "label_profile.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
