"""Carve Test2 out of U and write the Pool2 public view, per the frozen v0.10 rules.

Test2 logs come from a deterministic hash order; the guard extends the log
count until the corridor-event floor is met, exactly the rule the learning
curve used. Reading event-support counts for this guard is the same
metadata-level access Gate A used for cross-partition support. Windows whose
labels were already revealed in v0.8 are excluded from both sides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-windows", type=Path, required=True)
    parser.add_argument("--public-windows", type=Path, required=True)
    parser.add_argument("--revealed", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test2-logs", type=int, default=8)
    parser.add_argument("--min-corridor-events", type=int, default=55)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit("output directory must not already exist")

    revealed_ids: set[str] = set()
    for path in args.revealed:
        for line in path.open(encoding="utf-8"):
            revealed_ids.add(json.loads(line)["window_id"])

    u_rows: list[dict[str, object]] = []
    corridor_events: dict[str, set[str]] = {}
    for line in args.private_windows.open(encoding="utf-8"):
        row = json.loads(line)
        if row["partition"] != "u":
            continue
        u_rows.append(row)
        corridor = row["scenario_ids"].index("vehicle_relative_corridor_entry")
        bucket = corridor_events.setdefault(row["log_token"], set())
        bucket.update(row["event_group_ids"][corridor])

    logs = sorted(
        corridor_events,
        key=lambda t: hashlib.sha256(f"test2-v010\0{t}".encode()).hexdigest(),
    )
    count = args.test2_logs
    while True:
        test2_logs = set(logs[:count])
        events = len(set().union(*(corridor_events[t] for t in test2_logs)))
        if events >= args.min_corridor_events or count >= len(logs):
            break
        count += 1

    test2_eval_ids = [
        str(row["window_id"])
        for row in u_rows
        if row["log_token"] in test2_logs and row["window_id"] not in revealed_ids
    ]

    args.output_dir.mkdir(parents=True)
    pool2_count = 0
    with (args.output_dir / "pool2_windows.jsonl").open("w", encoding="utf-8") as out:
        for line in args.public_windows.open(encoding="utf-8"):
            row = json.loads(line)
            if row["log_token"] in test2_logs or row["window_id"] in revealed_ids:
                continue
            out.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            pool2_count += 1
    (args.output_dir / "test2_eval_ids.txt").write_text(
        "\n".join(test2_eval_ids) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "1.0",
        "artifact": "v010_partitions",
        "hash_salt": "test2-v010",
        "test2_log_count": count,
        "test2_logs_hash_ordered": logs[:count],
        "test2_corridor_events": events,
        "test2_eval_windows": len(test2_eval_ids),
        "excluded_previously_revealed": len(revealed_ids),
        "pool2_windows": pool2_count,
        "guard": f"corridor events >= {args.min_corridor_events}, extend by one log",
    }
    (args.output_dir / "v010_partitions_manifest.json").write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in ("test2_logs_hash_ordered",)}, sort_keys=True))


if __name__ == "__main__":
    main()
