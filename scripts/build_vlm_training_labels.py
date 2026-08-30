"""Turn frozen VLM replies into the label file a feedback arm can train on.

Reads only the VLM replies and the frozen ranking, so the training label file is
produced without the Oracle ever being opened -- which is what makes the arm a
real automatic-labeling pipeline rather than an Oracle-filtered one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from driving_scene_data_loop.vlm_labeling import (
    SCENARIO_DESCRIPTIONS,
    write_label_files,
)

JsonObject = dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-labels", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    ranking_rows = [
        row
        for row in _read_jsonl(args.ranking)
        if cast(int, row["rank"]) <= args.budget
    ]
    if [row["rank"] for row in ranking_rows] != list(range(1, args.budget + 1)):
        raise SystemExit("ranking prefix is not contiguously ranked")
    methods = {cast(str, row["method"]) for row in ranking_rows}
    if len(methods) != 1:
        raise SystemExit("ranking mixes selection methods")

    profile = write_label_files(
        label_rows=_read_jsonl(args.vlm_labels),
        ranking_rows=ranking_rows,
        scenario_ids=tuple(sorted(SCENARIO_DESCRIPTIONS)),
        method=methods.pop(),
        output_dir=args.output_dir,
    )
    print(json.dumps(profile, allow_nan=False, sort_keys=True))


def _read_jsonl(path: Path) -> list[JsonObject]:
    with path.open(encoding="utf-8") as source:
        return [cast(JsonObject, json.loads(line)) for line in source]


if __name__ == "__main__":
    main()
