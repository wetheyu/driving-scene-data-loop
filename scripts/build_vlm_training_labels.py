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

from driving_scene_data_loop.json_files import read_jsonl
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
    parser.add_argument(
        "--protocol",
        choices=("v0.11", "v0.13", "d7-seedlogs"),
        default="v0.11",
        help="Which labeler produced the replies; the training arm checks it.",
    )
    args = parser.parse_args()

    ranking_rows = [
        row
        for row in read_jsonl(args.ranking)
        if cast(int, row["rank"]) <= args.budget
    ]
    if [row["rank"] for row in ranking_rows] != list(range(1, args.budget + 1)):
        raise SystemExit("ranking prefix is not contiguously ranked")
    methods = {cast(str, row["method"]) for row in ranking_rows}
    if len(methods) != 1:
        raise SystemExit("ranking mixes selection methods")

    profile = write_label_files(
        label_rows=read_jsonl(args.vlm_labels),
        ranking_rows=ranking_rows,
        scenario_ids=tuple(sorted(SCENARIO_DESCRIPTIONS)),
        method=methods.pop(),
        protocol=args.protocol,
        output_dir=args.output_dir,
    )
    print(json.dumps(profile, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
