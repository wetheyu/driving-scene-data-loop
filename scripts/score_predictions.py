"""Score frozen prediction directories against one partition's labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from driving_scene_data_loop.prediction_scoring import score_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--partition", choices=["development", "frozen_test"], required=True)
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        required=True,
        metavar="NAME=PREDICTIONS_DIR",
        help="Repeatable. Names appear in the report and its paired contrasts.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs: dict[str, Path] = {}
    for value in args.runs:
        name, _, directory = value.partition("=")
        if not name or not directory:
            raise SystemExit(f"--run must look like NAME=DIR, got {value!r}")
        runs[name] = Path(directory)

    report = score_runs(
        windows_path=args.windows,
        partition=args.partition,
        runs=runs,
        output_path=args.output,
    )
    runs_summary = cast(dict[str, dict[str, object]], report["runs"])
    print(json.dumps({name: run["macro_average_precision_mean"]
                      for name, run in runs_summary.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
