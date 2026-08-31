"""Seed-paired Development contrasts from training reports, for D2/D3 reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driving_scene_data_loop.prediction_scoring import development_paired_contrasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        required=True,
        metavar="NAME=REPORT",
        help="Repeatable NAME=path/to/gru_report.json",
    )
    parser.add_argument(
        "--contrast",
        action="append",
        dest="contrasts",
        required=True,
        metavar="LEFT-RIGHT",
        help="Repeatable LEFT-RIGHT over the declared run names.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit("output file must not already exist")
    run_reports: dict[str, Path] = {}
    for item in args.runs:
        name, _, path = item.partition("=")
        if not name or not path:
            raise SystemExit(f"--run needs NAME=REPORT, got {item!r}")
        run_reports[name] = Path(path)
    contrasts = []
    for item in args.contrasts:
        left, _, right = item.partition("-")
        if not left or not right:
            raise SystemExit(f"--contrast needs LEFT-RIGHT, got {item!r}")
        contrasts.append((left, right))

    document = development_paired_contrasts(run_reports, tuple(contrasts))
    args.output.write_text(
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(document, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
