"""Reveal private Oracle labels for the already-frozen selected window IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driving_scene_data_loop.oracle_reveal import reveal_selected_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranking",
        type=Path,
        action="append",
        dest="rankings",
        required=True,
        help="One frozen ranked JSONL; repeat for every method being revealed.",
    )
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = reveal_selected_labels(
        ranking_paths=tuple(args.rankings),
        windows_path=args.windows,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
