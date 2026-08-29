"""Freeze extra Random public-Pool batches for selection-variance estimation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driving_scene_data_loop.selection import freeze_random_variance_rankings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = freeze_random_variance_rankings(
        pool_rows_path=args.pool_dir / "pool_windows.jsonl",
        pool_report_path=args.pool_dir / "pool_report.json",
        output_dir=args.output_dir,
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
