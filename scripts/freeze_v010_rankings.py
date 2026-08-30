"""Freeze the pre-registered v0.10 rankings on Pool2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driving_scene_data_loop.selection import freeze_v010_rankings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = freeze_v010_rankings(
        pool_rows_path=args.pool2_dir / "pool2_windows.jsonl",
        pool_report_path=args.pool2_dir / "pool2_report.json",
        output_dir=args.output_dir,
    )
    print(json.dumps({m: report["prefixes"][m]["900"]["unique_windows"]
                      for m in report["files"]}, sort_keys=True))


if __name__ == "__main__":
    main()
