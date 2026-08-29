"""Freeze Random and integrated Mining public-Pool rankings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driving_scene_data_loop.selection import freeze_selection_rankings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--fn-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = freeze_selection_rankings(
        pool_rows_path=args.pool_dir / "pool_windows.jsonl",
        pool_embeddings_path=args.pool_dir / "pool_embeddings.npy",
        pool_report_path=args.pool_dir / "pool_report.json",
        fn_rows_path=args.fn_dir / "fn_bank.jsonl",
        fn_embeddings_path=args.fn_dir / "fn_embeddings.npy",
        output_dir=args.output_dir,
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
