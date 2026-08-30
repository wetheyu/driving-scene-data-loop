"""Score frozen VLM labels against the revealed Oracle labels for the same IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driving_scene_data_loop.vlm_evaluation import evaluate_vlm_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-labels", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--method", default="disagreement_v010")
    parser.add_argument("--budget", type=int, default=900)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = evaluate_vlm_labels(
        vlm_labels_path=args.vlm_labels,
        oracle_dir=args.oracle_dir,
        method=args.method,
        budget=args.budget,
        output_dir=args.output_dir,
        run_manifest_path=args.run_manifest,
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
