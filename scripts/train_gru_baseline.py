"""Train the three-seed Global-GRU baseline on L0 and Development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driving_scene_data_loop.gru_baseline import train_gru_baselines
from driving_scene_data_loop.lr_baselines import (
    load_baseline_windows,
    load_frame_features,
    validate_formal_feature_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--scenario-id", action="append", dest="scenario_ids")
    args = parser.parse_args()

    window_data = load_baseline_windows(
        args.windows,
        args.feature_dir / "frame_index.jsonl",
    )
    frame_count = validate_formal_feature_manifest(args.feature_dir / "feature_manifest.json")
    frame_features = load_frame_features(
        args.feature_dir / "frame_features.npy",
        frame_count,
    )
    report = train_gru_baselines(
        window_data=window_data,
        frame_features=frame_features,
        output_dir=args.output_dir,
        num_threads=args.num_threads,
        scenario_ids=tuple(args.scenario_ids) if args.scenario_ids else None,
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
