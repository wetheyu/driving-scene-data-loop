"""Train LastFrame-LR and Mean5-LR on L0 and score Development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driving_scene_data_loop.lr_baselines import (
    load_baseline_windows,
    load_frame_features,
    train_lr_baselines,
    validate_formal_feature_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
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
    report = train_lr_baselines(
        window_data=window_data,
        frame_features=frame_features,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
