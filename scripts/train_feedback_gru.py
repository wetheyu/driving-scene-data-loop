"""Train one frozen Oracle-feedback GRU arm on L0 plus selected Pool windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from driving_scene_data_loop.feedback_retraining import (
    DEVELOPMENT_INFORMED_ARMS,
    ORACLE_ARMS,
    V010_ARMS,
    get_oracle_arm,
    load_feedback_windows,
)
from driving_scene_data_loop.gru_baseline import (
    TRAINING_CONFIGS,
    get_training_config,
    train_gru_baselines,
)
from driving_scene_data_loop.lr_baselines import (
    load_frame_features,
    validate_formal_feature_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=[arm.name for arm in ORACLE_ARMS + DEVELOPMENT_INFORMED_ARMS + V010_ARMS],
        required=True,
    )
    parser.add_argument("--private-windows", type=Path, required=True)
    parser.add_argument("--public-windows", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument(
        "--training-config",
        choices=sorted(TRAINING_CONFIGS),
        default="v1-frozen",
    )
    parser.add_argument(
        "--warm-start-dir",
        type=Path,
        help="Base run to resume from; selects incremental fine-tuning.",
    )
    args = parser.parse_args()

    arm = get_oracle_arm(args.arm)
    window_data = load_feedback_windows(
        private_windows_path=args.private_windows,
        public_pool_windows_path=args.public_windows,
        revealed_labels_path=args.oracle_dir / "revealed_labels.jsonl",
        reveal_profile_path=args.oracle_dir / "label_profile.json",
        frame_index_path=args.feature_dir / "frame_index.jsonl",
        arm=arm,
    )
    feature_cache = validate_formal_feature_manifest(
        args.feature_dir / "feature_manifest.json"
    )
    frame_count = int(cast(int, feature_cache["frame_count"]))
    frame_features = load_frame_features(
        args.feature_dir / "frame_features.npy",
        frame_count,
    )
    report = train_gru_baselines(
        window_data=window_data,
        frame_features=frame_features,
        output_dir=args.output_dir,
        num_threads=args.num_threads,
        config=get_training_config(args.training_config),
        warm_start_dir=args.warm_start_dir,
        feature_cache=feature_cache,
        run_name=arm.name,
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
