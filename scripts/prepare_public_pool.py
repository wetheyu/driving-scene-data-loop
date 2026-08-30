"""Run the three-class Base on public Pool windows and cache selection vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from driving_scene_data_loop.false_negative_bank import load_feature_index
from driving_scene_data_loop.lr_baselines import (
    load_frame_features,
    validate_formal_feature_manifest,
)
from driving_scene_data_loop.public_pool import prepare_public_pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-windows", type=Path, required=True)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    manifest_path = args.feature_dir / "feature_manifest.json"
    feature_cache = validate_formal_feature_manifest(manifest_path)
    frame_count = int(cast(int, feature_cache["frame_count"]))
    feature_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = prepare_public_pool(
        public_windows_path=args.public_windows,
        base_report_path=args.base_report,
        frame_features=load_frame_features(
            args.feature_dir / "frame_features.npy",
            frame_count,
        ),
        feature_index=load_feature_index(args.feature_dir / "frame_index.jsonl"),
        feature_model_id=feature_manifest["model_id"],
        feature_model_revision=feature_manifest["model_revision"],
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
