"""Build the private Development FN bank from the three-class Base run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driving_scene_data_loop.false_negative_bank import (
    build_false_negative_bank,
    load_feature_index,
)
from driving_scene_data_loop.lr_baselines import (
    load_frame_features,
    validate_formal_feature_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.feature_dir / "feature_manifest.json"
    frame_count = validate_formal_feature_manifest(manifest_path)
    feature_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = build_false_negative_bank(
        windows_path=args.windows,
        predictions_path=args.predictions,
        base_report_path=args.base_report,
        frame_features=load_frame_features(
            args.feature_dir / "frame_features.npy",
            frame_count,
        ),
        feature_index=load_feature_index(args.feature_dir / "frame_index.jsonl"),
        feature_model_id=feature_manifest["model_id"],
        feature_model_revision=feature_manifest["model_revision"],
        output_dir=args.output_dir,
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
