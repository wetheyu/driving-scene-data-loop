"""Build the frozen whole-log nuScenes trainval split manifest."""

import argparse
import json
from pathlib import Path

from nuscenes.nuscenes import NuScenes  # type: ignore[import-untyped]

from driving_scene_data_loop.splits import (
    SplitBuildError,
    build_nuscenes_trainval_split,
)


def parse_args() -> argparse.Namespace:
    """Parse one bounded split-manifest request."""

    parser = argparse.ArgumentParser(
        description="Build split-v1 without reading nuScenes camera media."
    )
    parser.add_argument(
        "--dataroot",
        required=True,
        type=Path,
        help="nuScenes root containing the v1.0-trainval metadata directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New JSON manifest path; existing files are not overwritten.",
    )
    return parser.parse_args()


def main() -> None:
    """Build, persist, and summarize one manifest."""

    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"split error: output already exists: {args.output}")
    try:
        nusc = NuScenes(
            version="v1.0-trainval",
            dataroot=str(args.dataroot),
            verbose=False,
        )
        manifest = build_nuscenes_trainval_split(nusc)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(manifest.to_json() + "\n", encoding="utf-8")
    except (OSError, SplitBuildError) as error:
        raise SystemExit(f"split error: {error}") from error

    print(
        json.dumps(
            {
                "log_counts": manifest.log_counts(),
                "manifest_sha256": manifest.manifest_sha256,
                "output_ref": str(args.output),
                "scene_counts": manifest.scene_counts(),
                "split_version": manifest.split_version,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
