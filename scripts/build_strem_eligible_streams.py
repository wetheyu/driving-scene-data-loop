"""Create private CAM_FRONT-eligible Strem streams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nuscenes.nuscenes import NuScenes  # type: ignore[import-untyped]

from driving_scene_data_loop.strem_eligibility import build_cam_front_eligible_streams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True, type=Path)
    parser.add_argument("--raw-stream-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nusc = NuScenes(
        version="v1.0-trainval",
        dataroot=str(args.dataroot),
        verbose=False,
    )
    result = build_cam_front_eligible_streams(
        nusc,
        args.raw_stream_dir,
        args.output_dir,
    )
    print(json.dumps(result.to_dict(), allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
