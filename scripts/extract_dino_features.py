"""Extract the frozen DINOv2-Small feature cache for model windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driving_scene_data_loop.dino_features import extract_feature_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument(
        "--pooling",
        choices=["cls", "patch_max"],
        default="cls",
        help="cls pools the CLS token; patch_max pools the encoder's patch tokens.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    manifest = extract_feature_cache(
        windows_path=args.windows,
        media_root=args.media_root,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        pooling=args.pooling,
        limit=args.limit,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(manifest, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
