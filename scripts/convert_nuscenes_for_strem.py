"""Convert complete nuScenes metadata into private Strem scene streams."""

import argparse
import json
from pathlib import Path

from driving_scene_data_loop.strem_converter import (
    StremConverterError,
    load_numeric_id_map,
    run_nuscenes_converter,
)


def parse_args() -> argparse.Namespace:
    """Parse one complete conversion request."""

    parser = argparse.ArgumentParser(
        description="Run the pinned external converter on all nuScenes scenes."
    )
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--converter", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run conversion and print a small machine-readable summary."""

    args = parse_args()
    try:
        result = run_nuscenes_converter(
            args.metadata_root,
            args.output_dir,
            args.converter,
        )
        instance_count = len(load_numeric_id_map(result.numeric_id_map_path))
    except StremConverterError as error:
        raise SystemExit(f"Strem conversion error: {error}") from error

    print(
        json.dumps(
            {
                "numeric_id_map_ref": str(result.numeric_id_map_path),
                "output_dir_ref": str(result.output_dir),
                "scene_streams": result.stream_count,
                "tracked_instances": instance_count,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
