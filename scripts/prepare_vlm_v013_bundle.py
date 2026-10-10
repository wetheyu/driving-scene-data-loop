"""Build the protocol v0.13 upload bundle for the rented GPU host.

Reads the private windows (for L0 targets and the Development sanity draw) and
the frozen v0.10 ranking plus the public Pool2 rows (for the 900 scored
windows). Only L0 targets and public rows enter the bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from driving_scene_data_loop.json_files import read_jsonl
from driving_scene_data_loop.vlm_finetune import build_bundle, public_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-windows", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--pool2-windows", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=900)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--host-root", default="/root/autodl-tmp/v013")
    parser.add_argument(
        "--model-path", default="/root/autodl-tmp/models/Qwen3-VL-8B-Instruct"
    )
    parser.add_argument("--lf-template", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    ranked = [row for row in read_jsonl(args.ranking) if cast(int, row["rank"]) <= args.budget]
    if [row["rank"] for row in ranked] != list(range(1, args.budget + 1)):
        raise SystemExit("ranking prefix is not contiguously ranked")
    wanted = [cast(str, row["window_id"]) for row in ranked]
    pool2 = {cast(str, row["window_id"]): row for row in read_jsonl(args.pool2_windows)}
    if set(wanted) - set(pool2):
        raise SystemExit("ranked windows are missing from the public Pool2 rows")
    scored_rows = [public_row(pool2[window_id]) for window_id in sorted(wanted)]

    manifest = build_bundle(
        private_rows=read_jsonl(args.private_windows),
        scored_rows=scored_rows,
        media_root=args.media_root,
        host_root=args.host_root,
        model_path=args.model_path,
        lf_template=args.lf_template,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
