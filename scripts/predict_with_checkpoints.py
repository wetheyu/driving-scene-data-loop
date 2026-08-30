"""Write per-seed predictions for one trained run on one partition.

The final evaluation is two stages: this script writes frozen prediction files
(window ID and probabilities only -- no labels), and score_predictions.py later
joins them with labels. Dry-run on development first: the scores it produces
there must reproduce the run's own training report before frozen_test is ever
named.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch  # type: ignore[import-not-found,unused-ignore]

from driving_scene_data_loop.gru_baseline import GlobalGRU, build_sequence_features
from driving_scene_data_loop.lr_baselines import (
    load_frame_features,
    validate_formal_feature_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument(
        "--partition", choices=["development", "frozen_test", "u"], required=True
    )
    parser.add_argument(
        "--window-list",
        type=Path,
        help="Restrict to these window IDs (one per line). Required for u, so a "
        "Test2 evaluation cannot silently widen to the whole Pool.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-threads", type=int, default=16)
    args = parser.parse_args()

    if args.partition == "u" and args.window_list is None:
        raise SystemExit("--window-list is required with --partition u")
    keep_ids: set[str] | None = None
    if args.window_list is not None:
        keep_ids = {
            line.strip()
            for line in args.window_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    if args.output_dir.exists():
        raise SystemExit("output directory must not already exist")
    torch.set_num_threads(args.num_threads)

    feature_cache = validate_formal_feature_manifest(args.feature_dir / "feature_manifest.json")
    frame_features = load_frame_features(
        args.feature_dir / "frame_features.npy",
        int(cast(int, feature_cache["frame_count"])),
    )
    feature_index: dict[str, int] = {}
    for line in (args.feature_dir / "frame_index.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        feature_index[row["media_ref"]] = row["feature_index"]

    window_ids: list[str] = []
    frame_rows: list[list[int]] = []
    with args.windows.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["partition"] != args.partition:
                continue
            if keep_ids is not None and row["window_id"] not in keep_ids:
                continue
            window_ids.append(row["window_id"])
            frame_rows.append([feature_index[ref] for ref in row["frame_refs"]])
    if not window_ids:
        raise SystemExit(f"no windows in partition {args.partition!r}")
    if keep_ids is not None and len(window_ids) != len(keep_ids):
        raise SystemExit(
            f"window list names {len(keep_ids)} windows but {len(window_ids)} were found"
        )

    sequences = build_sequence_features(
        frame_features,
        np.asarray(frame_rows, dtype=np.int64),
    )
    inputs = torch.from_numpy(sequences)

    train_report = json.loads((args.run_dir / "gru_report.json").read_text(encoding="utf-8"))
    checkpoints = sorted(args.run_dir.glob("gru_seed_*.pt"))
    if not checkpoints:
        raise SystemExit(f"no checkpoints in {args.run_dir}")

    args.output_dir.mkdir(parents=True)
    seeds: list[str] = []
    for checkpoint in checkpoints:
        seed = checkpoint.stem.removeprefix("gru_seed_")
        seeds.append(seed)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        hidden_size = int(state["classifier.weight"].shape[1])
        class_count = int(state["classifier.weight"].shape[0])
        model = GlobalGRU(class_count, hidden_size)
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(inputs)).numpy()
        with (args.output_dir / f"predictions_seed_{seed}.jsonl").open(
            "w", encoding="utf-8"
        ) as destination:
            for window_id, values in zip(window_ids, probabilities, strict=True):
                destination.write(
                    json.dumps(
                        {
                            "window_id": window_id,
                            "partition": args.partition,
                            "probabilities": [float(v) for v in values],
                        },
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    manifest = {
        "schema_version": "1.0",
        "artifact": "frozen_partition_predictions",
        "run_dir": args.run_dir.name,
        "run_name": train_report.get("run_name"),
        "training_config": train_report.get("training_config"),
        "feature_cache": feature_cache,
        "partition": args.partition,
        "window_count": len(window_ids),
        "seeds": seeds,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"run": args.run_dir.name, "partition": args.partition,
                      "windows": len(window_ids), "seeds": len(seeds)}))


if __name__ == "__main__":
    main()
