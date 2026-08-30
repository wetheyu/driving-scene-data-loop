"""Run the Base-small ensemble on Pool2: per-window probability mean and std."""

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
from driving_scene_data_loop.window_dataset import PUBLIC_U_FIELDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool2-windows", type=Path, required=True)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-threads", type=int, default=16)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit("output directory must not already exist")
    torch.set_num_threads(args.num_threads)

    feature_cache = validate_formal_feature_manifest(args.feature_dir / "feature_manifest.json")
    features = load_frame_features(
        args.feature_dir / "frame_features.npy",
        int(cast(int, feature_cache["frame_count"])),
    )
    feature_index: dict[str, int] = {}
    for line in (args.feature_dir / "frame_index.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        feature_index[row["media_ref"]] = row["feature_index"]

    windows: list[dict[str, object]] = []
    seen: set[str] = set()
    for line in args.pool2_windows.open(encoding="utf-8"):
        row = json.loads(line)
        if set(row) != set(PUBLIC_U_FIELDS) or row["partition"] != "u":
            raise SystemExit("pool2 rows must carry exactly the public fields")
        if row["window_id"] in seen:
            raise SystemExit(f"duplicate pool2 window: {row['window_id']}")
        seen.add(cast(str, row["window_id"]))
        windows.append(row)

    frame_rows = [[feature_index[r] for r in cast(list[str], w["frame_refs"])] for w in windows]
    inputs = torch.from_numpy(
        build_sequence_features(features, np.asarray(frame_rows, dtype=np.int64))
    )

    train_report = json.loads((args.base_run_dir / "gru_report.json").read_text())
    checkpoints = sorted(args.base_run_dir.glob("gru_seed_*.pt"))
    per_seed: list[np.ndarray[tuple[int, ...], np.dtype[np.float32]]] = []
    for checkpoint in checkpoints:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = GlobalGRU(
            int(state["classifier.weight"].shape[0]),
            int(state["classifier.weight"].shape[1]),
        )
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            per_seed.append(torch.sigmoid(model(inputs)).numpy())
    stack = np.stack(per_seed)                      # [seeds, windows, classes]
    mean, std = stack.mean(axis=0), stack.std(axis=0, ddof=1)
    if not (np.isfinite(mean).all() and np.isfinite(std).all()):
        raise SystemExit("ensemble produced non-finite values")

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "pool2_windows.jsonl").open("w", encoding="utf-8") as out:
        for index, window in enumerate(windows):
            row = dict(window)
            row["pool_index"] = index
            row["base_probabilities"] = [float(v) for v in mean[index]]
            row["probability_std"] = [float(v) for v in std[index]]
            out.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")

    report = {
        "schema_version": "1.0",
        "artifact": "pool2_ensemble_inference",
        "protocol": "v0.10",
        "scenario_ids": train_report["scenario_ids"],
        "base": {
            "run": args.base_run_dir.name,
            "ensemble": f"mean and std over {len(checkpoints)} seed checkpoints",
            "thresholds": {
                sid: train_report["base_thresholds"][sid]["threshold"]
                for sid in train_report["scenario_ids"]
            },
        },
        "feature_cache": feature_cache,
        "window_count": len(windows),
        "files": {"rows": "pool2_windows.jsonl"},
    }
    (args.output_dir / "pool2_report.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"windows": len(windows), "seeds": len(checkpoints)}))


if __name__ == "__main__":
    main()
