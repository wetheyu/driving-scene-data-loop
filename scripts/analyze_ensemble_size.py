"""D1: how many ensemble members does the disagreement selector actually need?

Recomputes per-seed Pool2 probabilities from the frozen Base-small checkpoints,
rebuilds the full v0.10 ranking pipeline from subset standard deviations, and
reports how much of the frozen top-300/top-900 selection each subset size
recovers. Sanity gate: the recomputed full ensemble must reproduce the frozen
selection sets, or the recomputation is wrong and the diagnostic stops.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import torch  # type: ignore[import-not-found,unused-ignore]

from driving_scene_data_loop.gru_baseline import GlobalGRU, build_sequence_features
from driving_scene_data_loop.lr_baselines import (
    load_frame_features,
    validate_formal_feature_manifest,
)
from driving_scene_data_loop.selection import freeze_v010_rankings

SUBSET_SIZES = (2, 4, 6, 8)
SUBSETS_PER_SIZE = 12
SUBSET_SEED = 17
BUDGETS = (300, 900)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--frozen-ranking", type=Path, required=True)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-threads", type=int, default=16)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit("output file must not already exist")
    torch.set_num_threads(args.num_threads)

    rows = [
        cast(dict[str, object], json.loads(line))
        for line in (args.pool_dir / "pool2_windows.jsonl").open(encoding="utf-8")
    ]
    report_path = args.pool_dir / "pool2_report.json"
    frozen = _top_ids(_read_ranking(args.frozen_ranking))

    feature_cache = validate_formal_feature_manifest(args.feature_dir / "feature_manifest.json")
    features = load_frame_features(
        args.feature_dir / "frame_features.npy",
        int(cast(int, feature_cache["frame_count"])),
    )
    feature_index: dict[str, int] = {}
    for line in (args.feature_dir / "frame_index.jsonl").open(encoding="utf-8"):
        entry = json.loads(line)
        feature_index[entry["media_ref"]] = entry["feature_index"]
    frame_rows = [
        [feature_index[ref] for ref in cast(list[str], row["frame_refs"])] for row in rows
    ]
    inputs = torch.from_numpy(
        build_sequence_features(features, np.asarray(frame_rows, dtype=np.int64))
    )

    per_seed = []
    for checkpoint in sorted(args.base_run_dir.glob("gru_seed_*.pt")):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = GlobalGRU(
            int(state["classifier.weight"].shape[0]),
            int(state["classifier.weight"].shape[1]),
        )
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            per_seed.append(torch.sigmoid(model(inputs)).numpy())
    stack = np.stack(per_seed)  # [seeds, windows, classes]
    seed_count = stack.shape[0]

    def selection_for(member_indices: list[int]) -> dict[int, set[str]]:
        std = stack[member_indices].std(axis=0, ddof=1)
        workdir = Path(tempfile.mkdtemp(prefix="d1-"))
        try:
            rows_path = workdir / "pool2_windows.jsonl"
            with rows_path.open("w", encoding="utf-8") as out:
                for index, row in enumerate(rows):
                    patched = dict(row)
                    patched["probability_std"] = [float(v) for v in std[index]]
                    out.write(json.dumps(patched, allow_nan=False, separators=(",", ":")) + "\n")
            freeze_v010_rankings(
                pool_rows_path=rows_path,
                pool_report_path=report_path,
                output_dir=workdir / "rankings",
            )
            return _top_ids(_read_ranking(workdir / "rankings" / "disagreement_v010_ranked.jsonl"))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    full = selection_for(list(range(seed_count)))
    sanity = {str(budget): _overlap(full[budget], frozen[budget]) for budget in BUDGETS}
    if any(value < 1.0 for value in sanity.values()):
        raise SystemExit(f"sanity gate failed: recomputed n={seed_count} differs from frozen "
                         f"({sanity}); the recomputation is wrong, not the frozen artifact")

    rng = random.Random(SUBSET_SEED)
    results: dict[str, dict[str, object]] = {}
    for size in SUBSET_SIZES:
        overlaps: dict[int, list[float]] = {budget: [] for budget in BUDGETS}
        for _ in range(SUBSETS_PER_SIZE):
            members = rng.sample(range(seed_count), size)
            selected = selection_for(members)
            for budget in BUDGETS:
                overlaps[budget].append(_overlap(selected[budget], frozen[budget]))
        results[str(size)] = {
            f"overlap_at_{budget}": {
                "mean": round(float(np.mean(overlaps[budget])), 6),
                "sd": round(float(np.std(overlaps[budget], ddof=1)), 6),
            }
            for budget in BUDGETS
        }

    document = {
        "schema_version": "1.0",
        "artifact": "ensemble_size_ablation",
        "protocol": "v0.12-D1",
        "question": "how many ensemble members does the disagreement selector need",
        "reference": "frozen disagreement_v010 ranking",
        "sanity_full_ensemble_reproduces_frozen": sanity,
        "subset_sizes": list(SUBSET_SIZES),
        "subsets_per_size": SUBSETS_PER_SIZE,
        "subset_seed": SUBSET_SEED,
        "metric": "fraction of the frozen top-K selection recovered by the subset ranking",
        "results": results,
    }
    args.output.write_text(
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(document, allow_nan=False, sort_keys=True))


def _read_ranking(path: Path) -> list[dict[str, object]]:
    return [cast(dict[str, object], json.loads(line)) for line in path.open(encoding="utf-8")]


def _top_ids(rows: list[dict[str, object]]) -> dict[int, set[str]]:
    return {
        budget: {
            cast(str, row["window_id"]) for row in rows if cast(int, row["rank"]) <= budget
        }
        for budget in BUDGETS
    }


def _overlap(selected: set[str], reference: set[str]) -> float:
    return round(len(selected & reference) / len(reference), 6)


if __name__ == "__main__":
    main()
