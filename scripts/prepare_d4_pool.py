"""Build the Pool2-restricted v0.8-style pool for the D4 similarity ablation.

The frozen v0.8 selection machinery reads a pool directory of rows, embeddings,
and a report with Base thresholds. D4 needs the same shape restricted to Pool2,
with probabilities and thresholds from Base-small: rows come from the Pool2
ensemble inference (std stripped, indexes rebuilt), embeddings are the frozen
768-d window vectors subset from the v0.8 pool, thresholds come from the
Base-small training report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

JsonObject = dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool2-inference-dir", type=Path, required=True)
    parser.add_argument("--v08-pool-dir", type=Path, required=True)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit("output directory must not already exist")

    rows = _read_jsonl(args.pool2_inference_dir / "pool2_windows.jsonl")
    rows.sort(key=lambda row: cast(int, row["pool_index"]))

    v08_index: dict[str, int] = {}
    for index, row in enumerate(_read_jsonl(args.v08_pool_dir / "pool_windows.jsonl")):
        v08_index[cast(str, row["window_id"])] = index
    embeddings = np.load(args.v08_pool_dir / "pool_embeddings.npy")

    base_report = cast(
        JsonObject, json.loads(args.base_report.read_text(encoding="utf-8"))
    )
    scenario_ids = cast(list[str], base_report["scenario_ids"])
    thresholds = {
        scenario_id: float(
            cast(JsonObject, base_report["base_thresholds"][scenario_id])["threshold"]
        )
        for scenario_id in scenario_ids
    }

    out_rows: list[JsonObject] = []
    picks: list[int] = []
    for new_index, row in enumerate(rows):
        window_id = cast(str, row["window_id"])
        if window_id not in v08_index:
            raise SystemExit(f"Pool2 window missing from the v0.8 pool: {window_id}")
        patched = {k: v for k, v in row.items() if k != "probability_std"}
        patched["pool_index"] = new_index
        out_rows.append(patched)
        picks.append(v08_index[window_id])

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "pool_windows.jsonl").open("w", encoding="utf-8") as out:
        for row in out_rows:
            out.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    np.save(args.output_dir / "pool_embeddings.npy", embeddings[picks])

    report = {
        "schema_version": "1.0",
        "artifact": "d4_pool2_similarity_pool",
        "protocol": "v0.12-D4",
        "scenario_ids": scenario_ids,
        "base": {
            "run": args.base_report.parent.name,
            "thresholds": thresholds,
            "probabilities": "Base-small twelve-seed ensemble mean",
        },
        "embedding_source": args.v08_pool_dir.name,
        "window_count": len(out_rows),
        "files": {"rows": "pool_windows.jsonl", "embeddings": "pool_embeddings.npy"},
    }
    (args.output_dir / "pool_report.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"windows": len(out_rows)}, sort_keys=True))


def _read_jsonl(path: Path) -> list[JsonObject]:
    with path.open(encoding="utf-8") as source:
        return [cast(JsonObject, json.loads(line)) for line in source]


if __name__ == "__main__":
    main()
