"""Score the v0.13 arms against the Oracle and apply the pre-registered criterion.

Arm A is the frozen v0.11 remote VLM; B and C are the open-weight arms. Each is
scored with the unchanged v0.11 label metrics, then C - A (primary), B - A, and
C - B are bootstrapped jointly over the 900 windows. C beats A only when the
C - A interval lies above zero and C's coverage is no lower than A's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driving_scene_data_loop.json_files import read_json, write_report
from driving_scene_data_loop.vlm_evaluation import (
    evaluate_vlm_labels,
    load_oracle_prefix,
    load_vlm_labels,
    paired_macro_f1_bootstrap,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--method", default="disagreement_v010")
    parser.add_argument("--budget", type=int, default=900)
    parser.add_argument("--arm-a", type=Path, required=True, help="v0.11 vlm_labels.jsonl")
    parser.add_argument("--arm-b", type=Path, required=True, help="parsed B directory")
    parser.add_argument("--arm-c", type=Path, required=True, help="parsed C directory")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit("output directory must not already exist")
    for directory in (args.arm_b, args.arm_c):
        manifest = read_json(directory / "vlm_run_manifest.json")
        if manifest.get("protocol") != "v0.13" or not manifest.get("replies_sha256"):
            raise SystemExit(f"{directory} is not a hashed v0.13 arm")

    paths = {
        "A": args.arm_a,
        "B": args.arm_b / "vlm_labels.jsonl",
        "C": args.arm_c / "vlm_labels.jsonl",
    }
    args.output_dir.mkdir(parents=True)
    reports = {
        arm: evaluate_vlm_labels(
            vlm_labels_path=path,
            oracle_dir=args.oracle_dir,
            method=args.method,
            budget=args.budget,
            output_dir=args.output_dir / f"arm_{arm}",
            protocol="v0.11" if arm == "A" else "v0.13",
        )
        for arm, path in paths.items()
    }
    scenario_ids, oracle_rows = load_oracle_prefix(args.oracle_dir, args.method, args.budget)
    labels = {arm: load_vlm_labels(path, oracle_rows, scenario_ids) for arm, path in paths.items()}
    contrasts = {
        f"{left} - {right}": paired_macro_f1_bootstrap(
            oracle_rows=oracle_rows,
            left=labels[left],
            right=labels[right],
            scenario_ids=scenario_ids,
            resamples=args.resamples,
            seed=args.seed,
        )
        for left, right in (("C", "A"), ("B", "A"), ("C", "B"))
    }
    primary = contrasts["C - A"]
    criterion = {
        "rule": "C beats A when the C - A 95% interval lies above zero "
        "and C's coverage is no lower than A's",
        "interval_above_zero": primary["ci95"][0] > 0,
        "coverage_not_lower": primary["left_coverage"] >= primary["right_coverage"],
    }
    criterion["c_beats_a"] = criterion["interval_above_zero"] and criterion["coverage_not_lower"]
    document = {
        "schema_version": "1.0",
        "artifact": "v013_labeler_comparison",
        "protocol": "v0.13",
        "method": args.method,
        "budget": args.budget,
        "scenario_ids": list(scenario_ids),
        "macro_f1": {arm: report["macro_f1"] for arm, report in reports.items()},
        "per_class_precision": {
            arm: {sid: report["per_class"][sid]["precision"] for sid in scenario_ids}
            for arm, report in reports.items()
        },
        "contrasts": contrasts,
        "criterion": criterion,
    }
    write_report(args.output_dir / "v013_comparison.json", document)
    print(json.dumps(document, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
