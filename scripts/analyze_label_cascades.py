"""Label-cascade analysis over the remote (A) and LoRA-adapted (C) labelers.

Descriptive, declared in the Evaluation Plan: for each fixed policy, the human
review load, the machine verdicts' quality on Oracle-decided windows, the
post-review quality with a reviewer assumed to return the Oracle label, and the
machine cost per window. No model is called and no label is revealed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driving_scene_data_loop.json_files import write_report
from driving_scene_data_loop.vlm_evaluation import (
    label_cascade_report,
    load_oracle_prefix,
    load_vlm_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--method", default="disagreement_v010")
    parser.add_argument("--budget", type=int, default=900)
    parser.add_argument("--arm-a", type=Path, required=True, help="v0.11 vlm_labels.jsonl")
    parser.add_argument("--arm-c", type=Path, required=True, help="v0.13 C vlm_labels.jsonl")
    parser.add_argument("--a-cost-usd", type=float, required=True, help="measured per window")
    parser.add_argument("--c-cost-cny", type=float, required=True, help="measured per window")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit("output must not already exist")
    scenario_ids, oracle_rows = load_oracle_prefix(args.oracle_dir, args.method, args.budget)
    report = {
        "schema_version": "1.0",
        "artifact": "label_cascade_analysis",
        "evidence_level": "label quality on the adversarial disagreement-900 slice",
        "reviewer_assumption": "a reviewer returns the Oracle label (upper bound)",
        "scenario_ids": list(scenario_ids),
        "window_count": len(oracle_rows),
        "unit_cost": {"A_usd": args.a_cost_usd, "C_cny": args.c_cost_cny},
        "policies": label_cascade_report(
            oracle_rows=oracle_rows,
            a_rows=load_vlm_labels(args.arm_a, oracle_rows, scenario_ids),
            c_rows=load_vlm_labels(args.arm_c, oracle_rows, scenario_ids),
            scenario_ids=scenario_ids,
            a_cost_usd=args.a_cost_usd,
            c_cost_cny=args.c_cost_cny,
        ),
    }
    write_report(args.output, report)
    print(json.dumps(report["policies"], sort_keys=True)[:2000])


if __name__ == "__main__":
    main()
