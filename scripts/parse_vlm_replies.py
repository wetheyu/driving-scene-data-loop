"""Turn one v0.13 arm's raw replies into VLM label rows, and hash them.

Runs locally on what the GPU host returned. The replies are hashed into the
manifest here; protocol v0.13 opens scoring only after both arms are hashed.
With `--sanity`, it also applies the label-free collapse check on the
Development draw: schema-valid rate and both verdicts present per class.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

from driving_scene_data_loop.json_files import read_json, read_jsonl, write_jsonl, write_report
from driving_scene_data_loop.vlm_labeling import SCENARIO_DESCRIPTIONS, parse_verdicts

SANITY_SCHEMA_VALID = 0.95


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replies-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit("output directory must not already exist")
    runner = read_json(args.replies_dir / "runner_manifest.json")
    replies_path = args.replies_dir / "replies.jsonl"
    digest = hashlib.sha256(replies_path.read_bytes()).hexdigest()
    if digest != runner["replies_sha256"]:
        raise SystemExit("replies changed after the host wrote them")

    scenario_ids = tuple(sorted(SCENARIO_DESCRIPTIONS))
    rows = []
    for reply in read_jsonl(replies_path):
        row = parse_verdicts(
            text=cast(str, reply["text"]),
            window_id=cast(str, reply["window_id"]),
            scenario_ids=scenario_ids,
        )
        row["status"] = reply["finish_reason"]
        row["model"] = f"qwen3-vl-8b-{runner['arm']}"
        rows.append(row)

    manifest = {
        "schema_version": "1.0",
        "artifact": "vlm_label_run",
        "protocol": "v0.13",
        "arm": runner["arm"],
        "identity": {
            "model_path": runner["model_path"],
            "lora_path": runner["lora_path"],
            "prompt_sha256": runner["prompt_sha256"],
        },
        "runner": runner,
        "replies_sha256": digest,
        "usage": {
            "requests": len(rows),
            "input_tokens": sum(cast(int, r["prompt_tokens"]) for r in read_jsonl(replies_path)),
            "output_tokens": sum(cast(int, r["output_tokens"]) for r in read_jsonl(replies_path)),
            "elapsed_seconds": runner["elapsed_seconds"],
        },
    }
    if args.sanity:
        valid = sum(bool(row["schema_valid"]) for row in rows) / len(rows)
        both = {
            scenario_id: {
                verdict: sum(
                    cast(list[str], row["verdicts"])[index] == verdict for row in rows
                )
                for verdict in ("positive", "negative", "uncertain", "invalid")
            }
            for index, scenario_id in enumerate(scenario_ids)
        }
        passed = valid >= SANITY_SCHEMA_VALID and all(
            counts["positive"] > 0 and counts["negative"] > 0 for counts in both.values()
        )
        manifest["sanity_gate"] = {
            "rule": "schema-valid >= 0.95 and every class gets a positive and a negative",
            "schema_valid_rate": round(valid, 6),
            "verdicts": both,
            "passed": passed,
            "labels_read": False,
        }

    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "vlm_labels.jsonl", rows)
    write_report(args.output_dir / "vlm_run_manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "runner"}, sort_keys=True))


if __name__ == "__main__":
    main()
