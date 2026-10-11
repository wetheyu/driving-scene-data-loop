"""Protocol v0.13: the host bundle stays inside its boundary, and the criterion is paired."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from driving_scene_data_loop.json_files import read_json, read_jsonl
from driving_scene_data_loop.vlm_evaluation import (
    _score_class,
    label_cascade_report,
    paired_macro_f1_bootstrap,
)
from driving_scene_data_loop.vlm_finetune import (
    IMAGE_SIZE,
    VlmFinetuneError,
    build_bundle,
    check_log_isolation,
    public_row,
    select_training_windows,
    sft_reply,
)
from driving_scene_data_loop.vlm_labeling import SCENARIO_DESCRIPTIONS, parse_verdicts
from driving_scene_data_loop.window_dataset import PUBLIC_U_FIELDS

SCENARIOS = tuple(sorted(SCENARIO_DESCRIPTIONS))


def _window(index: int, partition: str, states: list[str], log: str) -> dict[str, Any]:
    return {
        "window_id": f"{partition}-{index:04d}",
        "scene_token": f"scene-{partition}",
        "scene_name": f"scene-{partition}",
        "log_token": log,
        "partition": partition,
        "start_frame_index": index,
        "end_frame_index": index + 4,
        "sample_tokens": [f"s{index + k}" for k in range(5)],
        "timestamps_s": [0.5 * k for k in range(5)],
        "frame_refs": [f"samples/CAM_FRONT/{partition}-{index + k}.jpg" for k in range(5)],
        "scenario_ids": list(SCENARIOS),
        "labels": [1 if state == "positive" else 0 if state == "negative" else None
                   for state in states],
        "loss_mask": [state in ("positive", "negative") for state in states],
        "label_states": states,
        "event_group_ids": [[] for _ in states],
    }


def test_a_target_parses_as_the_frozen_schema_and_maps_ignore_to_uncertain() -> None:
    row = parse_verdicts(
        text=sft_reply(["positive", "ignore", "negative"], SCENARIOS),
        window_id="w",
        scenario_ids=SCENARIOS,
    )

    assert row["schema_valid"] is True
    assert row["verdicts"] == ["positive", "uncertain", "negative"]
    assert row["evidence_out_of_range"] == [False, False, False]
    with pytest.raises(VlmFinetuneError):
        sft_reply(["invalid", "negative", "negative"], SCENARIOS)


def test_training_keeps_every_eventful_l0_window_and_as_many_negatives() -> None:
    negative = ["negative"] * 3
    rows = (
        [_window(0, "l0", ["positive", "negative", "negative"], "a"),
         _window(1, "l0", ["negative", "ignore", "negative"], "a")]
        + [_window(i, "l0", negative, "a") for i in range(2, 9)]
        + [_window(0, "development", ["positive", "negative", "negative"], "d")]
    )

    chosen = select_training_windows(rows)

    assert len(chosen) == 4
    assert {row["partition"] for row in chosen} == {"l0"}
    assert {"l0-0000", "l0-0001"} <= {row["window_id"] for row in chosen}
    assert chosen == select_training_windows(rows)


def test_a_training_log_shared_with_an_evaluated_window_is_refused() -> None:
    train = [_window(0, "l0", ["negative"] * 3, "shared")]
    scored = [public_row(_window(0, "u", ["negative"] * 3, "shared"))]

    with pytest.raises(VlmFinetuneError, match="shares 1 logs"):
        check_log_isolation(train, scored)


def test_the_bundle_carries_no_scored_or_sanity_label(tmp_path: Path) -> None:
    media = tmp_path / "media"
    rows = (
        [_window(i, "l0", ["positive", "negative", "negative"], "l") for i in (0, 10)]
        + [_window(i, "l0", ["negative"] * 3, "l") for i in (20, 30, 40)]
        + [_window(i, "development", ["positive", "ignore", "negative"], "d") for i in (0, 10)]
    )
    scored = [public_row(_window(i, "u", ["positive"] * 3, "u")) for i in (0, 10)]
    for row in rows + scored:
        for ref in row["frame_refs"]:
            (media / ref).parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (160, 90), (1, 2, 3)).save(media / ref)

    manifest = build_bundle(
        private_rows=rows,
        scored_rows=scored,
        media_root=media,
        host_root="/host/v013",
        model_path="/host/model",
        lf_template="qwen3_vl",
        output_dir=tmp_path / "bundle",
        sanity_count=2,
    )

    bundle = tmp_path / "bundle"
    for name in ("scored_900.jsonl", "sanity_300.jsonl"):
        request_rows = read_jsonl(bundle / "requests" / name)
        assert all(list(row) == list(PUBLIC_U_FIELDS) for row in request_rows)
    train = json.loads((bundle / "sft" / "train.json").read_text(encoding="utf-8"))
    assert len(train) == 4 and manifest["training"]["windows"] == 4
    assert all(path.startswith("/host/v013/media/samples/CAM_FRONT/l0-") for record in train
               for path in record["images"])
    assert all(record["messages"][0]["content"].startswith("<image>" * 5) for record in train)
    with Image.open(bundle / "media" / scored[0]["frame_refs"][0]) as image:
        assert image.size == IMAGE_SIZE
    assert read_json(bundle / "prompt.json")["prompt_sha256"] == manifest["prompt_sha256"]
    assert "lora_rank: 16" in (bundle / "train_lora.yaml").read_text(encoding="utf-8")


def _labels(verdicts: list[list[str]]) -> dict[str, dict[str, Any]]:
    return {f"w{i}": {"verdicts": row} for i, row in enumerate(verdicts)}


def test_the_bootstrap_is_paired_reproducible_and_matches_the_point_scores() -> None:
    oracle_states = [["positive"] * 3, ["negative"] * 3, ["positive"] * 3, ["negative"] * 3,
                     ["ignore"] * 3]
    oracle = {f"w{i}": {"label_states": row} for i, row in enumerate(oracle_states)}
    left = _labels([["positive"] * 3, ["negative"] * 3, ["positive"] * 3, ["negative"] * 3,
                    ["positive"] * 3])
    right = _labels([["positive"] * 3, ["positive"] * 3, ["uncertain"] * 3, ["negative"] * 3,
                     ["negative"] * 3])

    first = paired_macro_f1_bootstrap(
        oracle_rows=oracle, left=left, right=right, scenario_ids=SCENARIOS, resamples=500
    )
    second = paired_macro_f1_bootstrap(
        oracle_rows=oracle, left=left, right=right, scenario_ids=SCENARIOS, resamples=500
    )

    pairs = [
        ({"label_states": oracle[w]["label_states"]},
         {"verdicts": right[w]["verdicts"], "evidence_out_of_range": [False] * 3})
        for w in sorted(oracle)
    ]
    assert first == second
    assert first["left_macro_f1"] == 1.0
    assert first["right_macro_f1"] == _score_class(pairs, 0)["f1"]
    assert first["difference"] == pytest.approx(1.0 - _score_class(pairs, 0)["f1"], abs=1e-6)
    assert first["left_coverage"] == 1.0 and first["right_coverage"] == 0.75
    assert first["ci95"][0] <= first["difference"] <= first["ci95"][1]


def test_cascade_policies_route_review_and_score_after_review() -> None:
    oracle = {
        "w0": {"label_states": ["positive", "negative", "positive"]},
        "w1": {"label_states": ["negative", "ignore", "negative"]},
    }
    a = _labels([["positive", "uncertain", "negative"], ["negative", "positive", "negative"]])
    c = _labels([["negative", "negative", "positive"], ["uncertain", "positive", "positive"]])

    report = label_cascade_report(
        oracle_rows=oracle, a_rows=a, c_rows=c, scenario_ids=SCENARIOS,
        a_cost_usd=0.02, c_cost_cny=0.001,
    )

    agree = report["agree"]["classes"][SCENARIOS[0]]
    assert agree["review_rate"] == 1.0  # w0 disagrees, w1 has a C abstention
    assert agree["after_review"]["f1"] == 1.0  # every decided window reviewed
    route = report["route"]["classes"][SCENARIOS[2]]  # corridor comes from C
    assert (route["machine"]["tp"], route["machine"]["fp"]) == (1, 1)
    assert report["C-then-A"]["share_of_windows_calling"]["A"] == 0.5
    assert report["A"]["machine_cost_per_window"] == {"usd": 0.02, "cny": 0.0}
