"""Tests for the remote-VLM request boundary, reply parsing, and label files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from driving_scene_data_loop.feedback_retraining import (
    FeedbackRetrainingError,
    OracleArm,
    get_oracle_arm,
    load_feedback_windows,
)
from driving_scene_data_loop.vlm_evaluation import evaluate_vlm_labels
from driving_scene_data_loop.vlm_labeling import (
    FROZEN_MODEL,
    SCENARIO_DESCRIPTIONS,
    VlmLabelingError,
    build_request_params,
    estimated_cost_usd,
    parse_verdicts,
    plan_chunks,
    write_label_files,
)
from driving_scene_data_loop.window_dataset import PUBLIC_U_FIELDS

SCENARIOS = tuple(sorted(SCENARIO_DESCRIPTIONS))
ORACLE_FIELDS = {
    "labels": [1, 0, 1],
    "loss_mask": [True, True, True],
    "label_states": ["positive", "negative", "positive"],
    "bindings": ["instance-token"],
    "base_probabilities": [0.9, 0.1, 0.8],
}


def test_request_carries_only_public_fields_and_images(tmp_path: Path) -> None:
    media_root = _write_media(tmp_path, 5)
    params = build_request_params(
        public_window=_public_window(),
        media_root=media_root,
        scenario_ids=SCENARIOS,
        model=FROZEN_MODEL,
    )

    content = params["messages"][0]["content"]
    assert [block["type"] for block in content] == ["image"] * 5 + ["text"]
    assert params["model"] == FROZEN_MODEL
    assert params["output_config"]["format"]["type"] == "json_schema"

    body = json.dumps(params)
    for field, value in ORACLE_FIELDS.items():
        assert field not in body
        assert json.dumps(value)[1:-1] not in body
    assert "20 m" in body and "corridor" in body


def test_request_rejects_a_window_row_carrying_an_oracle_field(tmp_path: Path) -> None:
    media_root = _write_media(tmp_path, 5)
    leaking = {**_public_window(), "label_states": ORACLE_FIELDS["label_states"]}

    with pytest.raises(VlmLabelingError, match="not a public window row"):
        build_request_params(
            public_window=leaking,
            media_root=media_root,
            scenario_ids=SCENARIOS,
        )


def test_chunks_skip_completed_windows_and_keep_ranked_order() -> None:
    window_ids = tuple(f"w{index}" for index in range(7))

    assert plan_chunks(window_ids, set(), 3) == (
        ("w0", "w1", "w2"),
        ("w3", "w4", "w5"),
        ("w6",),
    )
    assert plan_chunks(window_ids, {"w0", "w3", "w6"}, 3) == (("w1", "w2", "w4"), ("w5",))
    assert plan_chunks(window_ids, set(window_ids), 3) == ()


def test_cost_estimate_is_halved_for_batch_and_ordered_by_model() -> None:
    sync = estimated_cost_usd(900, model=FROZEN_MODEL, batch=False)
    batch = estimated_cost_usd(900, model=FROZEN_MODEL, batch=True)

    assert batch == pytest.approx(sync / 2)
    assert 10.0 < batch < 20.0


def test_parse_joins_verdicts_by_scenario_id_not_position() -> None:
    reply = json.dumps(
        {
            "verdicts": [
                _verdict(SCENARIOS[2], "positive", [3, 4]),
                _verdict(SCENARIOS[0], "negative", []),
                _verdict(SCENARIOS[1], "uncertain", [1]),
            ]
        }
    )

    row = parse_verdicts(text=reply, window_id="w0", scenario_ids=SCENARIOS)

    assert row["schema_valid"] is True
    assert row["verdicts"] == ["negative", "uncertain", "positive"]
    assert row["evidence_frames"] == [[], [1], [3, 4]]
    assert row["evidence_out_of_range"] == [False, False, False]


def test_parse_flags_unusable_replies_instead_of_raising() -> None:
    invalid = parse_verdicts(text="sorry, no JSON", window_id="w0", scenario_ids=SCENARIOS)
    assert invalid["schema_valid"] is False
    assert invalid["verdicts"] == ["invalid"] * 3

    partial = parse_verdicts(
        text=json.dumps({"verdicts": [_verdict(SCENARIOS[0], "positive", [0])]}),
        window_id="w0",
        scenario_ids=SCENARIOS,
    )
    assert partial["schema_valid"] is False
    assert "scenario set" in partial["parse_error"]


def test_parse_flags_unsupported_and_out_of_range_evidence() -> None:
    row = parse_verdicts(
        text=json.dumps(
            {
                "verdicts": [
                    _verdict(SCENARIOS[0], "positive", []),
                    _verdict(SCENARIOS[1], "positive", [7]),
                    _verdict(SCENARIOS[2], "negative", [0]),
                ]
            }
        ),
        window_id="w0",
        scenario_ids=SCENARIOS,
    )

    assert row["evidence_out_of_range"] == [True, True, False]
    assert row["evidence_frames"][1] == []


def test_label_file_masks_only_the_vlm_uncertain(tmp_path: Path) -> None:
    profile = write_label_files(
        label_rows=[
            _label_row("w0", ["positive", "uncertain", "negative"]),
            _label_row("w1", ["invalid", "negative", "positive"]),
        ],
        ranking_rows=_ranking_rows(),
        scenario_ids=SCENARIOS,
        method="disagreement_v010",
        output_dir=tmp_path / "vlm",
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "vlm" / "revealed_labels.jsonl").read_text().splitlines()
    ]
    assert profile["artifact"] == "vlm_labeled_windows"
    assert rows[0]["labels"] == [1, None, 0]
    assert rows[0]["loss_mask"] == [True, False, True]
    assert rows[1]["labels"] == [None, 0, 1]
    assert rows[1]["label_states"] == ["invalid", "negative", "positive"]


def test_vlm_arm_refuses_oracle_labels(tmp_path: Path) -> None:
    arm = get_oracle_arm("v010-disagreement-900-vlm")
    assert arm.label_source == "vlm"
    assert (arm.method, arm.budget) == ("disagreement_v010", 900)

    oracle_profile = tmp_path / "label_profile.json"
    oracle_profile.write_text(
        json.dumps({"artifact": "oracle_revealed_labels", "scenario_ids": list(SCENARIOS)}),
        encoding="utf-8",
    )
    with pytest.raises(FeedbackRetrainingError, match="declares vlm labels"):
        load_feedback_windows(
            private_windows_path=tmp_path / "windows.jsonl",
            public_pool_windows_path=tmp_path / "public.jsonl",
            revealed_labels_path=tmp_path / "revealed_labels.jsonl",
            reveal_profile_path=oracle_profile,
            frame_index_path=tmp_path / "frame_index.jsonl",
            arm=OracleArm("fixture-vlm", "disagreement_v010", 2, label_source="vlm"),
        )


def test_evaluation_scores_decided_windows_and_profiles_the_rest(tmp_path: Path) -> None:
    oracle_dir = tmp_path / "oracle"
    oracle_dir.mkdir()
    oracle_states = [
        ["positive", "negative", "ignore"],
        ["negative", "positive", "positive"],
    ]
    _write_jsonl(
        oracle_dir / "revealed_labels.jsonl",
        [
            {
                "method": "disagreement_v010",
                "rank": rank,
                "window_id": f"w{rank - 1}",
                "label_states": states,
            }
            for rank, states in enumerate(oracle_states, start=1)
        ],
    )
    (oracle_dir / "label_profile.json").write_text(
        json.dumps(
            {"artifact": "oracle_revealed_labels", "scenario_ids": list(SCENARIOS)}
        ),
        encoding="utf-8",
    )
    vlm_path = tmp_path / "vlm_labels.jsonl"
    _write_jsonl(
        vlm_path,
        [
            _label_row("w0", ["positive", "positive", "positive"]),
            _label_row("w1", ["negative", "uncertain", "positive"]),
        ],
    )

    report = evaluate_vlm_labels(
        vlm_labels_path=vlm_path,
        oracle_dir=oracle_dir,
        method="disagreement_v010",
        budget=2,
        output_dir=tmp_path / "report",
    )

    first = report["per_class"][SCENARIOS[0]]
    assert (first["true_positives"], first["true_negatives"]) == (1, 1)
    assert first["f1"] == 1.0
    second = report["per_class"][SCENARIOS[1]]
    assert second["false_positives"] == 1
    assert second["undecidable_rate_on_decided"] == 0.5
    third = report["per_class"][SCENARIOS[2]]
    assert third["oracle_decided_windows"] == 1
    assert third["ambiguous_oracle_windows"] == {
        "ignore": {"positive": 1, "negative": 0, "uncertain": 0, "invalid": 0}
    }
    assert report["downstream_gate"]["passed"] is False


def _public_window() -> dict[str, object]:
    return {
        "window_id": "w0",
        "scene_token": "scene-0",
        "scene_name": "scene-0000",
        "log_token": "log-0",
        "partition": "u",
        "start_frame_index": 3,
        "end_frame_index": 7,
        "sample_tokens": [f"sample-{index}" for index in range(5)],
        "timestamps_s": [0.0, 0.5, 1.0, 1.5, 2.0],
        "frame_refs": [f"samples/CAM_FRONT/f{index}.jpg" for index in range(5)],
    }


def _write_media(tmp_path: Path, count: int) -> Path:
    media_root = tmp_path / "media"
    (media_root / "samples" / "CAM_FRONT").mkdir(parents=True)
    for index in range(count):
        (media_root / "samples" / "CAM_FRONT" / f"f{index}.jpg").write_bytes(
            b"\xff\xd8\xff" + bytes([index])
        )
    return media_root


def _verdict(scenario_id: str, verdict: str, frames: list[int]) -> dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "verdict": verdict,
        "evidence_frames": frames,
        "confidence": 0.5,
        "limitation": "monocular distance is approximate",
    }


def _label_row(window_id: str, verdicts: list[str]) -> dict[str, object]:
    return {
        "window_id": window_id,
        "scenario_ids": list(SCENARIOS),
        "schema_valid": "invalid" not in verdicts,
        "verdicts": verdicts,
        "evidence_frames": [[0] for _ in verdicts],
        "evidence_out_of_range": [False for _ in verdicts],
        "confidence": [0.5 for _ in verdicts],
        "limitations": ["" for _ in verdicts],
        "parse_error": "",
    }


def _ranking_rows() -> list[dict[str, object]]:
    return [
        {
            "method": "disagreement_v010",
            "rank": rank,
            "window_id": f"w{rank - 1}",
            "scene_token": "scene-0",
            "log_token": "log-0",
            "start_frame_index": rank,
        }
        for rank in (1, 2)
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row) + "\n")


def test_public_field_set_is_what_the_builder_enforces() -> None:
    assert set(_public_window()) == set(PUBLIC_U_FIELDS)
