"""Build one frozen Oracle-feedback training arm."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from driving_scene_data_loop.false_negative_bank import load_feature_index
from driving_scene_data_loop.lr_baselines import BaselineWindowData, load_baseline_windows
from driving_scene_data_loop.window_dataset import PUBLIC_U_FIELDS

JsonObject = dict[str, Any]


class FeedbackRetrainingError(ValueError):
    """Raised when a feedback arm would violate the frozen comparison."""


@dataclass(frozen=True, slots=True)
class OracleArm:
    """One label-feedback arm: which frozen IDs, and whose labels for them."""

    name: str
    method: str
    budget: int
    pre_registered: bool = True
    label_source: str = "oracle"


# An arm must be trained on the label source it declares. Training the VLM arm
# on Oracle labels would silently turn the label-noise comparison into a rerun of
# the Oracle arm, so the profile's artifact identity is checked, not assumed.
LABEL_SOURCE_ARTIFACTS = {
    "oracle": "oracle_revealed_labels",
    "vlm": "vlm_labeled_windows",
}


ORACLE_ARMS = (
    OracleArm("random-150-seed101-oracle", "random", 150),
    OracleArm("random-300-seed101-oracle", "random", 300),
    OracleArm("random-600-seed101-oracle", "random", 600),
    OracleArm("random-300-seed102-oracle", "random_seed102", 300),
    OracleArm("random-300-seed103-oracle", "random_seed103", 300),
    OracleArm("mining-150-oracle", "mining", 150),
    OracleArm("mining-300-oracle", "mining", 300),
    OracleArm("mining-600-oracle", "mining", 600),
)

# Designed after round one diagnosed why ORACLE_ARMS found no Mining effect; see
# docs/FINDINGS.md. Never pre-registered: the Random baselines above were never
# tuned, so this selector enters any comparison with a stated advantage.
DEVELOPMENT_INFORMED_ARMS = (
    OracleArm(
        "mining-v2-300-score-ranked-oracle",
        "mining_v2_score_ranked",
        300,
        pre_registered=False,
    ),
)


# Protocol v0.10 arms, declared in the Evaluation Plan before any v0.10 reveal.
# pre_registered is True relative to their own protocol's reveals.
V010_ARMS = tuple(
    OracleArm(f"v010-{label}-{budget}", method, budget)
    for label, method in (
        ("disagreement", "disagreement_v010"),
        ("prob", "prob_ranked_v010"),
        ("random201", "random_seed201"),
        ("random202", "random_seed202"),
        ("random203", "random_seed203"),
    )
    for budget in (300, 600, 900)
)


# Protocol v0.11: the same frozen IDs and the same training recipe as the v0.10
# disagreement arm, with only the added batch's label source changed.
VLM_ARMS = (
    OracleArm(
        "v010-disagreement-900-vlm",
        "disagreement_v010",
        900,
        label_source="vlm",
    ),
)


def get_oracle_arm(name: str) -> OracleArm:
    """Return one arm, pre-registered or Development-informed, by name."""

    for arm in ORACLE_ARMS + DEVELOPMENT_INFORMED_ARMS + V010_ARMS + VLM_ARMS:
        if arm.name == name:
            return arm
    raise FeedbackRetrainingError(f"unknown feedback arm: {name}")


def load_feedback_windows(
    *,
    private_windows_path: Path,
    public_pool_windows_path: Path,
    revealed_labels_path: Path,
    reveal_profile_path: Path,
    frame_index_path: Path,
    arm: OracleArm,
) -> BaselineWindowData:
    """Join L0/Development with one selected, revealed Pool prefix."""

    profile = _read_json(reveal_profile_path)
    expected_artifact = LABEL_SOURCE_ARTIFACTS.get(arm.label_source)
    if expected_artifact is None or profile.get("artifact") != expected_artifact:
        raise FeedbackRetrainingError(
            f"{arm.name} declares {arm.label_source} labels but was given "
            f"{profile.get('artifact')!r}"
        )
    base = load_baseline_windows(private_windows_path, frame_index_path)
    reveal_scenarios = tuple(cast(list[str], profile["scenario_ids"]))
    if reveal_scenarios != base.scenario_ids:
        raise FeedbackRetrainingError("Oracle and training scenario order differ")

    revealed = _read_revealed_prefix(revealed_labels_path, arm)
    selected_ids = {cast(str, row["window_id"]) for row in revealed}
    public_rows = _read_selected_public_rows(public_pool_windows_path, selected_ids)
    feature_index = load_feature_index(frame_index_path)

    frame_indices: list[list[int]] = []
    labels: list[list[int]] = []
    masks: list[list[bool]] = []
    for reveal_row in revealed:
        window_id = cast(str, reveal_row["window_id"])
        public_row = public_rows[window_id]
        refs = cast(list[str], public_row["frame_refs"])
        if len(refs) != 5:
            raise FeedbackRetrainingError(f"selected window needs five frames: {window_id}")
        try:
            frame_indices.append([feature_index[ref] for ref in refs])
        except KeyError as error:
            raise FeedbackRetrainingError(
                f"selected window references an uncached frame: {window_id}"
            ) from error
        parsed_labels, parsed_mask = _parse_targets(
            reveal_row,
            len(base.scenario_ids),
            window_id,
        )
        labels.append(parsed_labels)
        masks.append(parsed_mask)

    added_count = len(revealed)
    return BaselineWindowData(
        scenario_ids=base.scenario_ids,
        window_ids=base.window_ids + tuple(cast(str, row["window_id"]) for row in revealed),
        partitions=base.partitions + ("feedback",) * added_count,
        frame_feature_indices=np.concatenate(
            (base.frame_feature_indices, np.asarray(frame_indices, dtype=np.int64)),
            axis=0,
        ),
        labels=np.concatenate(
            (base.labels, np.asarray(labels, dtype=np.int8)),
            axis=0,
        ),
        loss_mask=np.concatenate(
            (base.loss_mask, np.asarray(masks, dtype=np.bool_)),
            axis=0,
        ),
    )


def _read_revealed_prefix(path: Path, arm: OracleArm) -> list[JsonObject]:
    rows = [
        row
        for row in _read_jsonl(path)
        if row.get("method") == arm.method and cast(int, row.get("rank", 0)) <= arm.budget
    ]
    if [row.get("rank") for row in rows] != list(range(1, arm.budget + 1)):
        raise FeedbackRetrainingError(f"revealed prefix is incomplete for {arm.name}")
    if len({row.get("window_id") for row in rows}) != arm.budget:
        raise FeedbackRetrainingError(f"revealed prefix repeats a window for {arm.name}")
    return rows


def _read_selected_public_rows(
    path: Path,
    selected_ids: set[str],
) -> dict[str, JsonObject]:
    result: dict[str, JsonObject] = {}
    for row in _read_jsonl(path):
        window_id = row.get("window_id")
        if window_id not in selected_ids:
            continue
        if set(row) != set(PUBLIC_U_FIELDS) or row.get("partition") != "u":
            raise FeedbackRetrainingError(f"selected public row is invalid: {window_id}")
        result[cast(str, window_id)] = row
    if set(result) != selected_ids:
        raise FeedbackRetrainingError("selected IDs and public Pool rows differ")
    return result


def _parse_targets(
    row: JsonObject,
    class_count: int,
    window_id: str,
) -> tuple[list[int], list[bool]]:
    raw_labels = row.get("labels")
    raw_mask = row.get("loss_mask")
    if (
        not isinstance(raw_labels, list)
        or not isinstance(raw_mask, list)
        or len(raw_labels) != class_count
        or len(raw_mask) != class_count
    ):
        raise FeedbackRetrainingError(f"revealed targets are invalid: {window_id}")
    labels: list[int] = []
    masks: list[bool] = []
    for label, mask in zip(raw_labels, raw_mask, strict=True):
        if not isinstance(mask, bool):
            raise FeedbackRetrainingError(f"revealed mask is invalid: {window_id}")
        if mask and (isinstance(label, bool) or label not in (0, 1)):
            raise FeedbackRetrainingError(f"revealed label is invalid: {window_id}")
        if not mask and label is not None:
            raise FeedbackRetrainingError(f"masked revealed label must be null: {window_id}")
        labels.append(int(label) if mask else -1)
        masks.append(mask)
    return labels, masks


def _read_json(path: Path) -> JsonObject:
    return cast(JsonObject, json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[JsonObject]:
    with path.open(encoding="utf-8") as source:
        return [cast(JsonObject, json.loads(line)) for line in source]
