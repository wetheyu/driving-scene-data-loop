"""Build, parse, and store remote-VLM automatic labels for a frozen batch.

Protocol v0.11 in `docs/EVALUATION_PLAN.md` freezes what a request may contain:
five public CAM_FRONT frames, the public scenario descriptions, and the output
schema. Nothing here reads a private label file, so the request boundary is a
property of the code path rather than a reviewer's promise. Provider calls live
in `scripts/label_windows_with_vlm.py`; this module stays importable and testable
without the SDK, network, or credentials.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, cast

from driving_scene_data_loop.window_dataset import PUBLIC_U_FIELDS

JsonObject = dict[str, Any]

FROZEN_MODEL = "gpt-5.6-terra"
DIAGNOSTIC_MODEL = "gpt-5.6-sol"
RESPONSES_ENDPOINT = "/v1/responses"
MAX_OUTPUT_TOKENS = 2000
CHUNK_SIZE = 100
FRAMES_PER_WINDOW = 5
VERDICTS = ("positive", "negative", "uncertain")
LABELS_FILENAME = "vlm_labels.jsonl"
MANIFEST_FILENAME = "vlm_run_manifest.json"
LABEL_ARTIFACT = "vlm_labeled_windows"

# USD per million tokens at standard rates; the Batches API bills half of this.
MODEL_PRICES_USD_PER_MTOK = {
    FROZEN_MODEL: (2.0, 12.0),
    DIAGNOSTIC_MODEL: (5.0, 30.0),
}
# Only for the pre-submission ceiling: image tokenization is model-specific, so
# this deliberately overestimates. The run manifest records measured usage.
ESTIMATED_TOKENS_PER_IMAGE = 1550
TEXT_TOKENS_PER_REQUEST = 1500
OUTPUT_TOKENS_PER_REQUEST = 1000

# Public scenario descriptions with their Gate-A thresholds. These restate
# docs/PROJECT_SCOPE.md; they carry no per-window Oracle evidence.
SCENARIO_DESCRIPTIONS = {
    "pedestrian_ego_near_zone_entry": (
        "The same pedestrian is at least 20 m away from the ego vehicle in the "
        "ego ground plane, and within the next 2 seconds is at most 15 m away. "
        "Approach may be caused by the ego vehicle's own motion. It does not "
        "require continuous approach, and it is not a risk or intent judgement."
    ),
    "pedestrian_vehicle_center_proximity_hold": (
        "The same pedestrian and the same motor vehicle (car, truck, bus, "
        "trailer, or construction vehicle) stay at most 5 m apart, measured "
        "between their annotation centers in the ego ground plane, continuously "
        "for at least 1 second. This is center-to-center distance, not body "
        "clearance, and it is not a conflict or accident judgement."
    ),
    "vehicle_relative_corridor_entry": (
        "The same motor vehicle stays ahead of the ego vehicle (0 to 30 m "
        "forward) and its lateral offset moves from at least 3 m to at most "
        "1.5 m within 2 seconds. This is relative lateral position change; it "
        "does not require a deliberate lane change."
    ),
}

SYSTEM_PROMPT = """\
You are labeling five consecutive front-camera keyframes from a driving log for \
a scenario-mining dataset. The frames are about 0.5 seconds apart and span about \
2 seconds, in temporal order (frame 0 first).

For each scenario below, decide whether that scenario occurs within these five \
frames.

The scenario definitions are metric: they were originally computed from 3D \
annotations in the ego vehicle's ground plane. You only have monocular images, so \
distances must be estimated from apparent size, road geometry, and lane width \
(a typical lane is about 3.2 m wide). This is genuinely hard, and the honest \
answer is often "uncertain".

Rules:
- An object only counts if it is visible in these frames and identifiable as the \
same object across the frames it appears in.
- Answer "positive" only when the images actually support the transition or hold \
that the definition requires, not merely that the objects are present.
- Answer "negative" when the images show the scenario does not occur.
- Answer "uncertain" when the geometry cannot be judged from these frames. Do not \
guess a metric distance you cannot see, and do not describe an object that is not \
in the images.
- `evidence_frames` lists the frame indices (0 to 4) that support your verdict, \
and must be non-empty for a "positive" verdict.
- `limitation` states in one short sentence what limited this judgement.

Judge each scenario independently and return one verdict per scenario.\
"""

OUTPUT_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scenario_id": {
                        "type": "string",
                        "enum": list(SCENARIO_DESCRIPTIONS),
                    },
                    "verdict": {"type": "string", "enum": list(VERDICTS)},
                    "evidence_frames": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 4},
                    },
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "limitation": {"type": "string"},
                },
                "required": [
                    "scenario_id",
                    "verdict",
                    "evidence_frames",
                    "confidence",
                    "limitation",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


class VlmLabelingError(ValueError):
    """Raised when a request would leave the public boundary or a reply is unusable."""


def scenario_prompt(scenario_ids: tuple[str, ...]) -> str:
    """Render the frozen scenario block in the frozen scenario order."""

    if tuple(sorted(scenario_ids)) != tuple(sorted(SCENARIO_DESCRIPTIONS)):
        raise VlmLabelingError("scenario ids differ from the frozen scenario set")
    return "\n\n".join(
        f"{index + 1}. {scenario_id}\n{SCENARIO_DESCRIPTIONS[scenario_id]}"
        for index, scenario_id in enumerate(scenario_ids)
    )


def build_request_params(
    *,
    public_window: JsonObject,
    media_root: Path,
    scenario_ids: tuple[str, ...],
    model: str = FROZEN_MODEL,
) -> JsonObject:
    """Build one window's Responses request from public fields and media only.

    The public row must carry exactly `PUBLIC_U_FIELDS`, which is what makes an
    Oracle field structurally unable to reach the provider.
    """

    if set(public_window) != set(PUBLIC_U_FIELDS):
        raise VlmLabelingError(
            f"request input is not a public window row: {public_window.get('window_id')}"
        )
    frame_refs = cast(list[str], public_window["frame_refs"])
    if len(frame_refs) != FRAMES_PER_WINDOW:
        raise VlmLabelingError(f"window needs five frames: {public_window['window_id']}")

    content: list[JsonObject] = []
    for media_ref in frame_refs:
        path = media_root / media_ref
        if not path.is_file():
            raise VlmLabelingError(f"missing image below media_root: {media_ref}")
        encoded = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        content.append(
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded}"}
        )
    content.append(
        {
            "type": "input_text",
            "text": (
                "Scenarios to judge:\n\n"
                f"{scenario_prompt(scenario_ids)}\n\n"
                "Return one verdict for each of the three scenarios above."
            ),
        }
    )
    return {
        "model": model,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "instructions": SYSTEM_PROMPT,
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "window_verdicts",
                "schema": OUTPUT_SCHEMA,
                "strict": True,
            }
        },
    }


def extract_output_text(response_body: JsonObject) -> str:
    """Pull the assistant text out of a Responses payload, or return an empty string.

    A refusal or an empty output is an unusable reply rather than a crash: it
    becomes an `invalid` label row and is counted in the report.
    """

    parts: list[str] = []
    for item in cast(list[Any], response_body.get("output") or []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for block in cast(list[Any], item.get("content") or []):
            if isinstance(block, dict) and block.get("type") == "output_text":
                parts.append(cast(str, block.get("text", "")))
    return "".join(parts)


def prompt_identity(scenario_ids: tuple[str, ...], model: str) -> JsonObject:
    """Hash what the provider is asked, so the formal run's identity is checkable."""

    from hashlib import sha256

    payload = json.dumps(
        {
            "system": SYSTEM_PROMPT,
            "scenarios": scenario_prompt(scenario_ids),
            "schema": OUTPUT_SCHEMA,
        },
        allow_nan=False,
        sort_keys=True,
    )
    return {
        "model": model,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "prompt_sha256": sha256(payload.encode("utf-8")).hexdigest(),
        "scenario_order": list(scenario_ids),
    }


def estimated_cost_usd(window_count: int, *, model: str, batch: bool) -> float:
    """Estimate spend before it is committed, for the pre-submission ceiling."""

    if model not in MODEL_PRICES_USD_PER_MTOK:
        raise VlmLabelingError(f"unpriced model: {model}")
    input_price, output_price = MODEL_PRICES_USD_PER_MTOK[model]
    input_tokens = FRAMES_PER_WINDOW * ESTIMATED_TOKENS_PER_IMAGE + TEXT_TOKENS_PER_REQUEST
    per_window = (
        input_tokens * input_price + OUTPUT_TOKENS_PER_REQUEST * output_price
    ) / 1e6
    if batch:
        per_window *= 0.5
    return round(window_count * per_window, 4)


def plan_chunks(
    window_ids: tuple[str, ...],
    completed: set[str],
    chunk_size: int = CHUNK_SIZE,
) -> tuple[tuple[str, ...], ...]:
    """Split the still-missing windows into submittable chunks, preserving order.

    Five base64 frames run near a megabyte per request against a 200 MB batch
    file limit, so a batch is chunked rather than submitted whole, and a rerun
    re-requests only what is missing instead of paying twice for a window.
    """

    if chunk_size <= 0:
        raise VlmLabelingError("chunk size must be positive")
    pending = [window_id for window_id in window_ids if window_id not in completed]
    return tuple(
        tuple(pending[start : start + chunk_size])
        for start in range(0, len(pending), chunk_size)
    )


def parse_verdicts(
    *,
    text: str,
    window_id: str,
    scenario_ids: tuple[str, ...],
) -> JsonObject:
    """Parse one reply into a label row, recording invalidity instead of raising.

    A schema-invalid or incomplete reply is a measured outcome of automatic
    labeling, so it becomes an `invalid` row rather than an exception.
    """

    row: JsonObject = {
        "window_id": window_id,
        "scenario_ids": list(scenario_ids),
        "schema_valid": False,
        "verdicts": ["invalid"] * len(scenario_ids),
        "evidence_frames": [[] for _ in scenario_ids],
        "evidence_out_of_range": [False for _ in scenario_ids],
        "confidence": [None for _ in scenario_ids],
        "limitations": ["" for _ in scenario_ids],
        "parse_error": "",
    }
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        row["parse_error"] = f"not JSON: {error}"
        return row
    if not isinstance(payload, dict) or not isinstance(payload.get("verdicts"), list):
        row["parse_error"] = "missing verdicts array"
        return row

    by_scenario: dict[str, JsonObject] = {}
    for item in cast(list[Any], payload["verdicts"]):
        if not isinstance(item, dict):
            row["parse_error"] = "verdict entry is not an object"
            return row
        scenario_id = item.get("scenario_id")
        if not isinstance(scenario_id, str) or scenario_id in by_scenario:
            row["parse_error"] = "missing or repeated scenario_id"
            return row
        by_scenario[scenario_id] = cast(JsonObject, item)
    if set(by_scenario) != set(scenario_ids):
        row["parse_error"] = "verdict scenario set differs from the frozen set"
        return row

    verdicts: list[str] = []
    frames: list[list[int]] = []
    out_of_range: list[bool] = []
    confidences: list[float | None] = []
    limitations: list[str] = []
    for scenario_id in scenario_ids:
        item = by_scenario[scenario_id]
        verdict = item.get("verdict")
        if verdict not in VERDICTS:
            row["parse_error"] = f"invalid verdict for {scenario_id}"
            return row
        raw_frames = item.get("evidence_frames")
        if not isinstance(raw_frames, list) or any(
            not isinstance(value, int) or isinstance(value, bool) for value in raw_frames
        ):
            row["parse_error"] = f"invalid evidence_frames for {scenario_id}"
            return row
        window_frames = [int(value) for value in cast(list[int], raw_frames)]
        confidence = item.get("confidence")
        verdicts.append(cast(str, verdict))
        frames.append([value for value in window_frames if 0 <= value < FRAMES_PER_WINDOW])
        out_of_range.append(
            any(value < 0 or value >= FRAMES_PER_WINDOW for value in window_frames)
            or (verdict == "positive" and not window_frames)
        )
        confidences.append(
            float(confidence)
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
            else None
        )
        limitations.append(
            cast(str, item["limitation"]) if isinstance(item.get("limitation"), str) else ""
        )

    row.update(
        {
            "schema_valid": True,
            "verdicts": verdicts,
            "evidence_frames": frames,
            "evidence_out_of_range": out_of_range,
            "confidence": confidences,
            "limitations": limitations,
        }
    )
    return row


def write_label_files(
    *,
    label_rows: list[JsonObject],
    ranking_rows: list[JsonObject],
    scenario_ids: tuple[str, ...],
    method: str,
    output_dir: Path,
) -> JsonObject:
    """Write the VLM labels in the reveal file shape the training arms already read.

    Only the VLM's own `uncertain` (and an unusable reply) is masked. The Oracle's
    `ignore` states are deliberately not consulted: a real automatic-labeling
    pipeline has no oracle to filter its own output with.
    """

    if output_dir.exists():
        raise VlmLabelingError("output directory must not already exist")
    labels_by_id = {cast(str, row["window_id"]): row for row in label_rows}
    if len(labels_by_id) != len(label_rows):
        raise VlmLabelingError("label rows repeat a window")

    rows: list[JsonObject] = []
    for ranking_row in ranking_rows:
        window_id = cast(str, ranking_row["window_id"])
        label_row = labels_by_id.get(window_id)
        if label_row is None:
            raise VlmLabelingError(f"no VLM label for a ranked window: {window_id}")
        states = [
            verdict if verdict in VERDICTS else "invalid"
            for verdict in cast(list[str], label_row["verdicts"])
        ]
        mask = [state in ("positive", "negative") for state in states]
        rows.append(
            {
                "method": method,
                "rank": ranking_row["rank"],
                "window_id": window_id,
                "scene_token": ranking_row["scene_token"],
                "log_token": ranking_row["log_token"],
                "start_frame_index": ranking_row["start_frame_index"],
                "labels": [
                    1 if state == "positive" else 0 if state == "negative" else None
                    for state in states
                ],
                "loss_mask": mask,
                "label_states": states,
                "event_group_ids": [[] for _ in scenario_ids],
            }
        )

    profile: JsonObject = {
        "schema_version": "1.0",
        "artifact": LABEL_ARTIFACT,
        "protocol": "v0.11",
        "label_source": "vlm",
        "method": method,
        "scenario_ids": list(scenario_ids),
        "label_order": "labels, loss_mask, and label_states follow scenario_ids",
        "window_count": len(rows),
        "mask_rule": "only the VLM's own uncertain or unusable reply is masked",
        "per_class": {
            scenario_id: {
                state: sum(
                    cast(list[str], row["label_states"])[class_index] == state
                    for row in rows
                )
                for state in (*VERDICTS, "invalid")
            }
            for class_index, scenario_id in enumerate(scenario_ids)
        },
        "files": {"rows": "revealed_labels.jsonl"},
    }

    output_dir.mkdir(parents=True)
    with (output_dir / "revealed_labels.jsonl").open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    (output_dir / "label_profile.json").write_text(
        json.dumps(profile, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return profile
