"""Build what the rented GPU host receives for protocol v0.13, and nothing else.

v0.13 (`docs/EVALUATION_PLAN.md`) labels the v0.11 900 windows with an
open-weight VLM, zero-shot and with one LoRA adapter trained on L0. The host
gets L0 training targets, the public rows of the scored and sanity windows,
their frames resized once, and the frozen v0.11 prompt. The Oracle labels of the
scored and sanity windows are never written here (`docs/DATA_SPEC.md`, *Rented
GPU Boundary*): replies come back and are scored locally.
"""

from __future__ import annotations

import json
import random
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from PIL import Image

from driving_scene_data_loop.json_files import write_jsonl, write_report
from driving_scene_data_loop.vlm_labeling import (
    FRAMES_PER_WINDOW,
    MAX_OUTPUT_TOKENS,
    OUTPUT_SCHEMA,
    SYSTEM_PROMPT,
    prompt_identity,
    user_text,
)
from driving_scene_data_loop.window_dataset import PUBLIC_U_FIELDS

JsonObject = dict[str, Any]

# The v0.11 prompt, schema, and scenario text, as hashed in its run manifest.
# v0.13 reuses them unchanged, so the bundle refuses to build if they drift.
FROZEN_PROMPT_SHA256 = "d418e9bb5942e9751d04c642c3a92a9cd23c7175bb09a4e7981b1d36061b90fe"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
TRAIN_PARTITION = "l0"
SANITY_PARTITION = "development"
SANITY_WINDOWS = 300
SEED = 17
IMAGE_SIZE = (640, 360)
JPEG_QUALITY = 95

# Oracle state -> trained verdict. `ignore` is an event overlap without a bounded
# match: the definitional ambiguity `uncertain` exists for.
TARGET_VERDICT = {"positive": "positive", "negative": "negative", "ignore": "uncertain"}

# The declared recipe; LLaMA-Factory field names. Nothing here is tuned.
LORA_RECIPE: JsonObject = {
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target": "all",
    "freeze_vision_tower": True,
    "freeze_multi_modal_projector": True,
    "learning_rate": 1.0e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "num_train_epochs": 2.0,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "bf16": True,
    "seed": SEED,
    "cutoff_len": 4096,
    "image_max_pixels": IMAGE_SIZE[0] * IMAGE_SIZE[1],
    "save_strategy": "no",
    "logging_steps": 10,
    "report_to": "none",
    "overwrite_output_dir": False,
}


class VlmFinetuneError(ValueError):
    """Raised when the v0.13 bundle would cross the declared data boundary."""


def sft_reply(label_states: list[str], scenario_ids: tuple[str, ...]) -> str:
    """Render one window's Oracle states as a reply in the frozen v0.11 schema.

    `evidence_frames`, `confidence`, and `limitation` are constants of the target
    format, not judgements, and are never scored.
    """

    if len(label_states) != len(scenario_ids):
        raise VlmFinetuneError("label states and scenario order differ in length")
    verdicts: list[JsonObject] = []
    for scenario_id, state in zip(scenario_ids, label_states, strict=True):
        verdict = TARGET_VERDICT.get(state)
        if verdict is None:
            raise VlmFinetuneError(f"no training target for Oracle state {state!r}")
        verdicts.append(
            {
                "scenario_id": scenario_id,
                "verdict": verdict,
                "evidence_frames": list(range(FRAMES_PER_WINDOW)) if verdict == "positive" else [],
                "confidence": 0.5 if verdict == "uncertain" else 1.0,
                "limitation": "",
            }
        )
    return json.dumps({"verdicts": verdicts})


def select_training_windows(private_rows: list[JsonObject], seed: int = SEED) -> list[JsonObject]:
    """Every eventful L0 window plus as many all-negative ones, drawn with the seed."""

    rows = sorted(
        (row for row in private_rows if row["partition"] == TRAIN_PARTITION),
        key=lambda row: cast(str, row["window_id"]),
    )
    eventful = [
        row for row in rows
        if any(state in ("positive", "ignore") for state in row["label_states"])
    ]
    negatives = [
        row for row in rows if all(state == "negative" for state in row["label_states"])
    ]
    if len(negatives) < len(eventful):
        raise VlmFinetuneError("L0 has fewer all-negative windows than eventful ones")
    chosen = eventful + random.Random(seed).sample(negatives, len(eventful))
    return sorted(chosen, key=lambda row: cast(str, row["window_id"]))


def select_sanity_windows(
    private_rows: list[JsonObject],
    count: int = SANITY_WINDOWS,
    seed: int = SEED,
) -> list[JsonObject]:
    """Draw Development windows and keep only their public fields."""

    rows = sorted(
        (row for row in private_rows if row["partition"] == SANITY_PARTITION),
        key=lambda row: cast(str, row["window_id"]),
    )
    drawn = random.Random(seed).sample(rows, count)
    return [
        public_row(row) for row in sorted(drawn, key=lambda row: cast(str, row["window_id"]))
    ]


def public_row(row: JsonObject) -> JsonObject:
    """Keep exactly the public window fields, in their declared order."""

    return {field: row[field] for field in PUBLIC_U_FIELDS}


def check_log_isolation(train_rows: list[JsonObject], other_rows: list[JsonObject]) -> None:
    """No training window may share a log with a scored or sanity window."""

    shared = {row["log_token"] for row in train_rows} & {row["log_token"] for row in other_rows}
    if shared:
        raise VlmFinetuneError(f"training shares {len(shared)} logs with evaluated windows")
    if any(row["partition"] != TRAIN_PARTITION for row in train_rows):
        raise VlmFinetuneError("a training window is outside L0")


def resize_frame(source: Path, destination: Path) -> None:
    """Resize one CAM_FRONT frame to the declared size, the same for every arm."""

    with Image.open(source) as image:
        resized = image.convert("RGB").resize(IMAGE_SIZE, resample=Image.Resampling.BICUBIC)
    destination.parent.mkdir(parents=True, exist_ok=True)
    resized.save(destination, format="JPEG", quality=JPEG_QUALITY)


def build_bundle(
    *,
    private_rows: list[JsonObject],
    scored_rows: list[JsonObject],
    media_root: Path,
    host_root: str,
    model_path: str,
    lf_template: str,
    output_dir: Path,
    sanity_count: int = SANITY_WINDOWS,
) -> JsonObject:
    """Write the upload bundle: targets, public rows, frames, prompt, and recipe.

    `host_root` is where the bundle will sit on the GPU host; training rows name
    their frames by absolute host path because LLaMA-Factory reads them there.
    """

    if output_dir.exists():
        raise VlmFinetuneError("output directory must not already exist")
    if any(set(row) != set(PUBLIC_U_FIELDS) for row in scored_rows):
        raise VlmFinetuneError("a scored row is not a public window row")
    scenario_ids = tuple(cast(list[str], private_rows[0]["scenario_ids"]))
    identity = prompt_identity(scenario_ids, MODEL_ID)
    if identity["prompt_sha256"] != FROZEN_PROMPT_SHA256:
        raise VlmFinetuneError("the v0.11 prompt has drifted from its frozen hash")

    train_rows = select_training_windows(private_rows)
    sanity_rows = select_sanity_windows(private_rows, sanity_count)
    check_log_isolation(train_rows, scored_rows + sanity_rows)

    frame_refs = sorted(
        {ref for row in train_rows + scored_rows + sanity_rows for ref in row["frame_refs"]}
    )
    for media_ref in frame_refs:
        resize_frame(media_root / media_ref, output_dir / "media" / media_ref)

    text = user_text(scenario_ids)
    train_records = [
        {
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": "<image>" * FRAMES_PER_WINDOW + text},
                {
                    "role": "assistant",
                    "content": sft_reply(cast(list[str], row["label_states"]), scenario_ids),
                },
            ],
            "images": [f"{host_root}/media/{ref}" for ref in row["frame_refs"]],
        }
        for row in train_rows
    ]
    sft_dir = output_dir / "sft"
    sft_dir.mkdir(parents=True)
    train_bytes = json.dumps(train_records, ensure_ascii=False, indent=1).encode("utf-8")
    (sft_dir / "train.json").write_bytes(train_bytes)
    write_report(
        sft_dir / "dataset_info.json",
        {
            "v013_l0": {
                "file_name": "train.json",
                "formatting": "sharegpt",
                "columns": {"messages": "messages", "images": "images", "system": "system"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            }
        },
    )
    recipe = {
        **LORA_RECIPE,
        "model_name_or_path": model_path,
        "template": lf_template,
        "dataset": "v013_l0",
        "dataset_dir": f"{host_root}/sft",
        "output_dir": f"{host_root}/adapter",
    }
    (output_dir / "train_lora.yaml").write_text(
        "".join(f"{key}: {json.dumps(value)}\n" for key, value in recipe.items()),
        encoding="utf-8",
    )

    requests_dir = output_dir / "requests"
    requests_dir.mkdir()
    write_jsonl(requests_dir / "scored_900.jsonl", scored_rows)
    write_jsonl(requests_dir / "sanity_300.jsonl", sanity_rows)
    write_report(
        output_dir / "prompt.json",
        {
            "system": SYSTEM_PROMPT,
            "user_text": text,
            "output_schema": OUTPUT_SCHEMA,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "prompt_sha256": identity["prompt_sha256"],
            "scenario_ids": list(scenario_ids),
        },
    )

    manifest: JsonObject = {
        "schema_version": "1.0",
        "artifact": "v013_host_bundle",
        "protocol": "v0.13",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "path": model_path},
        "prompt_sha256": identity["prompt_sha256"],
        "scenario_ids": list(scenario_ids),
        "training": {
            "partition": TRAIN_PARTITION,
            "seed": SEED,
            "windows": len(train_rows),
            "eventful_windows": len(train_rows) // 2,
            "logs": len({row["log_token"] for row in train_rows}),
            "target_rule": "positive->positive, negative->negative, ignore->uncertain",
            "per_class_targets": {
                scenario_id: {
                    verdict: sum(
                        TARGET_VERDICT[cast(list[str], row["label_states"])[index]] == verdict
                        for row in train_rows
                    )
                    for verdict in ("positive", "negative", "uncertain")
                }
                for index, scenario_id in enumerate(scenario_ids)
            },
            "train_json_sha256": sha256(train_bytes).hexdigest(),
        },
        "scored_windows": len(scored_rows),
        "sanity_windows": {"partition": SANITY_PARTITION, "count": len(sanity_rows), "seed": SEED},
        "shared_logs_with_training": 0,
        "frames": {
            "count": len(frame_refs),
            "size": list(IMAGE_SIZE),
            "jpeg_quality": JPEG_QUALITY,
            "resample": "bicubic",
        },
        "recipe": recipe,
        "boundary": "L0 targets and public rows only; no scored or sanity Oracle label",
    }
    write_report(output_dir / "bundle_manifest.json", manifest)
    return manifest
