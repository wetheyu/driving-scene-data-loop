"""Label one frozen batch of public windows with the remote VLM.

The only stage that talks to a provider. It reads the frozen ranking, the public
window view, and the media files those windows reference -- never a label file.
Submission is chunked and resumable, and it refuses to spend more than the
declared ceiling without an explicit confirmation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, cast

from driving_scene_data_loop.vlm_labeling import (
    CHUNK_SIZE,
    DIAGNOSTIC_MODEL,
    FROZEN_MODEL,
    LABELS_FILENAME,
    MANIFEST_FILENAME,
    SCENARIO_DESCRIPTIONS,
    build_request_params,
    estimated_cost_usd,
    parse_verdicts,
    plan_chunks,
    prompt_identity,
)
from driving_scene_data_loop.window_dataset import PUBLIC_U_FIELDS

JsonObject = dict[str, Any]
SPEND_CEILING_USD = 35.0
POLL_SECONDS = 30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--public-windows", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=900)
    parser.add_argument("--model", choices=[FROZEN_MODEL, DIAGNOSTIC_MODEL], default=FROZEN_MODEL)
    parser.add_argument("--transport", choices=["batch", "sync"], default="batch")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing output directory, re-requesting only what is missing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check images, estimate spend, write contact sheets, and send nothing.",
    )
    parser.add_argument("--contact-sheets", type=int, default=3)
    parser.add_argument("--confirm-cost", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and not (args.resume or args.dry_run):
        raise SystemExit("output directory must not already exist; pass --resume to continue")

    ranking = [
        row
        for row in _read_jsonl(args.ranking)
        if cast(int, row["rank"]) <= args.budget
    ]
    if [row["rank"] for row in ranking] != list(range(1, len(ranking) + 1)):
        raise SystemExit("ranking prefix is not contiguously ranked")
    methods = {cast(str, row["method"]) for row in ranking}
    if len(methods) != 1:
        raise SystemExit("ranking mixes selection methods")
    method = methods.pop()
    window_ids = tuple(cast(str, row["window_id"]) for row in ranking)

    public_rows = _read_public_rows(args.public_windows, set(window_ids))
    scenario_ids = tuple(sorted(SCENARIO_DESCRIPTIONS))
    missing_images = [
        media_ref
        for window_id in window_ids
        for media_ref in cast(list[str], public_rows[window_id]["frame_refs"])
        if not (args.media_root / media_ref).is_file()
    ]
    if missing_images:
        raise SystemExit(f"{len(missing_images)} images missing; first={missing_images[0]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.output_dir / LABELS_FILENAME
    completed = {
        cast(str, row["window_id"])
        for row in (_read_jsonl(labels_path) if labels_path.is_file() else [])
    }
    chunks = plan_chunks(window_ids, completed, args.chunk_size)
    pending = sum(len(chunk) for chunk in chunks)
    estimate = estimated_cost_usd(pending, model=args.model, batch=args.transport == "batch")

    identity = prompt_identity(scenario_ids, args.model)
    plan = {
        "method": method,
        "budget": args.budget,
        "windows": len(window_ids),
        "already_labeled": len(completed),
        "pending": pending,
        "chunks": len(chunks),
        "transport": args.transport,
        "estimated_cost_usd": estimate,
        "identity": identity,
    }
    print(json.dumps(plan, allow_nan=False, sort_keys=True))

    if args.contact_sheets:
        sheets = _write_contact_sheets(
            [public_rows[window_id] for window_id in window_ids[: args.contact_sheets]],
            args.media_root,
            args.output_dir / "contact_sheets",
        )
        print(json.dumps({"contact_sheets": sheets}, sort_keys=True))
    if args.dry_run:
        return
    if estimate > SPEND_CEILING_USD and not args.confirm_cost:
        raise SystemExit(
            f"estimated ${estimate} exceeds the ${SPEND_CEILING_USD} ceiling; "
            "pass --confirm-cost to proceed"
        )
    if not pending:
        print(json.dumps({"status": "already complete"}))
        return
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("ANTHROPIC_API_KEY is not set")

    import anthropic

    client = anthropic.Anthropic()
    started = time.monotonic()
    usage: JsonObject = {
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "failures": 0,
    }
    chunk_records: list[JsonObject] = []
    for index, chunk in enumerate(chunks):
        requests = {
            window_id: build_request_params(
                public_window=public_rows[window_id],
                media_root=args.media_root,
                scenario_ids=scenario_ids,
                model=args.model,
            )
            for window_id in chunk
        }
        if args.transport == "batch":
            record = _run_batch(client, requests, scenario_ids, labels_path, usage)
        else:
            record = _run_sync(client, requests, scenario_ids, labels_path, usage)
        record["chunk"] = index
        chunk_records.append(record)
        print(json.dumps(record, allow_nan=False, sort_keys=True))

    usage["estimated_cost_usd"] = estimated_cost_usd(
        cast(int, usage["requests"]),
        model=args.model,
        batch=args.transport == "batch",
    )
    usage["elapsed_seconds"] = round(time.monotonic() - started, 3)
    manifest = {
        "schema_version": "1.0",
        "artifact": "vlm_label_run",
        "protocol": "v0.11",
        "method": method,
        "budget": args.budget,
        "transport": args.transport,
        "chunk_size": args.chunk_size,
        "identity": identity,
        "scenario_ids": list(scenario_ids),
        "request_boundary": "public window fields and CAM_FRONT media only",
        "chunks": chunk_records,
        "usage": usage,
    }
    (args.output_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "done", "usage": usage}, allow_nan=False, sort_keys=True))


def _run_batch(
    client: Any,
    requests: dict[str, JsonObject],
    scenario_ids: tuple[str, ...],
    labels_path: Path,
    usage: JsonObject,
) -> JsonObject:
    """Submit one chunk, wait for it, and append every reply as it is read."""

    batch = client.messages.batches.create(
        requests=[
            {"custom_id": window_id, "params": params}
            for window_id, params in requests.items()
        ]
    )
    while True:
        state = client.messages.batches.retrieve(batch.id)
        if state.processing_status == "ended":
            break
        time.sleep(POLL_SECONDS)

    rows: list[JsonObject] = []
    for result in client.messages.batches.results(batch.id):
        window_id = result.custom_id
        if result.result.type != "succeeded":
            usage["failures"] += 1
            continue
        rows.append(_row_from_message(result.result.message, window_id, scenario_ids, usage))
    _append_jsonl(labels_path, rows)
    return {"batch_id": batch.id, "requested": len(requests), "labeled": len(rows)}


def _run_sync(
    client: Any,
    requests: dict[str, JsonObject],
    scenario_ids: tuple[str, ...],
    labels_path: Path,
    usage: JsonObject,
) -> JsonObject:
    """Label one chunk with synchronous calls, used for the small smoke batch."""

    rows: list[JsonObject] = []
    for window_id, params in requests.items():
        try:
            message = client.messages.create(**params)
        except Exception as error:  # noqa: BLE001 - a failed call is a measured outcome
            usage["failures"] += 1
            print(json.dumps({"window_id": window_id, "error": str(error)}))
            continue
        rows.append(_row_from_message(message, window_id, scenario_ids, usage))
    _append_jsonl(labels_path, rows)
    return {"requested": len(requests), "labeled": len(rows)}


def _row_from_message(
    message: Any,
    window_id: str,
    scenario_ids: tuple[str, ...],
    usage: JsonObject,
) -> JsonObject:
    usage["requests"] += 1
    usage["input_tokens"] += int(message.usage.input_tokens)
    usage["output_tokens"] += int(message.usage.output_tokens)
    text = next((block.text for block in message.content if block.type == "text"), "")
    row = parse_verdicts(text=text, window_id=window_id, scenario_ids=scenario_ids)
    row["stop_reason"] = message.stop_reason
    row["model"] = message.model
    return row


def _write_contact_sheets(
    rows: list[JsonObject],
    media_root: Path,
    output_dir: Path,
) -> list[str]:
    """Render a few windows as five-frame strips so frame order is checked by eye."""

    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for row in rows:
        frames = []
        for media_ref in cast(list[str], row["frame_refs"]):
            with Image.open(media_root / media_ref) as source:
                frames.append(source.convert("RGB").resize((400, 225)))
        sheet = Image.new("RGB", (400 * len(frames), 225))
        for index, frame in enumerate(frames):
            sheet.paste(frame, (400 * index, 0))
        path = output_dir / f"{row['window_id']}.jpg"
        sheet.save(path, quality=88)
        written.append(path.name)
    return written


def _read_public_rows(path: Path, wanted: set[str]) -> dict[str, JsonObject]:
    rows: dict[str, JsonObject] = {}
    for row in _read_jsonl(path):
        window_id = cast(str, row.get("window_id"))
        if window_id not in wanted:
            continue
        if set(row) != set(PUBLIC_U_FIELDS):
            raise SystemExit(f"window row is not the public view: {window_id}")
        rows[window_id] = row
    if set(rows) != wanted:
        raise SystemExit("public window file does not cover the ranked IDs")
    return rows


def _read_jsonl(path: Path) -> list[JsonObject]:
    with path.open(encoding="utf-8") as source:
        return [cast(JsonObject, json.loads(line)) for line in source]


def _append_jsonl(path: Path, rows: list[JsonObject]) -> None:
    with path.open("a", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
