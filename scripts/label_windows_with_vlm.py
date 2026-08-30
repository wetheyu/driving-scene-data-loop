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
    RESPONSES_ENDPOINT,
    SCENARIO_DESCRIPTIONS,
    build_request_params,
    estimated_cost_usd,
    extract_output_text,
    parse_verdicts,
    plan_chunks,
    prompt_identity,
)
from driving_scene_data_loop.window_dataset import PUBLIC_U_FIELDS

JsonObject = dict[str, Any]
SPEND_CEILING_USD = 35.0
SUBMITTED_FILENAME = "submitted_batches.jsonl"
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
    submitted_path = args.output_dir / SUBMITTED_FILENAME
    unclaimed = [
        row
        for row in (_read_jsonl(submitted_path) if submitted_path.is_file() else [])
        if not set(cast(list[str], row["windows"])) <= completed
    ]
    if unclaimed:
        print(
            json.dumps(
                {
                    "warning": "a submitted batch has unwritten replies; "
                    "retrieve it before resubmitting or it is paid for twice",
                    "batch_ids": [row["batch_id"] for row in unclaimed],
                },
                sort_keys=True,
            )
        )
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
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("OPENAI_API_KEY is not set")

    from openai import OpenAI

    client = OpenAI()
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

    usage["measured_cost_usd"] = _measured_cost_usd(usage, args.model, args.transport)
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
    """Submit one chunk as a batch file, wait for it, and append every reply.

    The uploaded file carries the frames, so it is written under the private run
    directory, and both the local copy and the provider-side copy are removed
    once the results are in hand.
    """

    upload_path = labels_path.parent / "batch_input.jsonl"
    with upload_path.open("w", encoding="utf-8") as destination:
        for window_id, params in requests.items():
            destination.write(
                json.dumps(
                    {
                        "custom_id": window_id,
                        "method": "POST",
                        "url": RESPONSES_ENDPOINT,
                        "body": params,
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    with upload_path.open("rb") as source:
        uploaded = client.files.create(file=source, purpose="batch")
    upload_path.unlink()

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=RESPONSES_ENDPOINT,
        completion_window="24h",
    )
    _append_jsonl(
        labels_path.parent / SUBMITTED_FILENAME,
        [
            {
                "batch_id": batch.id,
                "input_file_id": uploaded.id,
                "windows": list(requests),
            }
        ],
    )
    while True:
        state = client.batches.retrieve(batch.id)
        if state.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(POLL_SECONDS)
    if state.status != "completed":
        raise SystemExit(f"batch {batch.id} ended as {state.status}")

    rows: list[JsonObject] = []
    for line in client.files.content(state.output_file_id).text.splitlines():
        if not line.strip():
            continue
        result = cast(JsonObject, json.loads(line))
        window_id = cast(str, result["custom_id"])
        response = cast(JsonObject, result.get("response") or {})
        body = cast(JsonObject, response.get("body") or {})
        if response.get("status_code") != 200 or not body:
            usage["failures"] = cast(int, usage["failures"]) + 1
            continue
        rows.append(_row_from_body(body, window_id, scenario_ids, usage))
    _append_jsonl(labels_path, rows)
    for file_id in (uploaded.id, state.output_file_id):
        try:
            client.files.delete(file_id)
        except Exception as error:  # noqa: BLE001 - cleanup must not lose results
            print(json.dumps({"cleanup_failed": file_id, "error": str(error)}))
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
            response = client.responses.create(**params)
        except Exception as error:  # noqa: BLE001 - a failed call is a measured outcome
            usage["failures"] = cast(int, usage["failures"]) + 1
            print(json.dumps({"window_id": window_id, "error": str(error)}))
            continue
        row = _row_from_body(response.model_dump(), window_id, scenario_ids, usage)
        _append_jsonl(labels_path, [row])
        rows.append(row)
    return {"requested": len(requests), "labeled": len(rows)}


def _row_from_body(
    body: JsonObject,
    window_id: str,
    scenario_ids: tuple[str, ...],
    usage: JsonObject,
) -> JsonObject:
    reply_usage = cast(JsonObject, body.get("usage") or {})
    usage["requests"] = cast(int, usage["requests"]) + 1
    usage["input_tokens"] = cast(int, usage["input_tokens"]) + int(
        cast(int, reply_usage.get("input_tokens", 0))
    )
    usage["output_tokens"] = cast(int, usage["output_tokens"]) + int(
        cast(int, reply_usage.get("output_tokens", 0))
    )
    row = parse_verdicts(
        text=extract_output_text(body),
        window_id=window_id,
        scenario_ids=scenario_ids,
    )
    row["status"] = body.get("status")
    row["model"] = body.get("model")
    incomplete = body.get("incomplete_details")
    if isinstance(incomplete, dict):
        row["incomplete_reason"] = incomplete.get("reason")
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


def _measured_cost_usd(usage: JsonObject, model: str, transport: str) -> float:
    """Price the tokens the provider actually reported, not the pre-run estimate."""

    from driving_scene_data_loop.vlm_labeling import MODEL_PRICES_USD_PER_MTOK

    input_price, output_price = MODEL_PRICES_USD_PER_MTOK[model]
    total = (
        cast(int, usage["input_tokens"]) * input_price
        + cast(int, usage["output_tokens"]) * output_price
    ) / 1e6
    if transport == "batch":
        total *= 0.5
    return round(total, 4)


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
