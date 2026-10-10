"""Label public windows with an open-weight VLM under vLLM, on the rented GPU host.

Protocol v0.13 arms B (zero-shot) and C (`--lora`). Runs in the host's vLLM
environment, not the project's: it imports nothing from this package and reads
the frozen prompt from the bundle's `prompt.json`, so the host needs only the
bundle. It writes raw replies; parsing and scoring happen on the local machine.
Decoding is greedy and constrained to the frozen v0.11 JSON schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

CHUNK = 100
# FlashInfer's sampler compiles CUDA kernels at start-up and the host has no
# nvcc. Decoding is greedy, so the PyTorch sampler returns the same argmax.
RUNTIME_ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lora", type=Path, help="LoRA adapter directory (arm C).")
    parser.add_argument("--arm", required=True, choices=["B", "C"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit("output directory must not already exist")
    if (args.arm == "C") != (args.lora is not None):
        raise SystemExit("arm C needs --lora and arm B must not have one")

    os.environ.update(RUNTIME_ENV)
    # Tools vLLM shells out to (ninja) live beside this interpreter.
    os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ['PATH']}"
    import vllm  # type: ignore[import-not-found,unused-ignore]
    from vllm import LLM, SamplingParams  # type: ignore[import-not-found,unused-ignore]

    prompt = json.loads((args.bundle / "prompt.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.requests.read_text(encoding="utf-8").splitlines()]
    media_root = (args.bundle / "media").resolve()

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=int(prompt["max_output_tokens"]),
        **_structured(prompt["output_schema"]),
    )
    llm = LLM(
        model=str(args.model),
        limit_mm_per_prompt={"image": 5},
        max_model_len=8192,
        allowed_local_media_path=str(media_root),
        enable_lora=args.lora is not None,
        max_lora_rank=16,
        seed=0,
    )
    lora_request = None
    if args.lora is not None:
        from vllm.lora.request import LoRARequest  # type: ignore[import-not-found,unused-ignore]

        lora_request = LoRARequest("v013-l0", 1, str(args.lora))

    args.output_dir.mkdir(parents=True)
    replies_path = args.output_dir / "replies.jsonl"
    started = time.monotonic()
    for start in range(0, len(rows), CHUNK):
        chunk = rows[start : start + CHUNK]
        conversations = [_conversation(row, prompt, media_root) for row in chunk]
        outputs = llm.chat(conversations, sampling, lora_request=lora_request, use_tqdm=False)
        with replies_path.open("a", encoding="utf-8") as destination:
            for row, output in zip(chunk, outputs, strict=True):
                completion = output.outputs[0]
                destination.write(
                    json.dumps(
                        {
                            "window_id": row["window_id"],
                            "text": completion.text,
                            "finish_reason": completion.finish_reason,
                            "prompt_tokens": len(output.prompt_token_ids or []),
                            "output_tokens": len(completion.token_ids),
                        },
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        print(json.dumps({"done": start + len(chunk), "of": len(rows)}), flush=True)
    elapsed = time.monotonic() - started

    import torch  # type: ignore[import-not-found,unused-ignore]

    manifest = {
        "schema_version": "1.0",
        "artifact": "v013_raw_replies",
        "protocol": "v0.13",
        "arm": args.arm,
        "requests": args.requests.name,
        "windows": len(rows),
        "model_path": str(args.model),
        "lora_path": str(args.lora) if args.lora else None,
        "prompt_sha256": prompt["prompt_sha256"],
        "decoding": {"temperature": 0.0, "max_tokens": prompt["max_output_tokens"],
                     "structured_output": "frozen v0.11 json schema"},
        "runtime_env": RUNTIME_ENV,
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_seconds": round(elapsed, 3),
        "replies_sha256": hashlib.sha256(replies_path.read_bytes()).hexdigest(),
    }
    (args.output_dir / "runner_manifest.json").write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


def _structured(schema: dict[str, Any]) -> dict[str, Any]:
    """Constrain decoding to the schema under whichever vLLM API is installed."""

    try:
        from vllm.sampling_params import (  # type: ignore[import-not-found,unused-ignore]
            StructuredOutputsParams,
        )

        return {"structured_outputs": StructuredOutputsParams(json=schema)}
    except ImportError:
        from vllm.sampling_params import (  # type: ignore[import-not-found,unused-ignore]
            GuidedDecodingParams,
        )

        return {"guided_decoding": GuidedDecodingParams(json=schema)}


def _conversation(
    row: dict[str, Any], prompt: dict[str, Any], media_root: Path
) -> list[dict[str, Any]]:
    """Five frames in temporal order, then the frozen text, as v0.11 sent them."""

    images = [
        {"type": "image_url", "image_url": {"url": (media_root / ref).as_uri()}}
        for ref in row["frame_refs"]
    ]
    return [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": [*images, {"type": "text", "text": prompt["user_text"]}]},
    ]


if __name__ == "__main__":
    main()
