"""Read and write the project's JSON artifacts in their one fixed byte format.

Rows are compact JSONL, one object per line; reports are indented, key-sorted
JSON with a trailing newline. Both refuse NaN. Every stage writes through these
helpers, so an artifact's bytes do not depend on which stage produced it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

JsonObject = dict[str, Any]


def read_json(path: Path) -> JsonObject:
    """Load one trusted project JSON document."""

    return cast(JsonObject, json.loads(path.read_text(encoding="utf-8")))


def read_jsonl(path: Path) -> list[JsonObject]:
    """Load every row of one trusted project JSONL file."""

    with path.open(encoding="utf-8") as source:
        return [cast(JsonObject, json.loads(line)) for line in source]


def jsonl_line(row: Mapping[str, Any]) -> str:
    """Serialize one row exactly as every JSONL artifact stores it."""

    return json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write rows as a new compact JSONL file."""

    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(jsonl_line(row))


def write_report(path: Path, document: Mapping[str, Any]) -> None:
    """Write one indented, key-sorted report document."""

    path.write_text(
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
