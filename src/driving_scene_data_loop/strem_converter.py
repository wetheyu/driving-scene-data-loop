"""Run the pinned external nuScenes-to-Strem converter."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

CONVERTER_SHA256 = "f1833b3a2ed6ed91d2a701a0e66695c5f92d2e05375d23e1043c8522828aa952"
STREMF_SCHEMA_VERSION = "0.2.1"
DEFAULT_CLASSES = "car,pedestrian,truck,bus,trailer,construction_vehicle"
DEFAULT_TIMEOUT_S = 3600.0
ID_MAP_FILENAME = "nuscenes_numeric_id_map.json"


class StremConverterError(RuntimeError):
    """Raised when the pinned converter cannot produce traceable scene streams."""


@dataclass(frozen=True, slots=True)
class ConverterRunResult:
    """Paths and count produced by one complete converter run."""

    output_dir: Path
    numeric_id_map_path: Path
    stream_count: int


def run_nuscenes_converter(
    metadata_root: Path,
    output_dir: Path,
    converter_path: Path | None = None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ConverterRunResult:
    """Convert all scenes to ego-frame STREMF streams with stable object IDs."""

    converter = (converter_path or _path_from_environment()).expanduser().resolve()
    metadata_root = metadata_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not converter.is_file():
        raise StremConverterError("STREM_CONVERTER must point to a file")
    if _sha256(converter) != CONVERTER_SHA256:
        raise StremConverterError("STREM_CONVERTER does not match the pinned converter")
    if not metadata_root.is_dir():
        raise StremConverterError("metadata root must be a directory")
    if output_dir.exists():
        raise StremConverterError("converter output directory must not already exist")

    command = [
        sys.executable,
        str(converter),
        "--version-dir",
        str(metadata_root),
        "--out-dir",
        str(output_dir),
        "--classes",
        DEFAULT_CLASSES,
        "--coordinate-frame",
        "ego",
        "--include-ego",
        "--version-field",
        STREMF_SCHEMA_VERSION,
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise StremConverterError(
            f"Strem converter exceeded the {timeout_s:g}s timeout"
        ) from error
    except OSError as error:
        raise StremConverterError(f"could not start Strem converter: {error}") from error

    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise StremConverterError(f"Strem converter failed: {message}")

    numeric_id_map_path = output_dir / ID_MAP_FILENAME
    load_numeric_id_map(numeric_id_map_path)
    stream_count = sum(
        path.name != ID_MAP_FILENAME for path in output_dir.glob("nuscenes_*.json")
    )
    if stream_count == 0:
        raise StremConverterError("Strem converter produced no scene streams")

    return ConverterRunResult(
        output_dir=output_dir,
        numeric_id_map_path=numeric_id_map_path,
        stream_count=stream_count,
    )


def load_numeric_id_map(path: Path) -> dict[int, str]:
    """Load numeric Strem object IDs back to nuScenes instance tokens."""

    try:
        document: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise StremConverterError(f"missing numeric ID map: {path}") from error
    except json.JSONDecodeError as error:
        raise StremConverterError("numeric ID map is not valid JSON") from error

    if not isinstance(document, dict) or document.get("schema_version") != "1.0":
        raise StremConverterError("numeric ID map must use schema 1.0")
    reserved_ids = document.get("reserved_ids")
    raw_mapping = document.get("numeric_id_to_instance_token")
    if reserved_ids != {"0": "ego"} or not isinstance(raw_mapping, dict):
        raise StremConverterError("numeric ID map has invalid identity fields")

    mapping: dict[int, str] = {}
    for raw_numeric_id, instance_token in raw_mapping.items():
        if not isinstance(raw_numeric_id, str) or not raw_numeric_id.isdigit():
            raise StremConverterError("numeric ID map contains an invalid numeric ID")
        numeric_id = int(raw_numeric_id)
        if numeric_id <= 0 or numeric_id >= 2**53:
            raise StremConverterError("numeric ID map contains an out-of-range ID")
        if not isinstance(instance_token, str) or not instance_token:
            raise StremConverterError("numeric ID map contains an invalid instance token")
        mapping[numeric_id] = instance_token

    if len(mapping) != len(raw_mapping) or len(set(mapping.values())) != len(mapping):
        raise StremConverterError("numeric ID map is not one-to-one")
    if not mapping:
        raise StremConverterError("numeric ID map contains no tracked instances")
    return mapping


def _path_from_environment() -> Path:
    value = os.environ.get("STREM_CONVERTER")
    if not value:
        raise StremConverterError("STREM_CONVERTER is not set")
    return Path(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
