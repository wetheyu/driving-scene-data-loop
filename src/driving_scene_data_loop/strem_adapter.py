"""Small Python boundary around the external Strem CLI."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

STREM_VERSION = "strem 0.3.0"
STREM_RESULT_SCHEMA_VERSION = "2.0"
DEFAULT_TIMEOUT_S = 30.0

# These are the two release binaries observed for this project. The identity
# check matters because older binaries also exist on the development machine
# and have different first-frame and JSON-result semantics.
STREM_SHA256_BY_PLATFORM = {
    ("Darwin", "arm64"): ("df656e0a8d020ffc268973812e8167867098b62b1215fd92b924ba84692d923c"),
    ("Linux", "x86_64"): ("0c031f8d9fd7cf350a5209119fbd5fbddf316bba0219ae4d7c51c9e242c3692c"),
}


class StremAdapterError(RuntimeError):
    """Raised when Strem cannot produce a trustworthy match result."""


@dataclass(frozen=True, slots=True)
class StremInterval:
    """One symbolic match region returned by Strem."""

    start_frame_index: int
    end_frame_index: int
    start_time_semantics: Literal["exact", "interval"]
    start_lower_timestamp: float
    start_upper_timestamp: float
    start_lower_inclusive: bool
    start_upper_inclusive: bool
    end_lower_timestamp: float
    end_upper_timestamp: float | None
    constraints: tuple[str, ...]
    bindings: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class StremRunResult:
    """The useful part of one Strem match or no-match response."""

    status: Literal["match", "no_match"]
    specification_name: str
    intervals: tuple[StremInterval, ...]


class StremAdapter:
    """Run a scene stream against one SpTA specification."""

    def __init__(
        self,
        binary_path: Path | None = None,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        configured_path = binary_path or _path_from_environment()
        self.binary_path = configured_path.expanduser().resolve()
        self.timeout_s = timeout_s
        self._check_binary()

    def run_scene(self, stream_path: Path, spec_path: Path) -> StremRunResult:
        """Return match/no-match; process failures are never converted to no-match."""

        command = [
            str(self.binary_path),
            "[unused]",
            str(stream_path),
            "--timed-automaton",
            str(spec_path),
            "--output",
            "json",
            "--first-frame-start",
            "exact",
        ]
        completed = self._run(command)

        if completed.returncode == 2:
            raise StremAdapterError(f"Strem error: {_error_message(completed.stderr)}")
        if completed.returncode not in (0, 1):
            raise StremAdapterError(
                f"Strem exited with code {completed.returncode}: {completed.stderr.strip()}"
            )

        expected_status: Literal["match", "no_match"] = (
            "match" if completed.returncode == 0 else "no_match"
        )
        return _parse_result(completed.stdout, expected_status)

    def _check_binary(self) -> None:
        if not self.binary_path.is_file() or not os.access(self.binary_path, os.X_OK):
            raise StremAdapterError("STREM_BIN must point to an executable file")

        platform_key = (platform.system(), platform.machine().lower())
        expected_sha256 = STREM_SHA256_BY_PLATFORM.get(platform_key)
        if expected_sha256 is None:
            raise StremAdapterError(f"no Strem release is registered for {platform_key}")
        if _sha256(self.binary_path) != expected_sha256:
            raise StremAdapterError("STREM_BIN does not match the pinned v0.3.0 release")

        version = self._run([str(self.binary_path), "--version"])
        if version.returncode != 0 or version.stdout.strip() != STREM_VERSION:
            raise StremAdapterError(f"STREM_BIN must report {STREM_VERSION!r}")

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise StremAdapterError(f"Strem exceeded the {self.timeout_s:g}s timeout") from error
        except OSError as error:
            raise StremAdapterError(f"could not start Strem: {error}") from error


def _path_from_environment() -> Path:
    value = os.environ.get("STREM_BIN")
    if not value:
        raise StremAdapterError("STREM_BIN is not set")
    return Path(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error_message(stderr: str) -> str:
    try:
        document = json.loads(stderr)
        diagnostic = document.get("diagnostic", {})
        message = diagnostic.get("message")
        return str(message) if message else stderr.strip()
    except (json.JSONDecodeError, AttributeError):
        return stderr.strip()


def _parse_result(
    stdout: str,
    expected_status: Literal["match", "no_match"],
) -> StremRunResult:
    try:
        document = json.loads(stdout)
        schema_version = document["schema_version"]
        status = document["status"]
        specification_name = document["specification"]["name"]
        first_frame_start = document["stream"]["first_frame_start"]
        raw_intervals = document["intervals"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise StremAdapterError("Strem returned malformed JSON") from error

    if schema_version != STREM_RESULT_SCHEMA_VERSION:
        raise StremAdapterError("Strem returned an unsupported result schema")
    if first_frame_start != "exact":
        raise StremAdapterError("Strem did not use exact first-frame semantics")
    if status != expected_status or not isinstance(specification_name, str):
        raise StremAdapterError("Strem JSON contradicts its exit code")
    if not isinstance(raw_intervals, list):
        raise StremAdapterError("Strem intervals must be a list")

    intervals = tuple(_parse_interval(item) for item in raw_intervals)
    if expected_status == "match" and not intervals:
        raise StremAdapterError("a match result must contain an interval")
    if expected_status == "no_match" and intervals:
        raise StremAdapterError("a no-match result cannot contain intervals")
    return StremRunResult(expected_status, specification_name, intervals)


def _parse_interval(value: object) -> StremInterval:
    if not isinstance(value, dict):
        raise StremAdapterError("each Strem interval must be an object")

    start = value.get("start_frame_index")
    end = value.get("end_frame_index")
    start_time_semantics = value.get("start_time_semantics")
    start_lower = _finite_number(value.get("start_lower_timestamp"), "start_lower_timestamp")
    start_upper = _finite_number(value.get("start_upper_timestamp"), "start_upper_timestamp")
    start_lower_inclusive = value.get("start_lower_inclusive")
    start_upper_inclusive = value.get("start_upper_inclusive")
    end_lower = _finite_number(value.get("end_lower_timestamp"), "end_lower_timestamp")
    raw_end_upper = value.get("end_upper_timestamp")
    end_upper = (
        None if raw_end_upper is None else _finite_number(raw_end_upper, "end_upper_timestamp")
    )
    constraints = value.get("constraints")
    bindings = value.get("bindings")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or start > end
    ):
        raise StremAdapterError("Strem interval has invalid frame bounds")
    if start_time_semantics not in ("exact", "interval"):
        raise StremAdapterError("Strem interval has invalid start semantics")
    if not isinstance(start_lower_inclusive, bool) or not isinstance(
        start_upper_inclusive, bool
    ):
        raise StremAdapterError("Strem interval has invalid start endpoints")
    start_is_empty = start_lower > start_upper or (
        start_lower == start_upper
        and not (start_lower_inclusive and start_upper_inclusive)
    )
    exact_is_invalid = start_time_semantics == "exact" and (
        start_lower != start_upper
        or not start_lower_inclusive
        or not start_upper_inclusive
    )
    if start_is_empty or exact_is_invalid or (
        end_upper is not None and end_lower >= end_upper
    ):
        raise StremAdapterError("Strem interval has invalid time bounds")
    if not isinstance(constraints, list) or any(
        not isinstance(constraint, str) for constraint in constraints
    ):
        raise StremAdapterError("Strem interval has invalid constraints")
    if not isinstance(bindings, dict) or any(
        not isinstance(name, str)
        or not name
        or isinstance(object_id, bool)
        or not isinstance(object_id, int)
        or object_id < 0
        for name, object_id in bindings.items()
    ):
        raise StremAdapterError("Strem interval has invalid bindings")

    return StremInterval(
        start_frame_index=start,
        end_frame_index=end,
        start_time_semantics=start_time_semantics,
        start_lower_timestamp=start_lower,
        start_upper_timestamp=start_upper,
        start_lower_inclusive=start_lower_inclusive,
        start_upper_inclusive=start_upper_inclusive,
        end_lower_timestamp=end_lower,
        end_upper_timestamp=end_upper,
        constraints=tuple(constraints),
        bindings=tuple(sorted(bindings.items())),
    )


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StremAdapterError(f"Strem interval {field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise StremAdapterError(f"Strem interval {field_name} must be a finite number")
    return number
