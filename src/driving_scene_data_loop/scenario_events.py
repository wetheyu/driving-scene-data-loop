"""Group Strem intervals that describe one continuous scenario event."""

from __future__ import annotations

from dataclasses import dataclass

from driving_scene_data_loop.strem_adapter import StremInterval, StremRunResult


@dataclass(frozen=True, slots=True)
class ScenarioEvent:
    """One continuous event for one fixed set of object bindings."""

    event_group_id: str
    scene_id: str
    scenario_id: str
    start_frame_index: int
    end_frame_index: int
    bindings: tuple[tuple[str, int], ...]
    source_intervals: tuple[StremInterval, ...]


def build_scenario_events(
    scene_id: str,
    run_result: StremRunResult,
) -> tuple[ScenarioEvent, ...]:
    """Merge overlapping or adjacent intervals only for identical bindings."""

    if run_result.status == "no_match":
        return ()

    grouped: dict[tuple[tuple[str, int], ...], list[StremInterval]] = {}
    for interval in run_result.intervals:
        grouped.setdefault(interval.bindings, []).append(interval)

    event_parts: list[tuple[tuple[tuple[str, int], ...], tuple[StremInterval, ...]]] = []
    for bindings, intervals in grouped.items():
        ordered = sorted(
            intervals,
            key=lambda item: (item.start_frame_index, item.end_frame_index),
        )
        current = [ordered[0]]
        current_end = ordered[0].end_frame_index
        for interval in ordered[1:]:
            if interval.start_frame_index <= current_end + 1:
                current.append(interval)
                current_end = max(current_end, interval.end_frame_index)
            else:
                event_parts.append((bindings, tuple(current)))
                current = [interval]
                current_end = interval.end_frame_index
        event_parts.append((bindings, tuple(current)))

    event_parts.sort(
        key=lambda item: (
            item[1][0].start_frame_index,
            max(interval.end_frame_index for interval in item[1]),
            item[0],
        )
    )
    return tuple(
        ScenarioEvent(
            event_group_id=(
                f"{scene_id}:{run_result.specification_name}:event-{index:04d}"
            ),
            scene_id=scene_id,
            scenario_id=run_result.specification_name,
            start_frame_index=intervals[0].start_frame_index,
            end_frame_index=max(interval.end_frame_index for interval in intervals),
            bindings=bindings,
            source_intervals=intervals,
        )
        for index, (bindings, intervals) in enumerate(event_parts, start=1)
    )
