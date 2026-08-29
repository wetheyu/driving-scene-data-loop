"""Tests for turning raw Strem intervals into independent events."""

from __future__ import annotations

from driving_scene_data_loop.scenario_events import build_scenario_events
from driving_scene_data_loop.strem_adapter import StremInterval, StremRunResult


def test_overlapping_and_adjacent_intervals_with_same_bindings_are_one_event() -> None:
    run = _match(
        _interval(0, 2, (("p", 17),)),
        _interval(1, 3, (("p", 17),)),
        _interval(4, 5, (("p", 17),)),
    )

    events = build_scenario_events("scene-1", run)

    assert len(events) == 1
    assert (events[0].start_frame_index, events[0].end_frame_index) == (0, 5)
    assert len(events[0].source_intervals) == 3


def test_different_bindings_or_a_frame_gap_create_different_events() -> None:
    run = _match(
        _interval(0, 2, (("p", 17),)),
        _interval(1, 3, (("p", 18),)),
        _interval(5, 7, (("p", 17),)),
    )

    events = build_scenario_events("scene-1", run)

    assert [
        (event.start_frame_index, event.end_frame_index, event.bindings)
        for event in events
    ] == [
        (0, 2, (("p", 17),)),
        (1, 3, (("p", 18),)),
        (5, 7, (("p", 17),)),
    ]


def test_no_match_creates_no_events() -> None:
    run = StremRunResult("no_match", "scenario", ())

    assert build_scenario_events("scene-1", run) == ()


def _match(*intervals: StremInterval) -> StremRunResult:
    return StremRunResult("match", "scenario", intervals)


def _interval(
    start: int,
    end: int,
    bindings: tuple[tuple[str, int], ...],
) -> StremInterval:
    return StremInterval(
        start_frame_index=start,
        end_frame_index=end,
        start_time_semantics="exact",
        start_lower_timestamp=float(start),
        start_upper_timestamp=float(start),
        start_lower_inclusive=True,
        start_upper_inclusive=True,
        end_lower_timestamp=float(end),
        end_upper_timestamp=float(end + 1),
        constraints=(),
        bindings=bindings,
    )
