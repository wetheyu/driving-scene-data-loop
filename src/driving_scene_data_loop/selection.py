"""Freeze label-free Random and integrated Mining rankings."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans  # type: ignore[import-untyped]

from driving_scene_data_loop.window_dataset import PUBLIC_U_FIELDS

JsonObject = dict[str, Any]

RANDOM_SEED = 101
VARIANCE_RANDOM_SEEDS = (102, 103)
CLUSTER_SEED = 17
SHORTLIST_SIZE = 2_000
CLUSTER_COUNT = 30
BUDGETS = (150, 300, 600)
TEMPORAL_SEPARATION = 5


class SelectionError(ValueError):
    """Raised when a frozen selection ranking would be invalid."""


@dataclass(frozen=True, slots=True)
class _Candidate:
    pool_index: int
    scenario_id: str
    probability: float
    threshold: float
    similarity: float
    boundary_margin: float
    cluster_id: int | None = None


def boundary_margin(probability: float, threshold: float) -> float:
    """Return absolute logit distance from the frozen class threshold."""

    epsilon = 1e-6
    probability = min(max(probability, epsilon), 1.0 - epsilon)
    threshold = min(max(threshold, epsilon), 1.0 - epsilon)
    probability_logit = math.log(probability / (1.0 - probability))
    threshold_logit = math.log(threshold / (1.0 - threshold))
    return abs(probability_logit - threshold_logit)


def freeze_selection_rankings(
    *,
    pool_rows_path: Path,
    pool_embeddings_path: Path,
    pool_report_path: Path,
    fn_rows_path: Path,
    fn_embeddings_path: Path,
    output_dir: Path,
    budgets: tuple[int, ...] = BUDGETS,
    shortlist_size: int = SHORTLIST_SIZE,
    cluster_count: int = CLUSTER_COUNT,
) -> JsonObject:
    """Build Random and Mining rankings without reading Pool Oracle evidence."""

    if output_dir.exists():
        raise SelectionError("output directory must not already exist")
    if not budgets or tuple(sorted(set(budgets))) != budgets:
        raise SelectionError("budgets must be positive, unique, and increasing")
    if shortlist_size <= 0 or cluster_count <= 0:
        raise SelectionError("shortlist size and cluster count must be positive")

    pool_report = _read_json(pool_report_path)
    scenario_ids = cast(tuple[str, ...], tuple(pool_report["scenario_ids"]))
    if len(scenario_ids) != 3 or any(
        budget <= 0 or budget % len(scenario_ids) != 0 for budget in budgets
    ):
        raise SelectionError("every budget must split evenly across three classes")
    thresholds = {
        scenario_id: float(pool_report["base"]["thresholds"][scenario_id])
        for scenario_id in scenario_ids
    }
    pool_rows = _read_pool_rows(pool_rows_path, len(scenario_ids))
    pool_embeddings = _load_embeddings(
        pool_embeddings_path,
        expected_rows=len(pool_rows),
    )
    fn_rows = _read_fn_rows(fn_rows_path, scenario_ids)
    fn_embeddings = _load_embeddings(
        fn_embeddings_path,
        expected_rows=len(fn_rows),
    )

    max_budget = budgets[-1]
    random_ranking = _random_ranking(pool_rows, max_budget, RANDOM_SEED, "random")
    mining_queues: dict[str, list[_Candidate]] = {}
    for class_index, scenario_id in enumerate(scenario_ids):
        candidates = _rank_class_candidates(
            scenario_id=scenario_id,
            class_index=class_index,
            threshold=thresholds[scenario_id],
            pool_rows=pool_rows,
            pool_embeddings=pool_embeddings,
            fn_rows=fn_rows,
            fn_embeddings=fn_embeddings,
            shortlist_size=shortlist_size,
        )
        mining_queues[scenario_id] = _diversify_candidates(
            candidates,
            pool_rows,
            pool_embeddings,
            cluster_count,
        )

    mining_ranking = _merge_class_queues(
        "mining",
        scenario_ids,
        mining_queues,
        pool_rows,
        max_budget,
    )

    output_dir.mkdir(parents=True)
    rankings = {
        "random": random_ranking,
        "mining": mining_ranking,
    }
    files: dict[str, str] = {}
    for method, rows in rankings.items():
        filename = f"{method}_ranked.jsonl"
        files[method] = filename
        _write_jsonl(output_dir / filename, rows)

    report: JsonObject = {
        "schema_version": "1.0",
        "artifact": "frozen_selection_rankings",
        "scenario_ids": list(scenario_ids),
        "budgets": list(budgets),
        "primary_budget": 300 if 300 in budgets else budgets[0],
        "maximum_ranked_count": max_budget,
        "public_pool_window_count": len(pool_rows),
        "policy": {
            "random_seed": RANDOM_SEED,
            "shortlist_size_per_class": shortlist_size,
            "uncertainty": "absolute logit distance to frozen class threshold",
            "temporal_separation_keyframes": TEMPORAL_SEPARATION,
            "cluster_algorithm": "MiniBatchKMeans",
            "cluster_count_per_class": cluster_count,
            "cluster_seed": CLUSTER_SEED,
            "class_merge": "round-robin in frozen scenario order",
        },
        "prefixes": {
            method: _prefix_summary(rows, budgets, scenario_ids)
            for method, rows in rankings.items()
        },
        "files": files,
    }
    (output_dir / "selection_report.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def freeze_random_variance_rankings(
    *,
    pool_rows_path: Path,
    pool_report_path: Path,
    output_dir: Path,
    seeds: tuple[int, ...] = VARIANCE_RANDOM_SEEDS,
    budgets: tuple[int, ...] = BUDGETS,
) -> JsonObject:
    """Freeze extra Random batches so Random is a distribution, not one draw.

    Random needs no bad-case bank, embeddings, or Base probabilities, so this
    reads only the public Pool rows. The frozen `RANDOM_SEED` batch is rejected
    here: reusing it would count one draw twice and overstate agreement.
    """

    if output_dir.exists():
        raise SelectionError("output directory must not already exist")
    if not seeds or tuple(sorted(set(seeds))) != seeds:
        raise SelectionError("variance seeds must be unique and increasing")
    if RANDOM_SEED in seeds:
        raise SelectionError("variance seeds must exclude the frozen Random seed")
    if not budgets or tuple(sorted(set(budgets))) != budgets:
        raise SelectionError("budgets must be positive, unique, and increasing")

    pool_report = _read_json(pool_report_path)
    scenario_ids = cast(tuple[str, ...], tuple(pool_report["scenario_ids"]))
    pool_rows = _read_pool_rows(pool_rows_path, len(scenario_ids))

    max_budget = budgets[-1]
    rankings = {
        seed: _random_ranking(pool_rows, max_budget, seed, f"random_seed{seed}")
        for seed in seeds
    }

    output_dir.mkdir(parents=True)
    files: dict[str, str] = {}
    for seed, rows in rankings.items():
        filename = f"random_seed{seed}_ranked.jsonl"
        files[str(seed)] = filename
        _write_jsonl(output_dir / filename, rows)

    report: JsonObject = {
        "schema_version": "1.0",
        "artifact": "random_variance_rankings",
        "purpose": (
            "compare the single frozen Mining batch against a Random "
            "distribution instead of one Random draw"
        ),
        "frozen_random_seed": RANDOM_SEED,
        "variance_seeds": list(seeds),
        "scenario_ids": list(scenario_ids),
        "budgets": list(budgets),
        "primary_budget": 300 if 300 in budgets else budgets[0],
        "public_pool_window_count": len(pool_rows),
        "policy": {
            "sampling": "uniform permutation of public Pool window IDs",
            "temporal_separation_keyframes": TEMPORAL_SEPARATION,
        },
        "prefixes": {
            str(seed): _prefix_summary(rows, budgets, scenario_ids)
            for seed, rows in rankings.items()
        },
        "files": files,
    }
    (output_dir / "random_variance_report.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def freeze_score_ranked_rankings(
    *,
    pool_rows_path: Path,
    pool_report_path: Path,
    output_dir: Path,
    budgets: tuple[int, ...] = BUDGETS,
    method: str = "mining_v2_score_ranked",
) -> JsonObject:
    """Freeze a Development-informed selector: pure Base-probability ranking.

    Round one measured the frozen `mining` selector against three Random
    batches and found no effect, then diagnosed three mechanisms behind the
    null result: the FN-similarity filter was anti-predictive (0.15x lift on
    the rarest class), MiniBatchKMeans round-robin diluted the one working
    signal (selections landed at the 23rd-32nd shortlist percentile instead of
    the top 10%), and the frozen "boundary margin" ranker sits in the far right
    tail of the probability distribution, so it was doing indirect positive
    retrieval rather than uncertainty sampling.

    This selector removes all three: no similarity filter, no shortlist, no
    clustering. Each class is ranked by the Base model's own probability,
    descending, with only temporal deduplication. It reads no FN bank and no
    embeddings.

    It must never be reported as pre-registered. The Random baselines were
    never tuned; this selector was designed after reading revealed Pool labels,
    so it carries a Development-informed advantage that has to be stated
    alongside any result.
    """

    if output_dir.exists():
        raise SelectionError("output directory must not already exist")
    if not budgets or tuple(sorted(set(budgets))) != budgets:
        raise SelectionError("budgets must be positive, unique, and increasing")

    pool_report = _read_json(pool_report_path)
    scenario_ids = cast(tuple[str, ...], tuple(pool_report["scenario_ids"]))
    if len(scenario_ids) != 3 or any(
        budget <= 0 or budget % len(scenario_ids) != 0 for budget in budgets
    ):
        raise SelectionError("every budget must split evenly across three classes")
    thresholds = {
        scenario_id: float(pool_report["base"]["thresholds"][scenario_id])
        for scenario_id in scenario_ids
    }
    pool_rows = _read_pool_rows(pool_rows_path, len(scenario_ids))

    max_budget = budgets[-1]
    queues: dict[str, list[_ScoreCandidate]] = {}
    for class_index, scenario_id in enumerate(scenario_ids):
        candidates = [
            _ScoreCandidate(
                pool_index=index,
                scenario_id=scenario_id,
                probability=float(row["base_probabilities"][class_index]),
                threshold=thresholds[scenario_id],
            )
            for index, row in enumerate(pool_rows)
        ]
        candidates.sort(
            key=lambda item: (
                -item.probability,
                pool_rows[item.pool_index]["window_id"],
            )
        )
        queues[scenario_id] = _score_temporal_filter(candidates, pool_rows)

    ranking = _merge_score_queues(method, scenario_ids, queues, pool_rows, max_budget)

    output_dir.mkdir(parents=True)
    filename = f"{method}_ranked.jsonl"
    _write_jsonl(output_dir / filename, ranking)

    report: JsonObject = {
        "schema_version": "1.0",
        "artifact": "score_ranked_selection",
        "development_informed": True,
        "diagnosis_source": "oracle-reveal-v1 unbiased-sample analysis; see docs/FINDINGS.md",
        "scenario_ids": list(scenario_ids),
        "budgets": list(budgets),
        "primary_budget": 300 if 300 in budgets else budgets[0],
        "maximum_ranked_count": max_budget,
        "public_pool_window_count": len(pool_rows),
        "policy": {
            "ranking": "Base probability, descending, per class",
            "similarity_filter": "none",
            "diversity_step": "none",
            "temporal_separation_keyframes": TEMPORAL_SEPARATION,
            "class_merge": "round-robin in frozen scenario order",
        },
        "prefixes": {method: _prefix_summary(ranking, budgets, scenario_ids)},
        "files": {method: filename},
    }
    (output_dir / "score_ranked_report.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


@dataclass(frozen=True, slots=True)
class _ScoreCandidate:
    pool_index: int
    scenario_id: str
    probability: float
    threshold: float


def _score_temporal_filter(
    candidates: list[_ScoreCandidate],
    pool_rows: list[JsonObject],
) -> list[_ScoreCandidate]:
    scene_starts: dict[str, list[int]] = {}
    result: list[_ScoreCandidate] = []
    for candidate in candidates:
        row = pool_rows[candidate.pool_index]
        if _temporally_allowed(row, scene_starts):
            _record_start(row, scene_starts)
            result.append(candidate)
    return result


def _merge_score_queues(
    method: str,
    scenario_ids: tuple[str, ...],
    queues: dict[str, list[_ScoreCandidate]],
    pool_rows: list[JsonObject],
    budget: int,
) -> list[JsonObject]:
    positions = {scenario_id: 0 for scenario_id in scenario_ids}
    selected_ids: set[str] = set()
    scene_starts: dict[str, list[int]] = {}
    selected: list[JsonObject] = []
    quota = budget // len(scenario_ids)
    for _ in range(quota):
        for scenario_id in scenario_ids:
            queue = queues[scenario_id]
            while positions[scenario_id] < len(queue):
                candidate = queue[positions[scenario_id]]
                positions[scenario_id] += 1
                row = pool_rows[candidate.pool_index]
                if row["window_id"] in selected_ids or not _temporally_allowed(
                    row, scene_starts
                ):
                    continue
                selected_ids.add(row["window_id"])
                _record_start(row, scene_starts)
                selected.append(
                    {
                        "rank": len(selected) + 1,
                        "method": method,
                        "window_id": row["window_id"],
                        "scene_token": row["scene_token"],
                        "scene_name": row["scene_name"],
                        "log_token": row["log_token"],
                        "start_frame_index": row["start_frame_index"],
                        "end_frame_index": row["end_frame_index"],
                        "query_scenario_id": candidate.scenario_id,
                        "base_probability": candidate.probability,
                        "base_threshold": candidate.threshold,
                    }
                )
                break
            else:
                raise SelectionError(f"{method} exhausted candidates for {scenario_id}")
    return selected


def _rank_class_candidates(
    *,
    scenario_id: str,
    class_index: int,
    threshold: float,
    pool_rows: list[JsonObject],
    pool_embeddings: NDArray[np.float32],
    fn_rows: list[JsonObject],
    fn_embeddings: NDArray[np.float32],
    shortlist_size: int,
) -> list[_Candidate]:
    bank_indices = [
        int(row["bank_index"])
        for row in fn_rows
        if row["scenario_id"] == scenario_id
    ]
    if not bank_indices:
        raise SelectionError(f"FN bank has no rows for {scenario_id}")
    similarities = np.max(
        pool_embeddings @ fn_embeddings[bank_indices].T,
        axis=1,
    )
    relevant = sorted(
        range(len(pool_rows)),
        key=lambda index: (-float(similarities[index]), pool_rows[index]["window_id"]),
    )[: min(shortlist_size, len(pool_rows))]
    candidates = [
        _Candidate(
            pool_index=index,
            scenario_id=scenario_id,
            probability=float(pool_rows[index]["base_probabilities"][class_index]),
            threshold=threshold,
            similarity=float(similarities[index]),
            boundary_margin=boundary_margin(
                float(pool_rows[index]["base_probabilities"][class_index]),
                threshold,
            ),
        )
        for index in relevant
    ]
    return sorted(
        candidates,
        key=lambda item: (
            item.boundary_margin,
            -item.similarity,
            pool_rows[item.pool_index]["window_id"],
        ),
    )


def _diversify_candidates(
    candidates: list[_Candidate],
    pool_rows: list[JsonObject],
    pool_embeddings: NDArray[np.float32],
    cluster_count: int,
) -> list[_Candidate]:
    deduplicated = _temporal_filter(candidates, pool_rows)
    count = min(cluster_count, len(deduplicated))
    if count <= 0:
        raise SelectionError("diversity candidate set is empty")
    indices = [candidate.pool_index for candidate in deduplicated]
    labels = MiniBatchKMeans(
        n_clusters=count,
        random_state=CLUSTER_SEED,
        n_init=10,
        batch_size=256,
    ).fit_predict(pool_embeddings[indices])

    buckets: dict[int, list[_Candidate]] = {}
    for candidate, label in zip(deduplicated, labels, strict=True):
        cluster_id = int(label)
        buckets.setdefault(cluster_id, []).append(
            replace(candidate, cluster_id=cluster_id)
        )
    core_positions = {
        candidate.pool_index: position
        for position, candidate in enumerate(deduplicated)
    }
    cluster_order = sorted(
        buckets,
        key=lambda cluster_id: core_positions[buckets[cluster_id][0].pool_index],
    )
    result: list[_Candidate] = []
    for position in range(max(len(bucket) for bucket in buckets.values())):
        for cluster_id in cluster_order:
            bucket = buckets[cluster_id]
            if position < len(bucket):
                result.append(bucket[position])
    return result


def _random_ranking(
    pool_rows: list[JsonObject],
    budget: int,
    seed: int,
    method: str,
) -> list[JsonObject]:
    order = np.random.default_rng(seed).permutation(len(pool_rows))
    selected: list[JsonObject] = []
    scene_starts: dict[str, list[int]] = {}
    for pool_index in order:
        row = pool_rows[int(pool_index)]
        if not _temporally_allowed(row, scene_starts):
            continue
        _record_start(row, scene_starts)
        selected.append(_selection_row(method, len(selected) + 1, row, None))
        if len(selected) == budget:
            return selected
    raise SelectionError("public Pool cannot satisfy the Random budget")


def _merge_class_queues(
    method: str,
    scenario_ids: tuple[str, ...],
    queues: dict[str, list[_Candidate]],
    pool_rows: list[JsonObject],
    budget: int,
) -> list[JsonObject]:
    positions = {scenario_id: 0 for scenario_id in scenario_ids}
    selected_ids: set[str] = set()
    scene_starts: dict[str, list[int]] = {}
    selected: list[JsonObject] = []
    quota = budget // len(scenario_ids)
    for _ in range(quota):
        for scenario_id in scenario_ids:
            queue = queues[scenario_id]
            while positions[scenario_id] < len(queue):
                candidate = queue[positions[scenario_id]]
                positions[scenario_id] += 1
                row = pool_rows[candidate.pool_index]
                if row["window_id"] in selected_ids or not _temporally_allowed(
                    row, scene_starts
                ):
                    continue
                selected_ids.add(row["window_id"])
                _record_start(row, scene_starts)
                selected.append(
                    _selection_row(method, len(selected) + 1, row, candidate)
                )
                break
            else:
                raise SelectionError(f"{method} exhausted candidates for {scenario_id}")
    return selected


def _temporal_filter(
    candidates: list[_Candidate],
    pool_rows: list[JsonObject],
) -> list[_Candidate]:
    scene_starts: dict[str, list[int]] = {}
    result: list[_Candidate] = []
    for candidate in candidates:
        row = pool_rows[candidate.pool_index]
        if _temporally_allowed(row, scene_starts):
            _record_start(row, scene_starts)
            result.append(candidate)
    return result


def _temporally_allowed(
    row: JsonObject,
    scene_starts: dict[str, list[int]],
) -> bool:
    scene_token = cast(str, row["scene_token"])
    start = int(row["start_frame_index"])
    return all(
        abs(start - selected_start) >= TEMPORAL_SEPARATION
        for selected_start in scene_starts.get(scene_token, [])
    )


def _record_start(row: JsonObject, scene_starts: dict[str, list[int]]) -> None:
    scene_token = cast(str, row["scene_token"])
    scene_starts.setdefault(scene_token, []).append(int(row["start_frame_index"]))


def _selection_row(
    method: str,
    rank: int,
    pool_row: JsonObject,
    candidate: _Candidate | None,
) -> JsonObject:
    result: JsonObject = {
        "rank": rank,
        "method": method,
        "window_id": pool_row["window_id"],
        "scene_token": pool_row["scene_token"],
        "scene_name": pool_row["scene_name"],
        "log_token": pool_row["log_token"],
        "start_frame_index": pool_row["start_frame_index"],
        "end_frame_index": pool_row["end_frame_index"],
    }
    if candidate is not None:
        result.update(
            {
                "query_scenario_id": candidate.scenario_id,
                "base_probability": candidate.probability,
                "base_threshold": candidate.threshold,
                "similarity": candidate.similarity,
                "boundary_margin": candidate.boundary_margin,
            }
        )
        if candidate.cluster_id is not None:
            result["cluster_id"] = candidate.cluster_id
    return result


def _prefix_summary(
    rows: list[JsonObject],
    budgets: tuple[int, ...],
    scenario_ids: tuple[str, ...],
) -> JsonObject:
    result: JsonObject = {}
    for budget in budgets:
        prefix = rows[:budget]
        class_counts = {
            scenario_id: sum(
                row.get("query_scenario_id") == scenario_id for row in prefix
            )
            for scenario_id in scenario_ids
        }
        cluster_profile: JsonObject = {}
        for scenario_id in scenario_ids:
            cluster_counts: dict[int, int] = {}
            for row in prefix:
                if row.get("query_scenario_id") != scenario_id or "cluster_id" not in row:
                    continue
                cluster_id = int(row["cluster_id"])
                cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
            if cluster_counts:
                total = sum(cluster_counts.values())
                cluster_profile[scenario_id] = {
                    "coverage": len(cluster_counts),
                    "maximum_cluster_share": max(cluster_counts.values()) / total,
                }
        result[str(budget)] = {
            "count": len(prefix),
            "unique_windows": len({row["window_id"] for row in prefix}),
            "log_coverage": len({row["log_token"] for row in prefix}),
            "query_class_counts": class_counts,
            "cluster_profile": cluster_profile,
        }
    return result


def _read_pool_rows(path: Path, class_count: int) -> list[JsonObject]:
    allowed = set(PUBLIC_U_FIELDS) | {"pool_index", "base_probabilities"}
    rows = _read_jsonl(path)
    for index, row in enumerate(rows):
        probabilities = np.asarray(row.get("base_probabilities"), dtype=np.float64)
        if (
            set(row) != allowed
            or row.get("partition") != "u"
            or row.get("pool_index") != index
            or probabilities.shape != (class_count,)
            or not np.isfinite(probabilities).all()
            or bool((probabilities < 0.0).any())
            or bool((probabilities > 1.0).any())
        ):
            raise SelectionError(f"invalid public Pool row at index {index}")
    return rows


def _read_fn_rows(path: Path, scenario_ids: tuple[str, ...]) -> list[JsonObject]:
    rows = _read_jsonl(path)
    for index, row in enumerate(rows):
        if row.get("bank_index") != index or row.get("scenario_id") not in scenario_ids:
            raise SelectionError(f"invalid FN row at index {index}")
    return rows


def _load_embeddings(path: Path, expected_rows: int) -> NDArray[np.float32]:
    value = np.load(path, mmap_mode="r")
    if value.shape != (expected_rows, 768) or value.dtype != np.float32:
        raise SelectionError(f"invalid embedding matrix: {path}")
    if not np.isfinite(value).all():
        raise SelectionError(f"non-finite embedding matrix: {path}")
    return cast(NDArray[np.float32], value)


def _write_jsonl(path: Path, rows: list[JsonObject]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(
                json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"
            )


def _read_json(path: Path) -> JsonObject:
    return cast(JsonObject, json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[JsonObject]:
    with path.open(encoding="utf-8") as source:
        return [cast(JsonObject, json.loads(line)) for line in source]
