# Data Specification

Status: Planned

Schema version: `0.1`

## Initial Data Strategy

Use a small, reviewable public-data subset before scaling. The first integration
may use nuScenes v1.0-mini because an existing converter and domain knowledge are
available. A later cross-dataset evaluation should use data not involved in model
or prompt development.

Raw datasets are external dependencies and are never committed to this repo.

## Data Layers

```text
data/
  registry/             Dataset and scene references
  derived/              Search documents and structured streams
  queries/              Canonical and paraphrased queries
  relevance/            Reviewed query-segment judgments
  model_outputs/        Versioned perception or VLM predictions
  runs/                 Agent and evaluation run records
  bad_cases/            Reproducible failure records
```

Generated content, indexes, weights, and private data will be ignored by Git.

## Scene Segment Record

```json
{
  "schema_version": "0.1",
  "dataset": "nuscenes",
  "dataset_version": "v1.0-mini",
  "scene_id": "scene_001",
  "segment_id": "scene_001_000120_000160",
  "start_time": 12.0,
  "end_time": 16.0,
  "frame_refs": [],
  "stream_ref": "stream_001",
  "object_classes": ["pedestrian", "car"],
  "metadata": {},
  "source_provenance": {}
}
```

## Query Record

```json
{
  "query_id": "query_001",
  "canonical_text": "pedestrian reappears within two seconds after occlusion",
  "scenario_family": "occlusion_reappearance",
  "entities": ["pedestrian", "vehicle"],
  "spatial_predicates": ["occluded", "visible"],
  "temporal_constraints": [{"operator": "within", "seconds": 2.0}],
  "failure_condition": null,
  "review_status": "human_reviewed",
  "source": "manual"
}
```

Paraphrases must retain a reference to the canonical query. Generated paraphrases
are not considered human labels until reviewed.

## Relevance Judgment

```json
{
  "query_id": "query_001",
  "segment_id": "scene_001_000120_000160",
  "grade": 2,
  "judgment_source": "human_and_monitor",
  "evidence": ["frame_124", "frame_138"],
  "notes": "temporal constraint satisfied"
}
```

Suggested grades:

- `0`: not relevant
- `1`: partially relevant or missing one required condition
- `2`: fully relevant

Formal matches may support a judgment but do not replace review when perception
annotations, natural-language intent, or failure labels are uncertain.

## Model Output Record

Store model name, version, configuration, input references, structured predictions,
confidence, latency, and provenance. Do not store API credentials or hidden model
reasoning.

## Agent Run Record

Every run includes:

- `run_id`, `query_id`, and dataset split
- Code, prompt, index, model, and monitor versions
- Planned steps and actual tool calls
- Candidate rankings and evidence references
- Validation results, retries, warnings, and failures
- Stage latency, token usage, and cost
- Final structured result

## Split Policy

- Split by complete scene or driving log.
- Keep all segments from one scene in one split.
- Keep paraphrases of one canonical query in one split unless performing an
  explicitly documented paraphrase-generalization test.
- Hold out scenario families when testing compositional generalization.
- Hold out at least one model version for failure-analysis evaluation.
- Never expose test relevance labels or formal match results to the Agent prompt.

## Initial Targets

The first learning dataset may contain:

- 100 to 300 scene segments
- At least 30 canonical queries
- Three to five reviewed paraphrases per canonical query
- Explicit positive, partial, and hard-negative judgments

These are development targets, not claims of statistical sufficiency. Scale only
after the labeling protocol and learning curves are understood.

## Quality Checks

- Schema and referential-integrity validation
- Timestamp order and segment-bound checks
- Duplicate and near-duplicate detection
- Class and scenario-family distribution reports
- Query-label and evidence consistency checks
- Train/test leakage checks
- Dataset license and source records
- Clear separation of human, monitor-derived, model-generated, and synthetic data

