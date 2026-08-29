# Driving Scene Mining and High-Value Data Feedback Loop

Specification: Frozen for `v0.8-three-class-loop`

Chinese title: **自动驾驶场景挖掘与高价值数据闭环**

Report subtitle: *Strem temporal event mining and fixed-budget selection based
on bad-case similarity, model uncertainty, and visual diversity.*

This evidence-oriented project studies one concrete autonomous-driving data-loop
question:

> Under the same simulated annotation budget `N=300`, can data selected by
> Development bad-case similarity, model-boundary uncertainty, and visual
> diversity improve Held-out Test Macro-AP more than random selection?

Improvement is a hypothesis. The project keeps negative results instead of
tuning the experiment until the desired answer appears.

## What the Project Does

```text
nuScenes v1.0-trainval
  -> split complete driving logs
  -> clean annotations that are unusable in CAM_FRONT
  -> Strem mines three temporal scene types
  -> create labeled five-frame windows
  -> DINOv2 extracts one feature per image
  -> LR and GRU learn scene classification
  -> collect Development false negatives
  -> select public-Pool windows without reading their labels
  -> reveal their hidden labels and retrain
  -> compare Random with integrated Mining on held-out data
```

In plain language: first define what scene to find, then automatically find it
in driving logs, train a visual model to recognize it, inspect what the model
misses, add more useful data, and measure whether the model actually improves.

## Frozen Scene Labels

Executable specifications live in [`specs/task_spec_v2`](specs/task_spec_v2).

| Label | Formal meaning | Not claimed |
| --- | --- | --- |
| `pedestrian_ego_near_zone_entry` | the same eligible pedestrian moves from at least 20 m to at most 15 m from ego within 2 s | collision risk, TTC, intent |
| `vehicle_relative_corridor_entry` | the same eligible motor vehicle moves from `abs(y)>=3 m` to `abs(y)<=1.5 m` while `0<=x<=30 m`, within 2 s | an intentional lane change |
| `pedestrian_vehicle_center_proximity_hold` | the same pedestrian and vehicle annotation centers remain within 5 m for at least 1 s | body clearance or conflict |

All three Gate-A classes enter the data loop. Gate B records how learnable each
class was for the initial model; it does not remove the weak proximity class.
That class is deliberately retained as the challenge class whose response to
targeted data is measured.

## Why Each Main Tool Exists

- **nuScenes devkit** loads official relational tables, token references,
  sample chains, boxes, calibrations, and camera records. The project does not
  rebuild these indexes.
- **Strem v0.3.0** evaluates precise temporal patterns and keeps the same
  object binding across frames. It is the scene miner and private simulated
  Oracle, not a visual model.
- **DINOv2-Small** converts each CAM_FRONT image into a frozen 384-dimensional
  visual feature.
- **Logistic Regression** provides simple last-frame and order-free baselines.
- **PyTorch GRU** processes the ordered five-frame sequence.
- **Similarity, uncertainty, and MiniBatchKMeans diversity** select Pool
  windows related to known failures without reading their hidden labels.

Strem is an external research dependency configured by `STREM_BIN`. Its source
repository is never copied into or modified by this project.

## Data and Evaluation Contract

Five consecutive annotated CAM_FRONT keyframes form one window. Complete
`log_token` groups are split before windows are generated:

| Partition | Logs | Use |
| --- | ---: | --- |
| Design Set (`task_design`) | 5 | define and audit labels |
| Initial Training Set (`l0`) | 13 | initial training |
| Development Set (`development`) | 7 | early stopping, thresholds, false negatives |
| Simulated Unlabeled Pool (`u`) | 25 | hidden-label candidate pool |
| Held-out Test Set (`frozen_test`) | 18 official-val logs | final evaluation only |

Strem labels each class independently:

- `positive`: the five-frame substream itself matches the specification;
- `negative`: the window does not overlap a complete-scene event;
- `ignore`: it overlaps an event but does not contain a valid five-frame match;
- `invalid`: required evidence or Strem execution failed.

`ignore` and `invalid` are masked from training. Public U records contain frame,
scene, log, and time references only; they contain no label, binding, 3D box, or
Strem result.

The primary comparison is:

```text
Base vs Random-300-Oracle (seeds 101/102/103) vs Mining-300-Oracle
```

Random is a distribution over batches while integrated Mining is deterministic,
so three independent Random batches are frozen before any label is revealed. The
question is whether the one Mining result falls outside the Random range, not
whether it beats one Random draw.

All feedback models start from the same Initial Training data and train the same GRU from
scratch. They differ only in the added windows. The final metrics are per-class
Average Precision and three-class Macro-AP on the Held-out Test Set. An
Oracle-only nested `150/300/600` budget curve checks whether the conclusion
depends on the chosen budget; `N=300` remains the primary comparison.

A remote VLM is only a later, bounded auto-label comparison on already-frozen
Mining-300 IDs. It is not the primary Oracle or a selector input.

## Code Map

The maintained pipeline intentionally has few modules:

| Module | Responsibility |
| --- | --- |
| `splits.py` | deterministic whole-log split using devkit records |
| `projection.py` | devkit 3D-box transform plus the frozen image filter |
| `strem_converter.py` | call the pinned external nuScenes-to-Strem converter |
| `strem_eligibility.py` | keep only CAM_FRONT-learnable objects in streams |
| `strem_adapter.py` | run Strem and parse its result |
| `scenario_events.py` | group Strem intervals with identical bindings |
| `window_dataset.py` | create five-frame positive/negative/ignore labels |
| `dino_features.py` | extract and cache frozen frame features |
| `lr_baselines.py` | train LastFrame-LR and Mean5-LR |
| `gru_baseline.py` | train and evaluate the ordered GRU |
| `false_negative_bank.py` | deduplicate Development false negatives by event |
| `public_pool.py` | run the frozen Base and build label-free selection vectors |
| `selection.py` | freeze Random and integrated Mining rankings |

The eleven scripts under `scripts/` are direct stage entry points. Historical
gallery builders, candidate-spec generators, archive extractors, and duplicate
raw-JSON indexers were removed after their results were frozen.

## Current Observed Results

Completed on private nuScenes trainval data:

- 850 scenes, 34,149 keyframes, 68 complete logs;
- 1,430 Strem event groups and 30,749 five-frame windows;
- 11,639 label-free public-U windows;
- 120,035 of 859,857 target annotations passed the CAM_FRONT eligibility rule;
- DINO cache shape `(34149, 384)`, finite `float32`;
- Development Macro-AP: LastFrame-LR `0.1499`, Mean5-LR `0.1942`;
- three-class GRU Development per-class AP `0.1675`, `0.0537`, and `0.5011`
  in scene-table order (three-seed means);
- a historical two-class diagnostic reached Macro-AP `0.3636 ± 0.0157`, with
  reversed-frame Macro-AP `0.3328 ± 0.0232`;
- the formal three-class FN bank has 69 event rows represented by 61 unique
  Development windows (`26/21/22` event rows by scenario-table order);
- public-Pool Base inference produced 11,639 unique rows and a finite,
  unit-normalized `(11639,768)` selection matrix with no Oracle fields.
- Random and integrated Mining each produced one frozen 600-window ranking with
  nested `150/300/600` prefixes. All IDs are unique, satisfy the five-keyframe
  temporal separation, and contain no Oracle fields.

These are Development, data-construction, or label-free selection observations.
Oracle reveal, feedback retraining, remote-VLM labeling, and Held-out Test
metrics are not yet completed.

## Environment and Checks

Python 3.12 and uv are used. The formal CPU environment is on Procyon; local
nuScenes mini is smoke-only.

```bash
uv sync --locked --all-groups
uv run ruff check .
uv run mypy src scripts tests
uv run pytest
uv lock --check
```

Core commands now use the official nuScenes root rather than individual JSON
table paths:

```bash
uv run python scripts/build_split_manifest.py \
  --dataroot /path/to/nuscenes \
  --output /path/to/split.json

uv run python scripts/build_strem_eligible_streams.py \
  --dataroot /path/to/nuscenes \
  --raw-stream-dir /path/to/raw_streams \
  --output-dir /path/to/eligible_streams
```

See [Architecture](docs/ARCHITECTURE.md), [Data Spec](docs/DATA_SPEC.md),
[Evaluation Plan](docs/EVALUATION_PLAN.md), and
[Stage Plan](docs/STAGE_PLAN.md) for the frozen experiment details.

## Scope Boundary

Core: scene mining, data cleaning, five-frame classification, bad-case mining,
fixed-budget data selection, controlled retraining, and evaluation.

Not core: Agent, RAG, LLM orchestration, SFT/LoRA, vector databases, Spark,
Airflow, multi-sensor fusion, prediction/planning/control, online learning, or
production safety claims.
