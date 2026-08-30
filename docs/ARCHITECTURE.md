# Architecture

Specification: Frozen for `v0.8-three-class-loop`

## Design Rule

This is a personal offline algorithm experiment. Architecture means that every
stage has a clear input and output; it does not mean adding many interfaces.

- Use the nuScenes devkit for standard dataset access and coordinate transforms.
- Use plain functions, small dataclasses, JSONL, and NumPy arrays.
- Keep only checks that protect labels, splits, hidden data, or fair evaluation.
- Add a shared abstraction only when two current stages really need it.
- Do not build registries, services, databases, or plugin systems in advance.

## Main Flow

```text
NuScenes(version, dataroot)
        |
        +--> whole-log split
        |
        +--> sample -> CAM_FRONT / annotations / calibrated boxes
        |
        v
CAM_FRONT eligibility + Strem scene streams
        |
        v
Strem full-scene matches + persistent bindings
        |
        v
five-frame private labels --------> label-free public U view
        |
        v
DINO frame cache -> LR / GRU -> Development FN bank
                                  |
                                  v
                       Random / Mining
                                  |
                                  v
                         Oracle label reveal
                                  |
                                  v
                         controlled retraining
                                  |
                                  v
                         Held-out Test
```

## Stage Boundaries

### 1. nuScenes Access

Each data-building script constructs one official devkit object:

```python
nusc = NuScenes(
    version="v1.0-trainval",
    dataroot="/path/to/nuscenes",
    verbose=False,
)
```

The devkit already provides:

- constant-time `nusc.get(table, token)` lookup;
- `sample["data"]["CAM_FRONT"]` for the annotated front-camera keyframe;
- `sample["anns"]` for annotations belonging to a sample;
- scene linked lists through `first_sample_token` and `sample["next"]`;
- calibrated annotation boxes in a camera coordinate frame.

The project therefore does not load nuScenes tables itself or rebuild token and
reverse indexes.

### 2. Split and Visual Cleaning

`splits.py` assigns complete logs before any window exists. Official validation
logs become Frozen Test.

`projection.py` asks the devkit to transform one annotation box into the
CAM_FRONT coordinate frame. Project code only calculates the clipped 2D box and
applies the frozen eligibility filter:

- every box corner is in front of the camera;
- at least one corner is inside the image;
- clipped area retains at least 50% of the raw projected area;
- projected height is at least 16 px after 518x518 letterbox resizing;
- nuScenes visibility rank is 2, 3, or 4.

### 3. Strem Scene Mining

The external converter creates one ego-BEV stream per complete scene. After
visual cleaning, `StremAdapter.run_scene(stream, spec)` returns:

```text
status + frame bounds + symbolic time bounds + clock constraints + bindings
```

Strem owns temporal pattern matching and persistent object identity. Python does
not translate the specification into a second rule implementation.

`scenario_events.py` only merges touching or overlapping intervals when their
scene, scenario, and object bindings are identical.

### 4. Five-Frame Labels

Full-scene matches locate possible events. Every overlapping five-frame window
is then passed to the same Strem specification:

- bounded match -> `positive`;
- overlap without bounded match -> `ignore`;
- no full-scene overlap -> `negative`;
- execution/evidence failure -> `invalid`.

The public U file is derived by removing labels, event IDs, bindings, boxes, and
all Strem evidence from private window records.

### 5. Visual Models

DINOv2-Small encodes each unique frame once into a 384-dimensional feature.
Overlapping windows reuse the cache.

- LastFrame-LR asks whether the final image alone is enough.
- Mean5-LR asks whether order-free aggregation is enough.
- Global-GRU asks whether ordered features add useful information.

The encoder is frozen. Only LR and GRU classifiers are trained.

### 6. Bad-Case Mining and Feedback

The seed-17 Base model and its frozen Development thresholds define false
negatives. Multiple overlapping windows from the same real event are reduced to
one representative.

The selector may read only public U metadata, DINO embeddings, and Base model
probabilities. It cannot read U labels or Strem evidence. Selected IDs freeze
before the private Oracle reveals their labels.

The causal comparison keeps architecture and training fixed:

```text
Base vs Random-Oracle x3 vs Mining-Oracle
```

### 7. Evaluation

Model inference first writes predictions. Final scoring reads those frozen
predictions and Frozen Test labels. Frozen Test never influences task design,
selection, thresholds, early stopping, or retraining.

## Maintained Files

One stage is one script plus its module. A stage reads the previous stage's
named artifact, so the pipeline is joined by filenames rather than by a runner
or a workflow engine.

| # | Script | Module | Output artifact |
| ---: | --- | --- | --- |
| 1 | `build_split_manifest.py` | `splits.py` | `split.json` |
| 2 | `convert_nuscenes_for_strem.py` | `strem_converter.py` | scene streams, `nuscenes_numeric_id_map.json` |
| 3 | `build_strem_eligible_streams.py` | `strem_eligibility.py`, `projection.py` | eligible streams, `cam_front_eligibility_summary.json` |
| 4 | `build_window_dataset.py` | `window_dataset.py`, `scenario_events.py`, `strem_adapter.py` | `events.jsonl`, `windows.jsonl`, `data_profile.json`, `u_windows.jsonl` |
| 5 | `extract_dino_features.py` | `dino_features.py` | `frame_features.npy`, `frame_index.jsonl`, `feature_manifest.json` |
| 6 | `train_lr_baselines.py` | `lr_baselines.py` | `lr_baseline_report.json` |
| 7 | `train_gru_baseline.py` | `gru_baseline.py` | `gru_report.json`, checkpoints, predictions |
| 8 | `build_false_negative_bank.py` | `false_negative_bank.py` | `fn_bank.jsonl`, `fn_embeddings.npy`, `fn_bank_report.json` |
| 9 | `prepare_public_pool.py` | `public_pool.py` | `pool_windows.jsonl`, `pool_embeddings.npy`, `pool_report.json` |
| 10 | `freeze_selection_rankings.py` | `selection.py` | `random_ranked.jsonl`, `mining_ranked.jsonl`, `selection_report.json` |
| 10b | `freeze_random_variance.py` | `selection.py` | `random_seed102_ranked.jsonl`, `random_seed103_ranked.jsonl`, `random_variance_report.json` |

Stage 4 writes the private and public views in one pass; that is where the
Oracle boundary is created. Stages 9, 10, and 10b may read only the public view.
Stage 10b adds Random batches for variance estimation into a separate directory
and never rewrites the frozen stage-10 artifact.
Every stage refuses to write into an existing output directory, so a rerun
produces a new named artifact instead of silently replacing one.

Oracle reveal now joins frozen selected IDs to `windows.jsonl` without changing
their ranks. Feedback retraining reuses the stage-7 GRU and changes only the
added selected windows; later Frozen-Test scoring reads already-written frozen
prediction files.

One-off TaskDesign galleries and candidate generators are no longer maintained
code. Their observed conclusions remain documented, while the three accepted
specifications remain under `specs/task_spec_v2`.

## Storage

```text
private_source/   nuScenes metadata, CAM_FRONT images, DINO feature cache
private_oracle/   Strem streams, events, and hidden labels
public_pool/      selector-safe U records, embeddings, probabilities
runs/             checkpoints, selections, predictions, reports
```

The observed formal artifact names under these roots are recorded in
[Development Environment](DEVELOPMENT_ENVIRONMENT.md), so a later stage can name
its exact input instead of rediscovering it.

JSON/JSONL is used for inspectable records and indexed `.npy` arrays for dense
features. SQLite and a vector database are unnecessary for this one-pass
experiment.

## Experiment-Critical Checks

- one log belongs to one partition;
- windows never cross scenes;
- persistent bindings keep the same object across an event;
- Strem errors never become negative labels;
- public U contains no Oracle fields;
- selectors receive the same budget;
- feedback runs differ only in added data;
- Frozen Test is read only after predictions freeze.
