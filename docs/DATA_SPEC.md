# Data Specification

Specification: Frozen for `v0.8-three-class-loop`

## Dataset Boundary

Formal experiments use nuScenes `v1.0-trainval`.

- Official train supplies TaskDesign, L0, Development, and U.
- Official validation is Frozen Test.
- Model input uses annotated `CAM_FRONT` keyframes only.
- Metadata supplies scenes, samples, poses, calibration, annotations,
  categories, instances, visibility, and logs.
- Sweeps, LiDAR, Radar, and the other five cameras are outside the model input.
- Local `v1.0-mini` is for smoke tests only.

Raw nuScenes JPEGs do not have boxes drawn into their pixels. The dataset stores
3D annotations separately in global coordinates. Every 2D rectangle used here is
derived by camera projection and clipping.

Formal media and derived features stay private and outside Git.

## Whole-Log Split

Split complete `log_token` groups before generating events or windows. The
`split-v1` order is deterministic:

1. serialize `split-v1 + NUL + v1.0-trainval + NUL + log_token`;
2. compute SHA-256;
3. sort by `(digest, log_token)`;
4. use cumulative rank boundaries at 10%, 35%, 50%, and 100%;
5. assign official validation directly to Frozen Test.

Observed split:

| Partition | Logs | Scenes | Label access |
| --- | ---: | ---: | --- |
| TaskDesign | 5 | 68 | visible for scenario design |
| L0 | 13 | 232 | visible for initial training |
| Development | 7 | 78 | visible for tuning and FN analysis |
| U | 25 | 322 | hidden until Oracle reveal |
| Frozen Test | 18 | 150 | final scoring only |

Official train and validation share no log. Windows, events, and adjacent frames
from one log therefore cannot cross partitions.

## Observed CAM_FRONT Preparation

The trainval metadata contains 850 scenes, 34,149 samples, and 68 logs. The
official devkit resolves one annotated `CAM_FRONT` keyframe for every sample;
TaskDesign contains 2,747 such keyframes.

All ten official blob volumes were scanned against that exact allowlist. Their
match counts were:

```text
0 / 319 / 0 / 0 / 690 / 569 / 0 / 685 / 484 / 0
```

The disjoint union is 2,747 verified `1600x900` JPEGs, all currently present in
the private Procyon data root. Zero-match archive volumes are storage-layout
observations, not scene labels.

## Camera and Visual-Eligibility Decision

A historical TaskDesign audit observed that all six annotated camera channels
were structurally complete. CAM_FRONT led none of four declared static coverage
proxies. On 2026-08-22, the author nevertheless retained CAM_FRONT as an
explicit front-view coherence and one-month scope choice—not as an empirically
best camera claim.

A second historical audit covered all 2,747 TaskDesign images and 18,182
projected pedestrian/motor annotations. Of 13,577 targets at least 16 letterbox
pixels high, 2,137 retained less than half of their projected area in CAM_FRONT.
Panoramic nuScenes visibility also could not replace a camera-specific clipping
check.

The frozen event-frame eligibility rule is:

```text
valid CAM_FRONT projection
and letterbox_height_px >= 16
and clipped_area_fraction >= 0.5
and visibility_token in {2, 3, 4}
```

The complete filter retained 7,915 TaskDesign annotations from the 13,577 that
passed the height threshold. This is a data-quality filter, not proof of clarity,
absence of occlusion, or model learnability. The private reports preserve the
detailed counts and reviewed images; their one-off audit/gallery implementation
is not maintained as core pipeline code.

The maintained full-trainval eligibility build later applied the same frozen
rule to all 34,149 frames. It observed 859,857 pedestrian/motor annotations and
retained 120,035:

| Project category | Eligible annotations |
| --- | ---: |
| pedestrian | 33,652 |
| car | 61,334 |
| truck | 15,007 |
| bus | 4,171 |
| trailer | 3,564 |
| construction_vehicle | 2,307 |

For Strem specifications only, the five motor categories are normalized to the
single formal class `motor_vehicle`. IDs, metric positions, timestamps, and the
numeric-ID-to-`instance_token` map remain unchanged. The unfiltered converter
streams are retained separately, so this normalization does not overwrite
source evidence.

## Data Cleaning and Profiling Contract

Cleaning creates a traceable derived dataset; it never edits the nuScenes
source files. The required layers are deliberately direct:

1. structural integrity: the devkit resolves standard token relations,
   timestamps are ordered, and every window stays inside one scene and log;
2. category normalization: the five declared motor categories map to
   `motor_vehicle` for Strem while original category and `instance_token`
   identity remain recoverable;
3. visual eligibility: invalid projection, targets behind the camera, no image
   overlap, height below 16 letterbox pixels, clipped-area fraction below 0.5,
   or visibility rank below 2 are filtered from formal event streams;
4. label quality: Strem errors and missing evidence become `invalid`, partial
   event windows become `ignore`, and neither is silently treated as negative.

The data-profile report records total source/retained/filtered annotations, the
frozen filter definitions, category/scene/log/split distributions, and
positive/negative/ignore/invalid window counts. The existing full-trainval v1
eligibility build did not persist mutually exclusive rejection-reason counts,
so the profile says so instead of reconstructing or inventing them. A later
selected-batch report adds positive yield, usable-label yield, log coverage,
duplicate rate, and ignore cost. Size, visibility, time-of-day, weather, and
scene complexity are added only when the metadata is available and the
analysis uses them. Pandas is sufficient at this project scale; PySpark is an
side topic, not a forced runtime dependency.

## Time and Window Contract

A window contains five consecutive annotated keyframes from one scene. Adjacent
windows move by one keyframe and overlap by four:

```text
W0 = frames 0..4
W1 = frames 1..5
W2 = frames 2..6
```

Use true timestamps, not an assumed fixed frame period. A window never crosses a
scene boundary.

Raw nuScenes sample timestamps are integer microseconds and are not zero-based.
The pinned converter turns them into scene-relative seconds:

```text
converted_timestamp[i] = (raw_timestamp[i] - raw_timestamp[0]) / 1_000_000
```

Its converted first timestamp is therefore `0.0`, but this is a converter
choice, not a nuScenes or Strem requirement. For any input first timestamp
`tau_0`, Strem v0.3.0 exact semantics use `{tau_0}`; later frame `i` starts in
`[tau_(i-1), tau_i)`. Pass converter timestamps through unchanged. Formal
labels do not use the optional left-censored policy, because that policy
intentionally leaves onset before the first observation unknown.

The pinned converter defines formal ego-BEV positions with the sample's
`LIDAR_TOP` keyframe ego pose. CAM_FRONT projection uses the camera sample-data
ego pose, because the sensors are not perfectly synchronized. The two derived
center distances can therefore differ slightly. Strem thresholds and gallery
labels use the formal stream distance; the camera-pose distance may be retained
only as projection evidence. A TaskDesign gallery audit observed a maximum
absolute difference of 0.322m across the 68 participating frames of the first
pedestrian-entry candidate selected for review.

## Scenario and Event Contract

The three current frozen scenario formulas are owned by `specs/task_spec_v2`
and documented in [Project Scope](PROJECT_SCOPE.md). Each event uses stable
object identity and real timestamps. Strem runs on a complete scene before
windows are created.

Python may merge touching or overlapping Strem result intervals only when all of these
are equal:

- scene;
- scenario and task-spec version;
- complete object bindings.

Different objects remain different events.

A Strem result interval contains supporting frame indices plus symbolic
continuous start/end regions and readable clock constraints. Strem can
aggregate touching regions with identical bindings and constraints, so the
aggregate frame bounds describe a match region rather than one minimal
five-frame witness. Gate A must verify each specification's terminal semantics.
Window labels are obtained by rerunning that same specification on bounded
five-frame substreams, not by parsing readable constraints in Python.

## Event-to-Window Labels

For every `(window_id, scenario_id)`:

- `positive`: the full-scene event overlaps the window and the same Strem
  specification returns `match` on the five-frame substream;
- `ignore`: the window overlaps an event but the bounded Strem run returns
  `no_match`;
- `negative`: the window has no event overlap;
- `invalid`: a required full-scene or bounded Strem run, or source evidence,
  failed.

`ignore` and `invalid` are masked from loss. A timeout, error, or malformed
result is never a negative.

Formal targets are three independent binary labels, so a window may be
`[1, 0, 1]`; there is no extra “all scenarios” class. Gate B is an initial
learnability diagnostic and never changes or removes a label.

Observed Gate B marked `pedestrian_vehicle_center_proximity_hold` as the weak
challenge class. A read-only audit found 21 independent Development FN events,
above the five-event minimum, so all three scenario labels remain in Base,
selection quotas, retraining, and the final Macro-AP.

## Minimal Persisted Data

Persist only data consumed by the next stage or needed to repeat the experiment.
The core shapes are:

### Scenario event

```text
scene_id, scenario_id,
start_frame_index, end_frame_index,
bindings, source_intervals
```

Each source interval preserves its own symbolic start/end bounds, endpoint
closure, readable constraints, and discrete frame indices. They provide event
evidence and candidate-window bounds. They do not replace the bounded Strem
run that decides whether the rule is satisfiable inside one five-frame input.

### Window and target

```text
window_id, scene_id, log_id, five frame refs, five timestamps, split
labels[C], loss_mask[C], event_refs[C]
```

### Feature metadata

```text
DINO model revision, preprocessing, dtype, array shape, frame-to-row index
```

### Selection

```text
method, selection seed, budget, ordered selected window IDs
```

### Development false-negative bank

```text
scenario_id, event_group_id, representative window_id,
scene/log/start/end metadata, Base probability, embedding row
```

The aligned float32 embedding is:

```text
L2(concat(mean(f0..f4), f4-f0)) -> [768]
```

The bank is a private Development-derived query artifact, not a public-U label
file. It deliberately omits bindings, intervals, 3D evidence, complete target
vectors, reversed predictions, and all U/Frozen-Test labels.

### VLM annotation

```text
selection ID, frozen provider/model/prompt/schema identity,
per-class verdict {positive, negative, uncertain},
evidence frame indices, confidence, limitations, latency, cost
```

The VLM record contains no copied Oracle label or Strem evidence. It is produced
only after Mining IDs freeze and is joined with the private Oracle afterward
for evaluation.

### Training/evaluation result

```text
data selection, model config, training seed, checkpoint, predictions, metrics
```

The existing split file remains the source of log assignments. JSON/JSONL and
indexed NumPy arrays are sufficient. Do not introduce a database, registry,
schema version on every row, or hash on every derived artifact without an actual
consumer.

## Public U and Private Oracle

The public selector view may contain:

- window, scene, log, time, and CAM_FRONT references;
- frozen DINO window embedding;
- Base logits and probabilities;
- non-label acquisition metadata.

It must not contain target labels, Strem results, bindings, instance tokens,
event evidence, 3D boxes, visibility, or other Oracle-derived facts.

Private Oracle files contain complete Strem-derived targets. After selected IDs
freeze, the Oracle reveals only those IDs. Every selected window consumes
budget; `ignore` is not replaced.

The frozen selection artifact contains one 600-row ranked list per method:

```text
rank, method, window_id, scene/log/time,
query_scenario_id (Mining only), similarity, boundary_margin,
cluster_id (integrated Mining diversity diagnostic)
```

It contains no labels or Strem evidence. Budget experiments use the exact
nested prefixes `150 ⊂ 300 ⊂ 600`; they do not rerun selection.

Separate public and private files plus a focused no-label-leak test are enough.

## Remote VLM Label Boundary

The remote VLM receives only the five selected CAM_FRONT images, the frozen
scenario descriptions, and the output schema. It does not receive hidden U
labels, Strem results, bindings, 3D boxes, or distances. Every retained class
must return `positive`, `negative`, or `uncertain`, plus evidence frames and a
short limitation statement.

The private Oracle later reveals the same selected IDs so VLM label quality can
be measured. Exact geometry and duration are difficult to recover from five
monocular frames, so the VLM is a semantic automatic labeler, not ground truth.
`uncertain` is masked in Mining-VLM training but still consumes the original
selection budget. Provider, model version, prompt, schema, image preprocessing,
latency, and cost freeze before the formal run. Frozen Test is never used to
edit the prompt.

## DINO Feature Contract

Use `facebook/dinov2-small` at revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`. Preserve aspect ratio with a
518x518 letterbox whose padding is RGB `(124,116,104)`. Then apply the pinned
processor normalization with resize and center crop disabled and
`use_fast=False`; otherwise the processor would change the already-frozen
geometry or may drift when its future default changes.

Core per-frame feature:

```text
global_cls: [384]
```

Selection representation per five-frame window:

```text
concat(mean(frame_0 ... frame_4), frame_4 - frame_0)
-> [768] -> L2 normalization
```

Cache `pooler_output` as the 384-dimensional CLS feature in one indexed
float32 NumPy array. The minimum loop does not store a spatial grid. Check
global-feature shape and finite values. Record the model revision and
preprocessing; per-file checksums are not required by default.

The formal private cache now contains all 34,149 unique CAM_FRONT frames with
shape `(34149,384)` and an accompanying frame-reference index. It took 2,120.6
seconds on Procyon CPU. The cache covers every partition only to avoid repeated
frozen inference; no scaler or classifier is fit outside L0.

The formal three-class Development FN artifact has 69 event-aligned rows and
shape `(69,768)`. All values are finite and unit-normalized. Multiple event rows
may share one representative window, so the 69 rows correspond to 61 unique
windows. The earlier two-class artifact had 51 rows over 44 unique windows and
remains historical evidence only.

## Critical Invariants

- no log crosses partitions and no window crosses a scene;
- five window timestamps are ordered;
- event bindings do not change identity;
- Strem failure is masked, not labeled negative;
- public U has no Oracle labels or evidence;
- VLM requests are created only from frozen selected IDs and contain no Oracle
  fields;
- selected IDs belong to the frozen public pool;
- all methods obey the same selection budget.
