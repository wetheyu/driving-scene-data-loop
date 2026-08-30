# Project Scope

Specification: Frozen for `v0.8-three-class-loop`

Gate A froze all three scenario specifications under `specs/task_spec_v2`.

## Goal

This one-month personal project studies one question:

> Under a fixed simulated annotation budget `N=300`, can selection driven by
> Development bad-case similarity, model-boundary uncertainty, and visual
> diversity improve Held-out Test Macro-AP more than random selection?

The downstream task is multi-label recognition of precise temporal events from
five consecutive annotated `CAM_FRONT` keyframes.

```text
formal event mining -> visual classifier -> Development bad cases
-> fixed-budget data selection -> retraining -> Held-out Test comparison
```

Strem supplies the private benchmark labels. The model and selector must not see
hidden U labels before a selection batch is fixed.

After the primary Oracle-label comparison is frozen, a secondary question asks
whether a remote VLM can produce useful structured labels on exactly the same
selected windows of the frozen v0.10 mining batch. This evaluates automatic
labeling separately from data-selection quality.

## Candidate Scenarios

All scenarios use real timestamps and stable `instance_token` identity.
`motor_vehicle` means `car|truck|bus|trailer|construction_vehicle`.

The three scenarios passed label-quality Gate A and all enter the quantitative
data loop. Gate B is a diagnostic of the initial visual model, not an exclusion
rule. This lets the experiment test whether targeted data helps the weakest
class instead of deleting it before the closed loop begins.

### 1. pedestrian_ego_near_zone_entry

The same CAM_FRONT-eligible pedestrian has ego-frame planar center distance at
least `outer`, then within two seconds has distance at most `inner`.

Candidates, tried in order:

```text
(outer, inner) = (20m, 12m), (20m, 15m), (25m, 15m), (25m, 20m)
```

It does not require monotonic approach or a post-entry hold. It is not TTC,
collision probability, pedestrian intent, or safety risk. Relative approach may
be caused mainly by ego motion. `20m -> 15m` means crossing two ego-relative
distance boundaries; it does not mean that the pedestrian walked five metres
toward the car.

### 2. vehicle_relative_corridor_entry

The same motor vehicle remains in front (`0 <= x <= 30m`) and moves from
`abs(y) >= outer_y` to `abs(y) <= inner_y` within two seconds, using the ego frame
at each timestamp.

Candidates, tried in order:

```text
(outer_y, inner_y) = (3m, 1.5m), (3m, 2m),
                     (2.5m, 1.5m), (2.5m, 2m)
```

This is relative position change. It does not prove that the target vehicle
actively changed lanes.

### 3. pedestrian_vehicle_center_proximity_hold

The same pedestrian and the same motor vehicle have planar distance between
their annotation centers, projected into the ego ground plane, at most
`distance` continuously for at least `hold`.

Candidates, tried in order:

```text
(distance, hold) = (4m, 1s), (5m, 1s), (6m, 1s)
```

This is not physical body clearance, conflict, intent, or accident risk.

## Shared Visual Eligibility

Every bound object in each participating event frame must satisfy:

```text
valid CAM_FRONT projection
and 518x518 letterbox box height >= 16 px
and projected-box area retained in the image >= 0.5
and nuScenes visibility rank in {2, 3, 4}
```

The rule was selected from TaskDesign only. Projection and visibility are
eligibility proxies, not proof that an object is clear or unoccluded.

## Gate A: Trustworthy Labels

For each scenario, choose the first declared threshold candidate that has:

- at least 12 independent positive events across at least 3 TaskDesign logs;
- at least 12 difficult negative windows across at least 3 logs;
- no systematic binding, coordinate, projection, or semantic error in review;
- target support of L0 `30/5 logs`, Development `15/5`, U `60/10`, and Frozen
  Test `20/5`.

A failing scenario is removed rather than weakened to keep a target count. Gate A
freezes retained parameters in a new immutable task-spec version; an existing
task spec is never edited in place.

Gate A result:

| Frozen scenario | TaskDesign | L0 | Development | U | Frozen Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| pedestrian ego `20m -> 15m` | 16/3 | 160/13 | 33/5 | 256/21 | 100/16 |
| vehicle corridor `3m -> 1.5m` | 46/5 | 153/13 | 74/6 | 201/22 | 69/15 |
| pedestrian-vehicle `<=5m` for `>=1s` | 16/4 | 85/12 | 28/6 | 156/22 | 37/9 |

Each cell is `event groups / matching logs`. All three classes passed compact
TaskDesign positive/hard-negative semantic review. These counts prove label
support, not visual learnability or model performance.

For the corridor, four declared hard-negative patterns yielded 20 temporally
separated windows across 4 TaskDesign logs after positive-overlap exclusion:
2 lateral near misses, 0 slow entries, 10 already-inside holds, and 8 entries
beyond 30m. A 24-window gallery covered 10 positives and 14 hard negatives. All
88 participating frames contained the eligible bound vehicle, and review found
no systematic binding, coordinate, projection, direction, or semantic error.
The zero-support slow-entry type is preserved as a negative result.

## Gate B: Initial Learnability Diagnostic

For each class, report whether Global-GRU Development AP exceeds positive
prevalence by at least `0.05`, using mean normal-order AP across the three
predeclared seeds. The result diagnoses the initial representation/model. A
failed class remains in the loop if it has at least five independent
Development false-negative events; the loop then measures whether selected
data improves it.

GRU results alone do not prove temporal learning. That claim also requires a
useful advantage over LastFrame-LR and Mean5-LR and a meaningful ordered-versus-
reversed diagnostic.

Observed Gate-B decision:

- pedestrian ego near-zone entry: pass, mean AP `0.1675` versus prevalence
  `0.0199`;
- vehicle relative corridor entry: pass, mean AP `0.5011` versus prevalence
  `0.0770`;
- pedestrian-vehicle proximity hold: fail, mean AP `0.0537` versus prevalence
  `0.0328`, short of the required `+0.05` margin.

The primary loop therefore keeps `C=3`. The proximity-hold class is reported as
the challenge class: its weak Base result is preserved and must not be presented
as evidence of strong visual learnability.

## Gate C: Fair Closed-Loop Comparison

Base, the three Random-Oracle batches, and Mining-Oracle share:

- L0 and the frozen DINO features;
- GRU architecture and training protocol;
- selection budget and Frozen Test.

In the primary comparison, only the selected additional training windows may
differ. The secondary Mining-VLM comparison reuses the exact Mining-Oracle IDs
and changes only their label source. Frozen Test never influences task design,
prompt design, tuning, selection, or retraining.

## Required One-Month Result

- trainval metadata and private CAM_FRONT preparation;
- three Gate-A-frozen scenarios and a Gate-B decision for each class;
- an explicit data-cleaning and data-profile report;
- Strem scene events and five-frame labels;
- frozen DINO global features;
- LastFrame-LR, Mean5-LR, and Global-GRU;
- Development false-negative bank;
- Base, three Random-Oracle batches, and Mining-Oracle experiments;
- structured VLM labels on the frozen v0.10 mining batch, evaluated against the
  Oracle, plus one controlled Mining-VLM retraining run;
- three training seeds and one-time Frozen Test scoring;
- four compact result tables, at least five deeply explained cases,
  reproducible commands, and write-up material.

The primary experiment uses `N=300`; an Oracle-only nested `N=150/300/600`
Random-versus-Mining curve is a light sensitivity check. Optional only
after the core works: more selector ablations, bootstrap analysis, a spatial
comparison, and small controlled image augmentation. VLM SFT,
LLM/Agent, data platforms, multi-sensor fusion, online learning, and deployment
are outside this month.

## Implementation Rule

Use the simplest code that preserves the experiment. Direct functions and plain
files are preferred. Add a class, registry, database, checksum, or generalized
interface only when a current downstream consumer or observed failure requires
it.

## Honest Claim Boundary

If completed, the project may claim a single-round offline bad-case-driven scene
mining and data-feedback experiment on nuScenes trainval, plus evaluation of a
remote VLM as a structured automatic labeler. It may not claim VLM training or
SFT, production active learning, PB-scale systems, perception detection,
prediction, planning, control, real safety validation, or improvements not
observed.

Current facts:

- whole-log `split-v1`, TaskDesign CAM_FRONT acquisition, projection, the
  minimal Strem CLI boundary, and full trainval conversion exist;
- 850 CAM_FRONT-eligible formal streams now retain 120,035 target annotations;
  the initial pedestrian-to-ego distance bands ending at 3--4m had zero
  TaskDesign events and the strict proximity candidate had only 10 events
  across 3 logs;
- all 11 final v0.5 candidates were scanned on TaskDesign. The first candidates
  clearing positive support are pedestrian-to-ego `(20m,15m)` at 16 events/3
  logs, vehicle-corridor `(3m,1.5m)` at 46/5, and
  pedestrian-vehicle hold `(5m,1s)` at 16/4;
- historical TaskDesign audits support the current camera and eligibility
  decisions and remain private evidence;
- 850 scene streams and 45,847 numeric-ID mappings were produced privately; a
  real TaskDesign fixture matched a stable pedestrian and rejected a mid-event
  replacement;
- all 16 TaskDesign pedestrian-entry event groups now have five-frame context
  and target-crop galleries. Review classified 13 as clear, 2 as
  partial, and 1 as severely rain-blurred. The difficult rain case remains a
  valid positive rather than being silently removed; no systematic target-box
  or binding error was observed;
- four pedestrian-entry hard-negative patterns were checked against Strem
  source semantics and the pinned real binary: inner distance near miss, outer
  band entry, slow entry, and already-near hold. TaskDesign produced 37 windows
  with no formal positive-interval overlap; requiring starts in the same scene
  to be at least five keyframes apart retained 25 windows across 5 logs
  (`4/11/1/9` by the four pattern types). A 15-window target-aware sample across
  all logs found no systematic binding, projection, or direction error;
- the first pedestrian-vehicle hard-negative attempt yielded only 4 windows.
  After the one disclosed TaskDesign widening to `(5m,10m]` for the near band
  plus transient-close candidates excluded from any positive event, 20
  temporally separated windows remained over 5 logs. A combined 14-window
  positive/negative review covered 10 scenes and found no systematic error; all
  52 participating displayed frames contained both eligible bound objects;
- a real full-scene result aggregated touching symbolic zones, while the same
  fixed specification returned contained witness intervals on its five-frame
  substream. Window labels will therefore use complete-scene events for
  candidate discovery and bounded Strem reruns for the final positive/ignore
  decision;
- generic event grouping, all three Gate-A audits, frozen `task_spec_v2`, 1,430
  formal event groups, and 30,749 private five-frame windows exist. The pinned
  DINO model/preprocessing produced the formal finite `(34149,384)` cache. The
  LastFrame-LR and Mean5-LR Development Macro-AP values are `0.1499` and
  `0.1942` over all three classes. Gate B diagnosed near-zone and corridor entry
  as above-margin and proximity hold as below-margin. A historical two-class
  diagnostic GRU Development Macro-AP is `0.3636 ± 0.0157`;
- the final seed-17 thresholds produced 46 near-zone and 48 corridor
  window-level Development false negatives. Event grouping retained 28 and 23
  event rows, respectively; those values remain historical two-class evidence.
  The formal three-class Base bank now contains `26/21/22` event rows in scene
  table order, 69 rows total, represented by 61 unique windows and a finite
  unit-normalized `(69,768)` matrix. Public-Pool inference produced 11,639
  unique label-free rows and a finite unit-normalized `(11639,768)` matrix with
  no Oracle fields. The 600-row Random and integrated Mining rankings are now
  frozen with nested `150/300/600` prefixes; every list has
  unique IDs and passes temporal separation. Oracle reveal retained 2,220 unique
  selected windows after the rankings froze; it found no invalid target. VLM
  labels, feedback-retraining results, and Held-out Test values do not yet exist.
