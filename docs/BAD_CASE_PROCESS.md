# Bad-Case Process

Specification: Frozen for `v0.8-three-class-loop`

Observed model status: the three-class Base model exists and its three-class
automatic FN bank is complete. Individual visual root-cause review remains a
later task. At class level,
pedestrian-vehicle proximity hold fell below the Gate B diagnostic margin and
is the first representation/model limitation to preserve and test with added
data.

## Purpose

Bad cases have two roles:

1. Development false negatives guide data selection.
2. A small reviewed set explains failures in data, labels, representations,
   models, selection, VLM automatic labeling, or evaluation.

A false negative is an observation, not automatically a root cause.

The failed proximity-hold class has Development AP `0.0537 ± 0.0056` against
prevalence `0.0328`, below the frozen `+0.05` Gate-B margin. The current
evidence does not distinguish whether the main cause is the global DINO
representation, monocular visual ambiguity, label semantics, data support, or
the GRU. It is therefore recorded as an observed failure with unknown root
cause, not explained away.

## Minimal Record

Each reviewed case needs only:

```text
case_id
layer
window_id or event_id
expected
observed
suspected_cause and confidence
evidence
next_action
```

Cause confidence is `confirmed`, `plausible`, or `unknown`. Do not present a
plausible story as fact. Add a regression test only when the case exposed a
repeatable code defect.

## Taxonomy

### Data and Label

- broken nuScenes join or coordinate transform;
- projected but visually unusable target;
- wrong Strem identity, binding, interval, or window mapping;
- scenario too rare or semantically weak.

### Representation and Model

- small target lost in a global DINO feature;
- preprocessing mismatch;
- LastFrame shortcut or Mean5 order loss;
- GRU overfitting, underfitting, seed instability, or order insensitivity.

### Selection

- similarity retrieves appearance-only duplicates;
- uncertainty favors noise or unlearnable samples;
- temporal suppression removes useful data;
- positive yield improves without downstream improvement.

### VLM Automatic Labeling

- object or relation hallucinated without supporting frames;
- exact distance or duration asserted from ambiguous monocular evidence;
- identity confused across the five frames;
- valid scene returned as `uncertain`, or an ambiguous scene returned with
  unjustified confidence;
- schema-invalid output or evidence frames outside the five inputs;
- semantic label agrees visually but conflicts with the metric Oracle definition.

### Evaluation

- log or hidden-label leakage;
- unequal retraining settings;
- premature Frozen Test access;
- AP or averaging error.

## Pre-declared Null-Result Diagnosis

A tie or regression is a valid outcome, but only if its explanation is chosen
from hypotheses written down beforehand. If Mining does not separate from the
Random batch range, these three are the candidates, in the order the evidence
should be checked:

1. **Representation ceiling.** The most likely cause. A frozen global DINO CLS
   vector summarizes the whole frame, while the bound target can be a pedestrian
   about 20 to 30 letterbox pixels high at 15 to 20 m. If the encoder cannot
   resolve the object, more windows containing that object cannot help. Evidence
   to check: whether added positives changed per-class Recall at all, and
   whether the weak classes stay weak across every arm and budget.

2. **Budget too small to move the metric.** `N=300` is 2.6% of the 11,639-window
   Pool, and at Development prevalence one Random draw adds roughly 6 near-zone
   positive labels. Evidence to check: whether the `150/300/600` curve shows any
   monotone trend, and whether the observed positive yield from the label profile
   was as low as that estimate.

3. **Label semantics.** The scenario definitions are metric and temporal, so
   visually similar windows can carry opposite labels. Evidence to check:
   `ignore` rate in the selected batches, and whether reviewed Mining selections
   look correct to a human but fail the bounded Strem rule.

These are distinguished by evidence, not asserted. Reporting "Mining did not
help" with an identified and checked cause is a stronger result than an
unexplained improvement.

## Review Procedure

For a selected case:

1. Preserve the original input, label, and prediction reference.
2. Check data and label correctness before blaming the model.
3. Compare simple baselines and neighboring windows when useful.
4. State observed behavior separately from the cause hypothesis.
5. Choose the smallest useful next action.
6. Re-run only Development or a small fixture; never tune from Frozen Test.

The formal FN bank contains 69 event rows from 61 unique representative
windows: 26 near-zone, 21 proximity-hold, and 22 corridor events. It first finds
Development windows whose seed-17 probability is below the frozen threshold,
then keeps the lowest-probability window for each real event. Therefore it means
"window-level false negatives deduplicated by event," not "every positive
window of the event was missed." In fact, 26 near-zone, 12 proximity-hold, and
11 corridor events were missed in all of their positive windows; the remaining
bank events contain at least one missed and at least one detected view.

The FN bank is automatic; final bad-case records are a small explanatory
sample. Frozen Test cases are reviewed only after predictions and scores freeze
and can influence only a future version.

## Historical Data-Design Findings

These TaskDesign observations remain valid evidence even though their one-off
gallery/audit code is not maintained as core:

- CAM_FRONT did not lead the declared six-camera static coverage proxies;
- a 16-pixel-height box could still be an edge sliver;
- geometric projection could pass while a pedestrian was heavily occluded;
- panoramic visibility could be high while the CAM_FRONT box was severely
  clipped;
- naive deterministic gallery sampling produced temporal, category, and log
  imbalance before the audit sample was corrected.

These findings motivated an explicit CAM_FRONT scope decision and the combined
height, clipping-retention, and visibility filter. They are not model results.

Development also exposed practical lessons: exact split serialization matters,
remote file layout must be verified, and long remote jobs need a persistent
session with an observable exit code. `sbatch` and `srun` are not currently
available in the project SSH shell, so scheduler use is not assumed. Detailed
shell incidents belong in execution logs, not in the core bad-case contract.

Two confirmed formal-label defects were caught before Gate A:

- the first candidate implementation used `@dist_m`, which includes vertical
  `z`, for rules defined by planar center distance; the corrected rules use the
  converter's ego-BEV box centers through `@dist`;
- the proximity-hold self-loop remained enabled after the hold threshold,
  allowing one candidate to finish at later frames; the loop now requires
  `x < hold`, while completion requires `x >= hold`.

Real Strem fixtures reproduce both semantics. These are engineering regression
cases, not model bad cases and not evidence of model quality.

The first target-aware gallery exposed a presentation-level coordinate-time
mismatch. Its initial labels recomputed pedestrian distance with the CAM_FRONT
ego pose, while Strem's formal stream uses the sample's LIDAR_TOP keyframe ego
pose. Because nuScenes sensors are asynchronous, the two values differed by up
to 0.322m in the reviewed participating frames. Formal labels were unchanged;
the gallery was corrected to display the exact stream distance and retain the
camera-pose value only as projection evidence.

The first TaskDesign design also produced useful negative evidence: requiring a
pedestrian to move from 8m to 3m (or the declared nearby variants) within two
seconds yielded zero eligible events, while the 4m pedestrian-vehicle hold
yielded only 10 events across 3 logs. A disclosed TaskDesign-only trajectory
audit showed that continuously eligible pedestrians more often cross farther
distance bands. The retained scenario therefore changed its candidates to
`(20m,12m)`, `(20m,15m)`, `(25m,15m)`, and `(25m,20m)`; `(20m,15m)` was the
first to clear positive support at 16 events across 3 logs. An intermediate
motor-vehicle entry experiment was not retained. No Development, U, or Frozen
Test labels were used for these decisions.

The first pedestrian-entry hard-negative draft also included a reversed
`<=15m -> >=20m` pattern. It produced zero TaskDesign events. The project keeps
that zero-support result instead of presenting the pattern as observed data.
The replacement outer-band pattern (`>=25m -> (17m,20m]` within 2s) was chosen
because it is still a clear boundary contrast to the positive definition and
has real TaskDesign support. After positive-overlap exclusion and same-scene
temporal suppression, the four retained patterns provide 25 audit windows over
5 logs. A compact target-aware review across all logs found no systematic
binding, projection, or direction error, while retaining genuinely ambiguous
weather, scale, and boundary examples.

The first proximity hard-negative design was also insufficient: its narrow
near band and short-hold construction yielded only 4 retained TaskDesign
windows. The project kept that result, then made one disclosed TaskDesign-only
boundary widening to `(5m,10m]` and a transient-close candidate. After excluding
every window overlapping the official positive event and applying temporal
suppression, 20 windows across 5 logs remained. The overlap exclusion is
essential: a transient-pattern match alone does not prove that the same window
fails the one-second positive rule.

A separate label-mapping failure exposed the difference between a Strem result
zone and a model window. On one complete scene, touching symbolic solutions for
fixed bindings were aggregated to frame bounds `9..14`, which cannot fit in a
five-frame input. Running the same specification on frames `9..13` returned
contained matches `9..12` and `10..13`. The corrected contract uses full-scene
results to find candidate regions and lets Strem re-evaluate the bounded
five-frame substream; Python does not infer a witness by clipping the aggregate
bounds or parsing constraint strings.

The corridor hard-negative scan retained 20 audit windows across 4 TaskDesign
logs: 2 lateral near misses, 10 already-inside holds, and 8 far-range entries.
The declared slow-entry type yielded zero usable windows after positive-overlap
exclusion, so it is preserved as a negative result rather than replaced until a
desired count appears. A 24-window positive/negative gallery showed no
systematic defect, but also showed that many positives are cross traffic. The
label is therefore a relative ego-corridor entry, not an active lane-change
claim.

## Release Requirement

Review at least five final cases deeply: one data/label case, one Base false
negative, one selector success or failure, one VLM label error or uncertainty,
and one retraining improvement/tie/regression. Prefer cases that connect the
whole chain: why Base missed, why Mining did or did not retrieve related data,
what label source supplied, and what changed after retraining. Small fixtures
may demonstrate failure handling, but they are not described as observed model
failures. Do not build a fault-injection subsystem or collect shallow cases
solely to satisfy a quota.
