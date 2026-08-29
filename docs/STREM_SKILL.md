# Strem-SpTA External Dependency Contract

Specification: Frozen for `v0.8-three-class-loop`

Strem is the author's independent external research project. This repository
consumes a fixed tagged release; it does not copy Strem source or silently use a
research-worktree build.

## Purpose

For this project, Strem does one job:

```text
complete structured scene stream + SpTA specification
-> match / no_match + symbolic time regions + clock constraints
   + frame indices + persistent object bindings
```

It is the rule-based scenario miner and private benchmark labeler. It does not
project camera boxes, train the visual model, rank U, select data, or evaluate AP.
It also remains the private benchmark when a remote VLM labels selected images;
the VLM does not replace Strem or change a Strem match.

## Fixed Release

Release identity:

```text
Version: strem 0.3.0
Commit: c7edd94c9fc2e6186ced32e6d2860313a942da19
Tag: v0.3.0
STREMF input schema: 0.2.1
Timed result JSON schema: 2.0
Source archive SHA-256:
425aea284cda45e2aa0dd82c1ab83337deb713390355eec4c5aca2b3409347be
Converter SHA-256:
f1833b3a2ed6ed91d2a701a0e66695c5f92d2e05375d23e1043c8522828aa952
```

macOS arm64:

```text
Path: ~/strem-releases/v0.3.0/strem-darwin-arm64
SHA-256: df656e0a8d020ffc268973812e8167867098b62b1215fd92b924ba84692d923c
```

Procyon Linux x86_64:

```text
Path: ~/strem-releases/v0.3.0/strem-linux-x86_64
SHA-256: 0c031f8d9fd7cf350a5209119fbd5fbddf316bba0219ae4d7c51c9e242c3692c
```

### Where the Two Artifacts Come From

Both the matcher and the converter are produced from the same pinned source
archive; neither is taken from the research worktree.

```text
strem-v0.3.0-source.tar.gz          the pinned archive
  -> extract
  -> cargo build --release          -> strem-<platform>   (STREM_BIN)
  -> source/scripts/                -> nuscenes_to_strem.py (STREM_CONVERTER)
```

The matcher is a compiled Rust binary. Only the archive's `src/` tree is
compiled; its experiments, benchmark results, docs, and poster are not part of
the executable. The result links against the platform C runtime only, so it runs
from any directory without the archive, Rust, or Cargo present.

The converter is not compiled. It is one 545-line Python file that imports only
the standard library, so it needs no environment beyond CPython and reads the
nuScenes JSON tables directly.

A `target/` directory beside the extracted source is local build scratch, not
archive content, and can be removed once the binary is extracted and its
SHA-256 verified.

Configure both through the environment:

```bash
export STREM_BIN=/path/to/strem-v0.3.0
export STREM_CONVERTER=/path/to/v0.3.0/source/scripts/nuscenes_to_strem.py
```

Do not use `~/strem/target/release/strem` or another research
worktree binary. Its contents are not the release contract.

The semantic audit for this milestone read the pinned v0.3.0 source archive.
The author's `~/strem` research worktree was dirty and was treated as read-only
context, never as release evidence and never modified. Its build differs from the
pinned release, and the adapter refuses it on the SHA-256 check.

The three version numbers above describe different interfaces. Strem `0.3.0`
still accepts input streams whose top-level `version` is `0.2.1`; only the timed
monitor result format changed to Schema `2.0`.

## Capabilities Used

Strem v0.3.0 provides:

- optional `persistent_bindings` keyed by annotation ID;
- exact and left-censored first-frame start policies;
- JSON intervals with start-set semantics, endpoint closure, frame bounds,
  clock constraints, and bindings;
- distinct intervals for distinct bindings;
- exits `0=match`, `1=no_match`, and `2=error`;
- a nuScenes converter using `sample -> sample_data -> ego_pose`;
- ego ID `0`, stable object numeric IDs, collision checks, and
  `nuscenes_numeric_id_map.json`.

Its `bbox` is an ego-frame BEV footprint, not a camera-image occlusion box.
The scenario specifications therefore use the operators as follows:

- `@dist(a, b)` measures two-dimensional Euclidean distance between BEV box
  centers and implements this project's planar center-distance rules;
- `@dist_m(a, b)` measures three-dimensional metric `position` distance,
  including `z`, and is not used for the two planar-distance scenarios;
- `@x_m(a)` and `@y_m(a)` read ego-frame metric coordinates for the corridor
  rule.

Strem has no general previous-frame numeric register, so a rule such as
“distance decreases on every frame” still needs an upstream fact or a future
formal extension.

The timed monitor starts a fresh candidate at every input frame. Each hold
candidate uses a strict `x < hold` self-loop and a separate `x >= hold`
completion transition, so that run reaches completion at its first observed
satisfying frame. The Strem controller can aggregate touching symbolic
solutions with identical bindings and constraints into a wider result zone.
Python may then group adjacent result zones with identical bindings into one
scenario event. Different bindings are never merged. Consequently, full-scene
result frame bounds are candidate-region evidence, not a minimal model-window
witness.

## Project Time Semantics

Formal project runs always pass:

```text
--first-frame-start exact
```

For observed timestamps `tau_0 < tau_1 < ...`:

- a match beginning at the first observed frame starts at the singleton
  `{tau_0}`;
- a candidate beginning at later frame `i` starts in
  `[tau_(i-1), tau_i)` before any guard narrowing;
- candidate ends lie in `(tau_j, tau_(j+1)]`, or after the final frame have no
  finite upper bound.

Strem does not require `tau_0` to be zero. The pinned nuScenes converter happens
to subtract the scene's first raw microsecond timestamp, so its output starts at
`0.0`; another valid stream may start at any finite value. Neither case needs an
extra timestamp shift. Python must preserve the converter's output timestamps.

Strem also supports `left-censored`, meaning the event may have started at or
before the first observation. This project does not use that policy for formal
training labels because it would intentionally leave the onset unknown.

## Execution Order and Ownership

```text
nuScenes joins
-> ego-frame facts and CAM_FRONT eligibility facts
-> one full-scene STREMF 0.2.1 stream
-> one scenario specification
-> Strem v0.3.0 with exact first-frame semantics
-> Schema 2.0 intervals and bindings
-> Python event grouping
-> same specification on event-overlapping five-frame substreams
-> five-frame labels
```

Do not replace complete-scene mining with independent window scans: complete
scenes are required to discover event regions and preserve traceable event
identity. However, full-scene JSON may aggregate touching symbolic match
regions. Therefore event-overlapping five-frame candidates are rechecked by
Strem as bounded substreams with original timestamps and IDs. This determines
whether the rule itself is satisfiable inside the model input without
reimplementing its clocks and guards in Python.

Strem owns temporal state progression, clocks, guards, bindings,
match/no-match, and raw symbolic intervals.

Python owns nuScenes joins, coordinate transforms, visual eligibility,
converter invocation, subprocess handling, Schema 2.0 parsing, ID-map lookup,
event grouping, and window labels. Python must not write a second if/else
implementation of the three scenario rules.

`specs/task_spec_v2` contains all three Gate-A-approved rules, including the
corridor `(3m,1.5m)` specification.

The binding grammar accepts one concrete class per variable. Therefore the
project's private eligibility stage maps the declared source classes
`car|truck|bus|trailer|construction_vehicle` to `motor_vehicle` in the formal
stream. It does not change object IDs, metric facts, timestamps, or the original
converter output. Strem still owns the actual corridor/proximity temporal rule.

## Minimal Python Boundary

```text
run_scene(stream_path: Path, spec_path: Path) -> StremRunResult
```

The adapter does only what can protect label semantics:

1. verify the platform release version and SHA-256;
2. call Strem with `--output json --first-frame-start exact` and a timeout;
3. distinguish exits 0, 1, and 2;
4. require result Schema `2.0` and exact boundary policy;
5. preserve frame bounds, start semantics and endpoint closure, end bounds,
   constraints, and bindings.

Focused tests cover match, no-match, error, timeout, wrong binary, malformed
JSON, and real-release first-frame matches at zero and positive timestamps. The next integration
fixture owns converter and numeric-ID-map behavior.

## Evidence Boundary

A match proves only that the versioned structured stream satisfied the
versioned SpTA specification under the declared first-frame policy. It does not
prove camera clarity, correct upstream annotation or transformation, collision
risk, intent, causality, safety, or visual-model generalization.

Current status: v0.3.0 is built and tested on macOS arm64 and Procyon Linux
x86_64, the Python boundary consumes Schema 2.0 with exact semantics, and
real-release zero and positive first-frame fixtures exist. The pinned converter
produced 850 private trainval scene streams and 45,847 traceable instance IDs.
Real-release scenario fixtures now cover all three candidate families, the
two-second boundary, reversed motion, stable versus replaced bindings, planar
distance, first-completion hold behavior, and event grouping. All 11 v0.5
final candidates were scanned on the 68 TaskDesign scenes. Their first
positive-support passes are pedestrian-to-ego `(20m,15m)`, vehicle corridor
`(3m,1.5m)`, and pedestrian-vehicle hold `(5m,1s)`. The first and third classes
have completed compact TaskDesign positive/hard-negative visual review without
a systematic binding, projection, or semantic defect. The pedestrian-entry
hard-negative SpTAs were additionally checked against `timed-spta.md`, timed
spec parsing, automaton execution, and S4m distance semantics in the external
repository. Real-binary fixtures verify their persistent bindings, distance
boundaries, and open/closed time bounds. They yield 25 temporally separated
TaskDesign audit windows across 5 logs after positive-overlap exclusion. The
preferred proximity class produced 20 retained boundary-negative windows after
one disclosed insufficient first attempt. A real five-frame substream run also
confirmed that bounded Strem re-evaluation yields contained witness intervals
when the full-scene result has aggregated them. All three specifications passed
cross-partition support and are recorded under `specs/task_spec_v2`. The
corridor additionally retained 20 hard-negative audit
windows across 4 logs, passed a 24-window visual-semantic review, and has
L0/Development/U/Frozen-Test support of `153/13, 74/6, 201/22, 69/15`. All
three rules are frozen in `specs/task_spec_v2`.
