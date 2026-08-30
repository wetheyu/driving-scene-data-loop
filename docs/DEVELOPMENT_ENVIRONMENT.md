# Development Environment

Specification: Frozen for `v0.8-three-class-loop`

## Local Environment

Observed:

- macOS arm64;
- project Python 3.12 managed by uv;
- src-layout Python package;
- Ruff, mypy, pytest, and a checked uv lockfile;
- local nuScenes mini metadata for smoke tests;
- external Strem v0.3.0 macOS release.

Use `uv run`; the shell's bare `python3` may point to another Python.

```bash
uv sync --locked --all-groups
uv run python --version
uv run ruff check .
uv run mypy src scripts tests
uv run pytest
uv lock --check
```

These commands verify implemented code only. Separate private run artifacts are
the evidence for completed formal event mining, feature extraction, model
training, the Development FN bank, label-free public-Pool inference, and the
frozen selection rankings. Oracle reveal is complete. Feedback retraining and
final evaluation remain incomplete.

## Dependencies

The dependency layout keeps dataset/model code small and explicit.

- `nuscenes-devkit`: official table lookup, reverse indexes, scene/sample
  relations, calibrated boxes, and coordinate transforms;
- `scikit-learn`: Logistic Regression and AP;
- `pillow`: image loading and letterbox preprocessing;
- `numpy`: feature arrays and numerical operations.

Pandas becomes a direct dependency when the Stage-A data-profile report begins.
PySpark is not installed for this single-machine pilot; learn its partitioning,
shuffle, and distributed-data use cases separately rather than forcing it into
the runtime.

PyTorch and Transformers remain in the explicit `ml` extra. The Procyon ML
environment now uses `torch 2.13.0+cpu` from the official PyTorch CPU index and
`transformers 4.57.6`; it does not install CUDA packages on a host without a
usable GPU. `pyproject.toml` routes Linux torch resolution to that explicit CPU
index, while macOS continues to use PyPI; the lock therefore contains the Linux
`+cpu` wheel and no NVIDIA CUDA or Triton packages. OpenAI,
sentence-transformers, BGE, and TinyBERT are not project dependencies.

## Procyon Environment

Observed on 2026-08-22:

```text
Host: shared university CPU host, reached by the SSH alias `procyon`
OS: Linux 6.8.0-111-generic x86_64
CPU: 256 logical CPUs
Memory: 1.0 TiB
Home storage: 30 TiB shared NFS, about 5.7 TiB free at audit time
GPU: no usable compute GPU observed
Scheduler: Slurm CPU partitions are available; no GPU GRES was observed
Python: uv-managed CPython 3.12.13
Rust/Cargo: 1.93.0
```

Project-owned paths are separate:

```text
Project: ~/projects/driving-scene-data-loop
Private data: ~/datasets/driving-scene-data-loop
Strem release: ~/strem-releases/v0.3.0
Thesis repository: ~/strem  # do not modify from this project
```

### Repository Sync

The two checkouts share one bare repository on Procyon instead of being copied
by hand:

```text
origin: procyon:git/driving-scene-data-loop.git   (bare)
laptop: ~/Documents/driving-scene-data-loop
procyon: ~/projects/driving-scene-data-loop
```

Push from the laptop, then fast-forward on Procyon:

```bash
git push origin main                                   # laptop
ssh procyon 'cd ~/projects/driving-scene-data-loop && git pull --ff-only'
```

Sync dependencies on Procyon with `--all-extras`, because the host has the `ml`
extra installed and a plain `uv sync --all-groups` would uninstall torch:

```bash
~/.local/bin/uv sync --locked --all-groups --all-extras
```

Procyon runs the full test suite with no skips, because torch and the pinned
Strem binary are both present there; the laptop skips those four tests. Run the
checks on Procyon before trusting a change that touches the GRU, public-Pool
inference, or the Strem boundary:

```bash
export STREM_BIN="$HOME/strem-releases/v0.3.0/strem-linux-x86_64"
~/.local/bin/uv run --no-sync pytest -q
```

Raw media, features, model weights, and run outputs stay under
`~/datasets/driving-scene-data-loop` and never enter this repository.

Use Slurm for long CPU jobs when the project account permits it, and keep
commands observable. The server is suitable for
metadata processing, Strem mining, Logistic Regression, and GRU training on
cached features. A real 58-frame DINO benchmark now confirms CPU feasibility:
64 threads and batch size 8 observed 10.51 frames/s at 518x518.

## Formal Data Status

The private Procyon data root currently contains:

- trainval metadata with 850 scenes, 34,149 samples, and 68 logs;
- the deterministic split file;
- all 2,747 planned TaskDesign `CAM_FRONT` JPEGs;
- 850 Strem v0.3.0 ego-BEV scene streams, a 45,847-instance numeric-ID map,
  and the private TaskDesign binding-audit report;
- private cross-partition support and TaskDesign audit evidence for the three
  frozen `task_spec_v2` scenarios;
- private historical camera/projection/visibility audit reports;
- the three-class Base run, a formal three-class Development FN bank containing
  69 event rows represented by 61 unique windows, and a finite `(69,768)`
  selection matrix;
- label-free public-Pool inference for 11,639 unique windows with a finite,
  unit-normalized `(11639,768)` selection matrix and no Oracle fields.
- frozen Random and integrated Mining lists,
  each with 600 unique temporally separated IDs and nested `150/300/600`
  prefixes;
- two further Random batches with seeds `102` and `103`, frozen on 2026-08-29
  before any reveal. Each has 600 unique temporally separated IDs, nested
  `150/300/600` prefixes, and no Oracle fields. At `N=300` each batch covers all
  25 Simulated Unlabeled Pool logs. Pairwise window overlap between the three
  `N=300` Random batches is 6, 7, and 9 windows against a uniform-draw
  expectation of `300*300/11639 = 7.7`; their union is 879 distinct windows and
  only one window appears in all three. `selection-rankings-v3` was not
  modified;
- `oracle-reveal-v1`, produced only after all rankings froze, contains labels for
  2,220 unique selected windows. Its reader parsed all 30,749 private window rows
  sequentially and retained/output only the selected IDs; it did not select or
  rerank any window.

All ten downloaded blob archives are still retained privately. Removing them is
a separate explicit cleanup decision.

The project uses the official devkit on a metadata-plus-CAM_FRONT subset. Devkit
metadata lookup and CAM_FRONT paths work normally; operations that try to open
the missing five camera channels, LiDAR, or radar media are outside project
scope. Raw images, features, model weights, and runs remain outside Git.

### Formal Artifact Paths

Later stages must name an exact input rather than rediscover it. Under the
private data root `~/datasets/driving-scene-data-loop` on Procyon:

```text
private_oracle/window-dataset-v1/windows.jsonl   private five-frame labels
public_pool/window-dataset-v1/u_windows.jsonl    label-free public U windows
private_source/features/dinov2-small-v1/         frozen DINO frame cache
runs/gru-baseline-v1/                            three-class Base and thresholds
runs/development-fn-bank-c3-v1/                  three-class Development FN bank
runs/public-pool-inference-v1/                   Pool probabilities and vectors
runs/selection-rankings-v3/                      frozen Random and Mining lists
runs/random-variance-v1/                         Random seeds 102/103
runs/oracle-reveal-v1/                           revealed labels and profile
runs/feedback-retraining-v1/                     controlled feedback models and Development predictions
```

Each run's aggregate report — and only the aggregate report, never its rows,
embeddings, or labels — is mirrored into the repository under `results/`, so a
documented number can be checked without access to this data root.

`selection-rankings-v3` is the current formal selection interface. Its Random IDs
match the earlier Random list and its Mining IDs match the earlier integrated
list; the rename did not reselect anything and no Oracle label was read. The
superseded `selection-rankings-v1` and `-v2` directories are development history
and must not be used as inputs.

## Strem Environment

Neither variable is set in the shell profile, so export both per session:

```bash
export STREM_BIN="$HOME/strem-releases/v0.3.0/strem-linux-x86_64"
export STREM_CONVERTER="$HOME/strem-releases/v0.3.0/source/scripts/nuscenes_to_strem.py"
```

The converter path is inside the extracted source archive; see
[Strem Skill](STREM_SKILL.md) for how both artifacts are produced from it.

Accepted releases:

```text
macOS arm64 SHA-256:
df656e0a8d020ffc268973812e8167867098b62b1215fd92b924ba84692d923c

Linux x86_64 SHA-256:
0c031f8d9fd7cf350a5209119fbd5fbddf316bba0219ae4d7c51c9e242c3692c

Version: strem 0.3.0
Commit: c7edd94c9fc2e6186ced32e6d2860313a942da19
STREMF input schema: 0.2.1
Timed result JSON schema: 2.0
Converter SHA-256:
f1833b3a2ed6ed91d2a701a0e66695c5f92d2e05375d23e1043c8522828aa952
```

The adapter directly accepts trusted stream/spec paths, runs Strem with a normal
timeout and explicit exact first-frame semantics, distinguishes result exits,
and preserves Schema 2.0 start sets, endpoint closure, end bounds, readable
clock constraints, frame indices, and bindings. Converter timestamps are not
rebased. No broader production process or security layer is part of this
project.

## DINO Gate and Formal Cache — Completed

- model: `facebook/dinov2-small`;
- immutable revision: `ed25f3a31f01632728cabb09d1542f84ab7b0056`;
- input: aspect-preserving 518x518 letterbox with RGB fill `(124,116,104)`;
- processor: normalization enabled, resize/center-crop disabled, slow processor
  explicitly frozen with `use_fast=False`;
- output: 384-dimensional `pooler_output` CLS feature in `float32`;
- observed benchmark: 58 real TaskDesign frames, batch 8, 64 CPU threads,
  10.51 frames/s; shape `(58,384)`, all finite and nonzero;
- formal cache: all 34,149 unique frames, shape `(34149,384)`, float32 and
  finite;
- formal runtime: 2,120.6 seconds at 16.10 frames/s with batch 8 and 64 CPU
  threads; the raw matrix is about 50 MiB.

Procyon CPU is sufficient. The indexed cache is private under the project data
root and is reused by every classifier instead of re-encoding overlapping
windows.

AWS Tokyo is not configured and is not part of the current plan. Reconsidering a
paid GPU requires an explicit budget decision.

## Remote VLM Gate

A remote VLM is permitted because the user has adequate API budget. Protocol
v0.11 in `docs/EVALUATION_PLAN.md` froze the provider (OpenAI API, Responses
endpoint), the model (`gpt-5.6-terra`, the volume-realistic tier rather than the
strongest one), the prompt, the JSON schema, and the image preprocessing;
`gpt-5.6-sol` is held as a gated diagnostic. `OPENAI_API_KEY` must be exported in
the Procyon shell that runs the stage. The gate rules that produced those choices
remain:

1. choose and record provider, exact model/version, image preprocessing, prompt,
   and JSON schema;
2. run a small TaskDesign/Development-only schema and cost smoke;
3. freeze those choices before labeling the selected Mining batch;
4. send only the five selected images and public scenario descriptions—never
   Oracle labels, Strem evidence, bindings, 3D boxes, or metric distances;
5. keep credentials in environment variables and responses in the private
   project data root;
6. record call count, latency, and cost without adding a provider framework.

The formal VLM run starts only after Mining IDs freeze. It does not require an
LLM planner, Agent, local VLM weights, or SFT infrastructure.

## Simple Reproducibility Record

For formal model runs, record:

- code revision and lockfile;
- split/task-spec/Strem/DINO identities;
- command, seed, model configuration, and selected window IDs;
- predictions and metrics.

Record runtime, memory, or storage only when they help evaluate feasibility. Do
not create per-file checksums, registries, or operational logs without a current
need.

## Secrets and Cleanup

Keep credentials and raw media out of Git, and out of every model-provider
upload except the v0.11 VLM request boundary declared in `docs/DATA_SPEC.md`. Before any
cleanup, confirm the exact project-owned target and preserve experiment inputs
and outputs that cannot be regenerated cheaply.

Current non-claims: the formal DINO cache, two LR baselines, three-class Gate-B
diagnostic, FN bank, label-free Pool inference, selection rankings, and selected
Oracle-label profile exist, as do the v0.10 feedback-retraining results and its
two spent held-out readings. The v0.11 VLM annotation is complete: 900 windows
labeled, scored against the Oracle, and one controlled downstream arm trained,
all recorded in `results/` and `docs/FINDINGS.md` section 15. No AWS resource
has been created.
