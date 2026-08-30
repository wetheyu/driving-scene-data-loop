# Instructions for AI Coding Agents

These instructions apply to this repository and every file below it.

## Read Before Editing

Read these documents in order:

1. `docs/PROJECT_SCOPE.md`
2. `docs/ARCHITECTURE.md`
3. `docs/STREM_SKILL.md`
4. `docs/DATA_SPEC.md`
5. `docs/EVALUATION_PLAN.md`
6. `docs/BAD_CASE_PROCESS.md`
7. `docs/FINDINGS.md`
8. `docs/ROADMAP.md`
9. `docs/DEVELOPMENT_ENVIRONMENT.md`

Do not silently change the research question, scenario definitions, data split,
annotation budget, Strem boundary, or evaluation protocol.

## Project Identity

- Project: Driving Scene Mining and High-Value Data Feedback Loop.
- Specification: Frozen for `v0.8-three-class-loop`.
- Formal data: nuScenes `v1.0-trainval`; mini is smoke-only.
- Model input: five consecutive annotated `CAM_FRONT` keyframes.
- Primary simulated annotation budget: `N=300` windows; Oracle-only budget
  sensitivity uses nested prefixes `N=150,300,600`.
- Primary metrics: per-class AP and three-class Macro-AP on the Held-out Test Set.
- Gate A froze three scenario specifications under `specs/task_spec_v2`.

## Language and Truthfulness

- Use English for code, identifiers, comments, logs, CLI text, and the
  specification documents that define the protocol.
- Use Chinese for discussion with the maintainer; maintainer-private notes live
  under the untracked `private/` directory and never enter Git.
- `docs/FINDINGS.md` is the English evidence record whose numbers trace to
  `results/`. Facts live there and nowhere else.
- Mark unimplemented work as `Planned`.
- Never report a plan, fixture, smoke test, or synthetic value as a real result.
- Preserve negative results and important bad cases.

## Incremental Development

- Advance one small, runnable, testable concept at a time.
- Before code, state the problem, input, transformation, output, and limitation.
- Combine repetitive mechanical work; never skip a design decision.
- End each milestone with a short written summary of what was measured.

## Simplicity and Scope Discipline

This is a one-person offline research project, not a production service or
a reusable platform. Its purpose is to demonstrate a clear algorithmic data
loop end to end and measure it honestly. Implement the smallest
direct solution that makes the current milestone runnable, testable, and
explainable.

- Prefer plain functions, small dataclasses, and explicit data flow over
  frameworks, registries, generic service layers, or speculative abstractions.
- Add an abstraction only for a real current second use case or repeated logic;
  do not design for hypothetical future requirements.
- Keep one responsibility in an existing module when that is clear enough. Do
  not add a new layer, manager, registry, interface, or record merely to make
  the project look engineered.
- Validate invariants that can change labels, splits, model inputs,
  experimental fairness, or reproducibility. These are experiment-integrity
  checks, not general defensive programming.
- Treat project-owned local files and fixed offline commands as trusted input.
  Do not implement security hardening, hostile-input handling, broad fallback
  trees, speculative retries, or exhaustive defensive checks unless a real
  observed failure blocks the current experiment.
- Test the main success path and failures that are likely to corrupt results.
  Do not build large hostile-input test matrices without an observed need.
- Persist only fields consumed by a later stage, evaluation, or reproduction
  command. Do not add records, registries, checksums, or manifests merely
  because they might be useful later.
- New complexity needs a concrete current requirement and a short explanation
  of why the simpler solution is insufficient.
- Leave future capabilities in documentation, not scaffolding. Add them when
  their milestone begins.

## Frozen Technical Boundaries

- Split by complete `log_token` before generating windows. The project uses
  official validation logs as its Held-out Test Set; this is not the hidden
  nuScenes test split.
- Use the official nuScenes devkit for table lookup, sample/annotation reverse
  relations, calibrated boxes, and coordinate transforms. Do not rebuild its
  token indexes in project code.
- Strem owns the candidate temporal patterns and persistent object bindings.
  Python prepares data, invokes Strem on complete scenes, parses event regions,
  and invokes the same specs on event-overlapping five-frame substreams for
  final window labels.
- Python preserves Strem's symbolic time bounds and readable clock constraints;
  it does not parse, reconstruct, or reimplement them.
- Use the external Strem v0.3.0 release through `STREM_BIN`; never copy or edit
  the Strem research repository.
- Formal runs pass `--first-frame-start exact` and preserve converter
  timestamps; do not rebase the stream onto a fabricated time origin.
- A Strem error or timeout is invalid evidence, never a negative label.
- DINOv2-Small stays frozen; only LR and GRU classifiers are trained.
- U labels and Strem evidence are never selector inputs; Oracle reveal begins
  only after a batch freezes.
- In the primary `N=300` comparison, Base, the three Random-Oracle batches, and
  Mining-Oracle differ only in added training windows. Random uses seeds
  `101/102/103`, all fixed before any Oracle reveal, so Mining is compared with a
  Random range rather than a single draw. Mining integrates bad-case similarity,
  boundary uncertainty, temporal deduplication, and cluster diversity.
  Mining-VLM reuses the exact Mining-Oracle IDs and changes only their label
  source.
- The primary causal comparison always uses the private Strem/nuScenes Oracle.
  A remote VLM may label only the already-frozen Mining-300 IDs and is evaluated
  against that Oracle; it is never the Oracle or a selector input.
- Frozen Test must not influence task design, selection, tuning, or retraining.

## Core Scope

Required: trainval preparation, projection and eligibility, Strem scene mining,
five-frame multi-label windows, frozen DINO features, LastFrame-LR, Mean5-LR,
Global-GRU, Development false-negative mining, fixed-budget selection,
controlled retraining, Frozen Test evaluation, and bad-case analysis.

Completion means this employability loop: three Gate-A-frozen scenarios,
leak-free windows and a data-quality profile, LR/GRU baselines, Development
false negatives, public-Pool Random and integrated Mining selection,
controlled Oracle-label retraining, structured VLM labeling on the same frozen
Mining-300 IDs, four compact result tables, and at least five deeply
explained cases. Gate B is a diagnostic, not a class-deletion rule: every Gate-A
class with at least five independent Development FN events enters the loop.

Not core: LLM, Agent, RAG, SFT, LoRA, vector databases, workflow platforms,
multi-sensor fusion, online learning, production deployment, or safety claims.
The VLM work is a bounded auto-label evaluation, not foundation-model training.

## Change and Completion Rules

- Preserve unrelated user changes in a dirty worktree.
- Add dependencies only for the active milestone.
- Update owning documentation when behavior changes.
- Add focused tests for the main path and experiment-critical failure modes.
- A milestone is done only when its focused checks pass, its status is truthful,
  and its purpose and limitation are documented.

Use these normal checks:

```bash
uv run ruff check .
uv run mypy src scripts tests
uv run pytest
uv lock --check
```
