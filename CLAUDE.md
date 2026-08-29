# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Agreement

`AGENTS.md` is the authoritative contract for this repository: read it first, and
follow its documentation reading order (`docs/PROJECT_SCOPE.md` →
`docs/STAGE_PLAN.md` → `docs/ARCHITECTURE.md` → `docs/STREM_SKILL.md` →
`docs/DATA_SPEC.md` → `docs/EVALUATION_PLAN.md` → `docs/BAD_CASE_PROCESS.md` →
`docs/ROADMAP.md` → `docs/ALIGNMENT.md` → `docs/DEVELOPMENT_ENVIRONMENT.md`)
before touching a stage you have not worked on. Key points that shape every edit:

- This is a one-person, research-first offline research experiment frozen at
  `v0.8-three-class-loop`, not a service or a reusable platform. Implement the
  smallest direct solution: plain functions, small dataclasses, JSONL + `.npy`.
  Do not add registries, managers, plugin layers, defensive fallback trees, or
  hostile-input handling without an observed failure that blocks the experiment.
- Validation exists only to protect labels, split isolation, model inputs,
  selector leakage, and fair evaluation. Those checks are required; general
  defensive programming is not.
- English for code, identifiers, comments, docs, logs, and CLI text. Chinese for
  teaching, progress explanation, and notes.
- Never present a plan, fixture, or synthetic value as a real result; mark
  unimplemented work `Planned`. Preserve negative results — this project does not
  tune until the desired answer appears.
- Do not silently change the research question, scenario definitions, split,
  budget, Strem boundary, or evaluation protocol. Update the owning document when
  behavior changes.

## Commands

```bash
uv sync --locked --all-groups        # runtime + dev (no torch)
uv sync --locked --all-groups --all-extras   # adds the `ml` extra (torch, transformers)
uv run ruff check .
uv run mypy src scripts tests        # strict mode
uv run pytest
uv lock --check
```

Always go through `uv run`; a bare `python3` may be a different interpreter.

Single test / single case:

```bash
uv run pytest tests/test_selection.py
uv run pytest tests/test_selection.py::test_selector_rejects_an_oracle_field -x
```

Test skips are expected and meaningful:

- `tests/test_gru_baseline.py` and `tests/test_public_pool.py` skip unless the
  `ml` extra is installed (`pytest.importorskip("torch")`).
- `tests/test_strem_exact_start.py` skips unless `STREM_BIN` points at the pinned
  Strem v0.3.0 release binary; it runs the real binary, not a fake.

`STREM_BIN` (and optionally `STREM_CONVERTER`) select the external Strem release.
The adapter verifies version and SHA-256 against the pins in
`strem_adapter.py` / `strem_converter.py` and refuses anything else — never point
it at a research-worktree build such as `~/strem/target/release/strem`.

## Pipeline Architecture

Eleven scripts under `scripts/` are the stage entry points; `src/driving_scene_data_loop/`
holds the logic. Stages chain by fixed artifact filenames, so the output directory
of one stage is the input directory of the next. Every stage refuses to write into an
existing output path, so a rerun needs a new directory.

```text
build_split_manifest.py      --dataroot --output               -> split.json
convert_nuscenes_for_strem.py --metadata-root --output-dir      -> nuscenes_<scene>.json + nuscenes_numeric_id_map.json
build_strem_eligible_streams.py --dataroot --raw-stream-dir --output-dir
                                                                -> filtered streams + cam_front_eligibility_summary.json
build_window_dataset.py      --dataroot --stream-dir --split-manifest --spec-dir
                             --strem-bin --eligibility-summary
                             --private-output-dir --public-output-dir
                                                                -> events.jsonl, windows.jsonl, data_profile.json
                                                                   (private) + u_windows.jsonl (public)
extract_dino_features.py     --windows --media-root --output-dir --cache-dir
                                                                -> frame_features.npy, frame_index.jsonl, feature_manifest.json
train_lr_baselines.py        --windows --feature-dir --output-dir -> lr_baseline_report.json
train_gru_baseline.py        --windows --feature-dir --output-dir -> gru_report.json + predictions
build_false_negative_bank.py --windows --predictions --base-report --feature-dir --output-dir
                                                                -> fn_bank.jsonl, fn_embeddings.npy, fn_bank_report.json
prepare_public_pool.py       --public-windows --base-report --feature-dir --output-dir
                                                                -> pool_windows.jsonl, pool_embeddings.npy, pool_report.json
freeze_selection_rankings.py --pool-dir --fn-dir --output-dir    -> selection_report.json + ranked lists
freeze_random_variance.py    --pool-dir --output-dir             -> random_seed{102,103}_ranked.jsonl + random_variance_report.json
```

Stage responsibilities that are easy to get wrong:

- **Ownership boundary.** Strem owns temporal pattern matching, clocks, guards,
  and persistent object bindings. Python owns nuScenes joins, coordinate
  transforms, visual eligibility, subprocess handling, Schema 2.0 parsing, event
  grouping, and window labels. Never write a second if/else implementation of the
  three scenario rules in Python, and never parse or reconstruct Strem's symbolic
  time bounds.
- **Two Strem passes.** The specs run on complete scenes to find event regions,
  then the *same* spec re-runs on each event-overlapping five-frame substream to
  decide the window label. Full-scene frame bounds aggregate touching regions and
  are candidate evidence, not a minimal witness — do not clip them into labels.
- **devkit is the index.** Use `NuScenes(version=..., dataroot=...)` for table
  lookup, sample chains, reverse relations, and calibrated boxes. Do not rebuild
  token indexes.
- **Label states.** Per `(window_id, scenario_id)`: `positive` (bounded match),
  `ignore` (event overlap, no bounded match), `negative` (no overlap), `invalid`
  (Strem/evidence failure). `ignore` and `invalid` are masked from loss; a Strem
  error or timeout is never a negative.
- **Scenario order is filename sort order** of `specs/task_spec_v2/*.json`:
  near-zone entry, proximity hold, corridor entry. Label vectors, thresholds, and
  every report table index in that order.
- **Public/private separation.** `u_windows.jsonl` and `pool_windows.jsonl` carry
  only the fields in `window_dataset.PUBLIC_U_FIELDS` plus DINO embeddings and
  Base probabilities. Labels, Strem results, bindings, instance tokens, 3D boxes,
  and visibility are Oracle-only. Selected IDs freeze *before* any Oracle reveal.

## Frozen Constants

These are pinned in code and documentation; changing one invalidates existing
private artifacts, so treat any change as a protocol change:

- split: `split-v1`, whole `log_token` groups, cumulative boundaries 10/35/50/100%,
  official validation → `frozen_test` (`splits.py`).
- eligibility filter: valid CAM_FRONT projection, letterbox height ≥ 16 px,
  clipped-area fraction ≥ 0.5, visibility token in `{2,3,4}` (`projection.py`).
- features: `facebook/dinov2-small` at revision
  `ed25f3a31f01632728cabb09d1542f84ab7b0056`, 518×518 letterbox with RGB fill
  `(124,116,104)`, `use_fast=False`, 384-d `pooler_output` float32
  (`dino_features.py`).
- window representation for selection: `L2(concat(mean(f0..f4), f4-f0))` → 768-d.
- GRU: seeds `(17, 29, 43)`, hidden 128, AdamW `lr=1e-3` / `wd=1e-4`, batch 128,
  ≤ 50 epochs, patience 5, Gate-B margin `+0.05` over prevalence
  (`gru_baseline.py`). Base thresholds come from seed 17 max-F1 on Development.
- selection: random seed `101`, cluster seed `17`, shortlist `M=2000`,
  `MiniBatchKMeans(K=30)`, temporal separation `5`, nested budgets
  `150 ⊂ 300 ⊂ 600` (`selection.py`). Budget experiments reuse prefixes of the
  frozen 600-row lists rather than re-running selection.
- Gate B is a diagnostic only. It never removes a Gate-A class from loss,
  selection, retraining, or the final Macro-AP.

## Repository Layout Notes

- src-layout package `driving_scene_data_loop`, Python 3.12 only, mypy `strict`,
  ruff line length 100 with `E4,E7,E9,F,I,UP,B`.
- Formal data is nuScenes `v1.0-trainval` on the remote Procyon CPU host; the
  local `data/raw/nuscenes` mini split is smoke-testing only.
- `data/`, `models/`, `artifacts/`, `dist/`, and all run outputs are gitignored.
  Private media, features, checkpoints, and run artifacts never enter Git or a
  model-provider upload.
- `specs/task_spec_v2/` holds the three immutable Gate-A scenario automata; they
  are frozen inputs, not code to edit.
