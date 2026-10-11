# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Agreement

`AGENTS.md` is the authoritative contract for this repository: read it first, and
follow its documentation reading order (`docs/PROJECT_SCOPE.md` →
`docs/ARCHITECTURE.md` → `docs/STREM_SKILL.md` →
`docs/DATA_SPEC.md` → `docs/EVALUATION_PLAN.md` → `docs/BAD_CASE_PROCESS.md` →
`docs/FINDINGS.md` → `docs/ROADMAP.md` → `docs/DEVELOPMENT_ENVIRONMENT.md`)
before touching a stage you have not worked on. Key points that shape every edit:

- This is a one-person offline research experiment, not a service or a reusable
  platform. The scenario specification is frozen at `v0.8-three-class-loop`;
  later protocols (v0.10, v0.11, diagnostics D1–D6, v0.13) build on
  it without editing it. Implement the smallest direct solution: plain
  functions, small dataclasses, JSONL + `.npy`. Do not add registries, managers,
  plugin layers, defensive fallback trees, or hostile-input handling without an
  observed failure that blocks the experiment.
- Validation exists only to protect labels, split isolation, model inputs,
  selector leakage, and fair evaluation. Those checks are required; general
  defensive programming is not.
- English for code, identifiers, comments, protocol documents, logs, and CLI
  text. Chinese for discussion with the maintainer. `FINDINGS.md` holds the facts.
- Never present a plan, fixture, or synthetic value as a real result; mark
  unimplemented work `Planned`. Preserve negative results — this project does not
  tune until the desired answer appears.
- A new experiment is declared in `docs/EVALUATION_PLAN.md` *before* it runs, and
  its reading rule is fixed there. Do not silently change the research question,
  scenario definitions, split, budget, Strem boundary, or evaluation protocol;
  update the owning document when behavior changes.
- The repository is public. Maintainer-private material lives in the gitignored
  `private/` directory (indexed by `private/README.md`) and never enters Git.

## Project State

| Protocol | Status | Where |
| --- | --- | --- |
| v0.8 bad-case similarity selection | done; held-out null (`+0.0033`, 0.3σ) | Findings §8–13 |
| v0.10 ensemble-disagreement selection | done; Test2 `+0.0619 ± 0.0082` corridor AP, 7.6σ | Findings §14 |
| v0.11 remote-VLM labels | done; Macro-F1 `0.494`, 86% of the Oracle gain (Development) | Findings §15 |
| D1–D6 scoping diagnostics | done; Development-level | Findings §16 |
| v0.13 open-weight VLM, zero-shot and LoRA | done; LoRA Macro-F1 `0.437` vs remote `0.494`, criterion not met | Findings §17 |
| v0.13 downstream arm on the LoRA labels | done; corridor gain ≈ Oracle's (`+0.037 ± 0.022`, Development), above the remote VLM's (3.0σ); above the Oracle's with a hidden-128 student (D6) | Findings §16–17 |
| label-cascade analysis | done; `agree` policy machine Macro-F1 `0.686` at review rate `0.372` (descriptive) | Findings §17 |
| bad-case review | done; six windows, every layer, checked against frames | Findings §17–18 |

Both held-out sets are spent (`frozen_test` for v0.8, Test2 for v0.10) and are
never reopened. Any new held-out claim needs a freshly carved set first.

## Commands

```bash
uv sync --locked --all-groups --all-extras   # runtime + dev + ml (torch, transformers) + vlm
uv run ruff check .
uv run mypy src scripts tests        # strict mode
uv run pytest
uv lock --check
```

Always go through `uv run`; a bare `python3` may be a different interpreter.
Sync with `--all-extras`: a plain `--all-groups` sync uninstalls torch.

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

## Environment

Every committed result was produced on Procyon, a university Linux CPU host that
is **no longer available**; all private artifacts survive in an offline backup.
The pinned macOS Strem binary is lost (a rebuild has a different hash and is
refused), so Strem-dependent stages and the real-binary tests need a Linux
x86_64 host with the pinned Linux binary. Everything else runs on the macOS
laptop. Measured on 2026-10-11: regenerating fourteen stages on macOS from the
backup reproduced rankings, reveals, and label files byte for byte and GRU
outputs within `1e-6`, but LR baselines differ by up to `0.0021` AP across
platforms. Details: `docs/DEVELOPMENT_ENVIRONMENT.md`.

## Pipeline Architecture

Thirty-two scripts under `scripts/` are the stage entry points;
`src/driving_scene_data_loop/` holds the logic. Stages chain by fixed artifact
filenames, so the output directory of one stage is the input directory of the
next. Every stage refuses to write into an existing output path, so a rerun needs
a new directory. Every JSON report and JSONL row is written through
`json_files.py` (`write_report`, `write_jsonl`, `jsonl_line`), which fixes the
byte format.

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
train_gru_baseline.py        --windows --feature-dir --output-dir -> gru_report.json + checkpoints + predictions
build_false_negative_bank.py --windows --predictions --base-report --feature-dir --output-dir
                                                                -> fn_bank.jsonl, fn_embeddings.npy, fn_bank_report.json
prepare_public_pool.py       --public-windows --base-report --feature-dir --output-dir
                                                                -> pool_windows.jsonl, pool_embeddings.npy, pool_report.json
freeze_selection_rankings.py --pool-dir --fn-dir --output-dir    -> selection_report.json + ranked lists
freeze_random_variance.py    --pool-dir --output-dir             -> random_seed{102,103}_ranked.jsonl + random_variance_report.json
reveal_selected_labels.py    --windows --ranking ... --output-dir
                                                                -> revealed_labels.jsonl + label_profile.json
train_feedback_gru.py        --arm --private-windows --public-windows --oracle-dir
                             --feature-dir --output-dir           -> gru_report.json + Development predictions
freeze_score_ranked_selection.py --pool-dir --output-dir          -> mining_v2_score_ranked_ranked.jsonl (Development-informed)
predict_with_checkpoints.py  --run-dir --windows --feature-dir --partition --output-dir
                                                                -> predictions_seed_*.jsonl + manifest.json
score_predictions.py         --windows --partition --run NAME=DIR ... --output
                                                                -> per-seed AP + seed-paired contrasts

# protocol v0.10
prepare_v010_partitions.py   --private-windows --public-windows --revealed ... --output-dir
                                                                -> pool2_windows.jsonl, test2_eval_ids.txt, manifest
prepare_pool2_inference.py   --pool2-windows --base-run-dir --feature-dir --output-dir
                                                                -> pool2_windows.jsonl (+ ensemble mean/std), pool2_report.json
freeze_v010_rankings.py      --pool2-dir --output-dir            -> disagreement / prob-ranked / random201-203 rankings

# protocol v0.11
prepare_vlm_smoke_batch.py   --windows --output-dir             -> smoke_windows.jsonl (public view)
                                                                   + smoke_ranking.jsonl + oracle/
label_windows_with_vlm.py    --ranking --public-windows --media-root --output-dir
                                                                -> vlm_labels.jsonl + vlm_run_manifest.json
evaluate_vlm_labels.py       --vlm-labels --oracle-dir --method --budget --output-dir
                                                                -> vlm_label_report.json
build_vlm_training_labels.py --vlm-labels --ranking --budget --output-dir
                                                                -> revealed_labels.jsonl + label_profile.json
                                                                   (vlm_labeled_windows identity, Oracle never opened)

# diagnostics D1-D6
analyze_ensemble_size.py     --pool-dir --frozen-ranking --base-run-dir --feature-dir --output   (D1)
build_noise_injected_labels.py --oracle-dir --rate-percent --output-dir                         (D2)
prepare_d4_pool.py           --pool2-inference-dir --v08-pool-dir --base-report --output-dir     (D4)
report_development_contrasts.py --run NAME=REPORT ... --contrast LEFT-RIGHT ... --output         (D2-D5)

# protocol v0.13: bundle locally, run on the rented GPU, score locally
prepare_vlm_v013_bundle.py   --private-windows --ranking --pool2-windows --media-root --lf-template --output-dir
                                                                -> L0 SFT targets, public request rows, 640x360 frames,
                                                                   prompt.json, train_lora.yaml (no scored/sanity label)
run_vlm_vllm.py              --bundle --requests --model [--lora] --arm --output-dir   (GPU host, vLLM env only)
                                                                -> replies.jsonl + runner_manifest.json
parse_vlm_replies.py         --replies-dir --output-dir [--sanity] -> vlm_labels.jsonl + hashed manifest (+ sanity gate)
compare_vlm_labelers.py      --oracle-dir --arm-a --arm-b --arm-c --output-dir
                                                                -> per-arm reports + paired-bootstrap criterion
analyze_label_cascades.py    --oracle-dir --arm-a --arm-c --a-cost-usd --c-cost-cny --output
                                                                -> label_cascade_report.json (descriptive, no model call)
```

Feedback arms are declared in `feedback_retraining.py` (`ALL_ARMS`); an arm's
label directory must carry the artifact identity of its declared label source
(`oracle`, `vlm`, `oracle_noise`). Trained checkpoints are reloaded through
`gru_baseline.load_checkpoint_model`, which reads width and class count from the
weights.

The final evaluation is two-stage: `predict_with_checkpoints.py` writes frozen
per-seed prediction files (no labels), `score_predictions.py` joins labels and
computes seed-paired contrasts. The scorer reproduced every training-report
Development AP bit-for-bit before being trusted. `--training-config` selects a
declared protocol from `TRAINING_CONFIGS` (`gru_baseline.py`); the current
measurement standard is `narrow-fast-12seed` (hidden 48, twelve seeds with
`17, 29, 43` as a nested prefix), chosen by pre-declared seed-variance criteria
after the frozen `v1-frozen` protocol proved noise-limited. Three-seed spreads
are unreliable — treat any `± x` from three seeds as unknown.

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
- **Public/private separation.** `u_windows.jsonl`, `pool_windows.jsonl`, and
  `pool2_windows.jsonl` carry only the fields in `window_dataset.PUBLIC_U_FIELDS`
  plus Base probabilities (and, for Pool2, their ensemble spread). Labels, Strem
  results, bindings, instance tokens, 3D boxes, and visibility are Oracle-only.
  Selected IDs freeze *before* any Oracle reveal, and VLM requests are built
  from public rows only.

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
- GRU `v1-frozen`: seeds `(17, 29, 43)`, hidden 128, AdamW `lr=1e-3` / `wd=1e-4`,
  batch 128, ≤ 50 epochs, patience 5, Gate-B margin `+0.05` over prevalence
  (`gru_baseline.py`). Base thresholds come from seed 17 max-F1 on Development.
- v0.8 selection: random seed `101` (variance seeds `102, 103`), cluster seed
  `17`, shortlist `M=2000`, `MiniBatchKMeans(K=30)`, temporal separation `5`,
  nested budgets `150 ⊂ 300 ⊂ 600` (`selection.py`). Budget experiments reuse
  prefixes of the frozen lists rather than re-running selection.
- v0.10: seed set `L0-small` (the lc25 subset), Test2 carved by
  `sha256("test2-v010\0" + log_token)` order with a 55-corridor-event guard,
  Random seeds `201, 202, 203`, nested budgets `300 ⊂ 600 ⊂ 900`.
- v0.11: `gpt-5.6-terra`, frozen prompt hash `d418e9bb…`, strict JSON schema
  (`vlm_labeling.py`).
- Gate B is a diagnostic only. It never removes a Gate-A class from loss,
  selection, retraining, or the final Macro-AP.

## Repository Layout Notes

- src-layout package `driving_scene_data_loop`, Python 3.12 only, mypy `strict`,
  ruff line length 100 with `E4,E5,E7,E9,F,I,UP,B`.
- Formal data is nuScenes `v1.0-trainval`; the local `data/raw/nuscenes` mini
  split is smoke-testing only.
- `data/`, `models/`, `artifacts/`, `dist/`, `private/`, and all run outputs are
  gitignored. Private media, features, checkpoints, and run artifacts never enter
  Git or a model-provider upload. The single declared exception is the v0.11 VLM
  stage, which sends the five CAM_FRONT frames of a frozen selected window and
  the public scenario descriptions — never labels, Strem evidence, bindings,
  boxes, or features — under the boundary in `docs/DATA_SPEC.md`. v0.13 runs
  open weights on a rented GPU under its own boundary in the same document.
- `results/` holds the committed aggregate report JSON for each private run, and
  is the source for every number quoted in the docs. Add a report there when a
  stage completes; never add its rows, embeddings, or labels.
- `specs/task_spec_v2/` holds the three immutable Gate-A scenario automata; they
  are frozen inputs, not code to edit.
- Commits carry no AI co-author trailer or other AI attribution.
