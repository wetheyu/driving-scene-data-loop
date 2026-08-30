# Committed Result Reports

Every number quoted in this repository's documentation comes from one of the
JSON files here, so a reader can check a claim without access to the private
data. Each directory name is the private run that produced the file, under
`~/datasets/driving-scene-data-loop` on the compute host.

| File | Answers |
| --- | --- |
| `window-dataset-v1/data_profile.json` | how many events, windows, and label states each partition holds |
| `dinov2-small-v1/feature_manifest.json` | the frozen DINO model, revision, preprocessing, and cache shape |
| `lr-baselines-v2/lr_baseline_report.json` | LastFrame-LR and Mean5-LR Development AP |
| `gru-baseline-v1/gru_report.json` | the three-class Base: config, three seeds, Gate B, frozen thresholds |
| `gru-base-c2-v1/gru_report.json` | the historical two-class order-sensitivity diagnostic |
| `development-fn-bank-c3-v1/fn_bank_report.json` | false-negative counts per class and the deduplication rule |
| `public-pool-inference-v1/pool_report.json` | label-free Pool inference and the selection-vector definition |
| `selection-rankings-v3/selection_report.json` | frozen Random and Mining rankings, budgets, cluster diagnostics |
| `random-variance-v1/random_variance_report.json` | the extra Random batches used to estimate selection variance |
| `oracle-reveal-v1/label_profile.json` | what each frozen batch bought once its labels were revealed |
| `feedback-scratch-*/gru_report.json` | round-one arms retrained from scratch on L0 plus 300 |
| `feedback-finetune-*/gru_report.json` | the same arms fine-tuned from the Base checkpoint |
| `score-ranked-v2/score_ranked_report.json` | the Development-informed pure-probability selector (Mining v2) |
| `oracle-reveal-v2-score-ranked/label_profile.json` | what Mining v2 bought once its labels were revealed |
| `feedback-scratch-mining-v2-300/gru_report.json` | Mining v2 retrained from scratch |
| `feedback-finetune-mining-v2-300/gru_report.json` | Mining v2 fine-tuned from the Base checkpoint |
| `n12-*/gru_report.json` | the rebuilt measurement: every arm under `narrow-fast` at twelve seeds, including the L0 learning curve (`n12-lc*`) and the 150/300/600 dose-response |
| `learning-curve-v1/manifest.json` | which whole-log L0 subsets the learning curve trained on |
| `frozen-test/frozen_test_scores.json` | the one Held-out Test opening: ten arms, twelve seeds, per-seed AP and every seed-paired contrast |

## What these files are not

They are aggregate reports, not data. They contain no image, no nuScenes table
row, no `window_id`, no scene or instance token, no embedding, and no label.
They cannot be used to reconstruct the dataset, and they are not a substitute
for nuScenes, which each user must obtain under its own licence.

They are also not a complete result set. VLM labelling has not run. The
Held-out Test has now been opened exactly once, and
`frozen-test/frozen_test_scores.json` is that reading; it will not be
regenerated, because reopening it would void the discipline that makes it
meaningful.

## Reading them

```bash
python -m json.tool results/gru-baseline-v1/gru_report.json | less
```

`gru_report.json` is the densest one. `gate_b` holds each class's mean AP against
its prevalence, `base_thresholds` holds the seed-17 maximum-F1 thresholds that
define a false negative, `runs` holds the per-seed detail, and `aggregate`
compares normal against reversed Development frame order.
