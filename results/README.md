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

## What these files are not

They are aggregate reports, not data. They contain no image, no nuScenes table
row, no `window_id`, no scene or instance token, no embedding, and no label.
They cannot be used to reconstruct the dataset, and they are not a substitute
for nuScenes, which each user must obtain under its own licence.

They are also not a complete result set. Oracle reveal, feedback retraining,
VLM labelling, and Held-out Test scoring have not run, so no file here contains
a Held-out Test number. When those stages complete, their reports join this
directory.

## Reading them

```bash
python -m json.tool results/gru-baseline-v1/gru_report.json | less
```

`gru_report.json` is the densest one. `gate_b` holds each class's mean AP against
its prevalence, `base_thresholds` holds the seed-17 maximum-F1 thresholds that
define a false negative, `runs` holds the per-seed detail, and `aggregate`
compares normal against reversed Development frame order.
