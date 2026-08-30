# Roadmap

Specification: Frozen for `v0.8-three-class-loop`

This is an accelerated, evidence-oriented personal-project plan. The remaining work
is organized by evidence needed for scene-mining and the data-loop record, not
by adding infrastructure. Expect roughly 12 to 14 focused sessions after the
current foundation; measured DINO CPU speed may move that estimate.

## Completion Standard

Stop when these results exist:

1. The three Gate-A-frozen Strem labels each receive a documented Gate-B
   decision.
2. Whole-log splits and five-frame windows have no known leakage.
3. A compact data-quality profile explains cleaning, filter reasons, label
   states, and split distributions.
4. DINOv2 features feed LastFrame-LR, Mean5-LR, and a PyTorch GRU.
5. Development false negatives produce one explainable integrated Mining rule.
6. Mining-Oracle and three independent Random-Oracle batches are retrained
   fairly with the same primary `N=300` budget.
7. A remote VLM labels the same frozen Mining-300 IDs, is measured
   against the Oracle, and supplies one controlled VLM retraining comparison.
8. Held-out Test metrics, four compact tables, and at least five deep cases are
   documented.

Extra selectors, UI, databases, Agent, SFT, distributed platforms, and
production engineering do not block completion.

## Operating Rules

- Procyon is the primary private environment; no compute GPU is observed.
- Raw media, features, weights, runs, and VLM requests/responses stay private
  and outside Git.
- Use direct functions and plain files. Add only validation that protects
  labels, split isolation, model inputs, selector leakage, or fair evaluation.
- Combine repeated scans and audits. Do not turn one-off analysis into a generic
  framework.
- Give the standard answer before connecting it to code.
- A tie or regression is a valid result; support and metrics are never invented.
- Finish the Oracle-label causal loop before interpreting the VLM-label result.

## Completed Foundation

- Python 3.12/uv environments on macOS and Procyon.
- nuScenes trainval metadata: 850 scenes, 34,149 samples, and 68 logs.
- Whole-log `split-v1`: TaskDesign/L0/Development/U plus official-val Frozen
  Test.
- All 2,747 TaskDesign CAM_FRONT keyframes and the frozen projection/visibility
  eligibility rule.
- 850 CAM_FRONT-eligible Strem streams with 120,035 retained target annotations
  from 859,857 candidates and 45,847 traceable object IDs.
- Fixed external Strem v0.3.0 boundary with exact timestamps, Schema 2.0,
  persistent bindings, and real-release semantic tests.
- All three candidate families pass positive-support prechecks on TaskDesign:
  pedestrian entry `(20m,15m)` at 16 events/3 logs, vehicle corridor
  `(3m,1.5m)` at 46/5, and pedestrian-vehicle hold `(5m,1s)` at 16/4.
- Pedestrian entry and pedestrian-vehicle hold passed compact positive,
  hard-negative, visual-semantic, and cross-partition audits.
- The corridor retained 20 hard-negative windows across 4 TaskDesign logs. A
  24-window review found no systematic defect, cross-partition support passed
  at `153/13, 74/6, 201/22, 69/15`, and all three rules are frozen in
  `specs/task_spec_v2`.
- Full-scene output can aggregate touching symbolic regions. A real five-frame
  substream check confirmed that the same fixed specification can decide the
  bounded model-window label.

Observed negative results remain part of the record: near-zone thresholds ending
at 3--4m had zero support, the 4m proximity hold had only 10 events/3 logs, a
receding pedestrian hard-negative rule had zero TaskDesign events, and the first
proximity hard-negative design yielded only 4 windows.

## Stage 0 — Audit the Third Scenario — Completed

Target: 1 focused session.

Completed evidence:

1. Kept the already-declared vehicle corridor definition:
   `0 <= x <= 30m`, `abs(y) >= 3m -> abs(y) <= 1.5m` within two seconds, for the
   same motor vehicle in each timestamp's ego frame.
2. Retained 20 difficult-negative windows across 4 TaskDesign logs after
   positive-overlap exclusion and temporal suppression.
3. Reviewed 10 positives and 14 negatives; all 88 participating frames had the
   eligible bound vehicle and no systematic defect was observed.
4. Passed L0/Development/U/Frozen-Test support thresholds.
5. Created immutable `task_spec_v2` with all three scenarios.

Key concepts: positive support versus label validity, hard negatives,
relative ego coordinates, persistent identity, and why this label does not
prove an active lane change.

## Stage A — Build Windows and the Data Profile — Completed

Observed formal output:

- 1,430 complete-scene event groups;
- 30,749 private five-frame windows across 850 scenes;
- 11,639 label-free public-U windows;
- zero Strem errors/timeouts and zero `invalid` labels;
- zero duplicate private IDs, malformed windows, or public/private U ID
  mismatches;
- 739,822 of 859,857 target annotations were removed by the frozen visual
  eligibility rule before formal event mining; the historical v1 eligibility
  summary did not retain mutually exclusive rejection-reason counts, and the
  profile records that limitation.

Implemented steps:

1. Generate complete-scene events and rerun the same Strem specs on
   event-overlapping five-frame substreams.
2. Produce positive/negative/ignore/invalid multi-label targets.
3. Write separate public-U and private-Oracle files.
4. Produce one data-quality report with join/filter reasons, category and split
   distributions, label states, log coverage, and later selected-batch yield.

Passed checks: stable binding mapping, bounded interval-to-window mapping, no
window crosses a scene, no log crosses partitions, all public/private U IDs
agree, and the single public U schema contains no Strem or label evidence.

Key concepts: data cleaning versus deleting source data, Pandas profiling,
timed intervals versus model windows, masked labels, and label leakage.

## Stage B — DINOv2 and Model Baselines — Completed

Target: 3 focused sessions.

DINO and LR observations:

- pinned revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`;
- 518x518 letterbox and normalization verified on real CAM_FRONT images;
- 58-frame CPU benchmark produced finite `(58,384)` float32 features at
  10.51 frames/s with batch 8 and 64 threads;
- the formal `(34149,384)` float32 cache completed in 2,120.6 seconds at 16.10
  frames/s; no GPU was required;
- LastFrame-LR and Mean5-LR completed on L0 and Development with Macro-AP
  `0.1499` and `0.1942` across the three Gate-A classes;
- the three-class GRU Gate-B run retained near-zone and corridor entry and
  rejected pedestrian-vehicle proximity hold;
- the final two-class Base GRU reached Development Macro-AP `0.3636 ± 0.0157`;
  reversed-input Macro-AP was `0.3328 ± 0.0232`. No U or Frozen-Test labels
  entered these runs.

1. Completed: pin DINOv2-Small, verify preprocessing, and cache each required
   frame's 384-dimensional global feature once.
2. Completed: train LastFrame-LR and Mean5-LR and save Development predictions.
3. Completed: implement `[B,5,384] -> GRU(128) -> Linear(C)` with masked BCE;
   the real tiny subset was memorized before formal training.
4. Completed: train seeds `17, 29, 43` and compare LR, GRU, normal order, and
   same-checkpoint reversed Development input.
5. Completed: report Gate B as an initial-learnability diagnostic. The
   three-output Base is the closed-loop model; the old two-output run remains a
   historical order-sensitivity diagnostic.

Required checks: DINO shape/finite values, train-only class weights, GRU
shape/loss/gradient/checkpoint reload, and a hand-checkable AP example.

Key concepts: ViT/CLS/self-supervision, frozen encoders, Logistic
Regression, BCE, imbalance, GRU gates, backpropagation, early stopping, AP, and
why a more complex model is not automatically better.

## Stage C — Primary Bad-Case-Driven Data Loop

Target: 3 focused sessions.

Prerequisite now frozen: three-class Base checkpoints, seed-17 thresholds, and
Development predictions.

1. Completed: rebuilt the FN bank from the three-class Base. It contains
   `26/21/22` event rows by scenario-table order, 69 total, represented by 61
   unique windows; every class clears the five-event minimum.
2. Completed: ran the three-class Base on all 11,639 label-free public Pool
   windows and cached finite, unit-normalized
   `L2(concat(mean(f0..f4), f4-f0))` 768-dimensional vectors. The public rows
   contain no Oracle fields.
3. Completed: froze 600-row Random and integrated Mining ranked lists. Each
   Mining class keeps its top 2,000 FN-similar candidates; Mining fits
   `MiniBatchKMeans(K=30)` per class only after this shortlist and temporal
   deduplication.
4. Completed: fixed the first 300 IDs for the primary comparison and nested
   prefixes `150/300/600` for the Oracle-only Random-versus-Mining
   budget curve. No Oracle labels were read.
5. Completed: froze two further Random batches with seeds `102` and `103` so
   Random is a distribution rather than one draw. Both reuse the same public
   Pool, budget, and temporal rule; `selection-rankings-v3` was not modified and
   no Oracle label was read. Observed `N=300` pairwise overlap is 6, 7, and 9
   windows, close to the uniform-draw expectation of 7.7.
6. Completed: revealed private Oracle labels for the four frozen rankings into
   `oracle-reveal-v1`. The reader parsed all 30,749 private rows and retained/
   output the 2,220 selected IDs; rankings were already frozen and were not
   changed. It produced no `invalid` label and refilled no `ignore`. Mining
   exceeded the Random range in positive yield on the proximity-hold class only.
7. Completed: round one trained the four frozen batches under both declared
   protocols. Adding 300 windows beat Base, but Mining and the three Random
   batches were indistinguishable, and fine-tuning absorbed the added data
   almost entirely. Reported as observed.
8. Completed: designed and tested Mining v2, a Development-informed selector
   that ranks each class by Base probability with no similarity filter and no
   clustering. It bought 2 to 5 times more positives than the best Random
   batch at `N=300` but produced no clean Macro-AP gain under either protocol,
   and its near-zone AP regressed with unusually high seed variance. Reported
   as Development-informed, not pre-registered.
9. Completed: rebuilt the measurement — `narrow-fast` config chosen by
   pre-declared criteria, twelve seeds per arm with `17, 29, 43` as a nested
   prefix, seed-paired statistics, a whole-log L0 learning curve, and a budget
   dose-response over the already-revealed 150/300/600 prefixes. Outcome: the
   three-seed spreads were unreliable; proximity hold is information-limited
   (flat learning curve); and the pre-registered Mining selector shows a
   corridor advantage of `+0.025` AP over the Random mean (3.7σ seed-paired,
   2.3σ with batch variance folded in, `+0.035` over Base) with a consistent
   sign across budgets. Recorded as Development-informed.
10a. Protocol v0.10 declared and frozen: small seed (the lc25 point),
    query/eval decoupling with a fresh 8-log Test2 carved from U, disagreement
    and prob-ranked selectors against three new Random batches, nested budgets
    300/600/1200 with the primary criterion at 1200. See the Evaluation Plan's
    v0.10 section; frozen before any v0.10 reveal.
10. Completed: froze 650,280 prediction rows across ten arms, verified they
    carry no label field, and opened the Held-out Test once. The pre-declared
    criterion reads `+0.0033 ± 0.0101` (0.3σ): the Development corridor effect
    did not transfer. The realised standard error, `0.0101`, would still have
    shown a Development-sized effect at about 2.4σ, so this is absence rather
    than lost power. Reported unchanged; no arm, threshold, or criterion was
    revised afterwards.

Required checks: public-U schema has no Oracle fields, equal budgets, temporal
separation, frozen selected IDs before reveal, Random variance seeds fixed
before reveal, and identical retraining config.

Key concepts: false negatives, cosine similarity, uncertainty versus
confidence, active data selection, Oracle simulation, and controlled variables.

## Stage D — VLM Automatic-Label Evaluation

Target: 2 focused sessions.

1. Freeze a remote VLM provider/model, prompt, input preprocessing, and JSON
   schema.
2. Send only the already-frozen Mining-300 five-frame images; do not send Strem
   evidence, 3D boxes, distances, bindings, or Oracle labels.
3. Collect per-class `positive|negative|uncertain`, evidence frames, confidence,
   limitations, latency, and cost.
4. Compare VLM labels with the Oracle on the same IDs.
5. Retrain Mining-VLM with the same model settings and compare it with
   Mining-Oracle. Mask `uncertain` labels without replacing windows.

Required checks: schema validity, no Oracle field in requests, exact ID equality
between Mining-VLM and Mining-Oracle, and no prompt tuning from Frozen Test.

Key concepts: VLM structured output, automatic-label precision/recall,
hallucination, uncertainty, monocular metric limitations, label noise, and why
API inference is not SFT.

## Stage E — Frozen Evaluation and Result Delivery

Target: 2 focused sessions.

1. Train every arm in the frozen arm list and freeze every prediction file
   before the first Held-out Test score. The Test is opened once, for all arms
   together.
2. Produce four compact tables:
   - data cleaning, scenario, event, and window profile;
   - LastFrame-LR versus Mean5-LR versus GRU, including reversed order;
   - Base versus the three Random-Oracle batches versus Mining-Oracle, with
     per-class AP, `AP - prevalence` margin, and Macro-AP;
   - VLM label quality and Mining-VLM versus Mining-Oracle.
3. Explain at least five representative cases covering data/label, Base FN,
   selector behavior, VLM labeling, and retraining outcome.
4. Finalize reproducible commands, limitations, a resume bullet, and 30-second,
   3-minute, and detailed written explanations.

No extra tuning occurs after the Held-out Test is opened. If Mining does
not beat Random or its VLM-label run trails Oracle labels, report and diagnose the result
instead of changing the protocol until it wins.

## Optional Only After Completion

- extra clustering algorithms or K tuning;
- optional similarity-only or uncertainty-only ablations;
- bootstrap analysis;
- small controlled photometric/low-light/rain augmentation;
- spatial DINO features;
- VLM SFT, Agent, UI, workflow systems, vector DB, or distributed platform work;
- production-scale or online-loop claims.

## Recovery Order

If a class remains weak after feedback, preserve it as a negative result and
separate data-limited, representation-limited, and label-ambiguity hypotheses.
If VLM outputs are too unreliable for useful retraining, report their label
metrics and bad cases without claiming Oracle replacement; the primary
Random-Oracle versus Mining-Oracle experiment remains valid. Never remove
whole-log isolation, stable Strem bindings, the public/private U boundary,
equal-budget from-scratch retraining, Frozen Test discipline, or honest
negative-result reporting.
