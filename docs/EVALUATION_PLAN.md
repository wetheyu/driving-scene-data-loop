# Evaluation Plan

Specification: Frozen for `v0.8-three-class-loop`

Observed model results: DINO cache, LR baselines, three-class GRU Gate B
diagnostics, a formal three-class Development FN bank, and label-free public
Pool inference. Selection and Oracle reveal are observed; feedback retraining
and held-out scoring remain unobserved.

## Questions

Primary:

> Does integrated Mining select a more useful `N=300` training batch than
> Random under an otherwise identical Oracle-label retraining protocol?

Secondary:

> On the same frozen Mining-300 IDs, how accurately does a remote VLM
> reproduce the Oracle labels, and what is the Held-out Test gap between
> Mining-VLM and Mining-Oracle retraining?

A tie or regression is a valid result. The project succeeds by answering the
question credibly, not by forcing a positive number.

## Data Protocol

1. Split complete official-train logs before generating windows.
2. Use TaskDesign to freeze the three-class `task_spec_v2` at Gate A.
3. Train the three-class Base on L0 and use Development for early stopping and thresholds.
4. Build false-negative banks from Base seed 17 on Development only.
5. Run selectors on public U without hidden labels.
6. Freeze selected window IDs before Oracle reveal.
7. Reveal Oracle labels for the primary comparison; independently obtain VLM
   labels for exactly the same Mining-300 IDs.
8. Retrain each feedback model from scratch on L0 plus its declared batch and
   label source.
9. Freeze all Test predictions before scoring official val once.

## Threshold and Hyperparameter Provenance

Thresholds come from different stages and must not be mixed:

| Value | How it was fixed | What it controls |
| --- | --- | --- |
| Scene distances and durations | first predeclared candidate passing Design Set support and audit Gate A | Strem benchmark labels |
| `16 px`, retained area `0.5`, visibility `2..4` | Design Set projection/visibility audit before training | whether an annotation is visually eligible |
| five frames, stride one | annotated keyframe rate and the approximately two-second task horizon | model window construction |
| start separation `5` | one complete five-frame window length | prevents overlapping selected windows |
| Base thresholds `0.885465/0.538358/0.637988` | seed-17 Development maximum F1, once per class | FN definition and uncertainty boundary |
| Gate B margin `+0.05` over prevalence | predeclared diagnostic heuristic | reports weak Base learnability; deletes no class |
| `M=2000` | ten times the maximum per-class quota `600/3=200` | relevant shortlist before uncertainty/diversity |
| `K=30` | fixed lightweight feature-space partition, not tuned for semantic quality | diversity constraint only |
| budgets `150/300/600` | primary engineering budget 300 plus half/double sensitivity | amount of revealed data |
| Random seeds `101/102/103` | predeclared before any Oracle label was revealed | estimates Random batch-to-batch variance |

No scene threshold, selection hyperparameter, or budget is chosen from Pool
Oracle labels or Held-out Test performance. The exact numeric values are
engineering choices or Design/Development decisions, not claimed optima.

## Scene Gate A

For each provisional scenario candidate, report positive events, hard negatives,
log coverage, projection/eligibility failures, and reviewed binding/semantic
errors. Choose the first candidate satisfying [Project Scope](PROJECT_SCOPE.md),
not the threshold producing the best model score. Review a compact stratified
gallery sufficient to detect systematic defects; exhaustive image-by-image
annotation is not a project deliverable. Complete-scene matches identify event
regions; event-overlapping five-frame candidates are labeled by rerunning the
same specification on the bounded substream. Aggregated full-scene frame bounds
are not clipped or interpreted as a minimal witness in Python.

All three `task_spec_v2` classes passed Gate A. The corridor `(3m,1.5m)` audit
retained 20 difficult-negative windows across 4 TaskDesign logs, reviewed 10
positives and 14 negatives without a systematic defect, and passed the declared
cross-partition support thresholds. Its slow-entry hard-negative type yielded
zero usable windows and remains reported as a negative result.

## Model Baselines

### LastFrame-LR

Input: final-frame DINO global feature `[384]`. Fit one independent Logistic
Regression per retained class.

### Mean5-LR

Input: mean of five DINO global features `[384]`. It deliberately loses order.

### Observed LR Development results

Both baselines fit a StandardScaler and three independent L2 Logistic
Regressions on L0 only. Positive class weights come from L0 only. AP uses only
valid Development labels; weighted LR outputs are ranking scores, not claimed
as calibrated probabilities.

| Baseline | Near-zone entry AP | Pedestrian-vehicle hold AP | Corridor entry AP | Macro-AP |
| --- | ---: | ---: | ---: | ---: |
| Development prevalence | 0.0199 | 0.0328 | 0.0770 | 0.0432 |
| LastFrame-LR | 0.0381 | 0.0603 | 0.3513 | 0.1499 |
| Mean5-LR | 0.0930 | 0.0671 | 0.4226 | 0.1942 |

All six fits converged in 133--186 iterations under the fixed 1,000-iteration
limit. Mean5 is higher for every class, showing that multi-frame aggregation is
useful here. Because averaging destroys frame order, this is not evidence of
temporal-order learning. No LR threshold or Frozen-Test score was produced.

### Global-GRU

```text
input: [batch, 5, 384]
one-layer unidirectional GRU, hidden size 128
Dropout(0.2) after final hidden state
Linear(128, C)
masked BCEWithLogitsLoss
AdamW(lr=1e-3, weight_decay=1e-4)
batch size 128, at most 50 epochs, patience 5
training seeds 17, 29, 43
```

Compute `pos_weight` from L0 once and use it for every method. Base seed 17
chooses each class threshold on Development by maximum F1; those thresholds then
freeze.

The three-seed checkpoint protocol uses normal-order Development Macro-AP and
patience 5. Gate B uses the mean normal-order AP across seeds `17,29,43`, never
the best seed. After each best checkpoint freezes, reversing only the
Development time axis supplies the order-sensitivity diagnostic; no reversed
model is retrained.

### Gate B Diagnostic

Global-GRU mean Development AP must exceed class prevalence by at least 0.05.
Every Gate-A class is evaluated independently; a failing class is removed from
the quantitative loop. A spatial model is optional after the minimum closed
loop.

Also compare LastFrame, Mean5, GRU, and normal/reversed order. Do not claim
temporal learning if the evidence does not support it.

Observed three-class Gate-B run:

| Scenario | GRU AP mean ± std | Prevalence | Margin | Gate B |
| --- | ---: | ---: | ---: | --- |
| pedestrian ego near-zone entry | 0.1675 ± 0.0215 | 0.0199 | 0.1476 | pass |
| pedestrian-vehicle proximity hold | 0.0537 ± 0.0056 | 0.0328 | 0.0209 | **fail** |
| vehicle relative corridor entry | 0.5011 ± 0.0113 | 0.0770 | 0.4241 | pass |

The proximity-hold label remains a valid Gate-A formal scene class and becomes
the challenge class in the quantitative loop. The failed diagnostic is kept as
the Base observation; it does not exclude the class from loss, selection,
retraining, or final metrics.

### Historical two-class diagnostic

This earlier run retrained only the two above-margin outputs. It remains useful
as a diagnostic but is not the Base model for the three-class closed loop.

| Baseline on retained classes | Near-zone entry AP | Corridor entry AP | Macro-AP |
| --- | ---: | ---: | ---: |
| LastFrame-LR | 0.0381 | 0.3513 | 0.1947 |
| Mean5-LR | 0.0930 | 0.4226 | 0.2578 |
| Global-GRU normal, mean ± std | 0.1815 ± 0.0056 | 0.5457 ± 0.0289 | 0.3636 ± 0.0157 |
| Same GRU checkpoints, reversed input | 0.1550 ± 0.0382 | 0.5106 ± 0.0414 | 0.3328 ± 0.0232 |

The normal-order GRU exceeds both simple controls on Development. Reversal
reduces mean Macro-AP by 0.0308, but not every class/seed declines. The result
supports limited order sensitivity, not a claim that the GRU understands
driving dynamics.

The three-class Base seed-17 maximum-F1 thresholds from `gru-baseline-v1` are
used for the new three-class Development FN bank. Thresholds decide FNs but do
not affect AP or the Gate B diagnostic.

## Development False-Negative Bank

Using Base seed 17 and frozen thresholds:

- include Ground Truth positive, predicted-negative Development windows;
- group them by the real `(scenario_id, event_group_id)` and keep the
  lowest-probability representative per event; ties use the smallest
  `window_id`;
- require at least five independent false-negative events per retained class.

Frozen Test failures never enter this bank.

Historical two-class result:

| Retained class | Raw FN windows | Event rows with any FN | Events missed in every positive window | Unique representative windows |
| --- | ---: | ---: | ---: | ---: |
| pedestrian ego near-zone entry | 46 | 28 | 26 | 22 |
| vehicle relative corridor entry | 48 | 23 | 11 | 22 |

The bank therefore contains 51 event rows backed by 44 unique windows. An event
enters when at least one of its positive windows is missed; it need not be
missed in every view. Each row has an aligned unit-normalized 768-dimensional
DINO vector `concat(mean(f0..f4), f4-f0)`. The artifact contains no U or
Frozen-Test labels.

The formal three-class rebuild uses the original three-output Base and contains
69 event rows represented by 61 unique windows:

| Class | Raw FN windows | Event rows with any FN | Events missed in every positive window | Unique representative windows |
| --- | ---: | ---: | ---: | ---: |
| pedestrian ego near-zone entry | 45 | 26 | 26 | 20 |
| pedestrian-vehicle proximity hold | 57 | 21 | 12 | 19 |
| vehicle relative corridor entry | 48 | 22 | 11 | 22 |

Its aligned selection matrix has shape `(69,768)`. Public-Pool inference has
11,639 unique rows and a finite unit-normalized `(11639,768)` matrix; its rows
contain only public metadata plus three Base probabilities.

## Selection Methods

All methods use the same public Pool and same-scene minimum separation of five
window starts. Oracle `ignore` consumes budget.

- Base: no added data;
- Random: uniformly shuffle public window IDs, then apply the common temporal
  suppression. Three independent batches use seeds `101`, `102`, and `103`;
  seed `101` remains the originally frozen list;
- Mining: per class, rank by maximum cosine similarity to that class's
  Development FN bank, keep the top `M=2000` relevant candidates, prefer points
  close to the frozen Base decision threshold, apply temporal suppression, fit
  `MiniBatchKMeans(K=30, random_state=17)`, and round-robin the clusters.

Boundary uncertainty is ranked by distance to the frozen class threshold, not
distance to `0.5`, because class weighting and maximum-F1 threshold selection
make `0.5` an arbitrary boundary for this experiment. Similarity answers
“related to a known failure,” uncertainty answers “near the current decision
boundary,” temporal suppression removes overlapping repetition, and clustering
spreads the remaining batch across visual feature regions. Clusters are not
treated as semantic scene labels.

The exact boundary margin is
`abs(logit(clamp(p))-logit(clamp(threshold)))`; smaller is more uncertain.
Mining ties use higher similarity and then `window_id`, and keep that order
inside each cluster. The three class queues are merged in
frozen scenario order, one item per class per round, so Mining prefixes contain
`50/100/200` query slots per class at budgets `150/300/600`. Cross-class window
duplicates and same-scene starts closer than five keyframes are skipped and
refilled from the same frozen queues.

The primary budget is `N=300`, allocated equally across three query classes
before cross-class duplicate merging and deterministic refill. A single ranked
list of 600 IDs freezes before reveal; prefixes `150 ⊂ 300 ⊂ 600` provide an
Oracle-only Random-versus-Mining budget sensitivity check. VLM labeling uses
only the 300-ID primary Mining batch.

All selectors see only DINO embeddings, Base probabilities, frozen thresholds,
and ordinary public window metadata. Pool labels, Strem matches, bindings, 3D
annotations, and event evidence are forbidden until selected IDs freeze.

### Observed label-free selection diagnostics

The formal `selection-rankings-v3` artifact freezes 600 unique rows per method;
both methods satisfy the common temporal rule and contain no Oracle fields.
At the primary 300-row prefix, every Mining class has exactly 100 query slots.

| Method/class at N=300 | Near-zone | Proximity hold | Corridor |
| --- | ---: | ---: | ---: |
| Mining cluster coverage | 30 | 29 | 30 |
| Mining maximum cluster share | 0.04 | 0.05 | 0.05 |

The selected Mining batch is spread across nearly all fitted visual regions.
This is a selection diagnostic only; it does not prove better labels or
downstream model improvement.

### Random Selection Variance

Random is a distribution over batches, while integrated Mining is deterministic
and produces exactly one batch. Comparing one Mining batch with one Random batch
therefore cannot separate method effect from draw luck, and the effect is large
here because the added positives are few: at `N=300` and Development prevalence,
one Random draw is expected to contain roughly 6 near-zone, 10 proximity-hold,
and 23 corridor positive labels. A count near 6 has a sampling spread of about
`sqrt(6)`, so two honest Random draws can differ by a factor of two in the
rarest class.

Three Random batches with seeds `101`, `102`, and `103` are therefore frozen
before any Oracle label is revealed. The primary reading is whether the single
Mining result falls outside the range spanned by the three Random batches, not
whether it beats one of them. Seeds were fixed in advance and no batch may be
added, replaced, or dropped after a reveal; doing so would turn a variance
estimate into selection of a favourable comparison.

The `150/600` budget curve keeps the original seed-`101` Random list only. It is
a sensitivity check on budget, not a second variance estimate.

### Pre-declared Selection Expectations

Recorded before the reveal so that the label profile confirms or refutes them
rather than explaining them afterwards:

- Mining should show a higher positive yield than Random, especially for the two
  rare classes, because Random adds almost no positives there;
- Mining should also show a higher `ignore` rate than Random. Mining deliberately
  retrieves windows near known events, and `ignore` is exactly the state of a
  window that overlaps an event without a bounded five-frame match. Mining may
  therefore contribute more positives and fewer total usable labels at the same
  budget. That trade-off is a property of the method, not a defect;
- the uncertainty component contributes least to `pedestrian_vehicle_center_proximity_hold`,
  because a class whose Base AP is near prevalence produces a nearly meaningless
  probability ranking, and boundary distance is computed from that ranking.

### Observed Oracle Reveal

`oracle-reveal-v1` joined the four frozen rankings to the private labels. The
implementation sequentially parsed all 30,749 private window rows, retained and
output only the 2,220 unique selected IDs, and did not rerank any selection. The
experimental isolation comes from freezing every ranking before this private
read, not from physically avoiding unselected rows. No `invalid` label appeared,
consistent with the zero Strem failures in the window build. Each method holds
exactly 300 windows at the primary budget, so no `ignore` was refilled.

Positive labels bought at `N=300`, against the L0 positive counts of 294, 317,
and 450:

| Class | Mining | Random 101/102/103 | Mining vs Random range |
| --- | ---: | ---: | --- |
| pedestrian ego near-zone entry | 12 | 14 / 9 / 10 | inside |
| pedestrian-vehicle proximity hold | 25 | 18 / 17 / 19 | above |
| vehicle relative corridor entry | 13 | 16 / 12 / 16 | inside |

The pre-declared expectations are therefore only partly supported, and this is
recorded as observed rather than reinterpreted:

- higher Mining positive yield holds for the proximity-hold class alone, where
  Mining bought 25 positives against a Random range of 17 to 19. On the other two
  classes Mining falls inside the Random range, so no selection effect is visible
  in yield. Against the seed-101 batch alone Mining would have appeared to lose
  two of three classes; the extra batches show those two as ordinary draws;
- the expected higher Mining `ignore` rate is contradicted on the near-zone
  class, where Mining took 8 against a Random range of 17 to 23. It holds for
  proximity hold, 22 against 8 to 12. Total usable labels are close across all
  four batches, 855 for Mining against 851, 864, and 866;
- Mining is far more concentrated: 21 logs and 130 scenes against 25 logs and
  212 to 214 scenes for every Random batch. Cluster diversity spreads a batch
  inside the FN-similar region; it does not spread it across the Pool.

The largest positive increase any method achieved over L0 is Mining's `+7.9%` on
proximity hold, and the largest gap between methods is about two percentage
points of positive count. A large Held-out Test movement is therefore not
expected from any arm, which is direct evidence for the pre-declared
budget-too-small hypothesis before any retraining has run.

## Controlled Retraining

Every feedback run:

- initializes GRU and classifier head from scratch;
- uses the same frozen DINO features and L0;
- uses the same loss, `pos_weight`, optimizer, batch size, and early stopping;
- changes only the revealed extra windows and declared training seed.

If another factor changes, Gate C fails and no data-selection conclusion is made.

### Frozen Arm List

The Held-out Test is opened once, so the complete set of scored arms must be
fixed before the first Test score — not before the first training run. Every
arm trains seeds `17, 29, 43`.

Improvement happens on Development, which is what Development is for. A round
trains its arms, is read on Development, and may motivate a further arm; the
Test stays sealed throughout. The arms below are the pre-registered set, frozen
before any label was revealed, and they are reported unchanged whatever later
rounds add:

| Group | Arm | Added windows |
| --- | --- | --- |
| control | Base | none |
| primary `N=300` | Random-300-seed101-Oracle | 300 |
| primary `N=300` | Random-300-seed102-Oracle | 300 |
| primary `N=300` | Random-300-seed103-Oracle | 300 |
| primary `N=300` | Mining-300-Oracle | 300 |
| budget curve | Random-150-seed101-Oracle | 150 |
| budget curve | Random-600-seed101-Oracle | 600 |
| budget curve | Mining-150-Oracle | 150 |
| budget curve | Mining-600-Oracle | 600 |
| label source | Mining-300-VLM | 300 |

Budget-curve arms reuse nested prefixes of the same frozen lists; they do not
rerun selection. `Mining-300-VLM` reuses the exact `Mining-300-Oracle` IDs and
changes only the label source. If the VLM stage is dropped for time, the first
nine arms remain a complete experiment.

A later round may add an improved selector. Such an arm is Development-informed
by construction and is reported as such, never as a pre-registered result: the
Random baselines were never tuned, so a Development-informed selector enters the
comparison with an advantage that has to be stated. Both the pre-registered
`Mining-300-Oracle` and any improved variant appear in the final tables.

All predictions for every listed arm freeze before the first Test score is
computed. Scoring the primary comparison first and then deciding to score the
budget curve would open the Test twice, and the second reading would no longer
be independent of the first.

### Controlled VLM-label retraining

Superseded in form by *Protocol v0.11* below, which retargets this comparison
from the v0.8 `Mining-300` batch to the v0.10 `disagreement_v010` 900 batch. The
rules in this section still hold.

Mining-VLM reuses the exact frozen Mining-Oracle window IDs. The only intended
change is the extra batch's label source:

- `positive` and `negative` become training targets;
- `uncertain` is masked for that class;
- the same L0, DINO features, architecture, optimizer, class weights, early
  stopping, and training seeds are reused.

This comparison estimates label-noise impact. It must not be interpreted as a
different data-selection strategy. Prompt/schema/model identity freezes before
the formal batch, and Frozen Test never guides prompt editing.

### Observed Round-One Retraining

Both declared protocols ran on the four frozen batches. Development Macro-AP,
mean and spread over training seeds `17, 29, 43`:

| Arm | P1 from scratch | P2 fine-tuned |
| --- | ---: | ---: |
| Base | `0.2408 ± 0.0088` | — |
| Base-continued, no added data | — | `0.2422 ± 0.0118` |
| Mining-300 | `0.2528 ± 0.0028` | `0.2428 ± 0.0082` |
| Random-300 seed 101 | `0.2542 ± 0.0178` | `0.2382 ± 0.0074` |
| Random-300 seed 102 | `0.2569 ± 0.0208` | `0.2425 ± 0.0121` |
| Random-300 seed 103 | `0.2561 ± 0.0188` | `0.2424 ± 0.0093` |

Three results, all reported as observed:

1. **Adding 300 windows helps; choosing them does not.** Under P1 every arm
   beats Base, `0.2408` against `0.2528` to `0.2569`, while the four arms cannot
   be told apart. The budget has value; the selection strategy shows none.
2. **Mining does not beat Random.** Under P1 its Macro-AP is below all three
   Random batches. Under P2 it exceeds them by `0.0003`, far inside its own
   `± 0.0082` seed spread. Neither direction is an effect.
3. **Fine-tuning absorbed the added data.** Every P2 arm sits between `0.2382`
   and `0.2428`, including `Base-continued`, which added nothing. Resuming a
   converged checkpoint at a tenth of the rate barely moved the model, so 300
   windows changed little.

Per-class Development AP shows why a single Random batch would have misled. On
`pedestrian_ego_near_zone_entry` under P1 the three Random batches scored
`0.1346`, `0.1831`, and `0.1859` while Mining scored `0.1704`. Against seed 101
alone Mining would have looked clearly ahead; against the three-batch range it is
ordinary. Batch-to-batch selection variance exceeds the Mining-versus-Random gap.

This matches the reveal-stage power estimate exactly: the two methods differed by
about seven positive windows, and the observed seed spread is `± 0.008` to
`± 0.021`, while the arms differ by less than `± 0.004`.

A Development-informed selector was designed from this diagnosis and tested
against the same Random batches; see [Findings](FINDINGS.md) section 8 for the
full result. It bought 2 to 5 times more positives than the best Random batch
but produced no clean Macro-AP gain under either protocol, which is recorded as
evidence against the budget being the sole limiting factor.

### Measurement revision

The three-seed numbers above were later shown to carry unreliable spreads:
twelve seeds under the declared `narrow-fast` protocol put the true seed
variance at two to three times the three-seed estimates, withdraw the
"adding data helps" reading (Random minus Base becomes `+0.0036 ± 0.0060`),
and reveal a corridor-class advantage for the pre-registered Mining selector
that the old instrument could not resolve. [Findings](FINDINGS.md) sections
10-12 hold the full revision; the tables above stand as what was observed
under the frozen protocol at the time.

## Protocol v0.10 — the small-seed decoupled cycle

Frozen 2026-08-30, before any v0.10 label is revealed. Motivated directly by
Findings 10-13: the v0.8 operating point capped every selector's effect at
about `+0.004` (full L0, N=300, flat region), the mining queries came from the
partition that judged them, and the one probe of positive-yield selection
showed confident positives teach a saturated model nothing. Each lesson maps
to one design change below. `v0.8` results stay sealed and reported as they
are; `frozen_test` is spent and is never reopened.

### Partition roles

| Role | Data | Rule |
| --- | --- | --- |
| Seed training | `L0-small` = the 3-log lc25 subset (1,931 windows; 92/47/58 positives) | the learning-curve point where corridor is steep and positives are scarce |
| Early stopping and sanity | Development (7 logs) | carries no headline claim in v0.10 |
| Selection pool | `Pool2` = U minus the Test2 logs | selectors may read only its public view and Base-small outputs |
| **Held-out** | `Test2` = 8 U logs chosen by `sha256("test2-v010\0" + log_token)` order | labels never previously revealed there are the eval set; any window already revealed in v0.8 is excluded from both Pool2 and Test2 scoring. Opened once, after all v0.10 predictions freeze |
| Sealed | `frozen_test`, the 10 unused L0 logs | untouched |

If the 8 hash-chosen Test2 logs hold fewer than 55 corridor events, extend by
one log in hash order until they do (the same guard rule the learning curve
used). The count check reads event-support metadata only, as Gate A did.

### Arms

Three selectors, all computed from `Base-small` (= the existing `n12-lc25`
run, twelve seeds) and the public Pool2 view, none using any bank or query
derived from an evaluation partition:

- **disagreement**: per class, rank by the standard deviation of the twelve
  Base-small seed probabilities, descending — epistemic uncertainty, untested
  so far and structurally incapable of the v0.8 query/eval coupling;
- **prob-ranked**: per class, rank by the twelve-seed mean probability,
  descending. At the small-seed point positives are the scarce resource, and
  the measured 53-84% of its top picks that are *not* positives are hard
  negatives by construction, so this arm operationalises positive mining and
  hard-negative mining jointly. An explicit FP-query bank (the gap the FN-only
  bank left) is deferred to a later round to cap the arm count;
- **random x3**: seeds `201, 202, 203`, the same distribution-not-a-draw rule
  as v0.8.

All rankings use the common temporal-separation and round-robin class-merge
rules, are frozen before any reveal, and use nested budgets `300 ⊂ 600 ⊂ 900`.

**Pre-reveal amendment (2026-08-30):** the budgets were declared as
`300/600/1200`, but the Test2 guard legitimately extended the held-out carve to
ten logs, leaving Pool2 at 5,311 windows across 192 scenes with an exact
temporal-rule capacity of 1,363 slots. Random placement at 1,200 is infeasible
(all three seeds fail; all succeed at 1,050), so the top budget becomes 900
with margin. This is a mechanical feasibility amendment made before any v0.10
label was revealed; the primary criterion moves with it to `N=900` unchanged in
form.

### Training and measurement

`narrow-fast-12seed` from scratch for every arm; `pos_weight` from L0-small
only; early stopping on Development; the bit-verified two-stage
predict-then-score path with a window-list extension for Test2.

### Pre-registered criterion

Primary: on Test2, the corridor-class seed-paired contrast (selector minus the
mean of the three randoms) at `N=900` (see the amendment note above), one per
selector; success is `≥ 2.5σ`, set above 2σ because two selectors are read. Secondary: the same contrasts at
300/600 (dose pattern) and on near-zone. Proximity hold and Macro are reported
only. Expected effect from the learning-curve slope (`≈ +0.045` AP per +100
corridor positives near the 25% point) and measured retrieval yields is
`+0.02` to `+0.05` against an anticipated paired stderr near `0.010-0.012`;
the power is moderate, which is stated here rather than discovered later.

Nothing in this section may change after the first v0.10 reveal. A null result
under this design closes the question for this dataset and representation: it
would be the third independent operating point at which selection shows no
transferable value.

## Protocol v0.11 — automatic labeling with a remote VLM

Frozen 2026-08-30, before any request is sent. This stage does not spend Oracle
budget, does not reopen a held-out set, and does not change any v0.10 result. It
answers the labeling half of the loop, which the project has so far bought from a
private metric Oracle:

> On the exact windows the v0.10 selector already chose and the Oracle already
> revealed, how accurately, at what uncertainty rate, and at what unit cost can a
> remote VLM reproduce the metric scenario labels from five monocular CAM_FRONT
> frames alone — and does training on those labels instead of the Oracle's keep
> the downstream gain?

The second half of that question is the same epistemic move that Finding 6 forced
on selection: label F1 is a proxy, and a proxy is only trusted after the
downstream measurement agrees with it.

### Batch amendment

`DATA_SPEC.md`, `STAGE_PLAN.md`, and the *Controlled VLM-label retraining*
section above name the v0.8 `Mining-300` batch, declared before v0.10 existed.
The labeled batch becomes the v0.10 `disagreement_v010` 900-row prefix — the
batch that actually closed the loop, whose Oracle labels are already revealed and
whose nested `300 ⊂ 600 ⊂ 900` structure is preserved. This is a mechanical
retarget made before the first request; no metric, gate, or red line moves with
it.

### Frozen identity

Everything below is fixed before the formal batch and recorded in
`vlm_run_manifest.json` with the prompt and schema hashes:

| Field | Value |
| --- | --- |
| provider | Anthropic API |
| model | `claude-sonnet-5` (the response's own `model` string is recorded too) |
| thinking | adaptive, default effort |
| output | structured output constrained by the frozen JSON schema |
| images | the five CAM_FRONT keyframes of the window, original 1600x900 JPEG, base64, in temporal order |
| text | the three frozen scenario descriptions and their frozen thresholds, plus the verdict instructions |
| temperature | not set (removed on this model family) |
| transport | Message Batches, 100 windows per submitted batch, `custom_id = window_id` |

The model is the volume-realistic tier, not the strongest tier, because the
industrial question is what auto-labeling costs at scale. Claude Opus 5 is held
in reserve as a *diagnostic* under the gate below, never as the default.

### Request boundary

One request carries exactly one window. The builder reads only the frozen ranking
file, the public `pool_windows.jsonl` fields in `PUBLIC_U_FIELDS`, and the media
files those `frame_refs` name. It never reads `revealed_labels.jsonl`,
`windows.jsonl`, Strem output, bindings, instance tokens, 3D boxes, distances,
visibility, or Base probabilities. A focused test asserts the built request body
contains no Oracle-derived field, in the same spirit as the existing
no-label-leak test. Credentials live in `ANTHROPIC_API_KEY`; every response is
written under the private data root and only aggregates reach `results/`.

### Output schema

Per window, one verdict per scenario, keyed by `scenario_id` and never by
position:

```text
{window_id, verdicts: [{scenario_id, verdict: positive|negative|uncertain,
                        evidence_frames: [0..4], confidence: 0..1,
                        limitation: str}]}
```

A response whose `scenario_id` set differs from the frozen scenario order is
rejected, not reordered — the Oracle reveal already rejected one class-order
mismatch and the same rule applies here.

### Smoke, then freeze

Up to three prompt iterations on about 30 **Development** windows (roughly four
positives and four negatives per class plus six `ignore` windows), synchronous
calls, at most about `$3`. Schema validity, uncertain rate, rough agreement, and
per-window cost are read; the prompt is edited only here. TaskDesign and
Development are the only partitions a prompt may ever see. Pool2, Test2, and
`frozen_test` never inform prompt editing. After the last smoke round the prompt,
schema, model, and preprocessing hash into the manifest and do not change.

### Execution and cost

Submission is chunked at 100 windows per batch because the five base64 images run
near 1 MB per request and a batch is size-limited; the nine chunk ids and their
states live in the manifest. Raw responses land on disk as each chunk completes,
so a rerun re-requests only missing or failed `window_id`s and never re-pays for a
completed one. Before any submission the builder verifies every referenced image
exists and is readable, and renders three windows as contact sheets for a
one-time human check that frame order and identity are correct. The script
refuses to submit when the estimated spend exceeds `$35` without an explicit
confirmation flag.

Estimated at batch pricing: about `$0.014` per window, `~$13` for the formal 900,
`~$3` for smoke, `~$11` if the Opus diagnostic below is triggered.

### Gates

Pre-declared so that no branch spends money without producing a reportable
result:

1. Formal Sonnet run on the 900 batch.
2. If Macro-F1 over the three classes is `< 0.35`, the Opus diagnostic runs on
   the nested 300 prefix only, and answers one question: is the ceiling the task
   or the model tier. Whether to then buy a full Opus batch is a separate
   decision, not pre-authorized here.
3. The downstream arm runs only if the VLM produces at least 15 `positive`
   verdicts that the Oracle also calls positive, summed over classes; below that
   the arm's outcome is arithmetically predetermined and the money is better not
   spent on compute time.

### Label metrics

Computed against the already-revealed Oracle labels for the same IDs:

- per class Precision, Recall, F1 over windows the Oracle calls `positive` or
  `negative`, with VLM `uncertain` excluded from the numerator and denominator
  but reported as its own rate;
- windows the Oracle calls `ignore` or `invalid` are profiled separately, never
  scored as errors — they are the definitionally ambiguous region;
- schema-invalid rate, and the rate of `evidence_frames` outside `0..4` or empty
  under a `positive` verdict, as the hallucination probe;
- latency distribution, total spend, spend per usable label.

Each pre-declared VLM failure mode in `BAD_CASE_PROCESS.md` maps to one of these
fields, so the diagnosis is chosen from a list written before the data existed.

### Controlled downstream arm

`v010-disagreement-900-vlm` reuses the exact 900 window IDs, the exact `L0-small`
seed set, `narrow-fast-12seed` from scratch, and every hyperparameter of
`v010-disagreement-900`. The only difference is that the added batch's labels come
from the VLM file instead of the reveal file. Only the VLM's own `uncertain`
verdicts are masked; the Oracle's `ignore` states must not be consulted to choose
which VLM labels to keep, because a real auto-labeling pipeline has no such
oracle. The arm loader rejects a label file whose artifact identity does not match
the arm's declared label source.

Both held-out sets are spent, so this contrast is measured on Development with
twelve seed-paired differences and is reported as Development-level evidence.
Three readings are pre-declared:

| Result | Reading |
| --- | --- |
| VLM arm ≈ Oracle arm | the automatic labeler substitutes for the expensive label source at this operating point |
| Base < VLM arm < Oracle arm | lossy but net-positive: noisy labels still beat no data |
| VLM arm ≤ Base | label noise cancels the data, and auto-labeling has a hard accuracy floor here |

### Red lines

No SFT and no weight update of any VLM: this stage is API inference, and it is
never described as VLM training experience. No held-out set is reopened for it.
The VLM never overrides a Strem label, and a disagreement is reported as a
disagreement rather than resolved in the VLM's favour.

## Metrics

Primary:

- per-class scikit-learn Average Precision;
- per-class `AP - prevalence` margin, using the same definition as Gate B;
- Macro-AP across all three Gate-A classes.

Call this metric AP, not trapezoidal AUPRC.

Per-class AP is the primary reading and Macro-AP is a summary. The three classes
sit at very different difficulty, so an unweighted mean is dominated in absolute
terms by `vehicle_relative_corridor_entry`: a `0.05` corridor gain moves Macro-AP
by about `0.017`, while the same relative gain on the weakest class moves it by
about `0.001`. Reporting the margin alongside AP keeps a weak class readable
instead of invisible.

`pedestrian_vehicle_center_proximity_hold` has the least Held-out Test support of
the three classes, at 37 event groups across 9 logs. Its Test AP is therefore the
noisiest of the three, and a change in it is not treated as a strong result. This
is recorded before scoring, not after.

Useful diagnostics:

- Precision, Recall, F1, FNR, and event recall;
- selected positive and usable-label yield;
- log coverage, duplicate rate, and ignore cost;
- runtime or resource use when it helps explain feasibility.

VLM-label diagnostics:

- schema-valid rate and per-class Precision, Recall, and F1 versus Oracle;
- `uncertain` rate and unsupported-assertion rate;
- label agreement by visibility, scale, weather, or other available subgroup;
- latency and remote-call cost;
- Mining-VLM versus Mining-Oracle downstream Macro-AP gap.

Report every training seed plus mean and spread. This small pilot does not
justify broad statistical-generalization claims. Two separate sources of
variation are reported separately: the three training seeds estimate
model-training variation within one batch, and the three Random batches estimate
selection variation across draws. Mining is deterministic, so it has training
spread but no batch spread; its single result is read against the Random batch
range. Three batches bound that range coarsely and do not support a significance
test.

## Core Result Tables

1. Data and labels: cleaning/filter-reason counts, scenario/event/window support,
   split/log coverage, and positive/negative/ignore/invalid distributions.
2. Baselines: LastFrame-LR, Mean5-LR, GRU, and reversed order, with per-class AP
   and Macro-AP.
3. Primary closed loop: Base, three Random-Oracle batches, and Mining-Oracle with added-data
   count, per-class AP, Macro-AP, Recall, F1, false-negative count, and selected
   usable-label yield.
4. Automatic labels: VLM-versus-Oracle label metrics, uncertainty/cost, and the
   Mining-VLM versus Mining-Oracle downstream comparison.

## Focused Tests

Add tests as each component is built, only for experiment-critical behavior:

- Strem match/no-match/error/timeout and persistent identity;
- event containment/overlap/no-overlap labels;
- no cross-log split or cross-scene window;
- no Oracle fields in public U and reveal only after selection;
- VLM requests contain only frozen selected IDs/images and no Oracle fields;
- VLM schema parsing and `uncertain` loss masking;
- DINO shape/finite values;
- GRU shape, masked loss, gradient, and checkpoint reload;
- equal Random-Oracle/Mining-Oracle budgets and temporal separation;
- Random variance batches are valid, mutually distinct, and cannot reuse the
  frozen Random seed;
- one hand-checkable AP example.

There is no need for a generic fault-injection framework. The enforceable rule
is simple: do not run Frozen Test scoring until every prediction file is final.

## Manual Review

Keep the remaining manual review bounded and job-focused: a small stratified
TaskDesign gallery plus at least five final explanatory cases, including one VLM
label error or uncertainty case. Strem labels simulate an Oracle; manual review
audits their quality and is not described as full human annotation.

## Release Check

Before scoring, confirm task spec, split, DINO revision, Base thresholds, FN
bank, selector policies, selected IDs, the three Random seeds, the frozen arm
list, VLM prompt/schema/model identity, and all predictions are frozen. Confirm
U selection saw no hidden labels and official val entered neither training nor
tuning. Confirm that every arm in the frozen list has a prediction file before
the first Test score, because the Test is opened once for all arms together.
