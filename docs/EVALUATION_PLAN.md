# Evaluation Plan

Specification: Frozen for `v0.8-three-class-loop`

Status: every stage through protocol v0.11 and diagnostics D1–D5 has run, and
both held-out sets are spent — `frozen_test` once for v0.8 and Test2 once for
v0.10. Protocol v0.13, its downstream arm, D6, and the label-cascade analysis have
run; their readings are Findings sections 16–17. Results live in
[Findings](FINDINGS.md); this document keeps each protocol as it was frozen.

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

`DATA_SPEC.md` and the *Controlled VLM-label retraining*
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
| provider | OpenAI API, Responses endpoint |
| model | `gpt-5.6-terra` (the reply's own `model` string is recorded too) |
| output | structured output, `json_schema` with `strict: true` |
| images | the five CAM_FRONT keyframes of the window, original 1600x900 JPEG, base64 data URLs, in temporal order |
| text | the three frozen scenario descriptions and their frozen thresholds, plus the verdict instructions |
| sampling | provider defaults, unset |
| transport | Batch API on `/v1/responses`, 100 windows per submitted file, `custom_id = window_id` |

**Provider amendment (2026-08-30, before any request):** the identity was first
frozen as Anthropic `claude-sonnet-5`. The available API budget turned out to
sit with OpenAI, so the provider moved and the model became the tier-matched
`gpt-5.6-terra` ($2/$12 per Mtok against $2/$10, and about `$13.5` for the 900
batch against `$12.75`). Nothing else in this protocol moves: the tier logic,
the request boundary, the schema, the gates, the metrics, and the downstream arm
are unchanged, and no request had been sent when this was written.

The model is the volume-realistic tier, not the strongest tier, because the
industrial question is what auto-labeling costs at scale. `gpt-5.6-sol` is held
in reserve as a *diagnostic* under the gate below, never as the default.

### Request boundary

One request carries exactly one window. The builder reads only the frozen ranking
file, the public `pool_windows.jsonl` fields in `PUBLIC_U_FIELDS`, and the media
files those `frame_refs` name. It never reads `revealed_labels.jsonl`,
`windows.jsonl`, Strem output, bindings, instance tokens, 3D boxes, distances,
visibility, or Base probabilities. A focused test asserts the built request body
contains no Oracle-derived field, in the same spirit as the existing
no-label-leak test. Credentials live in `OPENAI_API_KEY`; every response is
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

### Observed smoke, and the freeze it triggered

Run 2026-08-30 on 30 Development windows, synchronous, `$0.81` measured, 30 of
30 replies returned, ten seconds per window.

| Reading | Value |
| --- | --- |
| schema-valid rate | `1.00` |
| `evidence_frames` out of range or missing under a positive | `0` of 90 |
| Macro-F1 against the Oracle | `0.846` (per class `1.00 / 0.77 / 0.77`) |
| VLM `uncertain` on Oracle-decided windows | `14% / 17% / 4%` |
| Oracle `ignore` windows | near-zone `2/2` answered `uncertain`; corridor `3+/2-`; hold `1+` |
| measured input tokens per window | `9,365` (estimate assumed `9,250`) |

The prompt was frozen with **zero iterations**: the smoke exposed no failure to
fix, and editing a prompt that is already schema-clean and calibrated would be
tuning toward a nicer number rather than repairing a defect. Frozen prompt hash
`d418e9bb5942e9751d04c642c3a92a9cd23c7175bb09a4e7981b1d36061b90fe`.

Two limits on how far this reading travels, stated before the formal batch and
not after it. Thirty windows carry four to seven positives per class, so the
near-zone `1.00` rests on two true positives and is not a measurement; and
Development is not the mined batch — `disagreement_v010` selected exactly the
windows the Base ensemble disagreed on, which is an adversarial slice by
construction. The formal Macro-F1 is expected to land below `0.846`, and this
paragraph is the reason it will not be reported as a regression.

### Execution and cost

Submission is chunked at 100 windows per batch because the five base64 images run
near 1 MB per request against a 200 MB batch-file limit; the nine chunk ids and
their states live in the manifest. The uploaded input file and the returned
output file are deleted from the provider once results are in hand. Raw responses land on disk as each chunk completes,
so a rerun re-requests only missing or failed `window_id`s and never re-pays for a
completed one. Before any submission the builder verifies every referenced image
exists and is readable, and renders three windows as contact sheets for a
one-time human check that frame order and identity are correct. The script
refuses to submit when the estimated spend exceeds `$35` without an explicit
confirmation flag.

Estimated at batch pricing: about `$0.015` per window, `~$13.5` for the formal
900, `~$3` for smoke, `~$11` if the `gpt-5.6-sol` diagnostic below is triggered.
The manifest records measured token usage and prices it, so the reported cost is
metered rather than estimated.

### Observed execution (2026-08-30)

The Batch API is unusable on this organization: a batch rejects its own input
file as inaccessible, identically for a 207-byte file and a 105 MB one, while the
same key uploads that file, reads it back, and makes direct calls normally. The
cause was not diagnosed further because the remaining saving was `$11` and the
scarcer resource is time; the formal run therefore used direct calls with six
overlapping requests, at `$23.69` measured against the `$13.7` a batch would have
cost. This is the ordinary compliance-versus-cost tradeoff of a real labeling
pipeline and is recorded rather than hidden.

Formal run: 900 of 900 replies, zero failed calls, 23 minutes, 8,428,500 input
and 569,146 output tokens, `$0.0263` per window.

### Gates

Pre-declared so that no branch spends money without producing a reportable
result:

1. Formal Sonnet run on the 900 batch.
2. If Macro-F1 over the three classes is `< 0.35`, the `gpt-5.6-sol` diagnostic
   runs on the nested 300 prefix only, and answers one question: is the ceiling
   the task or the model tier. Whether to then buy a full flagship batch is a
   separate decision, not pre-authorized here.
3. The downstream arm runs only if the VLM produces at least 15 `positive`
   verdicts that the Oracle also calls positive, summed over classes; below that
   the arm's outcome is arithmetically predetermined and the money is better not
   spent on compute time.

### Label metrics

Computed against the already-revealed Oracle labels for the same IDs:

- per class Precision, Recall, F1 over windows the Oracle calls `positive` or
  `negative`, with VLM `uncertain` excluded from the numerator and denominator
  but reported as its own rate — and, since the decided-only recall reads like
  coverage but is not, recall including abstentions is reported beside it;
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

## Post-hoc diagnostics D1–D4 (v0.12)

Declared 2026-09-01, before any of the three runs. All three are
**Development-level, exploratory diagnostics**: they spend no Oracle budget
(only already-revealed labels), touch no held-out set, and none of them can
upgrade or downgrade the v0.10 held-out conclusion. Each answers one scoping
question a reviewer of the v0.10 result would ask.

### D1 — ensemble-size ablation (selector cost)

Question: how many of the twelve Base-small models does the disagreement
selector actually need? Method: recompute per-seed Pool2 probabilities from the
twelve frozen checkpoints; for each subset size `n ∈ {2, 4, 6, 8}` draw twelve
seeded random subsets, rebuild the full v0.10 ranking pipeline from the subset
standard deviation, and report the overlap of the top-300 and top-900 selections
with the frozen `disagreement_v010` ranking (mean ± sd over subsets). Sanity
gate: the recomputed `n = 12` ranking must reproduce the frozen ranking's
selection sets; if it does not, the recomputation — not the frozen artifact — is
wrong and the diagnostic stops. Reading: high overlap at small `n` means the
selector's inference cost can shrink proportionally.

### D2 — label-noise dose-response (how much noise the loop absorbs)

Question: the v0.11 result says F1-0.49 labels retrain as well as exact labels;
what does the whole curve look like? Method: take the revealed
`disagreement_v010` top-900 Oracle labels and flip each **decided** label
(positive ↔ negative; masked entries untouched) independently with probability
`p ∈ {0.10, 0.20, 0.30}`, deterministically seeded per `(window_id, class)`.
Train each level with the identical lc25 + `narrow-fast-12seed` recipe as
`v010-disagreement-900`. Reading: the corridor and Macro AP versus `p` curve on
Development, with the existing Oracle arm as `p = 0` and the VLM arm placed on
the same axes for context. A monotone, initially-flat curve explains the v0.11
result mechanically; a cliff locates the loop's noise tolerance.

### D3 — cross-operating-point check (does the effect survive more data?)

Question: v0.10 measured the selector at the lc25 point chosen for maximum
sensitivity; how much survives at lc50? Method: retrain
`v010-disagreement-900` and the three `v010-random20X-900` arms with
`lc_windows_50.jsonl` as the private windows file — same revealed labels, same
recipe, nothing else moved. Reading: the Development seed-paired contrast
(disagreement − random mean) at N=900, expected smaller than the lc25-point
effect; the interesting quantity is the decay, not a pass/fail.

### D4 — the clean-query similarity ablation (declared 2026-09-01, before any step ran)

The one ablation v0.10 deliberately did not buy: the v0.8 similarity strategy
with its coupling removed. Exploratory, Development-level, pre-registered here
before the query bank exists.

**Query source.** The ten L0 logs outside `L0-small`: Base-small never trained
on them, they are disjoint from Pool2 and Test2 (not U logs) and from
Development — a labeled query source isolated from both the selection pool and
the evaluation partition in exactly the way the v0.8 bank was not.

**Faithful reconstruction, declared deviations.** The bank is built as in v0.8
— seed-17 Base-small predictions, the run's frozen max-F1 thresholds, the same
FN definition and event-deduplication — but from the ten-log partition instead
of Development. Selection reuses the frozen v0.8 integrated recipe
(similarity + boundary uncertainty + KMeans diversity, `M=2000`, `K=30`,
temporal separation, round-robin) over a Pool2-restricted pool whose
probabilities come from the Base-small ensemble mean and whose thresholds come
from the Base-small report; embeddings are the frozen 768-d window vectors.
Budgets `300 ⊂ 600 ⊂ 900`.

**Pre-reveal amendment (2026-09-01):** at 900 the proximity-hold queue exhausts
its candidates under the frozen shortlist and temporal-separation rules on the
5,311-window pool — the same class of capacity infeasibility that moved the
v0.10 top budget, resolved the same way: the top budget becomes `600`
(`300 ⊂ 600`), the arm becomes `d4-similarity-clean-600`, and every comparison
target exists at 600 (the random-600 and disagreement-600 arms are already
trained). Recorded before any D4 label was revealed.

**Spend.** Revealing the 600-row prefix consumes up to 600 of Pool2's 2,212
unrevealed windows, approved as this diagnostic's cost; the remainder is
reserved for a future loop round.

**Training and reading.** Arm `d4-similarity-clean-600` (see the amendment):
identical lc25 + `narrow-fast-12seed` recipe, Development seed-paired contrast
against the three existing random-600 arms' mean, read beside the disagreement arm's same-
instrument reading (`+0.017`, 1.4σ). Development under-reads between-arm
contrasts (the D3 calibration), so the readout distinguishes "clearly weak"
from "comparable" and nothing finer. Either way it sharpens the v0.8
attribution: a weak clean-query arm supports "similarity was intrinsically
weak"; a comparable one supports "the coupling was the main killer" and
downgrades the selector-switch narrative. No held-out claim in either case.

These diagnostics report what they find. None of them is a registered
confirmatory claim, and their results are labeled Development-level wherever
they are quoted.

### D5 — student-capacity check on the v0.11 result (declared 2026-09-29, before any D5 run)

Question: the v0.11 VLM labels (Macro-F1 `0.494`) retrained to 86% of the
Oracle arm's corridor gain with the `narrow-fast-12seed` student (hidden 48).
Is that ratio a property of the labels, or of a low-capacity student that
cannot fit their errors? Method: repeat the three runs of the v0.11 comparison
with only the student's width changed — `wide-fast-12seed`, which is
`narrow-fast-12seed` with hidden 128 (the original v1 width), same twelve seeds,
rate, dropout, patience and batch. Runs: Base (`lc_windows_25.jsonl`, no
feedback), `v010-disagreement-900` on the revealed Oracle labels, and
`v010-disagreement-900-vlm` on the v0.11 VLM training labels. Same private
windows, public Pool2 file, label directories and feature cache as the hidden-48
runs; no label is revealed and no held-out set is touched.

**Path check before reading.** The same command lines with `narrow-fast-12seed`
must reproduce the committed hidden-48 VLM arm's per-seed Development corridor
AP; if they do not, the reconstruction is wrong and D5 stops.

**Reading, fixed now.** Development seed-paired contrasts on corridor and Macro
AP: VLM − Base, Oracle − Base, VLM − Oracle, and the gain ratio
`(VLM − Base) / (Oracle − Base)`. If VLM − Oracle stays within 2σ of zero, the
86% reading is not an artifact of the small student. If VLM − Oracle falls below
`−2σ`, a wider student is more sensitive to these label errors, and the v0.11
statement is scoped to the hidden-48 student. Development under-reads
between-arm contrasts (D3), so a within-2σ reading means "comparable at this
instrument's resolution", not "equal". No held-out claim either way.

**Report metadata fix, recorded with this declaration.** Until this change, the
`config` block of every `gru_report.json` printed the module defaults for
`hidden_size`, `dropout_after_final_hidden`, `weight_decay`, `batch_size`,
`max_epochs` and `patience` instead of the run's training config. The
`training_config` block always recorded the protocol actually used (checkpoint
weight shapes confirm hidden 48 for the `narrow-fast-12seed` runs); committed
reports keep their original bytes, and `training_config` is the authoritative
block for any run before this change.

## Protocol v0.13 — an open-weight VLM as the labeler, zero-shot and LoRA-tuned

Declared 2026-10-11, before any open-weight model has produced an output.
Executed 2026-10-11; the result is Findings section 17. This is a new stage with its own claim boundary; the v0.11
red line ("no SFT and no weight update of any VLM") governed v0.11 and still
does — no v0.11 number is rerun, edited, or reinterpreted here.

> v0.11 measured a remote VLM reproducing the Strem labels from five monocular
> frames at Macro-F1 `0.494`, with precision, not recall, as the failure
> (`0.40 / 0.30` precision on the two pedestrian classes against `0.80 / 0.86`
> recall). Can an open-weight VLM, adapted with LoRA on Oracle labels from
> training logs only, label the same 900 windows more accurately than the
> frozen remote VLM — and how much of any gain is the adaptation rather than
> the model?

This is the labeling half of the loop again: the Strem Oracle needs 3D
annotations, the Pool has only pixels, and the question is how close a model
that sees only pixels can get. Label quality only; see *Not in this protocol*.

### Arms

All three are scored on the same 900 window IDs, the `disagreement_v010`
900-row prefix, against the already-revealed Oracle labels in
`oracle-reveal-v010`.

| Arm | Labeler | State |
| --- | --- | --- |
| A | `gpt-5.6-terra`, zero-shot (v0.11) | existing `vlm-labels-v1`, not rerun |
| B | `Qwen3-VL-8B-Instruct`, zero-shot | new |
| C | `Qwen3-VL-8B-Instruct` + LoRA adapter trained on L0 | new |

The 900 windows are the adversarial slice the Base ensemble disagreed on.
Absolute F1 on them does not describe random windows; the arms are compared to
each other on identical inputs, which is what the slice supports.

### Frozen identity

| Field | Value |
| --- | --- |
| model | `Qwen/Qwen3-VL-8B-Instruct` at revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`; weight-file hashes recorded in the manifest |
| prompt | the v0.11 `SYSTEM_PROMPT` and scenario text unchanged (hash `d418e9bb…`); the user turn is the v0.11 text after the five images |
| output | the v0.11 `OUTPUT_SCHEMA`, enforced by vLLM structured output; replies parsed by the unchanged `parse_verdicts` |
| images | the five CAM_FRONT keyframes in temporal order, resized from 1600x900 to 640x360 before the processor; the processor's own rounding is recorded, not chosen |
| decoding | greedy (temperature 0), at most 2,000 output tokens |
| serving | vLLM, version recorded in the manifest |

B and C differ only in the adapter. The same images, prompt, schema, and
decoding serve both.

### Training set (arm C)

- **Partition:** L0 only — 8,390 windows from 13 logs, disjoint by whole-log
  split from Pool2 and therefore from all 900 scored windows (verified at
  declaration: zero shared logs). TaskDesign, Development, Pool2, Test2, and
  `frozen_test` contribute nothing.
- **Rows:** all 1,978 L0 windows with at least one `positive` or `ignore` state,
  plus 1,978 all-`negative` windows drawn with seed `17`: **3,956** rows.
- **Targets:** one reply in the v0.11 schema per window, in scenario-table
  order. `positive -> positive` with `evidence_frames [0,1,2,3,4]`;
  `negative -> negative` with `[]`; `ignore -> uncertain` with `[]` (an event
  overlaps the window but no bounded match exists — the definitional ambiguity
  `uncertain` was written for). `confidence` is `1.0` for decided verdicts and
  `0.5` for `uncertain`; `limitation` is `""`. These filler fields are
  constants of the target format, not model judgements, and are never scored.
  L0 has zero `invalid` states.

### Training recipe (arm C), fixed

LLaMA-Factory (version recorded), LoRA on every linear layer of the language
model, rank 16, alpha 32, dropout 0.05; vision encoder and projector frozen;
AdamW, learning rate `1e-4`, cosine schedule with 3% warmup, 2 epochs, effective
batch 16, bf16, seed `17`, maximum sequence length 4,096. The final checkpoint is
the adapter — no checkpoint selection, no hyperparameter search, one training
run.

### Gates, in order

1. **Pipeline smoke (B).** Ten Development windows, public fields only, no
   label read. Passes when every reply parses. A failure here may be fixed
   freely: no scored output exists yet.
2. **Fallback decisions close here.** Only three changes are allowed, each
   recorded with its cause, and each must be made before any 900-window output
   exists: training out of memory -> smaller per-device batch with gradient
   accumulation that keeps the effective batch at 16; still out of memory ->
   448x252 images for B and C together; the toolchain cannot load Qwen3-VL ->
   `Qwen2.5-VL-7B-Instruct` at a revision pinned at that moment.
3. **Formal B** on the 900 windows. Written and hashed; **not scored**.
4. **Train C** with the recipe above.
5. **Sanity (C).** 300 Development windows drawn with seed `17`, public fields
   only, **no label read**. Passes when the schema-valid rate is at least 95%
   and every class receives at least one `positive` and one `negative` verdict
   (a collapse check). On failure, one rerun is allowed for a pipeline defect
   (template, parsing, image order) found by inspection; the recipe never moves.
   A second failure is reported as the result.
6. **Formal C** on the 900 windows. Written and hashed.
7. **Scoring opens** only after both B and C outputs are hashed into the
   manifest, so B's score cannot inform C.

### Metrics and the pre-registered criterion

Per class and Macro, the v0.11 label metrics from `evaluate_vlm_labels`
unchanged: precision, recall, and F1 over Oracle-decided windows with
`uncertain` excluded, abstention rate, recall including abstentions, Oracle
`ignore` profiled separately, schema-valid rate, and the evidence probe.

**Primary:** `Macro-F1(C) − Macro-F1(A)`, with a paired percentile bootstrap
over the 900 windows (10,000 resamples, seed `17`; each resample recomputes the
three per-class F1s and their mean).

**C beats A** when the 95% interval lies above zero **and** C's mean coverage
(Oracle-decided windows that receive a decided verdict, averaged over classes)
is no lower than A's. F1 over decided windows rewards abstention, so a win
bought by abstaining more is not a win.

| Reading | Interpretation |
| --- | --- |
| C beats A | a pixel-only labeler adapted on training-log labels surpasses the frozen remote VLM on the hard slice |
| interval spans zero | no measurable difference at this sample size (34 to 47 Oracle positives per class) |
| interval below zero | adaptation on about 2,000 eventful windows does not reach the remote model |

Context contrasts, reported without a decision rule: `B − A` (open versus
remote at zero-shot) and `C − B` (what the adapter adds). Per-class precision
for C is read against A's precision failure.

**Cost:** GPU hours times the measured hourly price, reported separately for
training and for 900-window inference, beside v0.11's `$0.0263` per window. The
rental ceiling is CNY 150 (expected 40–50 at about CNY 6 per A800-hour);
reaching it stops the protocol, and whatever gate it stopped at is reported.

### Data boundary

Specified in `DATA_SPEC.md`, *Rented GPU Boundary (v0.13)*. In short: the rented
host receives L0 training targets and media plus public fields of the scored and
sanity windows; the Oracle labels of the 900 and of Development never leave the
local machine, and scoring runs locally.

### Observed execution (2026-10-11)

One AutoDL A800 80GB at CNY 5.59 per hour; vLLM 0.31.0, LLaMA-Factory commit
`1fded918`, template `qwen3_vl_nothink` (the Instruct model has no thinking
mode). All four weight files matched the published Hugging Face hashes.

- Gate 1: the first smoke failed at engine start-up because FlashInfer's
  sampler compiles kernels with `nvcc`, which the image lacks. The runner now
  sets `VLLM_USE_FLASHINFER_SAMPLER=0`; greedy decoding takes the same argmax.
  The rerun parsed 10 of 10 replies.
- Gate 2: a five-step training probe at 640x360 and per-device batch 4 fit in
  memory, so **no fallback was taken**.
- Gates 3–6: B labelled the 900 in 193 s; training ran 496 steps in 2 h 06 min;
  C passed the sanity gate (schema-valid `1.00`, both verdicts in every class)
  and labelled the 900 in 175 s. Both arms were hashed before scoring.
- Gate 7, the criterion: `C − A = −0.056`, 95% interval `[−0.125, +0.021]`.
  **C does not beat A.** Context: `C − B = +0.258 [+0.125, +0.388]`.
- Spend: about CNY 15 of GPU time against the CNY 150 ceiling.

### Not in this protocol

- No downstream retraining with C's labels. Both held-out sets are spent, so it
  could only be Development-level evidence. It was declared afterwards as its own
  stage: *Protocol v0.13 downstream* below.
- No held-out set is reopened, no prompt is edited, and no arm is rerun after
  scoring opens.
- The adapter never overrides a Strem label. A disagreement is reported as a
  disagreement.

## Protocol v0.13 downstream — retraining on the adapted labeler's labels

Declared 2026-10-11, after v0.13 scoring and before any training run on arm C's
labels. Development-level by construction: both held-out sets are spent.

> v0.11 showed that labels at Macro-F1 `0.494` retrain the student to 86% of the
> Oracle arm's corridor gain. Arm C of v0.13 labels the same 900 windows at
> Macro-F1 `0.437`, with a different error profile: corridor precision and recall
> both higher than the remote VLM's (`0.59 / 0.86` against `0.47 / 0.53`), and
> heavy over-calling on the two pedestrian classes (252 near-zone `positive`
> verdicts against the remote VLM's 47). How much of the Oracle gain do C's
> labels buy, per class?

**Arm.** `v010-disagreement-900-vlm-lora`: the exact 900 window IDs, `L0-small`
seed set, `narrow-fast-12seed` recipe, `pos_weight` source and every
hyperparameter of `v010-disagreement-900-vlm`; only the added batch's labels
change, built by `build_vlm_training_labels` from C's parsed 900-window labels
with the same mask rule (only C's own `uncertain` or unusable verdicts are
masked; the Oracle is not consulted). The label profile records protocol
`v0.13`, and the arm loader refuses a label file from any other protocol, so the
arm cannot silently train on the remote VLM's labels.

**Platform.** Every committed GRU run was trained on Procyon, which is gone.
All four arms of the comparison — Base-small (`lc_windows_25.jsonl`, no
feedback), `v010-disagreement-900` (Oracle labels), `v010-disagreement-900-vlm`
(remote-VLM labels) and the new arm — are therefore trained on this laptop from
the backed-up inputs, and every contrast is computed within the laptop runs.
Agreement of the three reference arms with their committed Procyon reports is
reported beside the result, not used to choose it.

**Reading, fixed now.** Development seed-paired contrasts, corridor (the
pre-registered class) and Macro AP:
`C − Base`, `Oracle − Base`, `C − Oracle`, `C − A` (A = the remote-VLM arm),
and the gain ratio `(C − Base) / (Oracle − Base)`.

| Result on corridor | Reading |
| --- | --- |
| `C − Oracle` within 2σ of zero | C's labels substitute for the Oracle at this instrument's resolution |
| `C − Oracle` below `−2σ`, `C − Base` above `+2σ` | lossy but net-positive |
| `C − Base` within 2σ of zero or below | C's labels buy nothing measurable |

`C − A` is reported with the same 2σ convention and no decision attached.
Expectation stated before the runs, from label quality alone: corridor at least
as good as the remote-VLM arm, and a near-zone regression below Base is
plausible because C over-calls that class. Development under-reads
between-arm contrasts (D3), so a within-2σ reading means "comparable", not
"equal". One run, no retries, nothing tuned after the first number exists.

### Observed execution (2026-10-11)

All four arms trained on the macOS laptop in a few minutes. The three
reference arms reproduced Procyon on every seed (identical best epochs, at most
`2.3e-5` AP apart). Corridor: `C − Oracle = +0.037 ± 0.022` (1.7σ), within 2σ —
**C's labels substitute for the Oracle's at this resolution**; `C − Base =
+0.166` (5.1σ), `C − A = +0.056` (3.0σ). Near-zone falls below Base (−3.7σ), as
expected. Findings section 17; `results/v013-downstream/`.

## D6 and the label-cascade analysis (declared 2026-10-12, before either ran)

Two closing analyses on artifacts that already exist. Neither reveals a label,
touches a held-out set, or calls a model.

### D6 — student-capacity check on the v0.13 downstream result

Question: the LoRA labels retrained the hidden-48 student to a corridor gain
indistinguishable from the Oracle's. D5 showed a hidden-128 student is more
sensitive to the remote VLM's label errors. Does the v0.13 downstream reading
survive the wider student? Method: the four arms of *Protocol v0.13
downstream* — Base-small, `v010-disagreement-900` (Oracle),
`v010-disagreement-900-vlm` (remote VLM), `v010-disagreement-900-vlm-lora`
(LoRA labels) — retrained with `wide-fast-12seed` and nothing else moved, all on
the laptop. Reading, as D5's: corridor `C − Oracle` within 2σ means the
downstream reading is not an artifact of the small student; below `−2σ` means
it is scoped to the hidden-48 student. `C − A`, Macro, and per-class
contrasts are reported beside it. Development-level.

### Label-cascade analysis — what an automatic labeling pipeline built from these labelers would deliver

Question: with two automatic labelers measured on the same 900 windows — the
remote VLM (A, `$0.0263` per window) and the LoRA-adapted open model (C, about
`CNY 0.0003` per window) — what label quality, human-review load, and machine
cost do simple cascades achieve? Policies, fixed now, applied per window and
class:

| Policy | Machine verdict | Sent to human review |
| --- | --- | --- |
| `A` | A's decided verdict | A abstains |
| `C` | C's decided verdict | C abstains |
| `route` | corridor from C, the two pedestrian classes from A | the chosen labeler abstains |
| `agree` | A and C both decide and agree | any disagreement or abstention |
| `C-then-A` | C's decided verdict; where C abstains, A's decided verdict (A is called only for windows with a C abstention) | both abstain |

Metrics per policy, per class and macro: the human-review rate over all 900
windows; the machine verdicts' precision, recall, and F1 on Oracle-decided
windows (Oracle `ignore` windows count toward review load but are never scored,
as in v0.11); and the final F1 after review, under the stated assumption that a
reviewer returns the Oracle label — an upper bound on human quality, not a
measurement. Machine cost per window is the share of windows on which each
labeler is called times its measured unit cost, reported per currency. The
analysis is descriptive: no policy is declared a winner, and these 900 windows
are the adversarial disagreement slice, not a random sample.

### Observed execution (2026-10-12)

D6: the four wide arms trained on the laptop; the three D5 arms reproduced
Procyon (identical best epochs, at most `1.1e-3` AP apart). Corridor
`C − Oracle = +0.058 ± 0.006` (9.3σ), not below −2σ: the downstream reading is
not a small-student artifact. Above +2σ was not a named reading; Findings §16
records it with the confound that C was adapted on ten L0 logs the student
never sees. Cascade: `agree` reaches machine Macro-F1 `0.686` with a review rate
of `0.372`; Findings §17.

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

Keep the remaining manual review bounded: a small stratified
TaskDesign gallery plus at least five final explanatory cases, including one VLM
label error or uncertainty case. Strem labels simulate an Oracle; manual review
audits their quality and is not described as full human annotation. The final
cases are Findings sections 17–18 (six windows, every layer).

## Release Check

Before scoring, confirm task spec, split, DINO revision, Base thresholds, FN
bank, selector policies, selected IDs, the three Random seeds, the frozen arm
list, VLM prompt/schema/model identity, and all predictions are frozen. Confirm
U selection saw no hidden labels and official val entered neither training nor
tuning. Confirm that every arm in the frozen list has a prediction file before
the first Test score, because the Test is opened once for all arms together.
