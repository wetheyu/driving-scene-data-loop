# Findings

Specification: Frozen for `v0.8-three-class-loop`

What the experiments actually found, in the order the evidence arrived. Every
number here comes from a report under [`results/`](../results/README.md), so a
claim can be checked against the file that produced it.

This document records observations and the reasoning built on them. It does not
restate the protocol; the owning contracts are
[Evaluation Plan](EVALUATION_PLAN.md) and [Bad-Case Process](BAD_CASE_PROCESS.md).

## The pre-registered question

> Under a fixed simulated budget of 300 windows, does selection driven by
> Development bad-case similarity, model-boundary uncertainty, temporal
> deduplication, and visual diversity improve Held-out Test Macro-AP more than
> random selection?

Everything below was frozen before any Oracle label was read: the three scenario
specifications, the split, the Base model and its thresholds, the false-negative
bank, both selection rankings, the three Random batches, the arm list, and the
expected outcomes. The answer is a negative result, and the value of the work is
in the diagnosis.

## 1. Selection variance is larger than the effect being measured

The first design decision that mattered was made before any label was revealed.
Integrated Mining is deterministic and produces exactly one batch, while Random
is a distribution over batches. Comparing one Mining batch with one Random batch
cannot separate method from draw luck.

Three Random batches were therefore frozen, with seeds `101`, `102`, and `103`.
They overlap by 6, 7, and 9 windows at `N=300` against a uniform-draw expectation
of `300 × 300 / 11639 = 7.7`; their union is 879 distinct windows, and one window
appears in all three. Three honest draws are almost disjoint.

That decision changed the conclusion. On `pedestrian_ego_near_zone_entry` the
three Random batches reached Development AP `0.1346`, `0.1831`, and `0.1859`,
while Mining reached `0.1704`. Against seed 101 alone Mining looks clearly ahead;
against the range it is an ordinary draw. **With one Random batch the reported
result would have been wrong, and nothing in the experiment would have revealed
it.**

## 2. The effect size was knowable before any model was trained

Revealing the frozen IDs bought these positive labels at `N=300`, against L0
positive counts of 294, 317, and 450:

| Class | Mining | Random 101/102/103 |
| --- | ---: | ---: |
| near-zone entry | 12 | 14 / 9 / 10 |
| proximity hold | 25 | 18 / 17 / 19 |
| corridor entry | 13 | 16 / 12 / 16 |

The two methods differ by about seven positive windows, in one class, on a base
of a few hundred. The GRU's own seed spread is `± 0.008` to `± 0.021` Macro-AP.
The experiment was therefore underpowered by roughly an order of magnitude, and
this was measurable before the first fit.

Stating it in advance mattered: the null result that followed is a confirmed
prediction, not an excuse constructed afterwards.

## 3. Adding data helped; choosing it did not

Round one, Development Macro-AP over training seeds `17, 29, 43`:

| Arm | From scratch | Fine-tuned from Base |
| --- | ---: | ---: |
| Base | `0.2408 ± 0.0088` | — |
| Base-continued, no added data | — | `0.2422 ± 0.0118` |
| Mining-300 | `0.2528 ± 0.0028` | `0.2428 ± 0.0082` |
| Random-300 seed 101 | `0.2542 ± 0.0178` | `0.2382 ± 0.0074` |
| Random-300 seed 102 | `0.2569 ± 0.0208` | `0.2425 ± 0.0121` |
| Random-300 seed 103 | `0.2561 ± 0.0188` | `0.2424 ± 0.0093` |

Every from-scratch arm beats Base, and the four arms cannot be told apart. The
budget has value; the selection strategy shows none. Mining sits below all three
Random batches from scratch and above them by `0.0003` when fine-tuned, well
inside its own seed spread.

Fine-tuning absorbed the added data entirely: every arm lands between `0.2382`
and `0.2428`, including `Base-continued`, which added nothing at all. Resuming a
converged checkpoint at a tenth of the learning rate barely moves the model, so
300 extra windows changed almost nothing. Both protocols were declared before
training and both are reported.

## 4. Why the selector failed: three measured mechanisms

Diagnosis used the windows selected by the three Random batches, 1,710 unique
windows that are a uniform sample of the Pool and whose labels were already
revealed. That makes the estimates unbiased for the Pool.

### 4.1 The similarity component was anti-predictive

Positive rate of the top decile by each signal, against the sample's base rate:

| Class | Base rate | FN similarity | Boundary margin |
| --- | ---: | ---: | ---: |
| near-zone entry | 3.92% | 0.58% (**0.15x**) | 10.53% (2.69x) |
| proximity hold | 5.03% | 8.77% (1.74x) | 14.04% (2.79x) |
| corridor entry | 4.85% | 4.68% (0.96x) | 15.79% (3.25x) |

On the rarest class the windows most similar to known failures are *less* likely
to be positive than a random draw. Because Mining uses similarity as a hard
filter that keeps only the top 2,000 of 11,639, it discards good candidates
before ranking begins. Among low-margin windows, those the filter removed had
higher positive rates than those it kept, on all three classes: 8.70% against
11.11%, 12.07% against 14.95%, and 13.04% against 17.91%. Each pair is
individually noisy; the agreement across three classes is the evidence.

### 4.2 The diversity component diluted the only working signal

Selecting 200 windows from a 2,000-window shortlist purely by uncertainty would
place every selection inside the shortlist's best 10% by margin. Round-robin
across 30 clusters instead produced a median at the 23rd to 32nd percentile,
with the worst selection at the 97th to 99.8th — the least uncertain window in
the entire shortlist was selected. This is a deterministic property of the frozen
ranking, not an estimate.

Cluster round-robin forces every cluster to contribute, including clusters whose
best member carries almost no signal.

### 4.3 The "uncertainty" component was not measuring uncertainty

The frozen thresholds sit in the far right tail of the probability distribution:

| Class | Threshold | Pool median probability | Pool fraction above |
| --- | ---: | ---: | ---: |
| near-zone entry | 0.8855 | 0.2290 | 2.4% |
| proximity hold | 0.5384 | 0.1321 | 10.4% |
| corridor entry | 0.6380 | 0.0704 | 7.1% |

Minimising `|logit(p) − logit(threshold)|` therefore selects high-probability
windows, not ambiguous ones. The component named "boundary uncertainty" was
performing positive retrieval indirectly, and less effectively than doing it
directly.

## 5. A threshold is a decision point, not a ranker

The thresholds themselves are correct. They were chosen by exhaustive maximum-F1
search over every observed probability on Development using Base seed 17, and an
independent recomputation reproduces `0.885465`, `0.538358`, and `0.637988` with
their F1 values exactly. At `0.885465` the model predicts 54 Development
positives against 54 real ones, which is what maximum-F1 selection does.

The defect is in what the threshold was then used for. Both bad-case mining and
data selection are ranking problems, and using a decision point to rank imports
the model's overall strength as a bias:

| Class | Development positives | Counted as false negatives |
| --- | ---: | ---: |
| near-zone entry | 54 | 45 (**83.3%**) |
| proximity hold | 90 | 57 (63.3%) |
| corridor entry | 206 | 48 (23.3%) |

With a weak model on a rare class, "false negative" degenerates into "positive".
The bank stopped selecting hard cases because almost every case qualified.

## 6. The bank's representative rule pointed the wrong way

The bank keeps the lowest-scoring window of each event as its representative.
The consequence:

| Class | Median score, all positives | Median score, bank representatives | Representative percentile |
| --- | ---: | ---: | ---: |
| near-zone entry | 0.5134 | 0.2856 | 31st |
| proximity hold | 0.4029 | 0.3040 | 42nd |
| corridor entry | 0.9136 | 0.3346 | 11th |

The bank holds the least typical view of each event. Its embedding centroid is
therefore a centroid of atypical scenes, and similarity search from it retrieves
more atypical scenes, which are less likely to be positive. This is the mechanism
behind the 0.15x lift in 4.1, and it is a design consequence rather than a bug:
each step was individually reasonable, and their composition inverted the signal.

This task has two layers of scarcity. The first is plain class imbalance:
3.50%, 3.78%, and 5.36% of L0 windows are positive. Inside those positives
sits the genuinely long-tailed layer — atypical views — rain-blurred, occluded, unusual
geometry — and the bank's representative rule targets exactly that second tail.

Retrieval works on the first layer and fails on the second, because typical
positives resemble each other while atypical ones do not resemble anything in
particular. "Atypical" is not a visual category, so those windows do not form a
cluster an embedding can retrieve against. This is also why production long-tail
mining leans on rule and metadata triggers, which name the tail explicitly,
rather than on embedding similarity, which assumes the tail is a cluster.

## 7. What does predict positives

Positive rate in the top 5% of the unbiased sample, `n = 85` per cell:

| Class | Base rate | Boundary margin | Probability, descending | Classic `\|p − 0.5\|` |
| --- | ---: | ---: | ---: | ---: |
| near-zone entry | 3.92% | 11.76% | **16.47%** | 10.59% |
| proximity hold | 5.03% | 14.12% | **24.71%** | 16.47% |
| corridor entry | 4.85% | 15.29% | **47.06%** | 12.94% |

Ranking by the model's own probability wins on every class. Textbook uncertainty
sampling at `p ≈ 0.5` is the weakest of the three, because in a problem with 4%
to 5% positives that region is almost entirely negative.

Simple probability ranking is not only the strongest measured signal here, it is
also the standard approach to rare-event mining. It retrieves two useful groups
at once: the rare positives, and the high-scoring negatives that are hard
negatives by definition — 53% to 84% of each selection.

Diversity machinery is not needed to keep such a batch spread out. Pure ranking
with the same temporal separation covers 44, 74, and 64 scenes across 17 to 19
logs per class, comparable to Mining's 130 scenes and 21 logs over all three.

## 8. A Development-informed selector was built and tested

Section 4 diagnosed three mechanisms and predicted that removing them would
recover the signal measured in section 7: ranking each class by the Base
model's own probability, descending, with only temporal deduplication — no
similarity filter, no shortlist, no clustering. This selector is
**Development-informed by construction**: it was designed after reading
revealed Pool labels, so unlike the pre-registered arms it carries a stated
advantage over Random, which was never tuned. It is reported as such, not as a
second pre-registered result.

### It fixed the yield gap decisively

Positive labels bought at `N=300`, using the same accounting as section 2 —
positives across the full selected batch, independent of which class's query
surfaced a window:

| Class | Mining v1 | Random range | **Mining v2** | v2 vs best Random |
| --- | ---: | ---: | ---: | ---: |
| near-zone entry | 12 | 9 – 14 | **24** | +10 |
| proximity hold | 25 | 17 – 19 | **31** | +12 |
| corridor entry | 13 | 12 – 16 | **66** | **+50** |

Corridor bought more than five times the best Random batch. Log and scene
coverage of the positive labels stayed reasonable rather than collapsing to a
few repeats: 10 to 16 logs and 18 to 44 scenes per class, comparable to the
spread Random achieved.

### It did not improve Development Macro-AP

Both declared protocols were run, exactly as for round one:

| Protocol | Mining v1 | Random range | **Mining v2** |
| --- | ---: | ---: | ---: |
| P1 from scratch | `0.2528 ± 0.0028` | `0.2542 – 0.2569` | `0.2527 ± 0.0024` |
| P2 fine-tuned | `0.2428 ± 0.0082` | `0.2382 – 0.2425` | `0.2420 ± 0.0107` |

Both Macro-AP figures land inside or below the Random range, indistinguishable
from round one's result despite the batch containing far more true positives.
Per class, from scratch:

| Class | Mining v2 AP | Random range | Verdict |
| --- | ---: | ---: | --- |
| near-zone entry | `0.1292 ± 0.0557` | `0.1346 – 0.1859` | below range, and below Mining v1's `0.1704` |
| proximity hold | `0.0899 ± 0.0265` | `0.0595 – 0.0847` | above range, by less than its own seed spread |
| corridor entry | `0.5391 ± 0.0282` | `0.5228 – 0.5433` | inside range |

The one class where v2 clears the Random range — proximity hold — clears it by
`0.0052`, smaller than its own `± 0.0265` seed spread. **No class shows a clean
win.** Near-zone is the sharpest result: despite the largest yield gain of the
three classes relative to L0, its Development AP fell below every Random batch
and below Mining v1, with a seed spread nearly three times the typical `± 0.02`
to `± 0.03` seen elsewhere in this project.

### What this confirms

**Positive yield is a proxy metric, not the outcome.** This was flagged before
training as a real risk — round one already showed Mining v1 buying more
proximity-hold positives than Random (25 against 18) while scoring lower on
that class — and this experiment is a direct, designed test of it, buying up
to five times more positives with no corresponding Macro-AP gain.

It also sharpens which open hypothesis in the previous section carries more
weight. Doubling to quintupling the positive count did not move the metric,
which argues against the budget alone being the limiting factor and for the
representation being closer to the true ceiling: a selector cannot make the
encoder resolve an object it cannot resolve, no matter how many correctly
labeled instances of that object it is shown.

One mechanism for the near-zone regression is plausible but not confirmed: pure
probability ranking on a weak classifier (Base near-zone AP `0.167`, prevalence
`2%`) surfaces windows the model is already confident about — the top-100
queried for this class had a median Base probability of `0.899` — which may
reinforce an already-learned response rather than teach a new one, while
shifting the decision boundary enough to raise seed-to-seed variance. This is
recorded as a hypothesis, not a finding: the evidence is one experiment at one
budget, and confirming it would need controlled variation of what is selected,
which round one's timeline did not include.

## 9. Open hypotheses

Two candidate explanations for the null result remain, both pre-declared in
[Bad-Case Process](BAD_CASE_PROCESS.md) before the result was known, and each is
directly testable without changing the task, the split, or the labels:

1. **The budget is too small.** At `N=300` the methods differ by seven positive
   windows. Raising the budget scales that difference roughly proportionally.
   Section 8 weakens this hypothesis somewhat: a selector that bought two to
   five times more positives than Random still produced no clean win, so a
   larger budget alone is not guaranteed to close the gap either.
2. **The representation is the ceiling.** Labels are defined by metric geometry
   in the ego ground plane, while the model input is one 384-dimensional global
   CLS vector per frame from a monocular camera. A pedestrian at 15 to 20 m is
   roughly 20 to 30 letterbox pixels tall, and global pooling discards where and
   how large it was. Spatial patch features would test this directly, and they
   need no Oracle information, so they remain a legitimate model input.

The second is the deeper limitation, and it caps what any selection method can
demonstrate here: adding windows containing an object the encoder cannot resolve
cannot help. Reporting this is more useful than reporting an unexplained gain.

## 10. The measurement itself was the largest error source

A review of the training curves found Base peaking on Development at epoch 2
while train loss fell from 0.96 to 0.14, with the Development curve swinging
about 0.036 between adjacent epochs — an order of magnitude above the
between-arm differences under test. The response was a declared protocol
change, selected by a criterion stated before any candidate ran: smallest
seed-to-seed spread on Base whose curve does not peak within three epochs,
with highest Macro-AP explicitly not the criterion. The winner, `narrow-fast`
(hidden 48, otherwise unchanged), was then run at twelve seeds per arm, keeping
`17, 29, 43` as a nested prefix.

Twelve seeds exposed how unreliable the three-seed spreads had been:

| Arm | 3-seed std (same runs, first three seeds) | 12-seed std |
| --- | ---: | ---: |
| Base | 0.0096 | 0.0263 |
| Random-300 seed 101 | 0.0391 | 0.0247 |
| Random-300 seed 102 | **0.0001** | **0.0194** |
| Random-300 seed 103 | 0.0054 | 0.0178 |

Three samples cannot estimate a standard deviation; every earlier `± 0.005`
was noise about noise. Two earlier readings are therefore corrected:

- **"Adding 300 windows helps" is withdrawn.** With twelve seeds and
  seed-paired statistics (same seed shares initialization and 97% of training
  data; observed correlation between arms `+0.37` to `+0.75`, pairing shrinks
  the standard error 1.4x), Random minus Base is `+0.0036 ± 0.0060` — nothing.
- **"The arms cannot be told apart" was a statement about the instrument**, not
  about the arms, as section 12 shows.

## 11. The learning curve: which classes can respond to data at all

Base was retrained on whole-log subsets of L0 (hash-ordered, guarded at twenty
or more positives per class), twelve seeds each:

| L0 | windows | near-zone | proximity hold | corridor |
| --- | ---: | ---: | ---: | ---: |
| 25% | 1,931 | 0.1040 | 0.0464 | 0.3677 |
| 50% | 3,812 | 0.0975 | 0.0452 | 0.4045 |
| 75% | 6,475 | 0.1164 | 0.0462 | 0.4573 |
| 100% | 8,390 | 0.1471 | 0.0586 | 0.4958 |

Corridor rises `+0.128` end to end at a steady `+0.02` AP per thousand windows
and is still climbing at 100%: the model remains data-hungry on the one class
it is competent at. Proximity hold is flat across a 4.3x data range — the
learning-curve confirmation of what Gate B diagnosed: the class is
information-limited in this input, and no selector can change that. Near-zone
is noisy with a rising tail. The curve also calibrates expectations: 300 extra
windows should move Macro-AP by roughly `+0.006`, exactly the size of the
Random-minus-Base reading above.

## 12. With the instrument fixed, the pre-registered selector works — where active learning says it should

The full twelve-seed grid, seed-paired against the mean of the three Random
batches:

| Contrast | Macro | near-zone | proximity hold | corridor |
| --- | ---: | ---: | ---: | ---: |
| Mining v1 − Random | `+0.0114 ± 0.0044` | `+0.0106 ± 0.0111` | `−0.0011 ± 0.0042` | **`+0.0247 ± 0.0066` (3.7σ)** |
| Mining v2 − Random | `+0.0003 ± 0.0047` | `−0.0131 ± 0.0123` | `+0.0032 ± 0.0029` | `+0.0107 ± 0.0125` |

The corridor effect survives every robustness check run against it: it beats
each Random batch separately (`+0.0376` at 4.2σ, `+0.0279` at 3.7σ, `+0.0085`
at 0.9σ against the strongest batch), stays at 2.3σ when the batch-to-batch
variance of the three Random draws is folded into the error, beats Base itself
(`+0.0350 ± 0.0116`, 3.0σ — Mining reaches 0.5308 against Base's 0.4958), and
holds at `+0.0239` (2.9σ) on the nine seeds added after the original three.
Across the already-revealed budget prefixes the sign is consistent —
`+0.0245 / +0.0376 / +0.0245` at 150/300/600 — though the curve is not
monotone and the 150/600 points have only the seed-101 batch as control.

A fifth check answers the sharpest remaining worry about Development's size:
its 74 corridor events cluster badly, with one log holding 43 of them (58%),
while the Held-out Test spreads 69 events over 15 logs at a 14% maximum share.
Splitting Development on that log, the effect stands in both disjoint halves
independently: `+0.0307 ± 0.0098` (3.1σ) against Random without the dominant
log — where Mining beats Base by `+0.0687` — and `+0.0279 ± 0.0068` (4.1σ)
inside it. The advantage is not one log's traffic pattern.

Where it works is exactly where the textbook says uncertainty sampling should:
corridor is the one class whose Base is competent (AP 0.50, so its uncertainty
is meaningful) and whose learning curve is still steep (so data still buys
improvement). Near-zone fails the first condition, proximity hold fails both.
And Mining v1 bought only 13 corridor positives against Random's 12 to 16 —
the advantage is the training value of near-threshold windows, largely hard
negatives, not positive count.

Two honest corrections come with this result. First, the section-4 diagnosis
judged v1's ranker by positive yield — the exact proxy this project later named
as a trap. By that metric v1 looked inferior to direct probability retrieval;
by training value, v1's boundary sampling is the only component that works, and
v2, built to maximise the proxy, delivers nothing. The diagnostic fell into the
trap before the experiment did. Second, an unexplained phenomenon is recorded
rather than smoothed over: at `N=600` both arms fall back to or below Base
(Random-600 lands 0.025 under it). A candidate mechanism — added windows are
overwhelmingly negative while `pos_weight` stays fixed to L0, diluting the
positives, compounded by Mining's ranks 301-600 sitting further from the
boundary — is a hypothesis, not a finding.

Status: the measurement protocol was selected after round one, by pre-declared
criteria, so everything in this section is Development-informed. The arbiter is
the Held-out Test, opened once, after all predictions freeze.

## 13. The Held-out Test: the effect did not transfer

Opened once, after all 650,280 prediction rows were frozen and verified to
carry only window ID, partition, and probabilities. Ten arms, twelve seeds
each, scored by the pre-declared rule: the corridor-class seed-paired
difference between Mining v1 and the mean of the three Random batches.

| Arm | Macro | near-zone | proximity hold | corridor |
| --- | ---: | ---: | ---: | ---: |
| Base | `0.2054 ± 0.0145` | `0.1785` | `0.0683` | `0.3696` |
| Random-300 seed 101 | `0.2211 ± 0.0185` | `0.1888` | `0.0866` | `0.3878` |
| Random-300 seed 102 | `0.2033 ± 0.0197` | `0.1844` | `0.0591` | `0.3662` |
| Random-300 seed 103 | `0.2084 ± 0.0137` | `0.1725` | `0.0699` | `0.3828` |
| Mining v1-300 | `0.2131 ± 0.0202` | `0.1999` | `0.0572` | `0.3822` |
| Mining v2-300 | `0.2229 ± 0.0200` | `0.1935` | `0.0891` | `0.3860` |

**Primary criterion: `+0.0033 ± 0.0101` (0.3σ). The Development corridor
effect did not transfer.** Against Base it is `+0.0127 ± 0.0113` (1.1σ), also
not significant. Broken out by batch the sign is not even stable: `−0.0056`,
`+0.0160`, `−0.0005` against seeds 101, 102, and 103.

This is not a power failure. The pre-opening estimate said the standard error
would be about `0.0068` and a transferred effect would read near 3.6σ; the
realised standard error is `0.0101`, so an effect of the Development size
(`+0.0247`) would still have shown at about 2.4σ. What is observed is
`+0.0033` — roughly a seventh of it. The effect is absent, not merely blurred.

Everything else is reported as it came out, none of it a pre-declared
criterion. Macro is flat (`+0.0022 ± 0.0042`). near-zone moves `+0.0180 ±
0.0100` (1.8σ), the largest Mining-favouring number on the Test and one that
Development did not predict. proximity hold moves `−0.0147 ± 0.0034` (4.4σ)
against Mining — the sharpest single number in the whole table, on the class
whose learning curve is flat and whose Test support is thinnest at 37 events.
v2 reads `+0.0071 ± 0.0098` on corridor, and the budget points give `+0.0088`,
`−0.0056`, `+0.0052` at 150, 300, and 600 — no dose structure survives.

### What this establishes

The corridor effect was real on Development and survived five robustness
checks there, including a leave-dominant-log-out split. It did not survive the
move to eighteen unseen logs. The two facts together are the result, and the
honest reading is the one the protocol was built to force: **a
Development-informed finding, however well defended on Development, is not a
finding until a held-out set agrees.**

Two candidate explanations are available and this experiment cannot separate
them. Development's corridor events concentrate 58% in a single log while the
Test spreads 69 events over fifteen logs at a 14% maximum share, so the
Development effect may have been specific to a traffic pattern that the
leave-one-log-out check could not detect because both halves came from the same
seven logs. Alternatively the measurement protocol, selected after seeing round
one, may have been tuned into a Development-specific optimum — the exact risk
the Development-informed label was attached to record.

### Post-hoc diagnosis: where the Development effect came from

Clearly labeled exploratory, run after the opening, changing nothing. Three
layers, ordered by how directly the evidence supports each.

**Instance-level proximity — verified, minor.** Splitting Development's 206
corridor positives at the median of their cosine similarity to the 300 added
Mining windows, the contrast is `+0.0284 ± 0.0077` on the similar half against
`+0.0194 ± 0.0100` on the dissimilar half: a real gradient in the predicted
direction, worth about `0.009` — not the collapse.

**Domain-level coupling — the main suspect.** The dissimilar half sits at
similarity levels comparable to the Test distribution (Development median
`0.655`, Test `0.613`, the far half below `0.651`), yet it still shows a 1.9σ
advantage that the Test does not. Cosine proximity to the batch therefore
cannot be the main carrier. What the far half shares with the batch is the
domain itself: every FN-bank query came from the same six or seven drives that
then evaluated the outcome. The selection was optimised, through its queries,
for the logs it would be judged on.

**Estimate inflation — structural.** Corridor became the primary criterion as
the largest of six contrasts read off the twelve-seed grid, so its Development
estimate carries winner's-curse inflation; on the Test the leading class
rotates to near-zone (`+0.0180`, unpredicted), the signature of reading noise
ranks rather than effects.

Why five robustness checks saw nothing: all of them resample inside
Development, and the leave-dominant-log-out split divides the evaluation logs
while the FN bank spans every one of them — the coupling covers both halves.

The design lesson in one sentence: **the selector's query set and the
evaluation set were the same partition.** No label ever leaked, but drawing
mining queries from the split that then measures the mining outcome couples
the two at instance and domain level, and no within-split check can see it.
The v0.10 protocol change follows directly: query logs and evaluation logs
must be disjoint.

No further arm was trained, no threshold moved, and no criterion was revised
after this table was produced. The pre-registered question — does bad-case
driven selection beat random selection at a fixed budget — is answered for this
dataset, representation, and budget: **no measurable advantage on held-out
data.**

## 14. Protocol v0.10: the loop closes, on the pre-registered criterion

Every diagnosed flaw was fixed by design, the protocol and criterion were
frozen before any v0.10 label was revealed, and the one budget amendment
(1,200 to 900, for measured placement capacity) was made pre-reveal and
recorded. The operating point moved to the small seed the learning curve
identified (L0-small, 1,931 windows, 58 corridor positives); the selectors use
no query bank at all — ensemble disagreement (probability spread over the
twelve Base-small seeds) and ensemble probability mean, both ranking Pool2
exhaustively with no recall stage; and the readout is Test2, ten U logs never
touched by any selector, query, reveal, or training run, carved by hash before
selection. Integrity audit: zero overlap between Test2 and revealed windows,
zero Test2-log windows in any ranking, training compositions exact.

On Test2 (3,639 windows; 139/129/117 positives per class):

| Contrast, selector − Random(3 batches), seed-paired | N=300 | N=600 | **N=900 (primary)** |
| --- | ---: | ---: | ---: |
| **disagreement, corridor** | `+0.0517` (3.9σ) | `+0.0372` (3.0σ) | **`+0.0619 ± 0.0082` (7.6σ) — passes the frozen ≥2.5σ bar** |
| disagreement, Macro | `+0.0149` (2.3σ) | `+0.0187` (3.6σ) | `+0.0148` (3.8σ) |
| prob-ranked, corridor | `+0.0563` (3.7σ) | `+0.0449` (3.7σ) | `+0.0254` (1.8σ) — fails the bar |

The loop's premise is also finally measurable: adding data itself
(Random minus Base-small) reads `+0.0425 / +0.0614 / +0.0691` Macro at the
three budgets, 8 to 11σ — the same quantity that was `+0.0036` (0.6σ) at the
v0.8 operating point. The small-seed redesign did exactly what the learning
curve predicted.

The two selectors separate in the theoretically expected direction. Yield-
maximising probability ranking, which buys the most positives, fails the
primary bar — the third independent demonstration that positive yield is not
training value. Epistemic disagreement, which asks where the ensemble
genuinely conflicts, wins decisively and at every budget. Its trade-off is
reported with it: at `N=900` it gives back `−0.0083` (−2.8σ) on near-zone and
`−0.0091` (−1.6σ) on proximity hold, and still carries Macro by +3.8σ.

A post-hoc efficiency reading (exploratory, same Test2 exam, full-L0 Base
predicted after the opening) puts the absolute numbers in their real currency.
On corridor, the small seed scores `0.1711`, full-L0 training (8,390 windows)
scores `0.3933`, and the disagreement arm — 2,831 windows, one third of the
data — scores `0.3594`: **85% of the full-data gain captured with 14% of the
additional labels**, against Random's 55% for the same 900-window budget.
Test2 prevalence is `3.3-4.1%`, so chance-level AP is about `0.03` and the
disagreement arm sits eleven times above it. The loop's value is measured in
label efficiency, not absolute AP: the absolute ceiling of this deliberately
small frozen stack is visible in the full-L0 number itself.

What the claim is: under this dataset, representation, and protocol, at a
seed-scarce operating point, **bad-model-driven selection by ensemble
disagreement measurably improves the model over random selection on held-out
logs it never saw** — the pre-registered question, answered positively, on the
second, corrected attempt. What it is not: evidence for the v0.8 similarity
pipeline, for yield-based selection, or for anything at the data-rich
operating point where v0.8 showed effects are unmeasurable.

## 15. Protocol v0.11: a VLM at Macro-F1 0.49 buys most of the same model

The loop so far bought its labels from a private metric Oracle. Industrially that
is the expensive half, so v0.11 asked what a remote VLM produces on exactly the
windows the v0.10 selector chose and the Oracle already revealed, and -- the part
that matters -- whether training on those labels instead keeps the gain.

`gpt-5.6-terra`, five CAM_FRONT frames and the public scenario text per window,
strict JSON-schema output, 900 windows, `$23.69` measured, 23 minutes, zero
failed calls. The provider never saw a label, a Strem result, a binding, a box,
or a distance: the request builder accepts only rows carrying exactly the public
field set, so the boundary is a property of the code path.

### The labels are mediocre

| Class | Precision | Recall (decided-only) | Recall incl. abstentions | F1 | VLM `uncertain` |
| --- | ---: | ---: | ---: | ---: | ---: |
| near-zone | 0.400 | 0.800 | **0.471** | 0.533 | 24.6% |
| proximity hold | 0.302 | 0.865 | **0.681** | 0.448 | 28.8% |
| corridor | 0.474 | 0.529 | **0.346** | 0.500 | 10.3% |

**Correction (2026-09-01 audit).** The recall column follows the plan's metric
definition — `uncertain` excluded from numerator and denominator — which means
it reads like positive coverage but is not: a positive the VLM abstained on is
invisible to it. Counting abstentions as misses, the VLM recovers only 35% of
all true corridor positives. The downstream comparison is unaffected (training
masks the same abstentions), but any coverage claim must quote the second
recall column, and the evaluator now emits both.

Macro-F1 `0.494`; schema-valid `99.1%`; `evidence_frames` out of range or missing
under a positive verdict: `0` of 2,700. The error is overwhelmingly
false-positive -- 118 false positives against 25 false negatives -- which is what
a monocular estimator does against a metric threshold: it sees the relation and
cannot see the boundary. The Development smoke reached `0.846` on the same prompt
five hours earlier; the drop is the selector working as designed, since
`disagreement` selects exactly the windows the ensemble finds hardest.

The Oracle's `ignore` windows are the check that the uncertainty is real rather
than decorative. On near-zone, 20 of 45 got `uncertain` -- the model declines
where the metric definition is itself ambiguous, instead of guessing.

### The downstream model does not notice

Twelve seed-paired Development contrasts, identical seeds, identical L0-small,
identical recipe, identical 900 window IDs. Only the added batch's label source
differs.

| Contrast | Corridor AP | Macro AP |
| --- | --- | --- |
| VLM − Base | `+0.1105 ± 0.0265` (4.2σ) | `+0.0258 ± 0.0085` (3.0σ) |
| Oracle − Base | `+0.1288 ± 0.0246` (5.2σ) | `+0.0299 ± 0.0090` (3.3σ) |
| **VLM − Oracle** | `−0.0183 ± 0.0201` (0.9σ) | `−0.0041 ± 0.0071` (0.6σ) |

VLM labels capture **86%** of the Oracle's corridor gain and are statistically
indistinguishable from it. This is the first of the three readings pre-declared
in the Evaluation Plan before the labels existed: at this operating point the
automatic labeler substitutes for the expensive label source.

The mechanism is not mysterious once the numbers sit together. The classifier
does not need correct labels on every window; it needs the added batch to shift
the decision boundary in the right direction. A precision-0.47 corridor labeler
that recovers half the positives still delivers most of that shift, because the
alternative is not perfect labels -- it is no labels at all. The near-zone
give-back reproduces in both arms (`−0.024` VLM, `−0.031` Oracle), so it belongs
to the batch, not to the label source.

### What this costs, and what it does not license

`$0.0263` per window, `$0.359` per label the Oracle also calls positive. That is
the number to plan a pipeline on, and the reason the mid-tier model was frozen
rather than the flagship: the industrial question is what labeling costs at
scale, not what the best available model can do once.

Two honest limits. The held-out sets are both spent, so this is a Development
contrast and cannot claim held-out transfer -- the same discipline section 13
enforced on v0.8. And label F1 is not the metric that mattered here: an F1 of
`0.49` predicted nothing about a downstream difference of `0.6σ`. That is the
same lesson as the positive-yield trap in section 6, arriving on the other leg of
the loop -- a label-quality proxy is only trusted after the downstream
measurement agrees with it.

Nothing here is VLM training. It is API inference against a frozen prompt, and it
is never to be described as fine-tuning or SFT experience.

## 16. Four scoping diagnostics: what the v0.10 result does and does not license

Declared as protocol v0.12 before running, executed 2026-09-01, all
Development-level; D1-D3 spend no Oracle budget and D4 declares its spend. Each answers one question a reviewer
of sections 14-15 would ask next. Reports: `results/v012-diagnostics/`.

### D1 — the selector needs its whole ensemble

Recomputing the twelve per-seed Pool2 probability sets and rebuilding the full
ranking pipeline from ensemble subsets (twelve seeded subsets per size), the
fraction of the frozen top-900 selection a subset recovers is:

| members | 2 | 4 | 6 | 8 | 12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| overlap@900 | 0.26 | 0.37 | 0.43 | 0.54 | 1.00 (sanity gate) |
| overlap@300 | 0.13 | 0.29 | 0.37 | 0.48 | 1.00 |

The sanity gate — the recomputed full ensemble must reproduce the frozen
selection exactly — passed, which doubles as an integrity re-verification of
the frozen artifact. The reading is a negative one, reported as such: the
disagreement *ranking* is not stable under ensemble shrinking, and the head of
the list is less stable than its tail, so the selector's inference cost cannot
be cut by dropping members while preserving the chosen set. Whether a cheaper
ensemble preserves the *downstream value* despite choosing different windows is
a separate question that would cost new reveals; overlap is the conservative
proxy, not the answer.

### D2 — the loop absorbs structured noise, not random noise

Flipping decided Oracle labels at declared rates (masks untouched,
deterministic per window and class) and retraining with the identical recipe:

| labels | corridor AP − Base | Macro − Base |
| --- | ---: | ---: |
| Oracle, 0% flips | `+0.1288 ± 0.0246` (5.2σ) | `+0.0299` (3.3σ) |
| 10% flips | `+0.0816 ± 0.0297` (2.7σ) | `+0.0154` (1.2σ) |
| 20% flips | `+0.0110 ± 0.0326` (0.3σ) | `−0.0149` (−1.2σ) |
| 30% flips | `−0.0653 ± 0.0454` (−1.4σ) | `−0.0477` (−3.3σ) |
| **VLM (Macro-F1 0.49)** | **`+0.1105 ± 0.0265` (4.2σ)** | `+0.0257` (3.0σ) |

The curve is monotone with its zero-crossing between 20% and 30% random flips,
and 30% is actively harmful — noisy enough labels are worse than no data. The
VLM row is the point of the exercise: labels that misfire on roughly two
thirds of their positive verdicts (118 of 184 on Oracle-decided windows, while
the overall decided error rate is only 7%) retrain like *well under 10%* random
noise. Section 15's result is therefore not "noise doesn't matter" but
"**the structure of the noise matters more than its rate**": the VLM's errors
are semantically correlated near-boundary false positives, which displace the
decision boundary far less than the same count of random contradictions. Two
caveats bound the comparison: the VLM arm masks its own `uncertain` verdicts
and so trains on fewer decided labels than the noise arms, and all rows are
Development readings.

### D3 — the selection effect shrinks with data, and Development under-reads it

Retraining the disagreement and three random arms at the lc50 operating point
with the same revealed labels: the corridor contrast against the random mean is
`+0.0101 ± 0.0143` (0.7σ), against `+0.0166 ± 0.0116` (1.4σ) for the same
instrument at lc25. Adding the 900 windows still helps strongly at lc50
(`+0.0802`, 8.2σ over its base), so the data is not exhausted — the *selector's
edge over random* is what shrinks.

The essential calibration: at lc25 this same Development contrast reads 1.4σ
while Test2 read 7.6σ. Development systematically under-reads between-arm
differences — plausibly because every arm's checkpoint is chosen on
Development, which compresses exactly the contrast being measured; that
mechanism is stated as a hypothesis, not a finding. D3 therefore bounds the
direction of the decay (consistent with the v0.8 null at full data), not its
magnitude; a held-out-grade answer at lc50 would need a fresh evaluation carve.

### D4 — with the coupling removed, similarity still buys nothing

The ablation v0.10 deliberately deferred: the v0.8 similarity recipe with a
clean query bank. Queries came from the ten L0 logs outside the small seed —
disjoint from the pool, from Test2, and from Development — through the same
bank construction (seed-17 Base-small predictions, frozen max-F1 thresholds),
yielding a bank of 271 events over 239 windows, four times richer than v0.8's.
The frozen v0.8 integrated recipe then ranked the Pool2-restricted pool; a
pre-reveal capacity amendment moved the top budget to 600 (the proximity-hold
queue exhausts at 900 — the same infeasibility class as v0.10's 1200), and the
reveal consumed 178 new windows (422 of the 600 were already revealed).

Twelve-seed Development contrasts against the random-600 mean, beside the
disagreement arm on the identical instrument:

| Arm (vs random-600 mean) | corridor AP | Macro |
| --- | ---: | ---: |
| clean-query similarity | `−0.0112 ± 0.0103` (−1.1σ) | `−0.0025` (−0.5σ) |
| disagreement | `+0.0444 ± 0.0158` (**+2.8σ**) | `+0.0128` (+2.1σ) |

The reading is sharp because both arms sit on the same compression-prone
instrument: it clearly shows disagreement's edge and shows none for similarity.
This sharpens the v0.8 attribution into its final form. The coupling explains
the *mirage* — why a worthless selection looked significant on Development —
and D4 shows there was nothing behind the mirage to recover: decoupled, with a
richer bank, at the sensitive operating point, similarity is indistinguishable
from random. The v0.8 failure was overdetermined, and the v0.10 selector
switch is vindicated by direct comparison rather than by argument. What this
does not test: other similarity variants (an FP-side bank, other embeddings)
remain open, and the readout is Development-level like every D-series number.

### D5 — a wider student keeps the corridor reading and loses the Macro one

Declared in the Evaluation Plan before any run (2026-09-29). The three runs
behind the v0.11 comparison were repeated with the student widened from
hidden 48 to hidden 128 (`wide-fast-12seed`) and nothing else moved. The same
command lines at hidden 48 reproduced the committed VLM arm's per-seed corridor
AP exactly (maximum difference 0.0), so the inputs are the v0.11 inputs.

| Development, 12 seed-paired | hidden 48 (v0.11) | hidden 128 (D5) |
| --- | --- | --- |
| corridor Oracle − Base | +0.1288 ± 0.0246 (5.2σ) | +0.0540 ± 0.0182 (3.0σ) |
| corridor VLM − Base | +0.1105 ± 0.0265 (4.2σ) | +0.0419 ± 0.0167 (2.5σ) |
| corridor VLM − Oracle | −0.0183 ± 0.0201 (0.9σ) | −0.0121 ± 0.0078 (−1.5σ) |
| corridor gain ratio | 86% | 78% |
| Macro VLM − Oracle | −0.0041 ± 0.0071 (0.6σ) | −0.0095 ± 0.0035 (−2.8σ) |
| Macro gain ratio | 86% | 27% |

Read by the declared rule. On corridor, the pre-registered class, VLM − Oracle
stays within 2σ: the v0.11 corridor reading is not an artifact of the small
student. On Macro it falls below −2σ, so Macro-level parity is scoped to the
hidden-48 student. The Macro gap comes from near-zone entry, where the wide
student trained on VLM labels ends below Base (−0.0273 ± 0.0084, −3.2σ) while the
Oracle arm does not (−0.0092, −1.1σ). A higher-capacity student is more
sensitive to the VLM's errors on the class where they are least structured.

Two caveats travel with it. The wide Base is already stronger on corridor
(0.437 against 0.368 at hidden 48), so both arms have less headroom and the
gains roughly halve; ratios at different headroom are not directly comparable.
And the wide VLM arm's checkpoints peak earlier than the wide Oracle arm's,
which fits a student memorising label noise but is an observation, not a
tested mechanism. Development-level, like every D-series number.

### D6 — the wider student does not shrink the v0.13 downstream reading

Declared 2026-10-12 before it ran. The four arms of the v0.13 downstream
comparison retrained with `wide-fast-12seed` (hidden 128), all on the laptop;
the three D5 arms reproduced their Procyon reports with identical best epochs
and at most `1.1e-3` AP apart (`results/d6-wide/platform_check.json`).

| Development, 12 seed-paired, hidden 128 | corridor AP | Macro AP |
| --- | ---: | ---: |
| C − Base | `+0.112 ± 0.018` (6.4σ) | `+0.023 ± 0.005` (4.2σ) |
| Oracle − Base | `+0.054 ± 0.018` (3.0σ) | `+0.013 ± 0.006` (2.3σ) |
| A − Base | `+0.042 ± 0.017` (2.5σ) | `+0.004 ± 0.007` (0.5σ) |
| **C − Oracle** | **`+0.058 ± 0.006` (9.3σ)** | `+0.010 ± 0.003` (2.8σ) |
| C − A | `+0.070 ± 0.007` (9.8σ) | `+0.019 ± 0.005` (3.9σ) |

By the declared rule the downstream reading is not an artifact of the small
student: `C − Oracle` is not below −2σ. It is in fact above +2σ, an outcome the
rule did not name, so it is reported as an observation and not as a claim that
C's labels are more accurate than the Oracle's — they are not (section 17). The
direction is the opposite of D5's: the wide student is hurt by the remote VLM's
errors but helped by C's labels. Near-zone still falls (`C − Base = −0.047`,
−5.6σ).

One confound explains a gain above the Oracle without any label being better.
C was adapted on all thirteen L0 logs; the student's seed set is three of them.
C's labels on the 900 Pool2 windows can therefore carry supervision from ten
labeled logs that neither the Base nor the Oracle arm ever sees — a
distillation channel, not a label-quality effect. The comparison answers "what
does a labeler trained on the available labeled logs buy", not "are C's labels
better than exact labels on equal information". Separating the two would need a
C adapted on the three seed logs only, which was not run. Development-level.

## 17. Protocol v0.13: an adapted open model closes most of the gap, not all of it

v0.11 bought labels from a remote VLM at Macro-F1 `0.494`, with precision as the
failure. v0.13 asked whether an open-weight VLM adapted on Oracle labels from
training logs only can do better on the same 900 windows, under a criterion
fixed before any open-weight output existed: the paired-bootstrap 95% interval
of `Macro-F1(C) − Macro-F1(A)` must lie above zero, and C must not buy it by
abstaining more.

`Qwen3-VL-8B-Instruct` at a pinned revision (all four weight files matched the
published hashes) labelled the 900 windows zero-shot (B) and with one LoRA
adapter (C): rank 16, language-model layers only, two epochs over 3,956 L0
windows, `ignore` trained as `uncertain`. Same frozen v0.11 prompt and schema,
greedy decoding constrained to that schema, frames resized once to 640x360. The
rented GPU never held a label for a scored or sanity window. One pipeline defect
was fixed at the smoke gate, before any scored output: FlashInfer's sampler
needs `nvcc`, which the image lacks, so the PyTorch sampler runs instead —
identical under greedy decoding. No declared fallback was needed. Both arms'
replies were hashed before scoring opened. Reports: `results/v013/`.

### The criterion does not pass

| Arm | Labeler | Macro-F1 | Coverage |
| --- | --- | ---: | ---: |
| A | `gpt-5.6-terra`, zero-shot (v0.11) | `0.494` | `0.788` |
| B | Qwen3-VL-8B, zero-shot | `0.180` | `0.759` |
| C | Qwen3-VL-8B + LoRA on L0 | `0.437` | `0.914` |

| Contrast (10,000 paired resamples) | Difference | 95% interval |
| --- | ---: | --- |
| **C − A** (primary) | `−0.056` | `[−0.125, +0.021]` |
| C − B | `+0.258` | `[+0.125, +0.388]` |
| B − A | `−0.314` | `[−0.464, −0.161]` |

C does not beat A: the interval spans zero, so by the declared reading there is
no measurable difference at this sample size (34 to 52 Oracle positives per
class). The coverage guard is not what failed — C abstains far less than A. The
two context contrasts are unambiguous: zero-shot, the open 8B model is far
behind the remote one, and the adapter recovers most of that distance.

### Per class, the adapter helps exactly where the GRU could learn

| Class | A precision / recall / F1 | C precision / recall / F1 |
| --- | --- | --- |
| vehicle corridor entry | `0.47 / 0.53 / 0.50` | **`0.59 / 0.86 / 0.70`** |
| pedestrian near-zone entry | `0.40 / 0.80 / 0.53` | `0.15 / 1.00 / 0.27` |
| pedestrian-vehicle proximity hold | `0.30 / 0.86 / 0.45` | `0.24 / 0.65 / 0.35` |

Recall here is over decided windows, as in section 15. On the corridor class C
fixes the very failure v0.11 had: precision rises, and of 52 Oracle positives C
says `positive` to 44 where A did to 18 (A abstained on 18). On the two
pedestrian classes C is worse than A, and in the same direction each time: it
over-calls. On near-zone entry C answers `positive` on 252 of 900 windows
against A's 47, and `uncertain` once — although 526 of its training targets for
that class were `uncertain`.

This is the class split the rest of the project already measured: corridor is
the class the frozen-feature GRU could learn and whose learning curve was steep
(sections 11–12); proximity hold was information-limited. The adapter learned
the class whose evidence is a large object crossing the frame, and did not learn
the classes whose evidence is a metric distance to a small pedestrian.

### Three cases

Each case is one window drawn with seed 17 from the cell it illustrates; the
scene name and window start identify it in the public nuScenes release.

1. **Corridor, C recovers what A abstained on** (`scene-1099`, start 13; one of
   15 such windows). At night a white SUV crosses from the left into the lane
   directly ahead and grows across the five frames. Oracle: positive. A:
   `uncertain` — "the ego corridor and metric lateral offsets are not reliably
   inferable at night". C: `positive`. The lateral entry is visible as motion
   across the image; the metric thresholds behind it did not need to be read
   off to get the label right.
2. **Near-zone, C invents an entry** (`scene-0241`, start 7; one of 77 windows
   where A was right to say `negative` and C said `positive`). A daytime street
   with parked trucks; pedestrians appear only far down the road and stay
   there. Oracle: negative. A: `negative`. C: `positive`. C fires on a
   visible pedestrian ahead, not on the 20 m → 15 m crossing the definition
   requires.
3. **Near-zone boundary, C says yes where the honest answer is unknown**
   (`scene-0006`, start 8; one of 14 Oracle-`ignore` windows where A abstained
   and C did not). A pedestrian crosses far ahead as the ego vehicle approaches.
   The event exists, but its bounded match does not complete inside these five
   frames, so the Oracle masks the window. A: `uncertain` — "monocular
   perspective does not establish a change from over 20 m to within 15 m". C:
   `positive`, as on 38 of the 45 near-zone `ignore` windows. Trained to say
   `uncertain` here, the adapter learned the look of the event instead.

Two explanations for the pedestrian over-calling are plausible and untested:
the 900 windows were chosen where the Base ensemble disagreed, which enriches
for scenes that look like positives; and an adapter trained on 1,978 eventful
windows learns "a pedestrian is ahead" more easily than a monocular metric
transition. Neither is claimed.

### What it costs

Training took 2 h 06 min on one A800 (`~CNY 12`); labelling the 900 windows
took 175 s, about `CNY 0.0003` per window against v0.11's `$0.026`, roughly
600 times cheaper per window once the adapter exists. The whole protocol,
including set-up and both arms, cost about `CNY 15` of GPU time against a
declared ceiling of `CNY 150`.

### Downstream: C's labels retrain the student as well as the Oracle's on corridor

Declared after scoring as its own stage (Evaluation Plan, *Protocol v0.13
downstream*), before any training on C's labels. The arm repeats
`v010-disagreement-900-vlm` exactly — 900 IDs, `L0-small`, `narrow-fast-12seed`,
`pos_weight` from L0-small — with only the added labels taken from C; only C's
own `uncertain` verdicts are masked and the Oracle is never consulted. Procyon
is gone, so all four arms were trained on the macOS laptop; the three
reference arms reproduced their Procyon reports on every seed with identical
best epochs and at most `2.3e-5` AP difference
(`results/v013-downstream/platform_check.json`).

| Development, 12 seed-paired | corridor AP | Macro AP |
| --- | ---: | ---: |
| C − Base | `+0.166 ± 0.032` (5.1σ) | `+0.038 ± 0.010` (4.0σ) |
| Oracle − Base | `+0.129 ± 0.025` (5.2σ) | `+0.030 ± 0.009` (3.3σ) |
| A − Base (remote VLM, v0.11) | `+0.110 ± 0.027` (4.2σ) | `+0.026 ± 0.009` (3.0σ) |
| **C − Oracle** | **`+0.037 ± 0.022` (1.7σ)** | `+0.008 ± 0.008` (1.1σ) |
| C − A | `+0.056 ± 0.018` (3.0σ) | `+0.013 ± 0.009` (1.5σ) |

Read by the declared rule: on corridor `C − Oracle` lies within 2σ, so C's
labels substitute for the Oracle's at this instrument's resolution. The point
estimate sits above the Oracle arm (gain ratio 129% on corridor, 128% on
Macro), but within the noise; it is not a claim that C's labels beat the
Oracle's. Against the remote VLM's labels C buys more corridor gain by 3.0σ,
the same ordering as label quality on that class (F1 `0.70` against `0.50`).
Mean corridor AP: Base-small `0.368`, Oracle arm `0.496`, A arm `0.478`, C arm
`0.534`.

The expectation stated before the runs held per class. Near-zone entry falls
below Base (`C − Base = −0.049 ± 0.013`, −3.7σ; the Oracle arm also drops,
−0.031, −2.6σ), where C called 181 Oracle negatives positive. Proximity hold
barely moves (`−0.002`, −0.7σ).

Against the Oracle on corridor, C's 92 positive targets fall on 44 Oracle
positives, 17 Oracle-`ignore` windows (the event exists but its bounded match
does not complete in the five frames) and 31 Oracle negatives; the Oracle arm
trains on 52 positives with its 38 `ignore` windows masked. A plausible,
untested reading is that C's extra positives on event-bearing `ignore` windows
carry real corridor signal the Oracle arm never sees. A second channel is
structural: C was adapted on all thirteen L0 logs while the student trains on
three, so C's labels can carry supervision from ten labeled logs the Oracle arm
never sees (D6). With a hidden-128 student the same comparison reads
`C − Oracle = +0.058 ± 0.006` on corridor (D6).

### A cascade of the two labelers

Declared 2026-10-12 before it ran (`results/label-cascades/`). Five fixed
policies over the 900 windows, each window and class either labeled by machine
or sent to review; review load counts all windows, quality is scored on
Oracle-decided windows, and "after review" assumes a reviewer returns the Oracle
label — an upper bound, not a measurement.

| Policy | Review rate | Machine Macro-F1 | After-review Macro-F1 | Machine cost / window |
| --- | ---: | ---: | ---: | --- |
| A alone | `0.221` | `0.494` | `0.621` | `$0.0263` |
| C alone | `0.086` | `0.437` | `0.444` | `CNY 0.0003` |
| route (corridor from C, pedestrians from A) | `0.229` | `0.560` | `0.633` | `$0.0263` + `CNY 0.0003` |
| agree (A and C both decide and agree) | `0.372` | **`0.686`** | **`0.812`** | `$0.0263` + `CNY 0.0003` |
| C-then-A (A only where C abstains) | `0.018` | `0.425` | `0.426` | `$0.0038` + `CNY 0.0003` |

Agreement lifts machine quality the most, well above either labeler
alone: machine corridor F1 `0.81` (precision and recall both `0.81`), at the
price of sending 37% of window-classes to review. The two labelers' errors are
partly independent, so their agreement is a usable confidence signal. Cheap
policies inherit C's pedestrian over-calling: C abstains so rarely (9%) that
routing its abstentions to A changes little. Descriptive only, on the
adversarial slice.

### What this does and does not license

It licenses: one LoRA adapter on one open-weight VLM, judged on label quality on
one adversarial 900-window slice, does not beat the remote VLM overall; it is
much better than the same model zero-shot; and its gains and losses split by
class along the same learnability line as the GRU's. On Development, its labels
retrain the hidden-48 student to a corridor gain statistically
indistinguishable from the Oracle's and larger than the remote VLM's. It does
not license any held-out claim, any statement about wider students, other
recipes, other models, or random windows. One recipe was run once; nothing was
tuned after scoring opened.

## 18. Six windows through the loop

Every number above is an aggregate. This section follows single windows, one
for each layer the bad-case contract names — data and label, Base miss,
selector, labeler, retraining — so the aggregates can be checked against what
the frames show. Cases 1–3 are the labeler cases of section 17; cases 4–6 come
from the v0.10 loop. Test2 windows are reviewed after its one opening and the
review informs no reported result. A window's standing is its **percentile**
among Test2 windows with a decided label for the class, ranked by the twelve-seed
mean probability (1.0 is first): ranking is what AP measures, and it needs no
threshold. Numbers: `results/cases-v1/case_review.json`.

| # | Layer | Scene, start | What it shows |
| ---: | --- | --- | --- |
| 1 | labeler | `scene-1099`, 13 | the adapted VLM reads a night-time corridor entry the remote VLM abstained on |
| 2 | labeler | `scene-0241`, 7 | the adapted VLM fires on a visible pedestrian, not on the metric entry |
| 3 | data and label | `scene-0006`, 8 | an Oracle `ignore`: the event exists, its bounded match does not fit the window |
| 4 | Base miss | `scene-0977`, 20 | cross traffic the seed model ranks low; added data lifts it |
| 5 | retraining | `scene-0414`, 18; `scene-0417`, 2 | curve-induced entries no arm learns; one moves down |
| 6 | selector | `scene-1101`, 12 | the disagreement selector's first corridor pick is a hard negative |

### 4. A cross-traffic positive the seed model ranks low

The ego vehicle waits at a red light; a dark sedan crosses from left to right in
front of it. By the frozen definition this is a corridor entry: the same vehicle
moves from `abs(y) ≥ 3 m` to `abs(y) ≤ 1.5 m` within 2 s while 0–30 m ahead. It
is not a lane change, and the scenario definition in the Project Scope says the
label never claims one. The Base-small ensemble ranks it at the `0.29` percentile, below most
negatives. After 900 added windows it stands at `0.63` with random data and
`0.72` with disagreement-selected data (`0.68` probability-ranked), and its
neighbours in the same scene at starts 19 and 26 move the same way. Most of the
lift comes from adding data at all; selection adds a little more.

That is the shape of the whole v0.10 effect at window level. Of the 117 Test2
corridor positives, the disagreement arm ranks 69 above the random-arm mean,
and the median percentile moves `0.898 → 0.921 → 0.935` from Base-small through
random to disagreement. The `+0.062` AP of section 14 is a broad shift, not a
handful of rescued windows. Case 4 was taken from the windows with the largest
disagreement-over-random lift, so it overstates the typical window; the
medians are the typical window.

### 5. Curve-induced entries no arm learns

Both windows were taken from the bottom of the disagreement arm's ranking of
Test2 corridor positives, so they show the failures, not the rate of failure.

A white van drives ahead of the ego vehicle on a curving road (`scene-0414`).
Its lateral offset in the ego frame crosses the corridor thresholds largely
because the road bends, so the Oracle calls it positive, while the frames show
no event — just a van ahead on a curve. Every arm leaves it in the lower half:
`0.34` under Base-small, `0.29` with random data, `0.40` with disagreement data.
A near-identical scene (`scene-0417`, a van ahead on a curving residential road)
moves *down* after retraining, `0.75 → 0.46` with disagreement data. The
reading — that an ego-frame metric definition includes cases whose visual
evidence is the road's shape rather than the vehicle's motion — is plausible and
untested; it is the corridor counterpart of the label-semantics hypothesis in
the Bad-Case Process.

### 6. The selector's first pick is a hard negative

The disagreement selector's first corridor-query pick (overall rank 3 of 900)
is a night scene in which a double-decker bus approaches on the right and passes
the ego vehicle. The twelve Base-small models gave it mean probability `0.33`
with standard deviation `0.255` — about as split as the pool gets. Revealed:
negative for all three classes, as were all five of its first corridor picks.

This is the mechanism behind section 14 on one window. At `N=900` disagreement
bought 52 corridor positives, barely more than random (42–48), while the
probability-ranked selector bought the most (62) and still failed the held-out
bar. What the winning selector adds is not positives but the windows on the
model's own decision boundary, many of them negatives that look like positives.

### What the cases add

The six windows agree with the aggregates and sharpen two of them. The v0.10
gain is diffuse, and its source is boundary data rather than positive yield.
And both the adapted VLM's failure (cases 2–3) and the GRU's stubborn misses
(case 5) sit where the label is a metric fact the image shows weakly — the same
learnability line sections 11 and 17 drew from aggregates. Cause confidence for
cases 3 and 5 is *plausible*, not confirmed.

## Claim boundary

Sections 1–12, 15, and 16 are Development and selection observations on one
dataset, one representation, and one downstream model; section 17 is a
label-quality reading on already-revealed Pool2 windows plus a Development-level
retraining on those labels and a descriptive cascade analysis, D6 extends
section 16, and section 18 reviews
single windows after every score froze; the only held-out readings are sections
13 (v0.8) and 14 (v0.10).
The unbiased-sample estimates rest on 1,710 windows with 67 to 86 positives per
class, so decile-level rates carry real uncertainty; the round-robin dilution and
the threshold arithmetic are exact computations and do not. Nothing here shows
that bad-case similarity is useless in general — only that in this
representation, with this bank construction, it was anti-predictive.

The Development-informed selector in section 8 is one experiment at one budget;
its near-zone regression is reported because it happened, not because one run
can support a confident mechanism. Sections 10-12 supersede the three-seed
readings quoted in sections 2, 3, and 8 wherever they disagree: the three-seed
spreads were unreliable. Section 13 in turn bounds section 12: the corridor
effect is a Development result that did not reproduce on held-out data, so it
must never be quoted as the project's outcome. The project's outcome is now
twofold: the v0.8 negative in section 13, and the v0.10 positive in section 14,
which stands on a pre-registered criterion, a decoupled held-out, and a clean
integrity audit — quote them together, since the second exists only because the
first was diagnosed honestly. The `N=600` downturn remains unexplained, and the
two explanations offered for the failed transfer are untested alternatives.
