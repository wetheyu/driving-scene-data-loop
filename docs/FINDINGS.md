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

Stated in long-tail terms, this task has two tails. The scenarios themselves are
the first: 3.50%, 3.78%, and 5.36% of L0 windows are positive. Inside those
positives sits a second tail of atypical views — rain-blurred, occluded, unusual
geometry — and the bank's representative rule targets exactly that second tail.

Retrieval works on the first tail and fails on the second, because typical rare
scenes resemble each other while atypical ones do not resemble anything in
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

## Claim boundary

These are Development and selection observations on one dataset, one
representation, and one downstream model. No Held-out Test number exists yet.
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
