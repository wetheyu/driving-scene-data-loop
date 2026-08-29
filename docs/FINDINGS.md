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

## 8. Open hypotheses

Two candidate explanations for the null result remain, both pre-declared in
[Bad-Case Process](BAD_CASE_PROCESS.md) before the result was known, and each is
directly testable without changing the task, the split, or the labels:

1. **The budget is too small.** At `N=300` the methods differ by seven positive
   windows. Raising the budget scales that difference roughly proportionally.
2. **The representation is the ceiling.** Labels are defined by metric geometry
   in the ego ground plane, while the model input is one 384-dimensional global
   CLS vector per frame from a monocular camera. A pedestrian at 15 to 20 m is
   roughly 20 to 30 letterbox pixels tall, and global pooling discards where and
   how large it was. Spatial patch features would test this directly, and they
   need no Oracle information, so they remain a legitimate model input.

The second is the deeper limitation, and it caps what any selection method can
demonstrate here: adding windows containing an object the encoder cannot resolve
cannot help. Reporting this is more useful than reporting an unexplained gain.

## Claim boundary

These are Development and selection observations on one dataset, one
representation, and one downstream model. No Held-out Test number exists yet.
The unbiased-sample estimates rest on 1,710 windows with 67 to 86 positives per
class, so decile-level rates carry real uncertainty; the round-robin dilution and
the threshold arithmetic are exact computations and do not. Nothing here shows
that bad-case similarity is useless in general — only that in this
representation, with this bank construction, it was anti-predictive.
