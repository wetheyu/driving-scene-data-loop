# Bad-Case Process

Status: Planned

## Purpose

A bad case is a reproducible input for which actual behavior differs from the
expected behavior or violates a documented system constraint. It is evidence for
debugging and evaluation, not an anecdote to be removed from a demo.

## Taxonomy

### Data

- Missing, incorrect, duplicated, or leaked annotation
- Timestamp or segmentation error
- Dataset-conversion error
- Train/test contamination

### Query Planning

- Entity, spatial predicate, temporal constraint, or failure condition misparsed
- Unsupported request accepted as supported
- Valid request rejected
- Invalid SpTA generated

### Retrieval

- Vocabulary mismatch
- Dense semantic confusion
- Hard negative ranked too highly
- Correct candidate retrieved but removed by reranking
- Index version or filtering error

### Perception and VLM

- Missed object or relation
- Hallucinated object or event
- Wrong evidence frame
- Poor calibration or unsupported explanation

### Formal Verification

- Input conversion error
- Unsupported dialect or malformed specification
- Timeout, crash, or parse failure
- Correct monitor result applied to the wrong stream or query

### Agent and Tool Use

- Wrong, repeated, irrelevant, or unauthorized tool call
- Context omission or stale memory
- Tool result ignored or contradicted
- Prompt injection followed
- Unsupported final claim

### System

- Excessive latency, memory, token use, or cost
- Partial failure hidden as success
- Missing provenance or non-reproducible run

## Triage Workflow

```text
Capture
  -> Reproduce
  -> Locate the failing layer
  -> Form a testable hypothesis
  -> Apply the smallest justified fix
  -> Add a regression test
  -> Rerun local and end-to-end metrics
  -> Record benefits and regressions
```

Do not modify a prompt before checking whether the root cause is data, retrieval,
schema, authorization, or tool integration.

## Bad-Case Record

```yaml
id: BC-0001
status: open
run_id: run_0042
query_id: query_001
segment_ids: [scene_031_000120_000160]
expected: "retrieve and verify the occlusion-reappearance interval"
actual: "correct segment removed by reranker"
layer: reranker
severity: high
reproduction_command: "to be added"
root_cause: "not confirmed"
fix: null
regression_test: null
metric_effects: {}
notes: []
```

## Root-Cause Standard

A root cause must explain the failure mechanism and predict a change in behavior.
Descriptions such as "the model is bad" or "the prompt needs improvement" are not
sufficient.

Examples:

- The reranker was trained only on object-class overlap and learned to ignore the
  temporal phrase "within two seconds."
- A generated scene description included test-only failure labels, leaking the
  answer into retrieval.
- The Agent passed a filesystem path instead of a registry reference, and the
  formal tool correctly rejected it.

## Regression Policy

- Add the smallest unit or integration test that reproduces the confirmed failure.
- Add the case to a held-out bad-case suite when it represents real usage.
- Do not move a test case into training without creating a replacement holdout.
- Rerun affected layer metrics and end-to-end metrics.
- Keep fixes that improve one metric but damage another visible in the report.

## Review Record

For each important bad case, be ready to explain:

1. How it was discovered.
2. Why the first hypothesis was right or wrong.
3. Which evidence located the failing layer.
4. What was changed and why.
5. How the regression test works.
6. What tradeoff or remaining limitation exists.

