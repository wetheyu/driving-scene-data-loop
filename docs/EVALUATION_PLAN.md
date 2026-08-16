# Evaluation Plan

Status: Planned

## Principle

Evaluate each layer independently before evaluating the full Agent. A good final
answer does not identify which component worked, and a bad final answer does not
identify which component failed.

## Query Planning

Baselines:

- Keyword and regular-expression parser
- Rule-based restricted query grammar
- Prompted structured-output LLM
- Optional fine-tuned query planner

Metrics:

- Valid structured-query rate
- Entity, predicate, and constraint precision/recall/F1
- SpTA compilation success rate
- Exact and semantic plan agreement
- Unsupported-constraint rate
- Latency and cost

## Retrieval

Baselines and comparisons:

- TF-IDF
- BM25
- Dense embedding retrieval
- Hybrid retrieval
- Cross-encoder reranking

Metrics:

- Recall@k
- MRR
- nDCG@k
- Precision@k where labels support it
- Index size and build time
- Mean and P95 query latency

Use BEIR only as an external implementation check. Project conclusions must come
from the driving-log query set and its documented labels.

## Perception and VLM Inspection

Metrics depend on the selected task and may include:

- Object-class precision, recall, and F1 against available annotations
- Attribute or relation accuracy
- Evidence-frame selection recall
- Unsupported-object and unsupported-relation rate
- Calibration and latency

A VLM explanation score alone is not a ground-truth metric.

## Strem-SpTA Adapter

Measure:

- Agreement with curated known-match and known-nonmatch cases
- Invalid-specification rejection rate
- Timeout and process-failure rate
- Output-parser correctness
- Runtime and memory for the project workload
- Compatibility across pinned monitor versions

## Agent and Tool Use

Measure:

- End-to-end task success
- Correct tool selection and argument validity
- Irrelevant or unauthorized tool-call rate
- Retry and fallback rate
- Evidence coverage of final claims
- Conflict-detection and unresolved-conflict rates
- Token usage, monetary cost, and stage latency

Function-calling evaluation patterns may be compared with BFCL. Prompt-injection
tests may be informed by AgentDojo, but project-specific attacks remain necessary.

## End-to-End Search and Analysis

Measure:

- Relevant interval Recall@k and nDCG@k
- Verified-match precision and recall
- Failure-layer classification Macro F1
- Unsupported-claim rate
- Human acceptance and correction rate on a held-out review set
- P50 and P95 end-to-end latency
- Cost per query

## Required Ablations

- BM25 versus dense versus hybrid versus reranked retrieval
- Rule parser versus prompted planner versus fine-tuned planner
- Retrieval with and without VLM evidence inspection
- Agent with and without Strem-SpTA verification
- Agent with and without claim-evidence validation
- Short context versus retrieved context

Change one controlled factor per ablation where possible.

## Reproducibility Record

Every reported experiment includes:

- Code revision
- Dataset, query-set, and split versions
- Index and retrieval configuration
- Model, adapter, prompt, and monitor versions
- Seeds and repeat count
- Hardware or API provider
- Failed-run count
- Metrics with uncertainty where appropriate

## Release Gates

Targets will be set after baselines exist. Before any public or resume claim:

- The evaluation command runs from a clean documented environment.
- Train/test leakage checks pass.
- Baseline and improved results use the same held-out data.
- Bad cases and regressions are preserved.
- Latency, cost, and failed runs are reported with quality metrics.

See [BAD_CASE_PROCESS.md](BAD_CASE_PROCESS.md) for failure handling.

