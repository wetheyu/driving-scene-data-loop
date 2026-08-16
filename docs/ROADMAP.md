# Roadmap

Status: Planning

A phase is complete only when its code, tests, measurements, documentation, and
learning checkpoint are complete.

## Phase 0: Definition

- [x] Define the user problem and input/output contract.
- [x] Define architecture and trust boundaries.
- [x] Define the Strem-SpTA integration contract.
- [x] Define data, evaluation, bad-case, learning, and job-alignment plans.
- [ ] Review and freeze project specification version `0.1`.

Exit condition: the first implementation task requires no new product assumptions.

## Phase 1: Python, Tests, and Data Exploration

- [x] Inspect the local Python, Git, database, and package-manager environment.
- [ ] Create the package, dependency lock, lint, type-check, and test configuration.
- [ ] Implement the first typed scene and query records.
- [ ] Load a tiny public-data sample without committing raw data.
- [ ] Produce a data-quality and distribution notebook or report.
- [ ] Store scene and query metadata in SQLite.

Exit condition: one command validates data and tests from a clean environment.

## Phase 2: Search Baselines

- [ ] Define and review the first canonical query set.
- [ ] Implement TF-IDF and BM25 baselines.
- [ ] Implement query-level train, development, and test evaluation.
- [ ] Add hard negatives and inspect ranking failures.
- [ ] Record Recall@k, MRR, nDCG, index time, and query latency.

Exit condition: baseline results are reproducible and every metric is understood.

## Phase 3: Dense, Hybrid, and Reranking

- [ ] Add one embedding model and local vector index.
- [ ] Add a documented hybrid-fusion method.
- [ ] Add a pretrained cross-encoder reranker.
- [ ] Train or fine-tune one small relevance model if labels support it.
- [ ] Compare quality, latency, memory, and bad cases.

Exit condition: the selected search stack has a measured reason for each component.

## Phase 4: Query Planning and Formal Skill

- [ ] Define a restricted structured-query schema.
- [ ] Implement a deterministic parser baseline.
- [ ] Implement and evaluate an LLM structured-output parser.
- [ ] Implement the fake Strem monitor and adapter tests.
- [ ] Integrate a pinned Strem-SpTA CLI executable.
- [ ] Compare retrieval with and without formal verification.

Exit condition: valid queries produce evidence-linked monitor results, and all
failure modes are visible.

## Phase 5: Multimodal Failure Analysis

- [ ] Add selected evidence-frame loading and rendering.
- [ ] Compare dataset annotations with one perception model or VLM.
- [ ] Define failure-layer labels and review a small held-out set.
- [ ] Train a baseline failure classifier if labels are sufficient.
- [ ] Add conflict records and bad-case regressions.

Exit condition: the system distinguishes retrieval, perception, query, and monitor
failures on reviewed cases.

## Phase 6: Agent and MCP

- [ ] Implement a bounded LangGraph workflow.
- [ ] Expose search, evidence, and formal verification through typed tools.
- [ ] Wrap the Strem adapter as an MCP tool.
- [ ] Add code-enforced authorization, limits, retries, and audit records.
- [ ] Add claim-evidence validation and prompt-injection tests.

Exit condition: the Agent completes held-out tasks without bypassing tool or evidence
boundaries.

## Phase 7: Fine-Tuning and Serving

- [ ] Build and audit an SFT dataset for query planning or tool calling.
- [ ] Run a small LoRA experiment.
- [ ] Compare base and adapted models on held-out tasks.
- [ ] Serve one model behind a versioned local endpoint.
- [ ] Measure batching, latency, throughput, memory, and cost.

Exit condition: the training and serving experiment is reproducible and includes
regressions as well as improvements.

## Phase 8: Portfolio Release

- [ ] Move experiment metadata to MySQL and write analysis queries.
- [ ] Add a minimal UI for query, evidence, intervals, and bad cases.
- [ ] Add continuous integration and a secret scan.
- [ ] Publish architecture, evaluation, limitations, and demo documentation.
- [ ] Prepare role-specific resume bullets and written explanations.

Exit condition: a reviewer can install, run, evaluate, and question the project
using documented evidence.

## Later Extensions

- Cross-dataset evaluation
- Local Spark batch-processing experiment
- OpenSearch deployment and load testing
- CARLA or another closed-loop scenario source
- Multi-agent design only if a measured task requires it
- Preference training with a defensible label protocol

## Next Task

Review and freeze project specification version `0.1`, then create the first
small Python package and environment smoke test.
