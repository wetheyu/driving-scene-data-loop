# Architecture

Status: Planned

## Design Goal

Build a measurable search and analysis pipeline in which probabilistic models
propose candidates and a deterministic monitor verifies structured conditions.
Keep each layer replaceable so that failures can be attributed correctly.

## Component View

```text
CLI / Web UI
      |
Query API
      |
LangGraph Orchestrator
  |        |          |             |
Planner  Retrieval  Perception   Formal Verification
           |          / VLM       Skill Adapter
           |                         |
     BM25 / Dense /                 Strem-SpTA
     Hybrid / Reranker
           |
Scene and Evidence Store ---- Evaluation and Bad-Case Store
```

## Main Components

### Ingestion

- Converts selected public dataset records into a stable internal schema.
- Segments logs into searchable units.
- Preserves timestamps, annotations, provenance, and model-version metadata.
- Produces human-readable scene documents without exposing test labels.

### Search Index

- Lexical baseline: TF-IDF and BM25.
- Semantic baseline: sentence embeddings and vector search.
- Hybrid retrieval: documented score fusion or reciprocal-rank fusion.
- Reranker: cross-encoder or comparable learned relevance model.

### Query Planner

- Extracts entities, spatial predicates, temporal constraints, and model filters.
- Starts with deterministic parsing and a restricted query language.
- Later compares prompting and a fine-tuned structured-output model.
- Produces a validated plan; it does not execute tools directly.

### Perception Inspector

- Reads dataset annotations and model predictions.
- Optionally invokes a VLM on selected evidence frames.
- Produces structured observations with confidence and provenance.
- Never treats VLM text as ground truth.

### Strem-SpTA Adapter

- Converts validated plan elements to a supported SpTA specification.
- Invokes a pinned external monitor version.
- Parses matches, diagnostics, runtime, and failures.
- Enforces path, size, timeout, and schema restrictions.

### Orchestrator

- Executes a bounded workflow using LangGraph.
- Chooses retrieval, inspection, formal verification, and reporting steps.
- Maintains run state, evidence references, retries, and failure status.
- Uses one agent initially; multi-agent behavior requires a measured need.

### Verifier and Report Builder

- Checks that claims reference evidence or tool output.
- Detects conflicts among annotations, model predictions, VLM output, and monitor
  results.
- Returns uncertainty and unresolved conflicts instead of guessing.
- Generates structured output before rendering natural language.

## Core Interfaces

```text
parse_query(text) -> QueryPlan
retrieve(plan, top_k) -> list[Candidate]
inspect(candidate, plan) -> EvidenceBundle
formal_match(stream, spta, timeout) -> MonitorResult
verify(run_state) -> VerificationResult
render(result) -> Report
```

Interfaces will use typed schemas. Provider-specific SDK objects must not cross
component boundaries.

## Storage

### Initial

- Local raw-data references
- JSONL or Parquet derived records
- SQLite for early learning and tests
- Local vector index for the first dense baseline

### Later

- MySQL for scenes, queries, runs, metrics, and bad cases
- OpenSearch or another documented search service for deployment experiments
- Model artifacts and index snapshots stored outside source control

## Technology Order

1. Python standard library, typing, tests, and SQLite
2. Pandas and scikit-learn
3. BM25 and a local vector index
4. PyTorch and Transformers
5. LangGraph and MCP
6. FastAPI, MySQL, and model serving

The first implementation should not depend on every planned technology.

## Trust Boundaries

- User queries and retrieved text are untrusted.
- Dataset annotations are reference data, not automatically perfect truth.
- Model and VLM output is probabilistic evidence.
- Formal monitor results are valid only for the supplied stream and specification.
- The report builder may explain results but may not override tool output.
- Secrets and unrestricted local filesystem paths are never exposed to an agent.

## Observability

Every run records:

- Dataset, split, query, and index versions
- Planner, retriever, reranker, model, prompt, and monitor versions
- Tool inputs represented by safe references, not secret data
- Tool outputs, validation decisions, retries, and errors
- Per-stage and end-to-end latency
- Token and monetary cost where applicable

