# Multimodal Driving Log Search and Failure Analysis Agent

Status: Planning

This project lets an engineer describe a driving scenario in natural language,
retrieve matching log intervals, verify spatial and temporal conditions, and
analyze why a perception, retrieval, or agent component failed.

A Python 3.12 and `uv` development baseline now exists. No runtime business
code, dataset, model, or experiment result has been created yet.

## Problem

Driving logs contain images, timestamps, object annotations, model predictions,
and vehicle metadata. Engineers need to find complex events such as:

```text
Find intervals where a pedestrian is occluded by a vehicle, becomes visible
again within two seconds, and is still missed by the detector.
```

Manual review is slow. Keyword search does not capture temporal relationships.
VLM output can be incomplete or unsupported. Exact formal monitoring requires a
structured specification that many users cannot write directly.

## Planned Workflow

```text
Natural-language query
  -> query planning and validation
  -> BM25, dense, and hybrid retrieval
  -> optional VLM or model-output inspection
  -> Strem-SpTA formal pattern verification
  -> conflict and failure analysis
  -> evidence-linked intervals and report
```

## Primary Outputs

- Ranked candidate scenes and time intervals
- The generated structured query or SpTA specification
- Evidence frames, object identifiers, and retrieval scores
- Formal match results and monitor diagnostics
- Failure-layer classification and reproducible bad-case records
- Search, model, agent, latency, and cost metrics

## Learning Goal

The project is designed for a beginner preparing for ML, deep-learning, search,
AI Agent, and data-analysis reviews. Each milestone follows this cycle:

```text
Learn -> Implement -> Test -> Measure -> Analyze bad cases -> Explain
```

Code completion alone is not sufficient. The author must be able to reproduce
the result, explain the design choice, identify limitations, and answer follow-up
questions without inventing experience.

## Relationship to Strem-SpTA

The thesis implementation remains a separate Rust project. This project will
consume a fixed version of Strem-SpTA as an external verification skill. It will
not copy the thesis source code or report thesis experiments as new results.

## Initial Scope

- One user, one orchestration agent, and a small public driving-log corpus
- Natural-language search over annotated scene segments
- BM25 baseline followed by dense, hybrid, and reranking experiments
- Strem-SpTA integration through a tested adapter and later an MCP tool
- Structured evaluation and bad-case regression

## Out of Scope Initially

- Vehicle steering, braking, or closed-loop autonomous driving
- Training an object detector or VLM from scratch
- Multi-agent coordination
- PB-scale infrastructure claims
- Full pretraining or full-scale RLHF
- Production deployment or unrestricted public access

## Documentation

- [Project scope and input/output](docs/PROJECT_SCOPE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Strem-SpTA skill contract](docs/STREM_SKILL.md)
- [Data specification](docs/DATA_SPEC.md)
- [Evaluation plan](docs/EVALUATION_PLAN.md)
- [Bad-case process](docs/BAD_CASE_PROCESS.md)
- [Learning path](docs/STAGE_PLAN.md)
- [Development environment](docs/DEVELOPMENT_ENVIRONMENT.md)
- [Roadmap](docs/ROADMAP.md)
- [Job alignment and review evidence](docs/ALIGNMENT.md)
- [Instructions for AI coding agents](AGENTS.md)
