# Strem-SpTA Skill Contract

Status: Planned

## Purpose

Use the thesis monitor as a deterministic spatial and temporal verification tool
inside a new Python search and Agent system. The new project does not reimplement
the monitor or treat it as an LLM.

## Ownership Boundary

The separate thesis repository owns:

- Timed perception-stream and SpTA semantics
- Rust implementation and monitor behavior
- SpTA JSON dialect
- Formal and implementation tests
- Thesis-specific experiments and results

This project owns:

- Natural-language query planning
- Search, ranking, and evidence selection
- Input adaptation and safe process invocation
- MCP exposure, authorization, timeout, and error handling
- End-to-end evaluation and failure analysis

## Tool Contract

Logical tool name:

```text
verify_spatial_timed_pattern
```

Input:

```json
{
  "stream_ref": "stream_0012",
  "spta_spec": {},
  "timeout_seconds": 10,
  "request_id": "request_0042"
}
```

Output:

```json
{
  "status": "success",
  "matched": true,
  "intervals": [[12.4, 13.8]],
  "constraints": [],
  "runtime_ms": 48,
  "monitor_version": "git:<commit>",
  "warnings": [],
  "error": null
}
```

The adapter resolves `stream_ref` through an allowlisted data registry. The agent
must never submit an unrestricted local path.

## Integration Stages

### Stage 1: Fake Monitor

A deterministic test double returns known results. This teaches interfaces and
allows the rest of the pipeline to be tested before external integration.

### Stage 2: CLI Adapter

The adapter invokes a pinned `strem` executable, captures stdout/stderr, applies a
timeout, parses structured output, and maps process failures to typed errors.

### Stage 3: Service Boundary

The adapter is exposed through a local service or MCP server with explicit input
schemas, authorization, limits, health checks, and version reporting.

### Stage 4: Agent Skill

A higher-level skill documents when the Agent should formulate an SpTA query,
when retrieval evidence is insufficient, and how to communicate monitor failures.

## Required Validation

- Stream reference exists and is allowlisted.
- SpTA JSON passes schema and supported-dialect validation.
- Requested timeout is within configured bounds.
- Input size and expected output size are limited.
- The process cannot access arbitrary credentials or files.
- The returned monitor version is recorded.
- Unknown output formats fail closed and remain visible in the run record.

## Guarantee Boundary

A successful monitor result means that the provided structured perception stream
matches the provided SpTA under the monitor's implemented semantics. It does not
prove that:

- The camera image was perceived correctly.
- Dataset annotations are complete.
- The LLM generated the intended specification.
- Conditions between sampled frames are known.
- The full driving system is safe.

These boundaries must appear in documentation and written explanations.

## Evaluation

Compare at least:

- Retrieval only
- Retrieval plus LLM explanation
- Retrieval plus Strem-SpTA verification
- Full pipeline with evidence and conflict checks

Measure interval retrieval quality, unsupported-claim rate, timeout/failure rate,
end-to-end latency, and the number of false rejections introduced by verification.

## Versioning

- Pin the monitor by release tag or Git commit.
- Record the executable hash in experiment metadata when practical.
- Keep thesis changes in the thesis repository.
- Upgrade the dependency through an explicit compatibility test.
- Never present prior thesis runtime results as new project measurements.

