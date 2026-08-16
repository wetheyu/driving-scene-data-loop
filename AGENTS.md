# Instructions for AI Coding Agents

These instructions apply to this project and every file below it.

## Read Before Editing

Read the following documents in order:

1. `docs/PROJECT_SCOPE.md`
2. `docs/STAGE_PLAN.md`
3. `docs/ARCHITECTURE.md`
4. `docs/STREM_SKILL.md`
5. `docs/DATA_SPEC.md`
6. `docs/EVALUATION_PLAN.md`
7. `docs/BAD_CASE_PROCESS.md`
8. `docs/ROADMAP.md`
9. `docs/ALIGNMENT.md`
10. `docs/DEVELOPMENT_ENVIRONMENT.md`

Do not silently change the product problem, input/output contract, data split,
evaluation protocol, or Strem-SpTA boundary.

## Language

- Use English for code, identifiers, comments, documentation, logs, and UI text.
- User-facing teaching and progress explanations may be in Chinese.
- Prefer precise technical wording over marketing language.
- Mark unimplemented functionality as `Planned`.

## Research-First Development

The user is learning from the beginning. Optimize for understanding and review
readiness as well as delivery.

- Break work into the smallest runnable and testable concept.
- Explain the problem before introducing a library or framework.
- Show the important command, execution path, output, and failure mode.
- Let the user predict, run, modify, or explain important code when practical.
- Do not generate a large advanced subsystem before prerequisites are learned.
- End each milestone with a knowledge check and written summary.
- Record genuine errors and bad cases instead of hiding them.

## Architecture Boundaries

- Retrieval, model inference, orchestration, formal verification, storage, and UI
  must have explicit interfaces.
- LLM and VLM output is untrusted and must pass schema and domain validation.
- The agent may propose an SpTA specification but cannot declare a formal match.
- Only the versioned Strem-SpTA adapter may return formal monitor results.
- Strem-SpTA remains an external dependency; do not copy its source into this repo.
- One orchestration agent is sufficient for the initial version.
- Every final claim must reference source data, evidence, or a tool result.

## Data and Evaluation

- Keep raw datasets, generated indexes, credentials, and model weights out of Git.
- Preserve dataset licenses and source provenance.
- Split by complete scene or log, not by frame or paraphrase alone.
- Never use test labels in prompts, retrieval documents, or training examples.
- Establish a baseline before adding a complex model.
- Report failed runs, latency, cost, and uncertainty where applicable.
- Do not invent performance values or resume claims.

## Security

- Load secrets from environment variables or ignored local configuration.
- Treat user queries, retrieved documents, scene text, and model output as
  untrusted content.
- Enforce tool permissions in code, not only in prompts.
- Apply input-size, timeout, path, and output-schema limits to external tools.
- Never persist hidden chain-of-thought; store structured decisions and evidence.
- Anonymize user-provided logs before exporting examples.

## Change Rules

- Update documentation with behavior-changing code.
- Add tests for normal, boundary, malformed, timeout, and unauthorized cases.
- Keep changes scoped to the current learning milestone.
- Add dependencies only with a documented reason and pinned version strategy.
- Do not modify the separate thesis repository unless explicitly requested.

## Definition of Done

A task is complete only when:

- The behavior and acceptance criteria are implemented.
- Relevant automated tests pass.
- The output remains reproducible and schema-valid.
- At least one likely failure case has been considered.
- Documentation and roadmap status are accurate.
- The author can explain the introduced concept and design tradeoff.

## Project Commands

- `uv sync --frozen`: reproduce the locked project environment.
- `uv run python --version`: verify the selected Python interpreter.
- `uv run ruff check .`: lint source and tests after they exist.
- `uv run mypy src tests`: type-check source and tests after they exist.
- `uv run pytest`: run automated tests after they exist.

Commands that depend on source or tests remain `Planned` until those files are
created.
