# Development Environment

Status: Active

## Goal

Keep the global machine setup small and keep project dependencies isolated and
reproducible. A command succeeding because of an undeclared global package is a
failure, not a convenience.

## Decisions

- Homebrew manages machine-level command-line tools.
- `uv` manages Python selection, the project virtual environment, and the lock.
- The user-level `uv` fallback is Python `3.12`.
- This project pins Python `3.12.13` in `.python-version`.
- Python libraries and development tools are installed into `.venv`, not into a
  Homebrew or system Python.
- `uv.lock` is the exact dependency lock. Dependency upgrades must be explicit.
- The global Git default branch is `main`.

The shell command `python3` may still resolve to Python 3.14. Project commands
must use `uv run`, which selects the project interpreter and environment.

## Initial Commands

Create or reproduce the environment:

```bash
uv sync --frozen
```

Inspect the selected tools:

```bash
uv run python --version
uv run pytest --version
uv run ruff --version
uv run mypy --version
```

The following validation commands become required after source code and tests
exist:

```bash
uv run ruff check .
uv run mypy src tests
uv run pytest
```

Until those files exist, these validation commands are `Planned`; version checks
only prove that the environment was installed, not that application behavior is
correct.

## Global Rollback

Remove the user-level `uv` Python fallback:

```bash
uv python pin --global --rm
```

Remove the global Git default-branch setting:

```bash
git config --global --unset init.defaultBranch
```

Project files and `.venv` are independent of these global settings.

