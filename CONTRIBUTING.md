# Contributing to FermiGates

Thanks for contributing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Before opening a PR

```bash
ruff check .
black --check .
isort --check-only .
pytest
```

## Guidelines

- Keep PRs focused and small.
- Add tests for new behavior.
- Preserve backward compatibility for existing public APIs when possible.
- Include short doc updates for any user-facing change.
