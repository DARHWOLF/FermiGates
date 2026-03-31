# Development Guide

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Local checks

```bash
ruff check .
black --check .
isort --check-only .
pytest
```

## Project layout

- `fermigates/base`: base abstractions and mixins
- `fermigates/gates`: differentiable gate implementations
- `fermigates/layers`: gated layers and blocks
- `fermigates/models`: reference model implementations
- `fermigates/losses`: Fermi-informed and auxiliary training losses
- `fermigates/training`: annealing schedules and adaptive controllers
- `fermigates/metrics`: occupancy and free-energy metric tracking
- `fermigates/export`: hard-mask conversion and pruning reports
- `tests`: unit and integration tests
- `examples`: runnable demos

## Contributing

1. Add or update tests with each behavior change.
2. Prefer backward-compatible APIs when possible.
3. Keep modules typed and documented with concise docstrings.
