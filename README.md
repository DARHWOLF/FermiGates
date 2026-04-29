# FermiGates
<div>
  <a href="https://www.github.com/DARHWOLF/FermiGates">
    <img src="logo/logo_blue.png" width="190" alt="FermiGates logo", align="left" />
  </a>

**FermiGates** is a PyTorch toolkit for differentiable sparsity with Fermi-inspired gating, practical training workflows, and export-ready pruning utilities. FermiGates feature:
</div>

- Differentiable gating with `FermiGate` for controllable sparsity during training.
- Ready-to-run classifier models for MLP, CNN, and Transformer use cases.
- Built-in sparsity tracking and occupancy metrics for gate behavior inspection.
- Post-pruning calibration and export helpers for deployment-oriented workflows.


## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e .[dev]
```

Using `uv`:

```bash
uv sync
uv sync --all-extras
```

## Quickstart

```python
import torch

from fermigates.gates import FermiGate
from fermigates.models import MLP

model = MLP(
    input_dim=32,
    hidden_dims=[64, 32],
    output_dim=4,
    gate=lambda: FermiGate(
        mode="weight",
        init_mu=-0.05,
        init_temperature=1.0,
    ),
)

x = torch.randn(16, 32)
logits = model(x)
kept, total, fraction_kept = model.compute_sparsity(threshold=0.5)

print(logits.shape)         # torch.Size([16, 4])
print(kept, total, fraction_kept)
```

## Example Workflows

Run the provided examples:

```bash
python examples/example_MLP_mnist.py
python examples/example_cnn_fashion_mnist_direct.py
python examples/example_transformer_cifar10.py
```

## Core Concepts

- **Weight gating** applies probabilities directly to layer weights; weight-gate sparsity is the primary structural signal.
- **Activation gating** applies probabilities to layer inputs or outputs; activation sparsity is the primary runtime signal.
- **Comparable sparsity** in experiments should follow the dominant gate location for the selected mode.
- Gate behavior depends strongly on initialization, especially `init_mu` and a strictly positive `init_temperature`.

## API Entry Points

- `fermigates.gates.FermiGate`
- `fermigates.models.MLP`, `fermigates.models.CNN`, `fermigates.models.Transformer`
- `fermigates.experiments.Experiment`
- `fermigates.losses` for Fermi-informed training terms
- `fermigates.metrics` for occupancy and free-energy tracking
- `fermigates.export` for hard-masked model export and pruning reports

For full API details, see [docs/api.md](docs/api.md).

## Documentation

- [Architecture](docs/architecture.md)
- [API Reference](docs/api.md)
- [Development Guide](docs/development.md)

## Quality Signals

- Automated tests with `pytest`
- Static analysis with `ruff`
- GitHub Actions CI across Python 3.10, 3.11, and 3.12

Run locally:

```bash
ruff check .
pytest
```

## License

Copyright (c) 2026 Raunak Dev. All rights reserved.
