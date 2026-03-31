# FermiGates

A complete PyTorch toolkit for differentiable pruning with Fermi-inspired gates.

FermiGates includes:
- Multiple gate families (`Fermi`, `BinaryConcrete`, `HardConcrete`, `Magnitude`, `GroupLasso`, `Gompertz`)
- Fermi-enabled layers (`Linear`, `Conv2d`, MLP/Residual blocks, Transformer encoder block)
- Ready-to-use models (MLP, CNN, Transformer classifier)
- Calibration utilities for post-pruning residual correction
- Full tests, examples, and CI

## Installation

```bash
pip install -e .
pip install -e .[dev]
```

Or with `uv`:

```bash
uv sync
uv sync --all-extras
```

## Quickstart

```python
import torch
from fermigates.models import FermiMLPClassifier

model = FermiMLPClassifier(input_dim=32, hidden_dims=(64, 32), num_classes=4)
x = torch.randn(16, 32)
logits = model(x)

model.set_temperature(0.5)
kept, total, frac = model.compute_sparsity(threshold=0.5)
print(kept, total, frac)
```

## Main API

### Base classes
- `BaseFermiLayer`: abstract base for gated layers
- `BaseFermiModel`: temperature, sparsity, and calibration utilities
- `BaseFermiBackbone`: reusable feature backbone base
- `BaseFermiClassifier`: reusable classifier base

### Gates
- `FermiGate`
- `BinaryConcreteGate`
- `HardConcreteGate`
- `MagnitudeGate`
- `GroupLassoGate`
- `GompertzGate`

### Layers
- `FermiGatedLinear`
- `FermiGatedConv2d`
- `FermiMLPBlock`
- `FermiResidualBlock`
- `FermiTransformerEncoderLayer`
- `LinearCalibration`

### Models
- `FermiMLPClassifier`
- `FermiConvClassifier`
- `FermiTransformerClassifier`

### Losses
- `fermi_informed_loss`
- `fermi_free_energy_loss`
- `binary_entropy_loss`
- `sparsity_l1_loss`
- `budget_penalty_loss`
- `consistency_loss`
- `kl_to_bernoulli_prior_loss`
- `group_sparsity_l21_loss`
- `hoyer_sparsity_loss`

### Training
- `AnnealingSchedule`
- `FermiAnnealingPlan`
- `AdaptiveBudgetController`

### Metrics
- `collect_gate_metrics`
- `free_energy_components`
- `MetricsTracker`

### Export
- `to_hard_masked_model`
- `hard_masked_state_dict`
- `pruning_report`

## Examples

Runnable scripts:
- `examples/train_mlp_synthetic.py`
- `examples/train_conv_synthetic.py`
- `examples/transformer_toy_forward.py`
- `examples/train_mlp_fermi_objective.py`

Run:

```bash
python examples/train_mlp_synthetic.py
python examples/train_conv_synthetic.py
python examples/transformer_toy_forward.py
python examples/train_mlp_fermi_objective.py
```

## Tests

```bash
pytest
```

## Lint & format

```bash
ruff check .
black .
isort .
```

## Documentation

See:
- `docs/architecture.md`
- `docs/api.md`
- `docs/development.md`

## CI

GitHub Actions workflow is included at `.github/workflows/ci.yml` and runs:
- Ruff
- Pytest (Python 3.10, 3.11, 3.12)

## License

MIT
