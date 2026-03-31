# Architecture

## Design goals

FermiGates is built around composable abstractions:
- Gates define probability maps in `[0, 1]`
- Layers apply gates to weights
- Models aggregate layers and expose global utilities

## Core abstractions

- `BaseGate`
  - Unified interface for gate probabilities, hard masks, and temperature control.
- `BaseFermiLayer`
  - Any layer that can expose gate probabilities and sparsity accounting.
- `BaseFermiModel`
  - Traverses registered gates/layers to provide global temperature and sparsity utilities.
  - Provides calibration helpers (`solve_ridge_cpu`, `calibrate_with_loader`, `apply_calibration`).
- `BaseFermiBackbone` and `BaseFermiClassifier`
  - Reusable templates for building custom models quickly.

## Data flow

1. A layer owns a dense weight tensor.
2. A gate converts either weights or internal logits to soft probabilities.
3. Effective weights are computed as element-wise product of dense weights and gate probabilities.
4. Forward uses effective weights.
5. Temperature can be annealed model-wide for sharper masks.

## Calibration flow

After pruning, the model can estimate residual error with linear ridge calibration:

- Build residual targets: `E = f_original(X) - f_pruned(X)`
- Solve ridge for `W_hat` (and optional `b_hat`)
- Apply correction: `f_calibrated(X) = f_pruned(X) + X @ W_hat + b_hat`

This provides fast post-pruning correction without full retraining.
