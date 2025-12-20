# Fermi-Gated Networks — README

> **Fermi-Dirac Statistical Masks: Unified framework for differentiable pruning and model design.**
> Learn soft, annealed masks via a physics-inspired gating function and recover pruning residuals with a light closed-form linear calibration — no full retraining required.

---
## What this is

A modular research codebase implementing:

* **Fermi-Dirac statistical masks** as differentiable retention probabilities per weight (or grouped by neuron/filter).
* **Fermi-gated layers** (`FermiGatedLinear`, `FermiGatedConv`) that multiply weights by learned probabilities.
* **Temperature annealing** to smoothly transition from soft masks to near-binary decisions.
* **Linear calibration**: closed-form ridge recovery that reconstructs the residual errors introduced by pruning using a small calibration dataset — eliminating or drastically reducing the need to retrain.
* **BaseFermiModel**: a single place for temperature control, μ initialization, sparsity reporting, and calibration plumbing so new models inherit the framework cleanly.

This repository is structured to be research-ready, reproducible, and extensible to CNNs / Transformers / graph models.

---

## Why it matters

* Enables **learned, end-to-end differentiable pruning** instead of brittle post-hoc magnitude rules.
* Integrates a **physics-inspired, interpretable control** (μ and T) for where/how to prune.
* Provides **closed-form calibration** that recovers pruned outputs cheaply — often avoiding expensive retraining.
* Facilitates efficient inference and lower energy footprints while preserving accuracy.

---

## Methodology

* Mask per weight:
  $$[
  P(w)=\frac{1}{\exp\big((|w|-\mu)/T\big)+1}
  ]$$
  Use $(\tilde w = P(w)\cdot w)$ in forward passes.
* Train weights and μ jointly. Anneal T from high → low.
* When T is low, threshold $(P(w))$ at 0.5 to hard-prune.
* Use ridge regression on a small calibration set to compute a linear correction $(\hat W)$ such that:
  $$[
  f_{\text{cal}}(X) = \widehat f(X) + X \hat W
  ]$$
  solves $$( \min_W |XW - (f(X)-\widehat f(X))|_F^2 + \lambda ||W||^2 )$$. Closed-form solution available.

---


## Project layout

```
FermiGates/
├── pyproject.toml
├── README.md
├── fermigates/               # core framework
│   ├── base/                 # houses base class for inhertance 
         |-- base.py
│   ├── masks/                # houses the implemented gating masks
        |-- fermimask.py        
│   ├── layers/
    |   |-- linear_layers/    # houses linear feed-forward layers 
            |--linear_layers.py
    │   ├── calibration/      # houses calibration layers
            |--linear_calibration.py        
    ├── models/               # houses different models  (fermimlp, fermitransformer,...)
    ├── tests/                # pytest files
    |-- examples/             # houses example useage files
```

---

## Quickstart

1. Create a virtual environment (preferably using astral-uv) and install in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```
If you are using astral-uv:

```bash
uv venv
uv sync
uv sync --all-extras
```

2. Run tests:

```bash
pytest fermigates/tests
```

3. Run an example experiment (MLP on MNIST):
(to:do)

```bash
python experiments/mlp_mnist.py --config configs/mlp.yaml
```

---

## Examples

### Train a gated MLP on MNIST

Example skeleton (see `experiments/mlp_mnist.py`):
(to:do)

```python
from fermi.base import BaseFermiModel
from models.mlp import GatedMLP  # subclass of BaseFermiModel

model = GatedMLP(init_mu=0.0, init_T=1.0).to(device)
model.init_mu_from_weights(percentile=0.5)    # sensible μ init
model.set_temperature(1.0)

for epoch in range(1, E+1):
    T = schedule(epoch)                       # linear/exponential anneal
    model.set_temperature(T)
    train_one_epoch(model, train_loader, optimizer, lambda_sparsity)
    acc = test(model, test_loader)
    log(epoch, T, acc, model.compute_sparsity())
```

### Apply linear calibration (closed-form)

Use `BaseFermiModel.calibrate_with_loader` or call `BaseFermiModel.solve_ridge_cpu`:

```python
# get closures that return (X_flat), f_orig(X), f_pruned(X) for the same batch
calib = model.calibrate_with_loader(layer_input_fn, orig_layer_fn, pruned_layer_fn,
                                    calibration_loader, lam=1e-3, add_bias=True, name="fc1_calib")

# apply during inference
y_pruned = pruned_layer(X_flat)
y_corrected = model.apply_calibration("fc1_calib", X_flat, y_pruned)
```

### Dynamic architecture shrinking (on-the-fly)

The shrinkable layers expose `.shrink(threshold)` to remove low-occupancy neurons/filters and rebuild downstream layers accordingly. Use shrinking once masks are sufficiently sharp to get real training/inference speedups.

---

## API reference (essential)

* `fermi.masks.FermiMask(shape, init_mu, init_T)`
  returns soft retention probabilities `P = mask(W)`.

* `fermi.layers.FermiGatedLinear(in_features, out_features, init_mu, init_T)`
  forward returns `(out, P)` where `out = F.linear(x, P*W, b)`.

* `fermi.base.BaseFermiModel`

  * `set_temperature(T)` — set temperature across all masks.
  * `init_mu_from_weights(percentile=0.5, per_layer_neuron=False)` — sensible μ init.
  * `compute_sparsity(threshold=0.5)` — returns `(kept, total, fraction_kept)`.
  * `calibrate_with_loader(layer_input_fn, original_layer_fn, pruned_layer_fn, calibration_loader, lam, add_bias, name)` — compute & store calibration.
  * `apply_calibration(name, X_flat, y_pruned)` — apply stored calibration.

* `fermi.calibration.LinearCalibration` — module: `forward(X)` returns `X @ W + b`.

See docstrings in `fermi/` for full details.

---

## How to cite

If you use this code in a paper, please cite the repository and the algorithm concept. A suggested citation block:

```
@software{fermi-gated-networks-2025,
  title = {Fermi-Gated Networks: A novel deep learning framework for efficient pruning and model design},
  author = {Raunak Dev},
  year = {2025},
  url = {https://github.com/DARHWOLF/FermiGates}
}
```

---

## Contributing

* Read `CONTRIBUTING.md` (if present). Keep PRs small and focused.
* Add tests for any new feature.
* Use pre-commit.

---

## License & contact

* Project licensed under `MIT` License
* For questions, issues, or collaboration: open an issue on the repository or email `raunak.r.dev@gmail.com`.


Which deliverable would you like next?
