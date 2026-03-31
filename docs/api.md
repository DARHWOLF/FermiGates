# API Reference

## Gates

- `FermiGate(shape, init_mu=0.0, init_T=1.0)`
- `BinaryConcreteGate(shape, init_log_alpha=0.0, init_T=1.0)`
- `HardConcreteGate(shape, init_log_alpha=0.0, init_T=2/3, lower=-0.1, upper=1.1)`
- `MagnitudeGate(shape, scores=None, inverted=False)`
- `GroupLassoGate(groups, group_size=1, init=0.0)`
- `GompertzGate(size, alpha=2.0, beta=1.0, learn_alpha=False)`

All gates support:
- `probabilities(...)`
- `set_temperature(T)` (where meaningful)
- `hard_mask(...)`
- `l0_penalty(...)`

## Layers

- `FermiGatedLinear(in_features, out_features, bias=True, init_mu=0.0, init_T=1.0, gate=None)`
- `FermiGatedConv2d(...)`
- `FermiMLPBlock(d_in, d_hidden, d_out, dropout=0.0, activation="gelu")`
- `FermiResidualBlock(dim, expansion=4, dropout=0.0, activation="gelu")`
- `FermiTransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1)`

## Models

- `FermiMLPClassifier(input_dim, num_classes, hidden_dims=(256, 128), ...)`
- `FermiConvClassifier(in_channels, num_classes, channels=(32, 64, 96), ...)`
- `FermiTransformerClassifier(vocab_size, num_classes, max_seq_len=256, ...)`

## Losses

- `fermi_informed_loss(task_loss, probabilities, energies, ...)`
- `fermi_free_energy_loss(probabilities, energies, interaction=None, temperature=1.0, ...)`
- `binary_entropy_loss(probabilities, reduction="sum")`
- `sparsity_l1_loss(probabilities, reduction="sum")`
- `budget_penalty_loss(probabilities, target, target_is_fraction=True)`
- `consistency_loss(probabilities, previous_probabilities, norm="l2", reduction="mean")`
- `kl_to_bernoulli_prior_loss(probabilities, prior_prob=0.5, ...)`
- `group_sparsity_l21_loss(values, group_dim=0, ...)`
- `hoyer_sparsity_loss(values, normalized=False, ...)`

## Training utilities

- `AnnealingSchedule(start, end, total_steps, mode=...)`
- `FermiAnnealingPlan(temperature=..., lambda_free_energy=..., budget_target=...)`
- `AdaptiveBudgetController(target_fraction_kept, lambda_budget=..., ...)`

## Export & reporting

- `to_hard_masked_model(model, threshold=0.5)`
- `hard_masked_state_dict(model, threshold=0.5)`
- `pruning_report(model, threshold=0.5, example_inputs=...)`

## Metrics

- `collect_gate_metrics(model, threshold=0.5)`
- `free_energy_components(probabilities, energies, interaction=None, temperature=1.0)`
- `MetricsTracker`

## Base utilities

`BaseFermiModel` provides:
- `set_temperature(T_new)`
- `init_mu_from_weights(percentile=0.5, per_layer_neuron=False)`
- `compute_sparsity(threshold=0.5)`
- `solve_ridge_cpu(X, E, lam=1e-3, add_bias=True)`
- `calibrate_with_loader(...)`
- `apply_calibration(name, X_flat, y_pruned)`
- `clear_calibrations()`
