# tests/test_calibration.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fermigates.base import BaseFermiModel
from fermigates.layers.calibration.linear_calibration import LinearCalibration


def _make_synthetic_linear_pair(
    n_samples=256,
    d_in=20,
    d_out=10,
    prune_fraction=0.3,
    seed=0,
    include_bias=True,    # NEW: control whether original layer has a bias
):
    """
    Create a synthetic 'original' linear mapping and a 'pruned' version (some input columns zeroed).
    Returns:
      X: (n_samples, d_in)
      f_orig: (n_samples, d_out)
      f_pruned: (n_samples, d_out)
      W_orig, W_pruned, b_orig, b_pruned
    """
    torch.manual_seed(seed)
    X = torch.randn(n_samples, d_in, dtype=torch.float32)

    W_orig = torch.randn(d_in, d_out, dtype=torch.float32)
    if include_bias:
        b_orig = torch.randn(d_out, dtype=torch.float32) * 0.1
    else:
        b_orig = torch.zeros(d_out, dtype=torch.float32)

    # Create a pruned weight matrix by zeroing out a fraction of input columns (structured pruning)
    mask = torch.ones(d_in, dtype=torch.bool)
    k = int(prune_fraction * d_in)
    if k > 0:
        zero_idx = torch.randperm(d_in)[:k]
        mask[zero_idx] = False

    W_pruned = W_orig.clone()
    W_pruned[~mask, :] = 0.0
    b_pruned = torch.zeros_like(b_orig)  # simulate pruned layer bias reset

    f_orig = X @ W_orig + b_orig.unsqueeze(0)
    f_pruned = X @ W_pruned + b_pruned.unsqueeze(0)

    return X, f_orig, f_pruned, W_orig, W_pruned, b_orig, b_pruned


def test_ridge_calibration_exact_recovery_no_bias():
    """
    When the original mapping has NO bias (b_orig == 0) and the pruned mapping is linear,
    the linear calibration (no bias term) should recover the residual essentially exactly.
    """
    X, f_orig, f_pruned, W_orig, W_pruned, b_orig, b_pruned = _make_synthetic_linear_pair(
        n_samples=300,
        d_in=24,
        d_out=12,
        prune_fraction=0.25,
        seed=42,
        include_bias=False,   # ensure no bias so add_bias=False is appropriate
    )

    E = (f_orig - f_pruned)  # residual to be fit

    # Solve ridge on CPU using BaseFermiModel helper (stable); no bias term
    W_hat, b_hat = BaseFermiModel.solve_ridge_cpu(X, E, lam=1e-8, add_bias=False)

    # Build calibration module and load solved parameters
    calib = LinearCalibration(d_in=X.shape[1], d_out=E.shape[1], learnable=False)
    calib.load_calibration(W_hat.to(calib.W.device), b_hat=None)

    # Apply calibration to a held-out subset and measure residual norms
    X_test = X[:64]
    f_orig_test = f_orig[:64]
    f_pruned_test = f_pruned[:64]

    with torch.no_grad():
        correction = calib(X_test)            # X @ W_hat
        f_calibrated = f_pruned_test + correction

        err_pruned = torch.norm((f_orig_test - f_pruned_test))
        err_cal = torch.norm((f_orig_test - f_calibrated))

    print(f"pre residual: {err_pruned.item():.6f}, post residual: {err_cal.item():.6f}")
    # Expect near exact recovery (numerical tolerance)
    assert err_cal < 1e-6 + 1e-6 * max(1.0, err_pruned)


def test_ridge_calibration_with_bias():
    """
    Test the ridge solver with an explicit bias term. When add_bias=True, the solver should
    recover both W_hat and b_hat so that f_pruned + X@W_hat + b_hat == f_orig (on calibration data).
    """
    X, f_orig, f_pruned, W_orig, W_pruned, b_orig, b_pruned = _make_synthetic_linear_pair(
        n_samples=320,
        d_in=16,
        d_out=8,
        prune_fraction=0.4,
        seed=1,
        include_bias=True,   # now include bias so solver must recover it
    )

    # Alter b_pruned to simulate common bias-reset after pruning
    b_pruned = 0.5 * b_orig  # not zero, just different
    f_pruned = X @ W_pruned + b_pruned.unsqueeze(0)

    E = (f_orig - f_pruned)  # residual

    # Solve ridge with bias
    W_hat, b_hat = BaseFermiModel.solve_ridge_cpu(X, E, lam=1e-4, add_bias=True)

    # Load into calibration module
    calib = LinearCalibration(d_in=X.shape[1], d_out=E.shape[1], learnable=False)
    calib.load_calibration(W_hat.to(calib.W.device), b_hat.to(calib.W.device))

    # Evaluate on held-out data
    X_test = X[50:120]
    f_orig_test = f_orig[50:120]
    f_pruned_test = f_pruned[50:120]

    with torch.no_grad():
        correction = calib(X_test)             # X @ W_hat + b_hat
        f_calibrated = f_pruned_test + correction

        err_pruned = torch.norm(f_orig_test - f_pruned_test)
        err_cal = torch.norm(f_orig_test - f_calibrated)

    assert err_cal < 1e-6 + 1e-6 * max(1.0, err_pruned)


def test_calibrate_with_loader_integration(tmp_path):
    """
    End-to-end integration: use BaseFermiModel.calibrate_with_loader helper which expects
    user-provided closures for extracting layer inputs and outputs on each batch.
    This test verifies the helper collects X and E and registers a calibration module.
    """
    class DummyModel(BaseFermiModel):
        def __init__(self, d_in=20, d_out=10):
            super().__init__()
            # use simple linear layers for the test
            self.fc = torch.nn.Linear(d_in, d_out)

        def forward(self, x):
            return self.fc(x)

    # Create dummy original and 'pruned' callables using closures over local dense/pruned layers
    d_in, d_out = 12, 6
    torch.manual_seed(0)
    orig_layer = torch.nn.Linear(d_in, d_out)
    pruned_layer = torch.nn.Linear(d_in, d_out)
    # simulate pruning by zeroing some columns of weight of pruned_layer
    pruned_layer.weight.data[:, :3] = 0.0
    pruned_layer.bias.data.zero_()

    # calibration dataset (no labels needed)
    X = torch.randn(200, d_in)
    ds = TensorDataset(X, torch.zeros(len(X), dtype=torch.long))
    loader = DataLoader(ds, batch_size=32)

    model = DummyModel(d_in=d_in, d_out=d_out)

    # closures that capture the batch and call the appropriate layer
    def make_layer_input_fn():
        def fn(inputs):
            # inputs already flattened for this dummy example
            return inputs
        return fn

    def make_orig_fn():
        def fn(inputs):
            return orig_layer(inputs)
        return fn

    def make_pruned_fn():
        def fn(inputs):
            return pruned_layer(inputs)
        return fn

    # run calibration helper
    calib = model.calibrate_with_loader(
        layer_input_fn=make_layer_input_fn(),
        original_layer_fn=make_orig_fn(),
        pruned_layer_fn=make_pruned_fn(),
        calibration_loader=loader,
        lam=1e-3,
        add_bias=True,
        name="fc_calib",
    )

    # helper should have stored calibration module under name
    assert "fc_calib" in model.calibrations
    assert isinstance(model.calibrations["fc_calib"], LinearCalibration)

    # Quick sanity: applying calibration reduces the residual on a small batch
    X_test = X[:40]
    with torch.no_grad():
        f_o = orig_layer(X_test)
        f_p = pruned_layer(X_test)
        f_c = model.apply_calibration("fc_calib", X_test, f_p)

        err_before = torch.norm(f_o - f_p)
        err_after = torch.norm(f_o - f_c)

    assert err_after < err_before, "Calibration did not reduce residual in integration test"