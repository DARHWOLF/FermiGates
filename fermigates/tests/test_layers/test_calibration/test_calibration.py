import torch
from torch.utils.data import DataLoader, TensorDataset

from fermigates.base import BaseFermiModel
from fermigates.layers.calibration import LinearCalibration


def test_ridge_solver_recovers_linear_residual_without_bias():
    torch.manual_seed(0)
    n, d_in, d_out = 200, 12, 5

    X = torch.randn(n, d_in)
    W_res = torch.randn(d_in, d_out)
    E = X @ W_res

    W_hat, b_hat = BaseFermiModel.solve_ridge_cpu(X, E, lam=1e-8, add_bias=False)

    assert b_hat is None
    assert torch.allclose(W_hat, W_res, atol=1e-4, rtol=1e-4)


def test_calibrate_with_loader_reduces_error():
    torch.manual_seed(1)
    n, d_in, d_out = 240, 16, 6

    X = torch.randn(n, d_in)
    W_orig = torch.randn(d_in, d_out)
    W_pruned = W_orig.clone()
    W_pruned[:4, :] = 0.0

    b_orig = torch.randn(d_out) * 0.1
    b_pruned = torch.zeros(d_out)

    def f_orig(inp):
        return inp @ W_orig + b_orig

    def f_pruned(inp):
        return inp @ W_pruned + b_pruned

    ds = TensorDataset(X, torch.zeros(n, dtype=torch.long))
    loader = DataLoader(ds, batch_size=32)

    model = BaseFermiModel()
    calib = model.calibrate_with_loader(
        layer_input_fn=lambda inp: inp,
        original_layer_fn=f_orig,
        pruned_layer_fn=f_pruned,
        calibration_loader=loader,
        add_bias=True,
        name="fc",
    )

    assert isinstance(calib, LinearCalibration)
    assert "fc" in model.calibrations

    X_test = X[:40]
    with torch.no_grad():
        err_before = torch.norm(f_orig(X_test) - f_pruned(X_test))
        corrected = model.apply_calibration("fc", X_test, f_pruned(X_test))
        err_after = torch.norm(f_orig(X_test) - corrected)

    assert err_after < err_before
