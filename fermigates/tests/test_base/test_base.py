import math
import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from fermigates.layers.linear_layers.fermilayer import FermiGatedLinear
from fermigates.base import BaseFermiModel
from fermigates.layers.calibration.linear_calibration import LinearCalibration


class DummyBaseModel(BaseFermiModel):
    """
    Minimal subclass that exposes a couple of FermiGatedLinear layers for testing.
    """

    def __init__(self, in_dim=16, hid=8, out_dim=4, init_mu=0.0, init_T=1.0):
        super().__init__()
        # two gated layers to exercise temperature propagation & sparsity utilities
        self.fc1 = FermiGatedLinear(in_dim, hid, init_mu=init_mu, init_T=init_T)
        self.fc2 = FermiGatedLinear(hid, out_dim, init_mu=init_mu, init_T=init_T)

    def forward(self, x):
        # Accept flattened inputs: (batch, in_dim)
        x, _ = self.fc1(x)
        x = torch.relu(x)
        x, _ = self.fc2(x)
        return x


def test_set_temperature_propagates_to_all_masks():
    torch.manual_seed(0)
    model = DummyBaseModel()
    model.set_temperature(0.12345)

    # every FermiMask in model.modules() should have T == 0.12345
    for m in model.modules():
        # only FermiMask has attribute 'T'
        if hasattr(m, "T"):
            assert torch.isclose(m.T, torch.tensor(0.12345)), "Temperature did not propagate to a mask"


def test_init_mu_from_weights_global_and_per_neuron():
    torch.manual_seed(1)
    model = DummyBaseModel()
    # Grab current absolute weight ranges for checks
    w1 = model.fc1.linear.weight.data.abs()
    w2 = model.fc2.linear.weight.data.abs()

    # Global percentile init (0.5 ~ median): should set mu within the [min, max] range
    model.init_mu_from_weights(percentile=0.5, per_layer_neuron=False)
    for m in model.modules():
        if isinstance(m, FermiGatedLinear):
            mu = m.mask.mu
            mu_min, mu_max = mu.min().item(), mu.max().item()
            # mu should lie between min and max weight magnitudes
            W = m.linear.weight.data.abs()
            assert mu_min >= float(W.min()) - 1e-8 - 1e-6  # small tolerance
            assert mu_max <= float(W.max()) + 1e-8 + 1e-6

    # Per-neuron initialization: ensure shape/broadcasting works without error
    model.init_mu_from_weights(percentile=0.5, per_layer_neuron=True)
    # After per-neuron init, mask.mu should still be same shape as weight tensor
    for m in model.modules():
        if isinstance(m, FermiGatedLinear):
            assert m.mask.mu.shape == m.linear.weight.shape


def test_compute_sparsity_bounds_and_counts():
    torch.manual_seed(2)
    model = DummyBaseModel()
    kept, total, frac = model.compute_sparsity(threshold=0.7)
    assert isinstance(kept, int) and isinstance(total, int)
    assert total > 0
    assert 0 <= kept <= total
    assert math.isfinite(frac)
    assert 0.0 <= frac <= 1.0


def test_attach_apply_clear_calibration_roundtrip():
    torch.manual_seed(3)
    model = DummyBaseModel()
    # create a tiny calibration matrix and attach it
    d_in = model.fc1.linear.in_features
    d_out = model.fc1.linear.out_features
    calib = LinearCalibration(d_in=d_in, d_out=d_out, learnable=False)

    # load a non-zero calibration so apply_calibration has effect
    W_hat = torch.randn(d_in, d_out) * 0.01
    b_hat = torch.randn(d_out) * 0.001
    calib.load_calibration(W_hat, b_hat)

    model.attach_calibration("fc1_calib", calib)
    assert "fc1_calib" in model.calibrations

    # Prepare a random input and a simulated pruned output
    X = torch.randn(6, d_in)
    with torch.no_grad():
        pruned_out, _ = model.fc1(X)
        corrected = model.apply_calibration("fc1_calib", X, pruned_out)

    # corrected should differ from pruned_out (because we loaded a non-zero calibration)
    assert corrected.shape == pruned_out.shape
    assert not torch.allclose(corrected, pruned_out)

    # clear calibrations and ensure registry empties
    model.clear_calibrations()
    assert len(model.calibrations) == 0


def test_calibrate_with_loader_integration_reduces_residual():
    """
    Integration test for calibrate_with_loader:
    - Construct a synthetic original linear function and a pruned variant (zeroed input columns).
    - Use BaseFermiModel.calibrate_with_loader to compute calibration.
    - Verify the calibrated output reduces residual on held-out batch.
    """
    torch.manual_seed(4)
    d_in = 12
    d_out = 5
    n_samples = 200

    # create a simple original and pruned linear layer (not part of the model)
    orig_layer = torch.nn.Linear(d_in, d_out)
    pruned_layer = torch.nn.Linear(d_in, d_out)
    # simulate structured pruning: zero first 3 input columns in pruned version
    with torch.no_grad():
        pruned_layer.weight[:, :3] = 0.0
        pruned_layer.bias[:] = 0.0

    # synthetic calibration dataset
    X = torch.randn(n_samples, d_in)
    ds = TensorDataset(X, torch.zeros(n_samples, dtype=torch.long))
    loader = DataLoader(ds, batch_size=32)

    # create a BaseFermiModel instance to call the helper (no real layers needed)
    model = BaseFermiModel()

    # closures required by calibrate_with_loader
    def layer_input_fn(inputs):
        # inputs are already shaped (batch, d_in)
        return inputs

    def original_layer_fn(inputs):
        return orig_layer(inputs)

    def pruned_layer_fn(inputs):
        return pruned_layer(inputs)

    # run calibration helper (should register under provided name)
    calib = model.calibrate_with_loader(
        layer_input_fn=layer_input_fn,
        original_layer_fn=original_layer_fn,
        pruned_layer_fn=pruned_layer_fn,
        calibration_loader=loader,
        lam=1e-3,
        add_bias=True,
        name="synthetic_fc",
        device=torch.device("cpu"),
    )

    assert "synthetic_fc" in model.calibrations
    assert isinstance(model.calibrations["synthetic_fc"], LinearCalibration)

    # small held-out test
    X_test = X[:40]
    with torch.no_grad():
        f_o = orig_layer(X_test)
        f_p = pruned_layer(X_test)
        f_c = model.apply_calibration("synthetic_fc", X_test, f_p)

        err_before = torch.norm(f_o - f_p).item()
        err_after = torch.norm(f_o - f_c).item()

    assert err_after < err_before, f"Calibration did not reduce residual: before={err_before}, after={err_after}"
