import torch
from fermigates.layers.linear_layers.fermilayer import FermiGatedLinear


def test_fermi_gated_linear_forward_shape():
    layer = FermiGatedLinear(16, 8)
    x = torch.randn(4, 16)

    y, P = layer(x)

    assert y.shape == (4, 8)
    assert P.shape == layer.linear.weight.shape


def test_mask_application_effect():
    layer = FermiGatedLinear(4, 2, init_mu=100.0, init_T=0.01)
    x = torch.randn(1, 4)

    y_masked, _ = layer(x)

    # All weights effectively masked out
    assert torch.allclose(y_masked, torch.zeros_like(y_masked), atol=5e-1)


def test_initialize_mu_from_weights():
    layer = FermiGatedLinear(10, 5)
    layer.initialize_mu_from_weight_percentile(percentile=0.5)

    mu = layer.mask.mu
    w = layer.linear.weight.abs()

    # μ should lie within the weight range
    assert mu.min() >= w.min()
    assert mu.max() <= w.max()
