import torch

from fermigates.layers import (
    FermiGatedConv2d,
    FermiGatedLinear,
    FermiMLPBlock,
    FermiTransformerEncoderLayer,
)


def test_fermi_gated_linear_forward_and_prune():
    layer = FermiGatedLinear(12, 7, init_mu=0.0, init_T=1.0)
    x = torch.randn(5, 12)

    y, p = layer(x)
    dense = layer.hard_pruned_linear(threshold=0.5)

    assert y.shape == (5, 7)
    assert p.shape == layer.linear.weight.shape
    assert dense.weight.shape == layer.linear.weight.shape


def test_initialize_mu_from_weight_percentile():
    torch.manual_seed(42)
    layer = FermiGatedLinear(10, 6)
    layer.initialize_mu_from_weight_percentile(percentile=0.5)

    mu = layer.mask.mu
    weight_mag = layer.linear.weight.abs()
    assert mu.min() >= weight_mag.min() - 1e-6
    assert mu.max() <= weight_mag.max() + 1e-6


def test_fermi_gated_conv2d_shape():
    layer = FermiGatedConv2d(3, 8, kernel_size=3, padding=1)
    x = torch.randn(4, 3, 16, 16)

    y, p = layer(x)
    dense = layer.hard_pruned_conv2d(threshold=0.5)

    assert y.shape == (4, 8, 16, 16)
    assert p.shape == layer.conv.weight.shape
    assert dense.weight.shape == layer.conv.weight.shape


def test_fermi_mlp_block_shape():
    block = FermiMLPBlock(d_in=32, d_hidden=64, d_out=16, dropout=0.1)
    x = torch.randn(6, 32)

    y, masks = block(x)

    assert y.shape == (6, 16)
    assert set(masks.keys()) == {"fc1", "fc2"}


def test_fermi_transformer_encoder_layer_shape():
    layer = FermiTransformerEncoderLayer(d_model=24, nhead=4, dim_feedforward=48)
    x = torch.randn(2, 10, 24)

    y, masks = layer(x)

    assert y.shape == x.shape
    assert set(masks.keys()) == {"ffn1", "ffn2"}
