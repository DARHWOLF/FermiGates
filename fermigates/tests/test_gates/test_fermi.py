import math

import pytest
import torch

from fermigates.gates.fermi import FermiGate


def test_forward_matches_expected_sigmoid_values():
    gate = FermiGate(shape=(2, 2), init_mu=1.0, init_temperature=2.0, annealer=None)
    x = torch.tensor([[1.0, 3.0], [-1.0, 5.0]], dtype=torch.float32)

    out = gate(x)
    expected = torch.sigmoid((x - 1.0) / 2.0)

    assert torch.allclose(out, expected)
    assert out.shape == x.shape
    assert torch.all((out >= 0.0) & (out <= 1.0))


def test_forward_is_differentiable():
    gate = FermiGate(shape=(2, 2), init_mu=0.0, init_temperature=1.0)

    x = torch.randn(2, 2, requires_grad=True)

    out = gate(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert gate.mu.grad is not None


def test_shape_mismatch_raises_error():
    gate = FermiGate(shape=(2, 2))

    x = torch.randn(3, 3)

    with pytest.raises(RuntimeError):
        gate(x)

def test_linear_annealing_uses_default_configuration():
    gate = FermiGate(shape=(2, 2), init_temperature=2.0, annealer="linear")

    gate.annealing_step(step=1000)

    expected_temperature = max(0.02, 2.0 - (2.0 - 0.02) * (1000 / 10000))
    assert gate.temperature == pytest.approx(expected_temperature)
    assert gate.annealing_config["T0"] == pytest.approx(2.0)
    assert gate.annealing_config["T_min"] == pytest.approx(0.02)
    assert gate.annealing_config["total_steps"] == 10000
    assert gate._step == 1000


def test_linear_annealing_respects_configure_overrides():
    gate = FermiGate(shape=(1,), init_temperature=5.0, annealer="linear")
    gate.configure(T0=3.0, T_min=0.5, total_steps=20)

    gate.annealing_step(step=10)

    expected_temperature = max(0.5, 3.0 - (3.0 - 0.5) * (10 / 20))
    assert gate.temperature == pytest.approx(expected_temperature)


def test_exponential_annealing_updates_temperature():
    gate = FermiGate(shape=(1,), init_temperature=4.0, annealer="exponential")
    gate.configure(decay=0.8)

    gate.annealing_step(step=3)

    assert gate.temperature == pytest.approx(4.0 * (0.8**3))


def test_cosine_annealing_reaches_t_min_at_total_steps():
    gate = FermiGate(shape=(1,), init_temperature=2.0, annealer="cosine")
    gate.configure(total_steps=20)

    gate.annealing_step(step=20)

    assert gate.temperature == pytest.approx(0.02)


def test_unknown_annealer_raises_value_error():
    gate = FermiGate(shape=(1,), init_temperature=1.0, annealer="invalid")

    with pytest.raises(ValueError, match="Unknown annealer type"):
        gate.annealing_step(step=1)


@torch.no_grad()
def test_initialize_mu_from_reference_global_percentile():
    gate = FermiGate(shape=(2, 3), init_mu=0.0, annealer=None)
    reference = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])

    gate.initialize_mu_from_reference(reference, percentile=0.5, per_output=False)

    expected = torch.full_like(reference, torch.quantile(reference, 0.5))
    assert torch.allclose(gate.mu, expected)


@torch.no_grad()
def test_initialize_mu_from_reference_per_output_percentile():
    gate = FermiGate(shape=(2, 3), init_mu=0.0, annealer=None)
    reference = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])

    gate.initialize_mu_from_reference(reference, percentile=0.5, per_output=True)

    expected = torch.tensor([[2.0, 2.0, 2.0], [20.0, 20.0, 20.0]])
    assert torch.allclose(gate.mu, expected)


def test_hard_mask_returns_binary_float_tensor():
    gate = FermiGate(shape=(2, 2), init_mu=0.0, init_temperature=1.0, annealer=None)
    x = torch.tensor([[-2.0, 0.0], [2.0, 5.0]])

    mask = gate.hard_mask(x, threshold=0.5)

    assert mask.dtype == torch.float32
    assert torch.all((mask == 0.0) | (mask == 1.0))
    assert torch.equal(mask, torch.tensor([[0.0, 1.0], [1.0, 1.0]]))


def test_hard_mask_threshold_edge():
    gate = FermiGate(shape=(1,), init_mu=0.0, init_temperature=1.0)

    x = torch.tensor([0.0])  # sigmoid(0)=0.5

    mask = gate.hard_mask(x, threshold=0.5)

    assert mask.item() == 1.0  # >= threshold


def test_annealing_initialization_runs_once():
    gate = FermiGate(shape=(1,), init_temperature=2.0, annealer="linear")

    gate.annealing_step(step=1)
    first_config = dict(gate.annealing_config)

    gate.annealing_step(step=2)
    second_config = dict(gate.annealing_config)

    assert first_config == second_config

def test_fermi_gate_range_and_temperature():
    gate = FermiGate(shape=(4, 5), init_mu=0.0, init_temperature=1.0)
    w = torch.randn(4, 5)

    p1 = gate(w)
    gate.set_temperature(0.25)
    p2 = gate(w)

    assert p1.shape == w.shape
    assert torch.all((p1 >= 0.0) & (p1 <= 1.0))
    assert gate.temperature == 0.25
    assert not torch.allclose(p1, p2)


def test_weight_mode_matches_elementwise_shape_behavior():
    gate = FermiGate(shape=(3, 4), mode="weight", init_mu=0.0, init_temperature=1.0, annealer=None)
    x = torch.randn(3, 4)

    out = gate(x)

    assert out.shape == x.shape
    assert gate.mu.shape == x.shape
    assert torch.all((out >= 0.0) & (out <= 1.0))
