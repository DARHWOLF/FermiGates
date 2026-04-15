import torch

from fermigates.gates import (
    BinaryConcreteGate,
    FermiGate,
    GompertzGate,
    GroupLassoGate,
    HardConcreteGate,
    MagnitudeGate,
)


def test_fermi_gate_range_and_temperature():
    gate = FermiGate(shape=(4, 5), init_mu=0.0, init_T=1.0)
    w = torch.randn(4, 5)

    p1 = gate(w)
    gate.set_temperature(0.25)
    p2 = gate(w)

    assert p1.shape == w.shape
    assert torch.all((p1 >= 0.0) & (p1 <= 1.0))
    assert gate.temperature == 0.25
    assert not torch.allclose(p1, p2)


def test_binary_concrete_gate_sampling():
    gate = BinaryConcreteGate(shape=(8,), init_log_alpha=0.0, init_T=0.5)
    deterministic = gate.probabilities()
    sampled = gate(sample=True)

    assert deterministic.shape == sampled.shape == (8,)
    assert torch.all((deterministic >= 0.0) & (deterministic <= 1.0))
    assert torch.all((sampled >= 0.0) & (sampled <= 1.0))


def test_hard_concrete_range():
    gate = HardConcreteGate(shape=(10,), init_log_alpha=1.0)
    p = gate.probabilities()
    z = gate(sample=True)

    assert torch.all((p >= 0.0) & (p <= 1.0))
    assert torch.all((z >= 0.0) & (z <= 1.0))


def test_magnitude_gate_normalization():
    scores = torch.tensor([0.0, 1.0, 2.0, 3.0])
    gate = MagnitudeGate(shape=scores.shape, scores=scores)
    p = gate.probabilities()

    assert torch.allclose(p, torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]))


def test_group_lasso_gate_repeat():
    gate = GroupLassoGate(groups=3, group_size=2, init=0.0)
    p = gate.probabilities()

    assert p.shape == (6,)
    assert torch.allclose(p[0:2], p[0].repeat(2))
    assert torch.allclose(p[2:4], p[2].repeat(2))


def test_gompertz_gate_range():
    gate = GompertzGate(size=5, alpha=2.0, beta=1.0)
    p = gate.probabilities()

    assert p.shape == (5,)
    assert torch.all((p > 0.0) & (p <= 1.0))
