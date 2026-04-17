import torch
import pytest

from fermigates.gates._base import BaseGate

# Dummy Gate for testing
class DummyGate(BaseGate):
    def __init__(self, annealer=None):
        super().__init__(annealer=annealer)
        self.temperature = 1.0

    def forward(self, x):
        return torch.sigmoid(x)

    def _apply_annealing(self, step: int):
        # Simple linear decay for testing
        new_temp = max(0.1, 1.0 - 0.1 * step)
        self._update_annealed_parameters(new_temp)

    def _update_annealed_parameters(self, value: float):
        self.temperature = value


def test_forward_not_implemented():
    gate = BaseGate()

    x = torch.randn(3, 3)

    with pytest.raises(NotImplementedError):
        gate.forward(x)


def test_get_probabilities_calls_forward():
    gate = DummyGate()

    x = torch.randn(2, 2)
    out1 = gate.forward(x)
    out2 = gate.get_probabilities(x)

    assert torch.allclose(out1, out2)


def test_hard_mask():
    gate = DummyGate()

    x = torch.tensor([[0.0, 2.0], [-2.0, 0.5]])
    mask = gate.hard_mask(x, threshold=0.5)

    assert mask.shape == x.shape
    assert torch.all((mask == 0) | (mask == 1))


def test_requires_step():
    gate_no_anneal = DummyGate(annealer=None)
    gate_with_anneal = DummyGate(annealer="linear")

    assert gate_no_anneal.requires_step() is False
    assert gate_with_anneal.requires_step() is True


def test_annealing_step_updates_temperature():
    gate = DummyGate(annealer="linear")

    initial_temp = gate.temperature

    gate.annealing_step(step=1)

    assert gate.temperature < initial_temp


def test_step_counter_increment():
    gate = DummyGate(annealer="linear")

    gate.annealing_step()
    gate.annealing_step()

    assert gate._step == 2


def test_configure_updates_config():
    gate = DummyGate(annealer="linear")

    gate.configure(total_steps=5000, T0=2.0)

    assert gate.annealing_config["total_steps"] == 5000
    assert gate.annealing_config["T0"] == 2.0


def test_get_gate_parameters():
    gate = DummyGate()

    params = gate.get_gate_parameters()

    assert isinstance(params, dict)


def test_get_state():
    gate = DummyGate()

    gate.annealing_step()

    state = gate.get_state()

    assert "step" in state
    assert state["step"] == 1