import torch

from fermigates.export import hard_masked_state_dict, pruning_report, to_hard_masked_model
from fermigates.metrics import MetricsTracker, collect_gate_metrics, free_energy_components
from fermigates.models import FermiMLPClassifier


def test_hard_masked_state_dict_is_non_destructive():
    torch.manual_seed(0)
    model = FermiMLPClassifier(input_dim=12, hidden_dims=(8,), num_classes=3)
    original = model.layers[0].linear.weight.detach().clone()

    state = hard_masked_state_dict(model, threshold=0.5)
    key = "layers.0.linear.weight"
    mask = (model.layers[0].gate_probabilities() >= 0.5).to(dtype=original.dtype)
    expected = original * mask

    assert key in state
    assert torch.allclose(state[key], expected)
    assert torch.allclose(model.layers[0].linear.weight, original)


def test_to_hard_masked_model_and_pruning_report():
    torch.manual_seed(1)
    model = FermiMLPClassifier(input_dim=16, hidden_dims=(10, 6), num_classes=4)
    x = torch.randn(5, 16)

    hard_model = to_hard_masked_model(model, threshold=0.5)
    report = pruning_report(model, threshold=0.5, example_inputs=x)
    hard_report = pruning_report(hard_model, threshold=0.5, example_inputs=x)

    assert report.total_weights > 0
    assert 0.0 <= report.fraction_kept <= 1.0
    assert len(report.layers) == len(model.layers)
    assert report.dense_macs >= report.kept_macs
    assert hard_report.saved_macs_fraction >= 0.0


def test_base_model_export_and_metrics_helpers():
    model = FermiMLPClassifier(input_dim=8, hidden_dims=(6,), num_classes=2)
    x = torch.randn(3, 8)

    metrics = model.collect_gate_metrics(threshold=0.5)
    report = model.pruning_report(threshold=0.5, example_inputs=x)
    hard_model = model.to_hard_masked_model(threshold=0.5)
    hard_state = model.hard_masked_state_dict(threshold=0.5)

    assert metrics.total > 0
    assert report.total_weights > 0
    assert isinstance(hard_model, FermiMLPClassifier)
    assert "layers.0.linear.weight" in hard_state


def test_collect_gate_metrics_and_tracker_history():
    model = FermiMLPClassifier(input_dim=10, hidden_dims=(6,), num_classes=2)
    snapshot = collect_gate_metrics(model, threshold=0.5)

    assert snapshot.total > 0
    assert 0.0 <= snapshot.fraction_kept <= 1.0
    assert len(snapshot.layers) == len(model.layers)

    tracker = MetricsTracker()
    tracker.log_gate_metrics(step=1, snapshot=snapshot, prefix="occ")
    tracker.log(step=1, loss=1.23)
    latest = tracker.latest()

    assert "step" in latest
    assert tracker.history("loss") == [1.23]
    assert len(tracker.as_dict()["step"]) == 2


def test_free_energy_components_matches_manual():
    probs = torch.tensor([0.2, 0.7], dtype=torch.float32)
    energies = torch.tensor([0.5, 1.2], dtype=torch.float32)
    interaction = torch.tensor([0.3, -0.1], dtype=torch.float32)
    temperature = 0.9

    comp = free_energy_components(
        probabilities=probs,
        energies=energies,
        interaction=interaction,
        temperature=temperature,
    )
    entropy = -(
        probs * torch.log(probs.clamp(1e-8, 1.0 - 1e-8))
        + (1.0 - probs) * torch.log((1.0 - probs).clamp(1e-8, 1.0 - 1e-8))
    ).sum()
    expected = (probs * energies).sum() + (0.5 * probs * interaction).sum()
    expected = expected - (temperature * entropy)
    assert torch.allclose(torch.tensor(comp.free_energy), expected, atol=1e-6)
