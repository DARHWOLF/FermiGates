import pytest
import torch

from fermigates.losses import (
    binary_entropy_loss,
    budget_penalty_loss,
    consistency_loss,
    fermi_free_energy_loss,
    fermi_informed_loss,
    sparsity_l1_loss,
)


def test_binary_entropy_loss_matches_manual_sum():
    probs = torch.tensor([0.2, 0.8], dtype=torch.float32)
    expected = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs)).sum()
    loss = binary_entropy_loss(probs, reduction="sum")
    assert torch.allclose(loss, expected)


def test_fermi_free_energy_loss_matches_manual_formula():
    probs = torch.tensor([0.2, 0.7], dtype=torch.float32)
    energies = torch.tensor([0.6, 1.3], dtype=torch.float32)
    interaction = torch.tensor([0.1, -0.2], dtype=torch.float32)
    temperature = torch.tensor([0.8, 1.2], dtype=torch.float32)

    entropy = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs))
    expected = (
        (probs * energies) + (0.5 * probs * interaction) - (temperature.mean() * entropy)
    ).sum()
    loss = fermi_free_energy_loss(probs, energies, interaction=interaction, temperature=temperature)
    assert torch.allclose(loss, expected)


def test_fermi_free_energy_loss_shape_validation():
    probs = torch.rand(3)
    energies = torch.rand(4)
    with pytest.raises(ValueError):
        fermi_free_energy_loss(probs, energies)


def test_sparsity_and_budget_losses():
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    assert torch.allclose(sparsity_l1_loss(probs), probs.sum())

    budget_fraction = budget_penalty_loss(probs, target=0.5, target_is_fraction=True)
    expected_fraction = (probs.sum() - 2.0).pow(2)
    assert torch.allclose(budget_fraction, expected_fraction)

    budget_absolute = budget_penalty_loss(probs, target=1.5, target_is_fraction=False)
    expected_absolute = (probs.sum() - 1.5).pow(2)
    assert torch.allclose(budget_absolute, expected_absolute)


def test_consistency_loss_zero_for_identical_probabilities():
    probs = torch.tensor([0.3, 0.6, 0.9], dtype=torch.float32)
    loss = consistency_loss(probs, probs, norm="l2", reduction="mean")
    assert torch.allclose(loss, torch.tensor(0.0))


def test_fermi_informed_loss_requires_optional_inputs():
    probs = torch.rand(5)
    energies = torch.rand(5)
    task = torch.tensor(0.5)

    with pytest.raises(ValueError):
        fermi_informed_loss(
            task_loss=task,
            probabilities=probs,
            energies=energies,
            lambda_budget=1.0,
        )

    with pytest.raises(ValueError):
        fermi_informed_loss(
            task_loss=task,
            probabilities=probs,
            energies=energies,
            lambda_consistency=1.0,
        )


def test_fermi_informed_loss_composition_and_backward():
    probs = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float32, requires_grad=True)
    energies = torch.tensor([0.3, 0.7, 1.1], dtype=torch.float32, requires_grad=True)
    interaction = torch.tensor([0.2, -0.1, 0.05], dtype=torch.float32)
    prev = torch.tensor([0.1, 0.55, 0.75], dtype=torch.float32)
    task_loss = (probs.square().mean() + 0.2 * energies.mean())

    terms = fermi_informed_loss(
        task_loss=task_loss,
        probabilities=probs,
        energies=energies,
        interaction=interaction,
        temperature=0.9,
        lambda_free_energy=1e-2,
        lambda_sparsity=1e-3,
        lambda_budget=2e-3,
        budget_target=0.6,
        budget_target_is_fraction=True,
        lambda_consistency=5e-3,
        previous_probabilities=prev,
    )

    expected_total = (
        terms.task
        + 1e-2 * terms.free_energy
        + 1e-3 * terms.sparsity
        + 2e-3 * terms.budget
        + 5e-3 * terms.consistency
    )
    assert torch.allclose(terms.total, expected_total)

    terms.total.backward()
    assert probs.grad is not None
    assert energies.grad is not None
