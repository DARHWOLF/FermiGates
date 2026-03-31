"""End-to-end MLP demo with Fermi losses, schedules, budget control, metrics, and export."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from fermigates.export import pruning_report, to_hard_masked_model
from fermigates.losses import (
    budget_penalty_loss,
    fermi_free_energy_loss,
    group_sparsity_l21_loss,
    hoyer_sparsity_loss,
    kl_to_bernoulli_prior_loss,
    sparsity_l1_loss,
)
from fermigates.metrics import MetricsTracker, collect_gate_metrics
from fermigates.models import FermiMLPClassifier
from fermigates.training import AdaptiveBudgetController, AnnealingSchedule, FermiAnnealingPlan


def make_dataset(n: int = 384, d_in: int = 32, num_classes: int = 4):
    x = torch.randn(n, d_in)
    w = torch.randn(d_in, num_classes)
    logits = x @ w
    y = logits.argmax(dim=1)
    return x, y


def run_demo(epochs: int = 6, batch_size: int = 64, seed: int = 7):
    torch.manual_seed(seed)
    x, y = make_dataset()
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    model = FermiMLPClassifier(input_dim=32, hidden_dims=(64, 32), num_classes=4, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    plan = FermiAnnealingPlan(
        temperature=AnnealingSchedule(start=1.3, end=0.35, total_steps=epochs, mode="cosine"),
        lambda_free_energy=AnnealingSchedule(
            start=0.0,
            end=1e-4,
            total_steps=epochs,
            mode="linear",
        ),
        budget_target=AnnealingSchedule(start=0.9, end=0.55, total_steps=epochs, mode="linear"),
    )
    controller = AdaptiveBudgetController(
        target_fraction_kept=0.9,
        lambda_budget=1e-3,
        gain=0.2,
        ema_beta=0.8,
        max_lambda=2.0,
    )
    tracker = MetricsTracker()

    for epoch in range(1, epochs + 1):
        state = plan.value(step=epoch)
        model.set_temperature(state.temperature)
        controller.set_target(state.budget_target)

        epoch_loss = 0.0
        correct = 0
        seen = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            task = F.cross_entropy(logits, yb)

            free_energy = torch.zeros((), dtype=task.dtype, device=task.device)
            sparsity = torch.zeros_like(free_energy)
            budget = torch.zeros_like(free_energy)
            prior = torch.zeros_like(free_energy)
            group = torch.zeros_like(free_energy)
            hoyer = torch.zeros_like(free_energy)
            for layer in model.layers:
                probs = layer.gate_probabilities()
                energies = layer.linear.weight.abs()
                free_energy = free_energy + fermi_free_energy_loss(
                    probs,
                    energies,
                    temperature=state.temperature,
                )
                sparsity = sparsity + sparsity_l1_loss(probs)
                budget = budget + budget_penalty_loss(
                    probs,
                    target=state.budget_target,
                    target_is_fraction=True,
                )
                prior = prior + kl_to_bernoulli_prior_loss(probs, prior_prob=state.budget_target)
                group = group + group_sparsity_l21_loss(probs, group_dim=0)
                hoyer = hoyer + hoyer_sparsity_loss(probs, normalized=True)

            loss = (
                task
                + (state.lambda_free_energy * free_energy)
                + (controller.lambda_budget * budget)
                + (1e-6 * sparsity)
                + (1e-6 * prior)
                + (1e-6 * group)
                + (1e-6 * hoyer)
            )
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            seen += int(yb.numel())

        snapshot = collect_gate_metrics(model, threshold=0.5)
        controller.update(snapshot.fraction_kept)
        tracker.log_gate_metrics(step=epoch, snapshot=snapshot, prefix="occupancy")
        tracker.log(
            step=epoch,
            loss=epoch_loss / len(loader),
            acc=float(correct) / float(seen),
            temperature=state.temperature,
            lambda_free_energy=state.lambda_free_energy,
            lambda_budget=controller.lambda_budget,
            budget_target=state.budget_target,
        )
        print(
            f"epoch={epoch:02d} loss={epoch_loss/len(loader):.4f} "
            f"acc={correct/seen:.3f} frac_kept={snapshot.fraction_kept:.3f} "
            f"lambda_budget={controller.lambda_budget:.4f}"
        )

    report = pruning_report(model, threshold=0.5, example_inputs=x[:32])
    hard_model = to_hard_masked_model(model, threshold=0.5)
    hard_report = pruning_report(hard_model, threshold=0.5, example_inputs=x[:32])
    print(
        f"soft_kept={report.kept_weights}/{report.total_weights} "
        f"hard_kept={hard_report.kept_weights}/{hard_report.total_weights} "
        f"saved_macs={hard_report.saved_macs_fraction:.2%}"
    )
    return model, tracker, report


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
