import math

from fermigates.training import AdaptiveBudgetController, AnnealingSchedule, FermiAnnealingPlan


def test_annealing_schedule_linear_boundaries_and_midpoint():
    schedule = AnnealingSchedule(start=1.0, end=0.2, total_steps=10, mode="linear")
    assert schedule.value(0) == 1.0
    assert schedule.value(10) == 0.2
    assert math.isclose(schedule.value(5), 0.6, rel_tol=1e-6)


def test_annealing_schedule_cosine_and_exponential():
    cosine = AnnealingSchedule(start=1.0, end=0.0, total_steps=8, mode="cosine")
    exp = AnnealingSchedule(start=1.0, end=0.125, total_steps=3, mode="exponential")

    assert cosine.value(0) == 1.0
    assert 0.0 < cosine.value(4) < 1.0
    assert cosine.value(8) == 0.0

    assert math.isclose(exp.value(0), 1.0, rel_tol=1e-6)
    assert math.isclose(exp.value(3), 0.125, rel_tol=1e-6)
    assert math.isclose(exp.value(1), 0.5, rel_tol=1e-6)


def test_fermi_annealing_plan_value_bundle():
    plan = FermiAnnealingPlan(
        temperature=AnnealingSchedule(start=1.0, end=0.5, total_steps=10),
        lambda_free_energy=AnnealingSchedule(start=0.0, end=0.1, total_steps=10),
        budget_target=AnnealingSchedule(start=0.9, end=0.6, total_steps=10),
    )
    state = plan.value(step=5)
    assert 0.5 < state.temperature < 1.0
    assert 0.0 < state.lambda_free_energy < 0.1
    assert 0.6 < state.budget_target < 0.9


def test_adaptive_budget_controller_moves_lambda_toward_target():
    controller = AdaptiveBudgetController(
        target_fraction_kept=0.5,
        lambda_budget=0.1,
        gain=0.5,
        ema_beta=0.0,
        min_lambda=0.0,
        max_lambda=1.0,
    )

    # Model is too dense -> increase budget penalty.
    increased = controller.update(0.8)
    assert increased > 0.1

    # Model is too sparse -> decrease budget penalty.
    decreased = controller.update(0.2)
    assert decreased < increased
