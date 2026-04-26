def _linear_annealing(T0: float, T_min: float, total_steps: int, step: int) -> float:
    return max(T_min, T0 - (T0 - T_min) * (step / total_steps))


def _exponential_annealing(T0: float, decay: float, step: int) -> float:
    return T0 * (decay ** step)


def _cosine_annealing(T0: float, T_min: float, total_steps: int, step: int) -> float:
    import math
    return T_min + 0.5 * (T0 - T_min) * (1 + math.cos(math.pi * step / total_steps))