from __future__ import annotations

import numpy as np

from app.models import KellyInputs


def kelly_fraction_closed_form(prob_success: float, gain_multiplier: float, loss_multiplier: float) -> float:
    """Closed-form Kelly fraction: p/a - (1-p)/b."""
    if not 0 <= prob_success <= 1:
        raise ValueError("prob_success must be in [0, 1]")
    if gain_multiplier <= 0 or loss_multiplier <= 0:
        raise ValueError("gain_multiplier and loss_multiplier must be > 0")
    return (prob_success / loss_multiplier) - ((1.0 - prob_success) / gain_multiplier)


def bounded_kelly_fraction(prob_success: float, gain_multiplier: float, loss_multiplier: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Kelly fraction clipped to practical bounds (default [0, 1])."""
    if lower > upper:
        raise ValueError("lower must be <= upper")
    f_star = kelly_fraction_closed_form(prob_success, gain_multiplier, loss_multiplier)
    return float(np.clip(f_star, lower, upper))


def expected_log_growth(fraction: np.ndarray, prob_success: float, gain_multiplier: float, loss_multiplier: float) -> np.ndarray:
    """Expected log growth for fraction(s) under binary payoff assumptions."""
    fraction = np.asarray(fraction)
    if np.any(1 + gain_multiplier * fraction <= 0) or np.any(1 - loss_multiplier * fraction <= 0):
        raise ValueError("fraction causes invalid log arguments")
    return (prob_success * np.log(1 + gain_multiplier * fraction)) + ((1.0 - prob_success) * np.log(1 - loss_multiplier * fraction))


def simulate_binary_wealth_paths(inputs: KellyInputs) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return wealth paths for all-in, full Kelly, and fractional Kelly allocations."""
    k = bounded_kelly_fraction(inputs.prob_success, inputs.gain_multiplier, inputs.loss_multiplier)
    fk = inputs.fractional_kelly * k

    outcomes = np.random.binomial(1, inputs.prob_success, size=inputs.n_periods)
    wealth_all = np.full(inputs.n_periods, inputs.initial_wealth, dtype=float)
    wealth_k = np.full(inputs.n_periods, inputs.initial_wealth, dtype=float)
    wealth_fk = np.full(inputs.n_periods, inputs.initial_wealth, dtype=float)

    for i in range(1, inputs.n_periods):
        if outcomes[i] == 0:
            wealth_all[i] = wealth_all[i - 1] * (1 - inputs.loss_multiplier)
            wealth_k[i] = wealth_k[i - 1] * ((k * (1 - inputs.loss_multiplier)) + (1 - k))
            wealth_fk[i] = wealth_fk[i - 1] * ((fk * (1 - inputs.loss_multiplier)) + (1 - fk))
        else:
            wealth_all[i] = wealth_all[i - 1] * (1 + inputs.gain_multiplier)
            wealth_k[i] = wealth_k[i - 1] * ((k * (1 + inputs.gain_multiplier)) + (1 - k))
            wealth_fk[i] = wealth_fk[i - 1] * ((fk * (1 + inputs.gain_multiplier)) + (1 - fk))

    return wealth_all, wealth_k, wealth_fk, k
