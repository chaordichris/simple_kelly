import numpy as np
import pytest

from app.kelly import (
    bounded_kelly_fraction,
    expected_log_growth,
    kelly_fraction_closed_form,
    simulate_binary_wealth_paths,
 )
from app.models import KellyInputs


def test_closed_form_matches_known_value():
    p = 0.60
    b = 1.5
    a = 0.5
    expected = (p / a) - ((1 - p) / b)
    assert kelly_fraction_closed_form(p, b, a) == pytest.approx(expected)


def test_bounded_kelly_clips_to_unit_interval():
    assert bounded_kelly_fraction(0.20, 0.2, 1.0) == 0.0
    assert bounded_kelly_fraction(0.95, 3.0, 0.05) == 1.0


def test_expected_log_growth_is_maximized_near_closed_form():
    p, b, a = 0.55, 1.2, 0.8
    f_star = bounded_kelly_fraction(p, b, a)
    grid = np.linspace(0.0, 1.0, 1001)
    growth = expected_log_growth(grid, p, b, a)
    f_grid = grid[np.argmax(growth)]
    assert abs(f_grid - f_star) < 0.02


def test_simulate_binary_wealth_paths_output_shapes():
    np.random.seed(123)
    n_periods = 20
    inputs = KellyInputs(
        prob_success=0.5,
        gain_multiplier=1.0,
        loss_multiplier=0.5,
        n_periods=n_periods,
        fractional_kelly=0.5,
    )
    all_in, kelly, frac_kelly, k = simulate_binary_wealth_paths(inputs)

    assert len(all_in) == n_periods
    assert len(kelly) == n_periods
    assert len(frac_kelly) == n_periods
    assert 0.0 <= k <= 1.0
