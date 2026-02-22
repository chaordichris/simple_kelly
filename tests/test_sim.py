import numpy as np

from app.models import SimulationConfig
from app.sim import simulate_kelly_paths


def test_simulate_kelly_paths_shape_and_columns():
    np.random.seed(42)
    per_sim, averaged = simulate_kelly_paths(SimulationConfig(
        mu=0.01,
        sigma=0.05,
        risk_free_rate=0.03,
        n_simulations=10,
        n_months=12,
        n_steps=5,
    ))

    assert per_sim is not None
    assert not per_sim.empty
    assert not averaged.empty
    assert set(["month", "sim", "kelly", "log_wealth", "wealth", "returns"]).issubset(per_sim.columns)
    assert set(["kelly", "month", "log_wealth"]).issubset(averaged.columns)


def test_simulate_kelly_kelly_grid_includes_endpoints():
    np.random.seed(7)
    _, averaged = simulate_kelly_paths(SimulationConfig(
        mu=0.01,
        sigma=0.04,
        risk_free_rate=0.02,
        n_simulations=5,
        n_months=8,
        n_steps=4,
    ))

    unique_kelly = sorted(averaged["kelly"].unique().tolist())
    assert unique_kelly[0] == 0.0
    assert unique_kelly[-1] == 1.0
