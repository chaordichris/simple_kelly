import pytest

from app.models import KellyInputs, SimulationConfig


def test_kelly_inputs_validation():
    with pytest.raises(ValueError):
        KellyInputs(prob_success=1.2, gain_multiplier=1.0, loss_multiplier=0.5)

    with pytest.raises(ValueError):
        KellyInputs(prob_success=0.5, gain_multiplier=1.0, loss_multiplier=0.5, fractional_kelly=1.5)


def test_simulation_config_validation_and_selected_idx():
    cfg = SimulationConfig(mu=0.01, sigma=0.05, risk_free_rate=0.03, n_simulations=10, n_months=12, n_steps=20, selected_fraction=0.5)
    assert cfg.selected_idx == 10

    with pytest.raises(ValueError):
        SimulationConfig(mu=0.01, sigma=0.0, risk_free_rate=0.03, n_simulations=10, n_months=12, n_steps=20)
