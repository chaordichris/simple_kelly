from dataclasses import dataclass


@dataclass(frozen=True)
class KellyInputs:
    prob_success: float
    gain_multiplier: float
    loss_multiplier: float
    fractional_kelly: float = 1.0
    n_periods: int = 50
    initial_wealth: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.prob_success <= 1.0:
            raise ValueError("prob_success must be in [0, 1]")
        if self.gain_multiplier <= 0.0 or self.loss_multiplier <= 0.0:
            raise ValueError("gain_multiplier and loss_multiplier must be > 0")
        if not 0.0 <= self.fractional_kelly <= 1.0:
            raise ValueError("fractional_kelly must be in [0, 1]")
        if self.n_periods < 2:
            raise ValueError("n_periods must be >= 2")
        if self.initial_wealth <= 0.0:
            raise ValueError("initial_wealth must be > 0")


@dataclass(frozen=True)
class SimulationConfig:
    mu: float
    sigma: float
    risk_free_rate: float
    n_simulations: int
    n_months: int
    n_steps: int
    selected_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError("sigma must be > 0")
        if self.n_simulations < 1:
            raise ValueError("n_simulations must be >= 1")
        if self.n_months < 2:
            raise ValueError("n_months must be >= 2")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if not 0.0 <= self.selected_fraction <= 1.0:
            raise ValueError("selected_fraction must be in [0, 1]")

    @property
    def selected_idx(self) -> int:
        return int(round(self.selected_fraction * self.n_steps))
