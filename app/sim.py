from __future__ import annotations

import numpy as np
import pandas as pd

from app.models import SimulationConfig


def simulate_kelly_paths(config: SimulationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Kelly-fraction simulations and return (selected_fraction_paths, averaged_paths)."""
    monthly_rf = config.risk_free_rate / 12.0

    grouped_frames: list[pd.DataFrame] = []
    selected_fraction_paths = pd.DataFrame()

    for u in range(config.n_steps + 1):
        kelly_fraction = u / config.n_steps
        sim_frames: list[pd.DataFrame] = []

        for i in range(config.n_simulations):
            rvs = np.random.normal(loc=config.mu, scale=config.sigma, size=config.n_months)
            rvs = np.clip(rvs, -0.99, 0.99)

            wealth = np.ones(config.n_months, dtype=float)
            for j in range(1, config.n_months):
                port_ret = (kelly_fraction * (rvs[j] - monthly_rf)) + (1 + monthly_rf)
                wealth[j] = wealth[j - 1] * port_ret

            dfa = pd.DataFrame(
                {
                    "returns": rvs,
                    "wealth": wealth,
                    "month": np.arange(1, config.n_months + 1),
                    "sim": i,
                    "kelly": kelly_fraction,
                    "log_wealth": np.log(wealth),
                }
            )
            sim_frames.append(dfa)

        df_concat = pd.concat(sim_frames, ignore_index=True)
        grouped_frames.append(df_concat.groupby(["kelly", "month"], as_index=False).mean(numeric_only=True))

        if u == config.selected_idx:
            selected_fraction_paths = df_concat.copy()

    averaged = pd.concat(grouped_frames, ignore_index=True)
    return selected_fraction_paths, averaged
