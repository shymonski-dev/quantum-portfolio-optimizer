"""Generate sample datasets for unit tests and examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SyntheticDataset:
    prices: pd.DataFrame
    returns: pd.DataFrame


def generate_synthetic_dataset(
    num_assets: int = 3,
    num_points: int = 32,
    mu: float = 0.001,
    sigma: float = 0.01,
    seed: Optional[int] = None,
) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=num_points, freq="D")
    shocks = rng.normal(loc=mu, scale=sigma, size=(num_points, num_assets))
    prices = 100 * np.exp(np.cumsum(shocks, axis=0))
    cols = [f"Asset_{i}" for i in range(num_assets)]
    df_prices = pd.DataFrame(prices, index=dates, columns=cols)
    returns = df_prices.pct_change().dropna()
    return SyntheticDataset(prices=df_prices, returns=returns)
