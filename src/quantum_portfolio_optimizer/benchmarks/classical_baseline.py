"""Classical mean-variance baseline for benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import optimize


@dataclass
class BaselineResult:
    allocations: np.ndarray
    expected_return: float
    variance: float
    sharpe_ratio: float
    success: bool
    message: str


def markowitz_baseline(
    expected_returns: Sequence[float],
    covariance: Sequence[Sequence[float]],
    budget: float = 1.0,
    bounds: Tuple[float, float] = (0.0, 1.0),
    risk_aversion: float = 0.5,
    risk_free_rate: float = 0.0,
    tol: float = 1e-9,
    maxiter: int = 500,
) -> BaselineResult:
    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    num_assets = mu.size

    if sigma.shape != (num_assets, num_assets):
        raise ValueError("Covariance matrix dimensions must align with expected returns.")

    def objective(weights: np.ndarray) -> float:
        variance = weights @ sigma @ weights
        return risk_aversion * variance - mu @ weights

    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - budget},
    )
    bounds_list = [bounds] * num_assets
    initial = np.full(num_assets, budget / num_assets, dtype=float)

    result = optimize.minimize(objective, x0=initial, bounds=bounds_list, constraints=constraints, tol=tol, options={"maxiter": maxiter})
    weights = np.clip(result.x, bounds[0], bounds[1])
    returns = float(mu @ weights)
    variance = float(weights @ sigma @ weights)
    volatility = np.sqrt(max(variance, 1e-12))
    sharpe = (returns - risk_free_rate) / volatility if volatility > 0 else 0.0

    return BaselineResult(
        allocations=weights,
        expected_return=returns,
        variance=variance,
        sharpe_ratio=sharpe,
        success=result.success,
        message=result.message,
    )
