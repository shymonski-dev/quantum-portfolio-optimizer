"""Classical mean-variance baseline for benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
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


def mip_baseline(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    budget: float,
    num_assets: int,
    bounds: tuple = (0.0, 1.0),
    risk_aversion: float = 0.5,
    risk_free_rate: float = 0.0,
) -> BaselineResult:
    """Integer portfolio optimization selecting exactly num_assets from n.

    For n_assets <= 15: exhaustive enumeration over all C(n, num_assets) subsets,
    with QP weight optimization for each. Returns the globally optimal solution.

    For n_assets > 15: greedy selection by individual Sharpe ratio, then QP for
    weights. Not globally optimal but computationally tractable.

    Uses only scipy (already a core dependency), no additional packages needed.

    Args:
        expected_returns: Array of shape (n_assets,) with expected returns
        covariance: Positive semi-definite covariance matrix (n_assets x n_assets)
        budget: Total portfolio weight sum (typically 1.0)
        num_assets: Exactly this many assets to select (K)
        bounds: (min_weight, max_weight) per asset when selected
        risk_aversion: Risk aversion parameter (0 = return only, 1 = risk only)
        risk_free_rate: Risk-free rate for Sharpe ratio calculation

    Returns:
        BaselineResult with optimal allocations, metrics, and solver info
    """
    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    n = len(mu)

    if sigma.shape != (n, n):
        raise ValueError("Covariance matrix dimensions must align with expected returns.")
    if num_assets > n:
        raise ValueError(
            f"num_assets ({num_assets}) must be <= n_assets ({n})."
        )
    if num_assets < 1:
        raise ValueError("num_assets must be >= 1.")

    def _solve_subset_qp(selected_indices: list) -> Optional[np.ndarray]:
        k = len(selected_indices)
        mu_sub = mu[selected_indices]
        sigma_sub = sigma[np.ix_(selected_indices, selected_indices)]

        def objective(w):
            return risk_aversion * (w @ sigma_sub @ w) - (1 - risk_aversion) * (mu_sub @ w)

        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - budget},)
        bounds_list = [(bounds[0], bounds[1])] * k
        x0 = np.full(k, budget / k, dtype=float)

        result = optimize.minimize(
            objective, x0=x0, bounds=bounds_list,
            constraints=constraints, method="SLSQP",
            options={"maxiter": 500},
        )
        if not result.success:
            return None

        full_weights = np.zeros(n)
        for i, idx in enumerate(selected_indices):
            full_weights[idx] = np.clip(result.x[i], bounds[0], bounds[1])
        return full_weights

    def _portfolio_cost(weights: np.ndarray) -> float:
        return risk_aversion * (weights @ sigma @ weights) - (1 - risk_aversion) * (mu @ weights)

    ENUMERATE_THRESHOLD = 15

    if n <= ENUMERATE_THRESHOLD:
        best_cost = np.inf
        best_weights = None
        solver_msg = (
            f"Exhaustive enumeration over C({n},{num_assets})="
            f"{comb(n, num_assets)} subsets"
        )

        for selected in combinations(range(n), num_assets):
            weights = _solve_subset_qp(list(selected))
            if weights is not None:
                cost = _portfolio_cost(weights)
                if cost < best_cost:
                    best_cost = cost
                    best_weights = weights

        greedy_fallback = False
    else:
        individual_sharpe = (mu - risk_free_rate) / np.sqrt(np.diag(sigma) + 1e-10)
        selected = np.argsort(individual_sharpe)[-num_assets:].tolist()
        best_weights = _solve_subset_qp(selected)
        solver_msg = f"Greedy selection (n={n} > {ENUMERATE_THRESHOLD} threshold)"
        greedy_fallback = True

    if best_weights is None:
        return BaselineResult(
            allocations=np.zeros(n),
            expected_return=0.0,
            variance=0.0,
            sharpe_ratio=0.0,
            success=False,
            message=f"Optimization failed. {solver_msg}",
        )

    ret = float(mu @ best_weights)
    var = float(best_weights @ sigma @ best_weights)
    vol = np.sqrt(max(var, 1e-12))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0

    msg = solver_msg
    if greedy_fallback:
        msg += " (greedy heuristic, not globally optimal)"

    return BaselineResult(
        allocations=best_weights,
        expected_return=ret,
        variance=var,
        sharpe_ratio=sharpe,
        success=True,
        message=msg,
    )
